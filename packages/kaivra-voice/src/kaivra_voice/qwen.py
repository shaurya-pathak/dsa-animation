"""Local Qwen3-TTS provider backed by the resident TTSKit CoreML worker."""

from __future__ import annotations

import base64
import binascii
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from kaivra.audio.base import AudioResult, VoiceProvider

_PROTOCOL_PREFIX = "CALLBOX_QWEN_JSON:"
_DEFAULT_SPEAKER = "serena"
_DEFAULT_INSTRUCTION = (
    "A warm, clear American teacher explaining an idea to a curious beginner. "
    "Speak naturally at about 165 words per minute. Use short, meaningful pauses, "
    "connect phrases smoothly, and avoid an announcer-like or theatrical delivery."
)
_DEFAULT_DECODER_MODE = "throughputOptimized"
_DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
_DEFAULT_GENERATION_TIMEOUT_SECONDS = 180.0
_DEFAULT_MAX_CHARS_PER_REQUEST = 900


@dataclass(frozen=True)
class QwenTTSPaths:
    """Resolved worker and model locations for Qwen3-TTS."""

    worker_path: str
    models_path: str


@dataclass(frozen=True)
class _WorkerFailure:
    message: str


ProcessFactory = Callable[..., subprocess.Popen[bytes]]


class QwenProvider(VoiceProvider):
    """Generate narration with a persistent local Qwen3-TTS 1.7B worker."""

    def __init__(
        self,
        worker_path: str | None = None,
        models_path: str | None = None,
        voice_id: str | None = None,
        instruction: str | None = None,
        decoder_mode: str | None = None,
        startup_timeout_seconds: float | None = None,
        generation_timeout_seconds: float | None = None,
        max_chars_per_request: int | None = None,
        *,
        process_factory: ProcessFactory = subprocess.Popen,
    ) -> None:
        self.worker_path = worker_path
        self.models_path = models_path
        self.voice_id = voice_id
        self.instruction = instruction
        self.decoder_mode = decoder_mode
        self.startup_timeout_seconds = startup_timeout_seconds
        self.generation_timeout_seconds = generation_timeout_seconds
        self.max_chars_per_request = max_chars_per_request
        self._process_factory = process_factory
        self._process: subprocess.Popen[bytes] | None = None
        self._events: queue.Queue[dict[str, Any] | _WorkerFailure] = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=30)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._generation_lock = threading.Lock()
        self._active_speaker: str | None = None
        self._ready_metadata: dict[str, Any] = {}

    def generate(self, scene_id: str, text: str, **kwargs: Any) -> AudioResult:
        """Generate a PCM16 WAV for one scene while reusing the resident model."""
        if not text.strip():
            raise RuntimeError("Qwen3-TTS requires non-empty narration text.")

        requested_speaker = str(kwargs.get("voice_id") or self.voice_id or _speaker_from_env())
        with self._generation_lock:
            self._ensure_started(requested_speaker)
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError(self._worker_error("Qwen3-TTS worker is not running."))

            pcm_parts: list[bytes] = []
            sample_rate: int | None = None
            for text_chunk in _split_synthesis_text(text, self._max_chars_per_request()):
                request_id = uuid.uuid4().hex
                self._send({"id": request_id, "action": "synthesize_stream", "text": text_chunk})
                request_pcm_parts: list[bytes] = []
                expected_sequence = 0
                # Each bounded request receives a full generation window. A
                # long lesson should not fail merely because several healthy
                # chunks collectively take longer than one short-scene timeout.
                deadline = time.monotonic() + self._generation_timeout()

                while True:
                    event = self._next_event(deadline)
                    if event.get("id") != request_id:
                        continue
                    if not event.get("ok", True):
                        raise RuntimeError(
                            self._worker_error(
                                str(event.get("error") or "Qwen3-TTS synthesis failed.")
                            )
                        )

                    event_name = str(event.get("event") or "")
                    if event_name == "audio":
                        sequence = int(event.get("sequence", -1))
                        if sequence != expected_sequence:
                            raise RuntimeError(
                                "Qwen3-TTS returned out-of-order audio chunks: "
                                f"expected {expected_sequence}, received {sequence}."
                            )
                        chunk_rate = int(event.get("sample_rate", 0))
                        if chunk_rate <= 0:
                            raise RuntimeError("Qwen3-TTS returned an invalid sample rate.")
                        if sample_rate is None:
                            sample_rate = chunk_rate
                        elif sample_rate != chunk_rate:
                            raise RuntimeError("Qwen3-TTS changed sample rate within one scene.")
                        request_pcm_parts.append(_decode_pcm(event))
                        expected_sequence += 1
                        continue

                    if event_name == "complete":
                        break

                if not request_pcm_parts:
                    raise RuntimeError("Qwen3-TTS completed a text chunk without producing audio.")
                pcm_parts.extend(request_pcm_parts)

            if sample_rate is None or not pcm_parts:
                raise RuntimeError("Qwen3-TTS completed without producing audio.")

            pcm = b"".join(pcm_parts)
            output_path = _temporary_wav_path(scene_id)
            _write_pcm16_wav(output_path, pcm, sample_rate)
            return AudioResult(
                audio_path=str(output_path),
                duration_seconds=len(pcm) / 2 / sample_rate,
                scene_id=scene_id,
                cues=(),
            )

    def close(self) -> None:
        """Ask the resident worker to unload, then terminate it if necessary."""
        process = self._process
        self._process = None
        if process is None:
            return

        if process.poll() is None:
            try:
                payload = {"id": uuid.uuid4().hex, "action": "shutdown"}
                assert process.stdin is not None
                process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
                process.stdin.flush()
                process.wait(timeout=8)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)

        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def _ensure_started(self, requested_speaker: str) -> None:
        normalized_speaker = requested_speaker.strip().lower()
        if not normalized_speaker:
            raise RuntimeError("Qwen3-TTS voice_id must not be empty.")
        if self._process is not None and self._process.poll() is None:
            if self._active_speaker != normalized_speaker:
                raise RuntimeError(
                    "A Qwen3-TTS render must use one speaker for every scene. "
                    f"The worker is already using {self._active_speaker!r}, not "
                    f"{normalized_speaker!r}."
                )
            return

        resolved = resolve_qwen_tts_paths(
            worker_path=self.worker_path,
            models_path=self.models_path,
        )
        instruction = self.instruction or _instruction_from_env()
        decoder_mode = self.decoder_mode or _decoder_mode_from_env()
        if decoder_mode not in {"latencyOptimized", "throughputOptimized"}:
            raise RuntimeError(
                "KAIVRA_QWEN_TTS_DECODER_MODE must be latencyOptimized or throughputOptimized."
            )

        command = [
            resolved.worker_path,
            "--models-path",
            resolved.models_path,
            "--speaker",
            normalized_speaker,
            "--decoder-mode",
            decoder_mode,
        ]
        if instruction:
            command.extend(["--instruction", instruction])

        try:
            process = self._process_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start Qwen3-TTS worker: {exc}") from exc

        self._process = process
        self._active_speaker = normalized_speaker
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            name="kaivra-qwen-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name="kaivra-qwen-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        deadline = time.monotonic() + self._startup_timeout()
        while True:
            event = self._next_event(deadline)
            if event.get("id") != "__ready__" and event.get("event") != "ready":
                continue
            if not event.get("ok", False):
                self.close()
                raise RuntimeError(
                    self._worker_error(
                        str(event.get("error") or "Qwen3-TTS worker failed during startup.")
                    )
                )
            self._ready_metadata = event
            return

    def _read_stdout(self, stream: BinaryIO | None) -> None:
        if stream is None:
            self._events.put(_WorkerFailure("Qwen3-TTS worker stdout is unavailable."))
            return
        try:
            for raw_line in iter(stream.readline, b""):
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith(_PROTOCOL_PREFIX):
                    continue
                try:
                    message = json.loads(line[len(_PROTOCOL_PREFIX) :])
                except json.JSONDecodeError as exc:
                    self._events.put(_WorkerFailure(f"Qwen3-TTS emitted invalid JSON: {exc.msg}."))
                    return
                if not isinstance(message, dict):
                    self._events.put(_WorkerFailure("Qwen3-TTS emitted a non-object event."))
                    return
                self._events.put(message)
        finally:
            process = self._process
            code = process.poll() if process is not None else None
            self._events.put(_WorkerFailure(f"Qwen3-TTS worker exited unexpectedly ({code})."))

    def _read_stderr(self, stream: BinaryIO | None) -> None:
        if stream is None:
            return
        for raw_line in iter(stream.readline, b""):
            line = raw_line.decode(errors="replace").strip()
            if line:
                self._stderr_lines.append(line)

    def _next_event(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(self._worker_error("Qwen3-TTS timed out."))
        try:
            event = self._events.get(timeout=remaining)
        except queue.Empty as exc:
            raise RuntimeError(self._worker_error("Qwen3-TTS timed out.")) from exc
        if isinstance(event, _WorkerFailure):
            raise RuntimeError(self._worker_error(event.message))
        return event

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Qwen3-TTS worker stdin is unavailable.")
        try:
            process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(self._worker_error("Qwen3-TTS worker pipe failed.")) from exc

    def _startup_timeout(self) -> float:
        return _positive_timeout(
            self.startup_timeout_seconds,
            "KAIVRA_QWEN_TTS_STARTUP_TIMEOUT_SECONDS",
            _DEFAULT_STARTUP_TIMEOUT_SECONDS,
        )

    def _generation_timeout(self) -> float:
        return _positive_timeout(
            self.generation_timeout_seconds,
            "KAIVRA_QWEN_TTS_TIMEOUT_SECONDS",
            _DEFAULT_GENERATION_TIMEOUT_SECONDS,
        )

    def _max_chars_per_request(self) -> int:
        return _positive_int(
            self.max_chars_per_request,
            "KAIVRA_QWEN_TTS_MAX_CHARS",
            _DEFAULT_MAX_CHARS_PER_REQUEST,
        )

    def _worker_error(self, message: str) -> str:
        if not self._stderr_lines:
            return message
        return message + " Worker stderr: " + " | ".join(self._stderr_lines)


def resolve_qwen_tts_paths(
    *,
    worker_path: str | None,
    models_path: str | None,
) -> QwenTTSPaths:
    """Resolve generic Kaivra paths plus the existing Callbox installation."""
    worker = _resolve_existing_file(
        explicit=worker_path,
        env_names=("KAIVRA_QWEN_TTS_BIN", "CALLBOX_QWEN_TTS_BIN"),
        candidates=(
            Path.home() / ".kaivra/bin/callbox-qwen-tts",
            Path.home() / "Documents/end-to-end-voice/native/.build/release/callbox-qwen-tts",
        ),
        description="Qwen3-TTS worker",
    )
    if not os.access(worker, os.X_OK):
        raise RuntimeError(f"Qwen3-TTS worker is not executable: {worker}")

    models = _resolve_existing_directory(
        explicit=models_path,
        env_names=("KAIVRA_QWEN_TTS_MODELS_PATH", "CALLBOX_QWEN_TTS_MODELS_PATH"),
        candidates=(
            Path.home() / ".kaivra/models/qwen3-tts",
            Path.home() / "Documents/huggingface/models/argmaxinc/ttskit-coreml",
        ),
        description="Qwen3-TTS models directory",
    )
    return QwenTTSPaths(worker_path=str(worker), models_path=str(models))


def _resolve_existing_file(
    *,
    explicit: str | None,
    env_names: tuple[str, ...],
    candidates: tuple[Path, ...],
    description: str,
) -> Path:
    path = _configured_path(explicit, env_names, candidates, require_directory=False)
    if path is None:
        names = " or ".join(env_names)
        raise RuntimeError(f"Could not locate the {description}. Set {names}.")
    return path


def _resolve_existing_directory(
    *,
    explicit: str | None,
    env_names: tuple[str, ...],
    candidates: tuple[Path, ...],
    description: str,
) -> Path:
    path = _configured_path(explicit, env_names, candidates, require_directory=True)
    if path is None:
        names = " or ".join(env_names)
        raise RuntimeError(f"Could not locate the {description}. Set {names}.")
    return path


def _configured_path(
    explicit: str | None,
    env_names: tuple[str, ...],
    candidates: tuple[Path, ...],
    *,
    require_directory: bool,
) -> Path | None:
    configured: list[Path] = []
    if explicit and explicit.strip():
        configured.append(Path(explicit).expanduser())
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            configured.append(Path(value).expanduser())
    configured.extend(candidates)

    for candidate in configured:
        if require_directory and candidate.is_dir():
            return candidate.resolve()
        if not require_directory and candidate.is_file():
            return candidate.resolve()
    return None


def _decode_pcm(event: dict[str, Any]) -> bytes:
    encoded = str(event.get("pcm_s16le_b64") or "")
    try:
        pcm = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("Qwen3-TTS returned invalid base64 audio.") from exc
    if not pcm or len(pcm) % 2:
        raise RuntimeError("Qwen3-TTS returned invalid PCM16 audio.")
    return pcm


def _split_synthesis_text(text: str, max_chars: int) -> tuple[str, ...]:
    """Split long narration at sentence/word boundaries for stable local TTS.

    Whitespace is normalized deterministically so retries produce the same
    requests. Sentence boundaries are preferred; a single unusually long
    sentence falls back to word-bounded chunks without dropping text.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return ()
    if len(normalized) <= max_chars:
        return (normalized,)

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in sentences:
        if len(sentence) <= max_chars:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                flush()
                current = sentence
            else:
                current = candidate
            continue

        flush()
        for word in sentence.split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                flush()
                current = word
            else:
                current = candidate

    flush()
    return tuple(chunks)


def _temporary_wav_path(scene_id: str) -> Path:
    safe_scene_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in scene_id)
    descriptor, path = tempfile.mkstemp(prefix=f"kaivra_qwen_{safe_scene_id}_", suffix=".wav")
    os.close(descriptor)
    return Path(path)


def _write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def _speaker_from_env() -> str:
    return (
        os.environ.get("KAIVRA_QWEN_TTS_SPEAKER")
        or os.environ.get("CALLBOX_QWEN_TTS_SPEAKER")
        or _DEFAULT_SPEAKER
    )


def _instruction_from_env() -> str:
    return (
        os.environ.get("KAIVRA_QWEN_TTS_INSTRUCTION")
        or os.environ.get("CALLBOX_QWEN_TTS_INSTRUCTION")
        or _DEFAULT_INSTRUCTION
    ).strip()


def _decoder_mode_from_env() -> str:
    return (
        os.environ.get("KAIVRA_QWEN_TTS_DECODER_MODE")
        or os.environ.get("CALLBOX_QWEN_TTS_DECODER_MODE")
        or _DEFAULT_DECODER_MODE
    ).strip()


def _positive_timeout(explicit: float | None, env_name: str, default: float) -> float:
    value = explicit if explicit is not None else float(os.environ.get(env_name, default))
    if value <= 0:
        raise RuntimeError(f"{env_name} must be greater than zero.")
    return value


def _positive_int(explicit: int | None, env_name: str, default: int) -> int:
    value = explicit if explicit is not None else int(os.environ.get(env_name, default))
    if value <= 0:
        raise RuntimeError(f"{env_name} must be greater than zero.")
    return value
