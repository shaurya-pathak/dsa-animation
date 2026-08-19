from __future__ import annotations

import os
import stat
import wave
from pathlib import Path

import pytest
from kaivra_voice.qwen import QwenProvider, resolve_qwen_tts_paths


def _fake_worker(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import base64
import json
import sys

PREFIX = "CALLBOX_QWEN_JSON:"

def emit(payload):
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)

emit({"id": "__ready__", "ok": True, "event": "ready", "model": "fake-qwen"})
for line in sys.stdin:
    command = json.loads(line)
    if command["action"] == "shutdown":
        emit({"id": command["id"], "ok": True, "event": "shutdown"})
        break
    request_id = command["id"]
    pcm = b"\\x00\\x00\\xff\\x7f\\x01\\x80\\x00\\x00"
    emit({
        "id": request_id,
        "ok": True,
        "event": "audio",
        "sequence": 0,
        "sample_rate": 24000,
        "pcm_s16le_b64": base64.b64encode(pcm).decode(),
    })
    emit({"id": request_id, "ok": True, "event": "complete"})
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_qwen_provider_reuses_worker_and_writes_pcm16_wav(tmp_path: Path) -> None:
    worker = _fake_worker(tmp_path / "fake-qwen-worker")
    models = tmp_path / "models"
    models.mkdir()
    provider = QwenProvider(
        worker_path=str(worker),
        models_path=str(models),
        startup_timeout_seconds=2,
        generation_timeout_seconds=2,
    )

    first = provider.generate("opening", "How does the model decide?", voice_id="serena")
    process = provider._process
    second = provider.generate("answer", "The clues become one answer.", voice_id="serena")

    assert process is not None
    assert provider._process is process
    assert first.duration_seconds == pytest.approx(4 / 24000)
    assert second.duration_seconds == pytest.approx(4 / 24000)
    for result in (first, second):
        with wave.open(result.audio_path, "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 24000
            assert audio.getnframes() == 4
        Path(result.audio_path).unlink()

    provider.close()
    assert process.poll() == 0


def test_qwen_provider_chunks_long_narration_and_concatenates_audio(tmp_path: Path) -> None:
    worker = _fake_worker(tmp_path / "fake-qwen-worker")
    models = tmp_path / "models"
    models.mkdir()
    provider = QwenProvider(
        worker_path=str(worker),
        models_path=str(models),
        startup_timeout_seconds=2,
        generation_timeout_seconds=2,
        max_chars_per_request=24,
    )

    result = provider.generate(
        "lesson",
        "First sentence here. Second sentence here.",
        voice_id="serena",
    )

    assert result.duration_seconds == pytest.approx(8 / 24000)
    with wave.open(result.audio_path, "rb") as audio:
        assert audio.getnframes() == 8

    Path(result.audio_path).unlink()
    provider.close()


def test_qwen_provider_rejects_speaker_change_in_one_render(tmp_path: Path) -> None:
    worker = _fake_worker(tmp_path / "fake-qwen-worker")
    models = tmp_path / "models"
    models.mkdir()
    provider = QwenProvider(worker_path=str(worker), models_path=str(models))
    result = provider.generate("one", "First scene.", voice_id="serena")
    Path(result.audio_path).unlink()

    with pytest.raises(RuntimeError, match="one speaker"):
        provider.generate("two", "Second scene.", voice_id="vivian")

    provider.close()


def test_qwen_path_resolution_uses_environment(tmp_path: Path, monkeypatch) -> None:
    worker = _fake_worker(tmp_path / "worker")
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("KAIVRA_QWEN_TTS_BIN", str(worker))
    monkeypatch.setenv("KAIVRA_QWEN_TTS_MODELS_PATH", str(models))

    resolved = resolve_qwen_tts_paths(worker_path=None, models_path=None)

    assert resolved.worker_path == str(worker.resolve())
    assert resolved.models_path == str(models.resolve())


def test_qwen_path_resolution_reports_missing_install(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("kaivra_voice.qwen.Path.home", classmethod(lambda cls: tmp_path))
    for name in (
        "KAIVRA_QWEN_TTS_BIN",
        "CALLBOX_QWEN_TTS_BIN",
        "KAIVRA_QWEN_TTS_MODELS_PATH",
        "CALLBOX_QWEN_TTS_MODELS_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="KAIVRA_QWEN_TTS_BIN"):
        resolve_qwen_tts_paths(worker_path=None, models_path=None)


def test_qwen_provider_reads_callbox_compatible_environment(tmp_path: Path, monkeypatch) -> None:
    worker = _fake_worker(tmp_path / "worker")
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("CALLBOX_QWEN_TTS_BIN", str(worker))
    monkeypatch.setenv("CALLBOX_QWEN_TTS_MODELS_PATH", str(models))
    monkeypatch.setenv("CALLBOX_QWEN_TTS_SPEAKER", "VIVIAN")

    provider = QwenProvider(startup_timeout_seconds=2, generation_timeout_seconds=2)
    result = provider.generate("scene", "A locally generated sentence.")

    assert provider._active_speaker == "vivian"
    assert os.path.exists(result.audio_path)
    Path(result.audio_path).unlink()
    provider.close()
