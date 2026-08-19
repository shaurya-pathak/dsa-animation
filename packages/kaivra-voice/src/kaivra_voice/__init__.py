"""Voice synthesis providers for kaivra animations."""

from kaivra_voice.elevenlabs import ElevenLabsProvider
from kaivra_voice.openai import OpenAIProvider
from kaivra_voice.qwen import QwenProvider

__all__ = ["ElevenLabsProvider", "OpenAIProvider", "QwenProvider"]

try:
    _local_module = __import__("kaivra_voice.local", fromlist=["LocalProvider"])
    LocalProvider = _local_module.LocalProvider
    __all__.append("LocalProvider")
except ImportError:
    pass
