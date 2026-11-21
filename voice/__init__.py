"""Module 2: Voice I/O — speech in (ASR) and out (TTS) for JanSetu AI.

Public surface (lazily imported, so ``python voice/speech.py`` stays warning-free):
    from voice import synthesize_hindi, synthesize_hindi_sync, transcribe_audio
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from voice.speech import (
        asr_backend,
        synthesize_hindi,
        synthesize_hindi_sync,
        transcribe_audio,
    )

_LAZY = {"asr_backend", "synthesize_hindi", "synthesize_hindi_sync", "transcribe_audio"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from voice import speech as _speech

        return getattr(_speech, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY)


__all__ = sorted(_LAZY)
