"""
voice/speech.py — Module 2: Hindi/Bhojpuri speech in (ASR) and out (TTS).

    ASR : Sarvam AI (saaras:v4, mode="transcribe") or Groq (whisper-large-v3-turbo)
    TTS : edge-tts (hi-IN-SwaraNeural) — free, no API key required

Both ASR providers are optional. When ``MOCK_MODE=true`` (the default) or when no
key is configured, transcription returns a canned Bhojpuri pension transcript so
the UI and the agent pipeline keep working offline and in CI.
"""

from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Works for both `uv run python -m voice.speech` and `uv run python voice/speech.py`:
# the latter puts voice/ (not the repo root) on sys.path, so `import schemas` fails.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import edge_tts
import httpx
from dotenv import load_dotenv

from schemas import CitizenQuery

load_dotenv(ROOT / ".env")

DEFAULT_VOICE = "hi-IN-SwaraNeural"
OUTPUT_DIR = ROOT / "outputs"

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saaras:v4"
GROQ_MODEL = "whisper-large-v3-turbo"

AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}

MOCK_TRANSCRIPT = (
    "हमका बुढ़ापा पेंशन खातिर आवेदन करे के बा, हमार नाम रामेश्वर यादव ह, "
    "उमर साठ बरिस बा, हमार गाँव बलिया जिला में बा।"
)


# --------------------------------------------------------------------- config


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def mock_mode() -> bool:
    """MOCK_MODE defaults to true, so a fresh clone never hits a paid API by accident."""
    return _flag("MOCK_MODE", default=True)


def _sarvam_key() -> str | None:
    return (os.getenv("SARVAM_API_KEY") or "").strip() or None


def _groq_key() -> str | None:
    return (os.getenv("GROQ_API_KEY") or "").strip() or None


def asr_backend() -> str:
    """Which ASR path ``transcribe_audio`` would take: ``mock``, ``sarvam`` or ``groq``."""
    sarvam, groq = _sarvam_key(), _groq_key()
    if mock_mode() or not (sarvam or groq):
        return "mock"
    return "sarvam" if sarvam else "groq"


# ------------------------------------------------------------------------ TTS


async def synthesize_hindi(
    text: str,
    output_path: str | Path,
    voice: str = DEFAULT_VOICE,
) -> Path:
    """Render ``text`` to an MP3 at ``output_path`` (parent dirs are created)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    await edge_tts.Communicate(text, voice).save(str(path))
    return path


def synthesize_hindi_sync(
    text: str,
    output_path: str | Path,
    voice: str = DEFAULT_VOICE,
) -> Path:
    """Blocking wrapper around :func:`synthesize_hindi` for CLI and sync callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(synthesize_hindi(text, output_path, voice))

    # Already inside an event loop (e.g. a FastAPI handler): asyncio.run would raise,
    # so hand the coroutine to a worker thread that owns its own loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run, synthesize_hindi(text, output_path, voice)
        ).result()


# ------------------------------------------------------------------------ ASR


def transcribe_audio(
    audio_path: str | Path,
    session_id: str = "demo-session",
) -> CitizenQuery:
    """Transcribe a Hindi/Bhojpuri recording into a :class:`CitizenQuery`."""
    backend = asr_backend()
    if backend == "mock":
        return CitizenQuery(
            session_id=session_id,
            raw_text=MOCK_TRANSCRIPT,
            dialect_hint="bhojpuri",
        )

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if backend == "sarvam":
        return _transcribe_sarvam(path, session_id)
    return _transcribe_groq(path, session_id)


def _mime_type(path: Path) -> str:
    return AUDIO_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _transcribe_sarvam(path: Path, session_id: str) -> CitizenQuery:
    with path.open("rb") as handle:
        response = httpx.post(
            SARVAM_URL,
            headers={"api-subscription-key": _sarvam_key() or ""},
            files={"file": (path.name, handle, _mime_type(path))},
            data={"model": SARVAM_MODEL, "mode": "transcribe"},
            timeout=60.0,
        )
    response.raise_for_status()

    payload = response.json()
    return CitizenQuery(
        session_id=session_id,
        raw_text=(payload.get("transcript") or "").strip(),
        dialect_hint="hindi",
    )


def _transcribe_groq(path: Path, session_id: str) -> CitizenQuery:
    from groq import Groq  # imported lazily so the module loads without it

    client = Groq(api_key=_groq_key() or "")
    with path.open("rb") as handle:
        result = client.audio.transcriptions.create(
            file=(path.name, handle.read()),
            model=GROQ_MODEL,
            language="hi",
            temperature=0.0,
        )
    return CitizenQuery(
        session_id=session_id,
        raw_text=(result.text or "").strip(),
        dialect_hint="hindi",
    )


# ---------------------------------------------------------------- CLI self-test

if __name__ == "__main__":
    # Windows consoles default to cp1252, which cannot encode Devanagari.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample_text = (
        "नमस्ते, मुझे वृद्धावस्था पेंशन के लिए आवेदन करना है, "
        "मेरा नाम रामेश्वर यादव है।"
    )
    target = OUTPUT_DIR / "test_voice.mp3"

    audio = synthesize_hindi_sync(sample_text, target)
    print(f"TTS -> {audio}  ({audio.stat().st_size:,} bytes)")
    print(f"ASR backend: {asr_backend()}")

    query = transcribe_audio(audio)
    print(query.model_dump_json(indent=2))
