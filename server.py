"""
server.py — Module 6: FastAPI backend for JanSetu AI.

    uv run uvicorn server:app --port 8000
    uv run python server.py

Serves the split-screen demo UI at ``/``, the static assets at ``/static`` and
generated audio/PDFs at ``/outputs``. One :class:`JanSetuAgent` (and therefore
one retrieval index) is shared by every request; per-visitor state is kept in
memory under ``session_id`` so a demo take can be replayed or reset cleanly.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import AsyncIterator

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.agent import get_agent, llm_backend
from forms.pdf_builder import generate_application_pdf
from schemas import (
    AgentTurnResponse,
    CitizenQuery,
    DocCheckResult,
    DocumentType,
    EligibilityResult,
    FormSlots,
    PDFGenerationResponse,
)
from vision.doc_guard import check_document
from voice.speech import synthesize_hindi, transcribe_audio

load_dotenv(ROOT / ".env")

STATIC_DIR = ROOT / "static"
OUTPUTS_DIR = ROOT / "outputs"
AUDIO_DIR = OUTPUTS_DIR / "audio"
PDF_DIR = OUTPUTS_DIR / "pdfs"
UPLOAD_DIR = OUTPUTS_DIR / "uploads"

for directory in (AUDIO_DIR, PDF_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------- sessions

@dataclass
class SessionState:
    """Everything one browser tab has taught the agent so far."""

    slots: FormSlots = field(default_factory=FormSlots)
    eligibility: EligibilityResult | None = None


_sessions: dict[str, SessionState] = {}
_sessions_lock = Lock()


def get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        return _sessions.setdefault(session_id, SessionState())


def store_session(
    session_id: str, slots: FormSlots, eligibility: EligibilityResult | None
) -> None:
    with _sessions_lock:
        _sessions[session_id] = SessionState(slots=slots, eligibility=eligibility)


def clear_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)


# ---------------------------------------------------------------- uploads

_SAFE_SUFFIX = re.compile(r"^\.[a-z0-9]{1,5}$")
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")
_MIME_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
}


def save_upload(upload: UploadFile, default_suffix: str = ".bin") -> Path:
    """Copy an upload into ``outputs/uploads`` under a collision-free name.

    A sanitized prefix of the original stem is kept (for debugging and for the
    mock document extractor, which keys off names such as ``*mismatch*``); the
    UUID guarantees uniqueness. The stem is allow-listed, so it cannot inject
    path separators or unusual characters.
    """
    original = Path(upload.filename or "")
    suffix = original.suffix.lower()
    if not _SAFE_SUFFIX.match(suffix):
        mime = (upload.content_type or "").split(";")[0].strip().lower()
        suffix = _MIME_SUFFIX.get(mime, default_suffix)

    stem = _SAFE_STEM.sub("-", original.stem).strip("-_")[:40]
    name = f"{uuid.uuid4().hex}-{stem}" if stem else uuid.uuid4().hex
    path = UPLOAD_DIR / f"{name}{suffix}"
    with path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path


# ---------------------------------------------------------------- application

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load the retriever + embedding model once, before the first request.
    app.state.agent = get_agent()
    print(f"[server] JanSetu AI ready | LLM backend: {llm_backend()}", file=sys.stderr)
    yield


app = FastAPI(title="JanSetu AI", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/voice", response_model=AgentTurnResponse)
async def voice(
    request: Request,
    session_id: str = Form(...),
    text: str | None = Form(default=None),
    audio: UploadFile | None = File(default=None),
) -> AgentTurnResponse:
    """Transcribe an upload (or take typed text), then run one agent turn."""
    if audio is None and not (text and text.strip()):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' or an 'audio' upload.",
        )

    state = get_session(session_id)

    if audio is not None:
        saved = save_upload(audio, default_suffix=".webm")
        try:
            query = await asyncio.to_thread(transcribe_audio, saved, session_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc
    else:
        query = CitizenQuery(session_id=session_id, raw_text=(text or "").strip())

    agent = request.app.state.agent
    try:
        response = await asyncio.to_thread(agent.process_turn, query, state.slots)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc

    # The reply is useful even if the TTS service is unreachable, so never let
    # a synthesis failure cost the citizen their answer.
    audio_name = f"{uuid.uuid4().hex}.mp3"
    try:
        await synthesize_hindi(response.bot_reply_hindi, AUDIO_DIR / audio_name)
        response.tts_audio_url = f"/outputs/audio/{audio_name}"
    except Exception as exc:
        print(f"[server] TTS failed ({exc}); text reply only", file=sys.stderr)
        response.tts_audio_url = None

    store_session(session_id, response.slots, response.eligibility)
    return response


@app.post("/api/check-doc", response_model=DocCheckResult)
def check_doc(
    session_id: str = Form(...),
    document_type: DocumentType = Form(default="income_certificate"),
    file: UploadFile = File(...),
) -> DocCheckResult:
    """Run the blur + name-mismatch guard against the session's application."""
    state = get_session(session_id)
    saved = save_upload(file)

    try:
        return check_document(
            saved,
            document_type=document_type,
            expected_name=state.slots.applicant_name,
            expected_income_inr=state.slots.annual_income_inr,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class GeneratePdfRequest(BaseModel):
    session_id: str


@app.post("/api/generate-pdf", response_model=PDFGenerationResponse)
def generate_pdf(
    session_id: str | None = Query(default=None),
    payload: GeneratePdfRequest | None = Body(default=None),
) -> PDFGenerationResponse:
    """Render the draft PDF for an existing session (query string or JSON body)."""
    sid = (session_id or (payload.session_id if payload else "") or "").strip()
    if not sid:
        raise HTTPException(
            status_code=422,
            detail="session_id is required in the query string or JSON body.",
        )

    with _sessions_lock:
        state = _sessions.get(sid)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {sid}")

    try:
        return generate_application_pdf(state.slots, state.eligibility, output_dir=PDF_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/reset")
def reset(session_id: str = Form(default="demo-session")) -> dict[str, object]:
    """Forget a session so the next demo take starts from a clean slate."""
    clear_session(session_id)
    return {"ok": True, "session_id": session_id}


def run(host: str | None = None, port: int | None = None) -> None:
    """Start the production-style Uvicorn server (also used by ``main.py``)."""
    import uvicorn

    uvicorn.run(
        app,
        host=host or os.getenv("HOST", "127.0.0.1"),
        port=port or int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
