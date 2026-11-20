# JanSetu AI — Team Build Plan & Progress Tracker

**Event:** AI Kaushal Hackathon (IIIT Lucknow)  
**Use Case:** Vernacular Citizen Services Assistant (Uttar Pradesh)  
**Team Size:** 3 Members  
**Submission Deadline:** November 27, 2026  
**Deliverable:** Working Local PoC + 60–90s Demo Video + Pitch Deck (packed in a `.zip` under 20MB)

---

## 1. Architecture & Tech Stack

Lightweight, zero-bundler stack designed for parallel 3-person development, minimal merge conflicts, and clean live demos:

* **Package Manager:** `uv` (fast, deterministic installs across Windows, Linux, and macOS)
* **Backend:** FastAPI + Uvicorn (serves both API routes and static frontend on one port)
* **Frontend:** Vanilla HTML, CSS, and JavaScript (`MediaRecorder` API for voice, `<audio>` for TTS playback, `fetch()` for API calls)
* **Voice (ASR + TTS):**
  * **ASR:** Groq (`whisper-large-v3-turbo`) or Sarvam AI (`saaras:v4` with `mode="transcribe"`)
  * **TTS:** `edge-tts` (`hi-IN-SwaraNeural` — free, no API key required)
* **Grounded RAG:** Hybrid `faiss-cpu` + `rank-bm25` with Reciprocal Rank Fusion (RRF) over 5 curated UP scheme Markdown files
* **Agentic Core:** Single structured JSON LLM call (dialect normalization + eligibility check + multi-turn slot-filling)
* **Document Pre-Check Guard:** OpenCV Laplacian blur check + Vision/OCR extraction + `rapidfuzz` name-mismatch guard
* **PDF Generator:** `reportlab` with bundled `NotoSansDevanagari-Regular.ttf`

---

## 2. Team Ownership & Parallel Work Tracks

All 3 members align on **Module 0 (`schemas.py`)** first. Once `schemas.py` is committed to `main`, everyone builds in parallel without blocking each other:

| Track | Owner | Assigned Modules | Can Work Independently Using |
| :--- | :--- | :--- | :--- |
| **Track A: Core AI & RAG** | *(Assign Name)* | Module 1 (`rag/retriever.py`) + Module 3 (`core/agent.py`) + `data/schemes/` | CLI test scripts + `schemas.py` |
| **Track B: Voice, Vision & PDF** | *(Assign Name)* | Module 2 (`voice/speech.py`) + Module 4 (`vision/doc_guard.py`) + Module 5 (`forms/pdf_builder.py`) | Sample audio/images + hardcoded `FormSlots` |
| **Track C: Backend, UI & Pitch** | *(Assign Name)* | Module 6 (`server.py` + `static/`) + Architecture Diagram + Pitch Deck | `MOCK_MODE=true` dummy responses matching `schemas.py` |

> **Why this split works:** Because every module only communicates through `schemas.py`, Track C can build and polish the entire WhatsApp-style UI on Day 1 using mock responses while Track A and Track B build the real AI/CV pipelines.

---

## 3. Repository Structure

```text
jansetu-ai/
├── PLAN.md                 # This file (architecture + live team checklist)
├── .gitignore              # Ignore .env, .venv/, __pycache__/, outputs/
├── .env                    # Local API keys + MOCK_MODE toggle (never commit)
├── .env.example            # Shared template for environment variables
├── pyproject.toml          # Managed by uv
├── schemas.py              # Module 0: Shared Pydantic I/O contracts (FROZEN after Day 1)
├── data/
│   ├── schemes/            # 5 clean UP scheme Markdown files [Track A]
│   │   ├── old_age_pension.md
│   │   ├── income_cert.md
│   │   ├── caste_cert.md
│   │   ├── pm_kisan.md
│   │   └── jansunwai.md
│   ├── fonts/              # NotoSansDevanagari-Regular.ttf [Track B]
│   └── test_samples/       # 1 clear + 1 blurry dummy doc + test audio [Track B]
├── rag/
│   └── retriever.py        # Module 1: Hybrid FAISS + BM25 search [Track A]
├── voice/
│   └── speech.py           # Module 2: ASR + Edge-TTS wrapper [Track B]
├── core/
│   └── agent.py            # Module 3: Dialect normalizer + eligibility + slot-filler [Track A]
├── vision/
│   └── doc_guard.py        # Module 4: OpenCV blur check + OCR mismatch guard [Track B]
├── forms/
│   └── pdf_builder.py      # Module 5: FormSlots -> Pre-filled UP Draft PDF [Track B]
├── outputs/                # Generated TTS audio & pre-filled PDFs (gitignored)
├── server.py               # Module 6a: FastAPI app (3 endpoints + static mount) [Track C]
└── static/                 # Module 6b: Split-screen WhatsApp Chat + AI Inspector [Track C]
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 4. Local Setup & Git Workflow

### One-Time Repo Setup (Project Lead)
```bash
mkdir jansetu-ai && cd jansetu-ai
git init
uv init --python 3.11
uv add fastapi uvicorn python-multipart pydantic python-dotenv \
       faiss-cpu rank-bm25 sentence-transformers \
       opencv-python-headless rapidfuzz edge-tts reportlab httpx
```

### Teammate Onboarding (Clone & Run)
```bash
git clone <repo-url>
cd jansetu-ai
uv sync
cp .env.example .env
# Fill in your API keys (or keep MOCK_MODE=true for offline UI/logic work)
```

### `.env.example` Template
```env
MOCK_MODE=true
GROQ_API_KEY=your_key_here
SARVAM_API_KEY=your_key_here
LLM_API_KEY=your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

### Simple Git Rules (Avoid Merge Headaches)
1. **Branch per module:** Never push directly to `main`. Use short branches like `feat/rag`, `feat/doc-guard`, or `feat/ui`.
2. **Freeze `schemas.py`:** Do not rename or delete fields in `schemas.py` without messaging the team first.
3. **Test before PR:** Every module must pass its own `if __name__ == "__main__":` CLI test before merging into `main`.
4. **Never commit secrets or artifacts:** Keep `.env`, `.venv/`, `__pycache__/`, and `outputs/` in `.gitignore`.

---

## 5. Module-by-Module Implementation Plan

### Module 0: Contracts & Seed Setup (`schemas.py`) — *All Members Sync First*
**Goal:** Lock down strict Pydantic models so every teammate's code plugs together cleanly on integration day.

* [ ] Create `schemas.py` with 5 models:
  * `FormSlots`: `scheme_type`, `applicant_name`, `father_or_husband_name`, `age`, `district`, `residency_type` (`rural`/`urban`), `annual_income`, `grievance_summary`, `missing_fields: list[str]`
  * `EligibilityResult`: `is_eligible: bool | None`, `reasons: list[str]`, `citations: list[str]`
  * `AgentResponse`: `normalized_hindi: str`, `reply_hindi: str`, `slots: FormSlots`, `eligibility: EligibilityResult`, `ready_for_docs: bool`
  * `DocCheckResult`: `passed: bool`, `is_blurry: bool`, `blur_score: float`, `extracted_name: str | None`, `name_match_score: float | None`, `warning_hindi: str`
  * `SessionState`: `session_id`, `history: list[dict]`, `slots: FormSlots`, `doc_verified: bool`
* [ ] Push `schemas.py`, `.gitignore`, `.env.example`, and empty folder stubs to `main`.

### Module 1: Grounded RAG (`rag/retriever.py`) — *[Track A]*
**Goal:** Retrieve cited UP scheme rules with zero hallucination.

* [ ] Write 5 short, accurate Markdown files in `data/schemes/` (UP Old-Age Pension, Income Certificate, Caste Certificate, PM-KISAN, Jansunwai Grievance) with exact income caps, age criteria, and required documents.
* [ ] Load `.md` files and chunk by `##` headings (attach `source` filename to each chunk).
* [ ] Build BM25 index (`rank_bm25`) and vector index (`faiss-cpu` with `paraphrase-multilingual-MiniLM-L12-v2`).
* [ ] Implement `retrieve(query: str, top_k: int = 3) -> list[dict]` combining both rankings via Reciprocal Rank Fusion (RRF).
* [ ] **CLI Verification:** `uv run python -m rag.retriever` with `"बुढ़ापा पेंशन के लिए सालाना आय सीमा क्या है?"` returns the exact UP rural/urban income limit with `old_age_pension.md` cited.

### Module 2: Voice I/O (`voice/speech.py`) — *[Track B]*
**Goal:** Convert spoken Hindi/dialect audio to text and synthesize Hindi replies.

* [ ] Implement `transcribe_audio(file_path: str) -> str`:
  * Support Groq `whisper-large-v3-turbo` or Sarvam `saaras:v4` (`mode="transcribe"`).
  * Respect `MOCK_MODE=true` by returning a realistic Hindi test transcript.
* [ ] Implement `synthesize_speech(text: str, output_path: str) -> str` using `edge-tts` (`hi-IN-SwaraNeural`).
* [ ] **CLI Verification:** `uv run python -m voice.speech` transcribes a 5-second test `.wav`/`.webm` and generates `outputs/test_reply.mp3`.

### Module 3: Agentic Slot-Filler & Eligibility Core (`core/agent.py`) — *[Track A]*
**Goal:** Normalize regional phrasing (Bhojpuri/Awadhi/Hindi), check eligibility via RAG, and fill `FormSlots` across turns.

* [ ] Implement `process_turn(user_text: str, current_slots: FormSlots, history: list[dict]) -> AgentResponse`:
  * Fetch top-3 RAG chunks via `rag.retriever.retrieve()`.
  * Make 1 LLM call in strict JSON mode enforcing the `AgentResponse` schema.
  * Prompt rules: normalize dialect phrasing, never invent eligibility rules outside the RAG context, preserve already-filled slots, ask for only 1–2 missing fields at a time in polite Hindi.
* [ ] **CLI Verification:** `uv run python -m core.agent` runs a simulated 3-turn Hindi/Awadhi conversation and outputs populated `FormSlots` + citations.

### Module 4: Document Pre-Check Guard (`vision/doc_guard.py`) — *[Track B]*
**Goal:** Catch blurry photos and name mismatches before form submission.

* [ ] Create 2 dummy test document images in `data/test_samples/` (1 crisp, 1 blurred or name-mismatched).
* [ ] **Stage 1 — Local Blur Gate (OpenCV):**
  * Load image in grayscale and compute `blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()`.
  * If `blur_score < 100.0` (calibrate on sample images), short-circuit immediately with `passed=False` and Hindi warning: *"फोटो धुंधली है, कृपया रोशनी में साफ फोटो खींचकर भेजें।"*
* [ ] **Stage 2 — OCR & Name Match Gate:**
  * Extract applicant name and document type using a Vision LLM API (or local `rapidocr-onnxruntime`).
  * Compare extracted name against `FormSlots.applicant_name` using `rapidfuzz.fuzz.token_sort_ratio()`.
  * If score `< 80`, return `passed=False` with a specific Hindi mismatch warning.
* [ ] **CLI Verification:** `uv run python -m vision.doc_guard` passes the clear dummy doc and flags the blurry/mismatched one.

### Module 5: Pre-Filled PDF Generator (`forms/pdf_builder.py`) — *[Track B]*
**Goal:** Turn verified `FormSlots` into a clean, submission-ready UP e-District draft PDF.

* [ ] Download `NotoSansDevanagari-Regular.ttf` into `data/fonts/` and register it in `reportlab`.
* [ ] Implement `build_application_pdf(slots: FormSlots, eligibility: EligibilityResult, output_path: str) -> str`.
* [ ] Layout: Official-style header (*"Uttar Pradesh e-District Application Draft — Generated via JanSetu AI"*), applicant details table, eligibility status + rule citations, and a structured JSON payload footer.
* [ ] **CLI Verification:** `uv run python -m forms.pdf_builder` outputs `outputs/sample_draft.pdf` with clean Hindi + English text.

### Module 6: FastAPI Server & Split-Screen UI (`server.py` + `static/`) — *[Track C]*
**Goal:** Single-command web app with a realistic WhatsApp mobile chat on the left and a live "Under the Hood" AI inspector on the right.

* [ ] **`server.py` (FastAPI):**
  * In-memory `SESSIONS: dict[str, SessionState]` store.
  * `POST /api/voice` — accepts audio blob (or text fallback), runs ASR → Agent → TTS, returns JSON + audio URL.
  * `POST /api/check-doc` — accepts image upload, runs `doc_guard`, returns `DocCheckResult` + TTS warning URL.
  * `POST /api/generate-pdf` — generates and streams the pre-filled PDF for download.
  * Mount `/outputs` for audio/PDF downloads and `/` to `static/`.
  * Include a `MOCK_MODE=true` path so the UI works end-to-end before Track A & Track B merge.
* [ ] **`static/` (`index.html`, `style.css`, `app.js`):**
  * Left pane: WhatsApp-style chat bubbles, mic record button (`MediaRecorder`), text input fallback, document upload button, inline audio player, PDF download card.
  * Right pane ("AI Inspector"): Live formatted JSON of `FormSlots`, active RAG citations, and Document Pre-Check telemetry (`blur_score`, `name_match_score`, pass/fail badge).
* [ ] **End-to-End Verification:** Run `uv run uvicorn server:app --reload`, open `http://localhost:8000`, complete a full voice + blurry doc + clear doc + PDF download flow.

---

## 6. Universal AI-Assisted Coding Guidelines

Regardless of which editor or AI assistant each teammate uses (Cursor, Copilot, Windsurf, `opencode`, Claude Code, Gemini, or ChatGPT), follow these four rules to keep code clean and consistent:

1. **Minimal Context Rule:** Only share `schemas.py`, the relevant module section from `PLAN.md`, and the single file you are editing. Avoid pasting unrelated files.
2. **Self-Testing Modules:** Every module must end with a standalone `if __name__ == "__main__":` test block so you can verify it locally before opening a PR.
3. **Use `MOCK_MODE=true` First:** Develop and debug flows with mock responses before switching to live API keys.
4. **Reusable Prompt Template (works in any AI tool):**
   ```text
   Read `schemas.py` and the Module [X] specification in `PLAN.md`.
   Implement `[filepath]` cleanly using Python 3.11 type hints.
   - Respect `MOCK_MODE` from `.env`.
   - Keep code concise, modular, and dependency-light.
   - Add an `if __name__ == "__main__":` block to test the module from the CLI.
   Do not modify `schemas.py` or any other files outside your assigned module.
   ```

---

## 7. 7-Day Parallel Sprint Schedule (Nov 21 – Nov 27, 2026)

| Date | Track A (Core AI & RAG) | Track B (Voice, Vision & PDF) | Track C (Backend, UI & Pitch) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Nov 21** | Sync on Module 0 (`schemas.py`) + Write 5 UP scheme `.md` files | Sync on Module 0 + Gather font & dummy doc/audio samples | Sync on Module 0 + Scaffold `server.py` (`MOCK_MODE`) & `static/index.html` | [ ] |
| **Nov 22** | Build & CLI-test Module 1 (`rag/retriever.py`) | Build & CLI-test Module 2 (`voice/speech.py`) & Module 5 (`pdf_builder.py`) | Finish split-screen WhatsApp UI + AI Inspector (`style.css`, `app.js`) | [ ] |
| **Nov 23** | Build & CLI-test Module 3 (`core/agent.py`) | Build & CLI-test Module 4 (`vision/doc_guard.py`) | Draft clean Excalidraw/Mermaid Architecture Diagram + 5-Slide Deck | [ ] |
| **Nov 24** | **Integration Day:** Wire Track A, Track B, and Track C into `server.py` with `MOCK_MODE=false` | | | [ ] |
| **Nov 25** | End-to-end bug fixing, calibrate OpenCV blur threshold, polish Hindi prompts & PDF formatting | | | [ ] |
| **Nov 26** | Record 60–90s PoC demo video, compress with FFmpeg, finalize Pitch Deck, pack & verify `.zip` (< 20MB) | | | [ ] |
| **Nov 27** | Final form text check (add *"Working PoC video attached in .zip"* to Q5) & submit early | | | [ ] |

---

## 8. Final `.zip` Submission Checklist (Max 20MB)

Target folder contents before zipping (`JanSetu_AI_Submission.zip`):

* [ ] `01_JanSetu_AI_Pitch_Deck.pdf` — 5-slide visual deck (Problem, 2 USPs, Architecture, Sample Output, 24h Scope vs. PoC)
* [ ] `02_System_Architecture.png` — Crisp Excalidraw/Mermaid diagram (no garbled AI text)
* [ ] `03_PoC_Demo.mp4` — 60–90s screen recording showing Voice Intake → Blurry Doc Warning → Clear Doc → Pre-filled PDF Download
* [ ] `04_Sample_PreFilled_Form.pdf` — Real PDF output generated by Module 5

**FFmpeg command to compress demo video under ~10MB:**
```bash
ffmpeg -i raw_demo.mp4 -vcodec libx264 -crf 28 -preset slow -vf "scale=1280:-2" -acodec aac -b:a 96k 03_PoC_Demo.mp4
```

* [ ] Verify total `.zip` size is well under **20MB** and contains no `__pycache__`, `.env`, or nested junk folders.