# JanSetu AI

**Vernacular citizen-services assistant for Uttar Pradesh.** A citizen speaks, types, or
photographs a document in Hindi / Awadhi / Bhojpuri; JanSetu AI understands the request,
grounds it in official UP scheme rules, collects the missing form fields over multiple
turns, pre-checks document photos, and generates a pre-filled UP e-District PDF draft.

Built as a working local proof-of-concept for the **AI Kaushal Hackathon (IIIT Lucknow)**.

---

## What it does

| Module | Capability |
| :--- | :--- |
| **RAG** (`rag/retriever.py`) | Hybrid FAISS (multilingual embeddings) + BM25 retrieval with Reciprocal Rank Fusion over 5 curated UP scheme documents. |
| **Voice** (`voice/speech.py`) | Hindi/Bhojpuri speech-to-text (Sarvam `saaras:v4` or Groq Whisper) and Hindi text-to-speech (`edge-tts`, `hi-IN-SwaraNeural`). |
| **Agent** (`core/agent.py`) | Dialect normalization, multi-turn slot filling, and eligibility decisions grounded strictly in retrieved scheme text; offline deterministic rule engine by default. |
| **Document guard** (`vision/doc_guard.py`) | OpenCV Laplacian blur gate + vision-LLM field extraction + RapidFuzz name-mismatch check. |
| **PDF builder** (`forms/pdf_builder.py`) | Bilingual (Devanagari + Latin) pre-filled application draft via `fpdf2`, with eligibility reasons and a document checklist. |
| **Web app** (`server.py`, `static/`) | FastAPI backend plus a split-screen WhatsApp-style chat (left) and live AI pipeline inspector (right). |

Supported schemes: **Old Age Pension, Income Certificate, Caste Certificate, PM-KISAN,
Jansunwai grievance**.

---

## Requirements

- **Python 3.13+** (declared in `.python-version` / `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** (recommended) or plain `pip`
- Internet access on first run for:
  - the ~470 MB embedding model download, and
  - `edge-tts` speech synthesis (the free TTS endpoint needs network access)
- Optional API keys for live ASR / LLM / vision. **No keys are required** — the default
  mock mode exercises the full UI and pipeline without any paid API (the only network
  dependencies are the first-run embedding download and the free `edge-tts` voice).

---

## Quick start (recommended: uv)

```bash
git clone <repo-url>
cd jansetu-ai

uv sync                              # installs Python 3.13 if needed + locked deps
cp .env.example .env                 # optional: keep MOCK_MODE=true for an offline demo

uv run python main.py                # http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000** in a browser.

For auto-reload while editing:

```bash
uv run uvicorn server:app --reload --port 8000
```

### Alternative: plain pip

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
cp .env.example .env                 # optional

python main.py                       # http://127.0.0.1:8000
```

`requirements.txt` is a fully pinned export of `uv.lock`; regenerate it after changing
dependencies with the command documented at the top of the file.

---

## Configuration

Copy `.env.example` to `.env`. **Every value is optional** — missing keys simply fall
back to the offline paths.

| Variable | Default | Used for |
| :--- | :--- | :--- |
| `MOCK_MODE` | `true` | When true, ASR, the agent brain, and document extraction all use deterministic offline stand-ins. Blur detection, retrieval, and PDF generation stay real. |
| `SARVAM_API_KEY` | empty | Speech-to-text via Sarvam (`saaras:v4`). Takes priority over Groq ASR. |
| `GROQ_API_KEY` | empty | Speech-to-text (`whisper-large-v3-turbo`), LLM brain (`openai/gpt-oss-120b`), and document vision (`llama-4-scout`). |
| `DEEPSEEK_API_KEY` | empty | LLM brain fallback (`deepseek-chat`) when no Groq key is set. |
| `GEMINI_API_KEY` | empty | Document extraction via Gemini 2.5 Flash. Takes priority over Groq Vision. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address for `python main.py`. |

Provider priority (only when `MOCK_MODE=false`):

- **ASR:** Sarvam → Groq → mock
- **Agent:** Groq → DeepSeek → offline rule engine
- **Vision extraction:** Gemini → Groq Vision → mock

> With `MOCK_MODE=true`, `/api/voice` accepts typed text and returns a canned Bhojpuri
> transcript for audio; `/api/check-doc` still performs the real blur check. TTS is never
> mocked, so an offline machine gets the text reply without audio (handled gracefully).

### First run: embedding model

`rag/retriever.py` uses `paraphrase-multilingual-MiniLM-L12-v2`. If a local copy exists at
`data/models/paraphrase-multilingual-MiniLM-L12-v2/`, it is used offline; otherwise
`sentence-transformers` downloads it (~470 MB) on first use and caches it in your Hugging
Face cache. For flaky networks, set the optional Hugging Face variables documented in
`.env.example`.

---

## Using the demo

1. **Describe the need** — type, use a suggestion chip, or record with the mic
   (browser microphone permission is required; use `localhost`, not a remote IP).
2. **Answer follow-ups** — the agent asks for 1–2 missing fields per turn and fills the
   inspector on the right.
3. **Upload a document** — the paperclip opens the image picker. A blurry photo is
   rejected with Hindi guidance (`blur_score < 100`); a clear document is checked for a
   name match (`RapidFuzz ≥ 80`).
4. **Generate the PDF** — once all required slots are filled, click
   *Generate & Download Pre-Filled PDF*. Files land in `outputs/pdfs/`.
5. **Reset Demo** — clears server session state and starts a fresh browser session.

Sample documents are generated on demand by `vision/doc_guard.py` into
`outputs/test_docs/` (`cert_valid.jpg`, `cert_blurry.jpg`, `cert_mismatch.jpg`) and can be
uploaded through the UI. Uploaded files and generated audio/PDFs live under `outputs/`
(gitignored).

---

## Verify each module (CLI self-tests)

Each module has an `if __name__ == "__main__":` block that exercises real behaviour and
asserts its core guarantees:

```bash
uv run python schemas.py             # validates the shared Pydantic contracts
uv run python -m rag.retriever       # indexes 20 chunks; checks Hindi/English/Hinglish queries
uv run python -m voice.speech        # TTS to outputs/test_voice.mp3 + mock ASR (TTS needs network)
uv run python -m core.agent          # simulated 2-turn Bhojpuri pension conversation
uv run python -m vision.doc_guard    # blurry rejected, name mismatch rejected, clear accepted
uv run python -m forms.pdf_builder   # writes a sample bilingual draft into outputs/pdfs/
```

Expected final lines look like `OK: ...`. The very first retrieval run downloads the
embedding model; after that, indexing and queries finish in a few seconds.

---

## API

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Split-screen demo UI. |
| `POST` | `/api/voice` | Multipart form: `session_id`, plus `text` **or** `audio`. Returns transcription, slots, eligibility, RAG citations, Hindi reply, and a TTS URL. |
| `POST` | `/api/check-doc` | Multipart form: `session_id`, `document_type`, `file`. Returns blur/name-match telemetry and a Hindi warning. |
| `POST` | `/api/generate-pdf` | JSON `{"session_id": "..."}` (or `?session_id=`). Returns the application ID, filename, and PDF URL. |
| `POST` | `/api/reset` | Form field `session_id`. Forgets that session. |
| `GET` | `/outputs/...` | Generated audio and PDF downloads. |
| `GET` | `/docs` | FastAPI's interactive OpenAPI docs. |

Example typed turn:

```bash
curl -X POST http://127.0.0.1:8000/api/voice \
  -F "session_id=demo-session" \
  -F "text=मुझे बुढ़ापा पेंशन के लिए आवेदन करना है, मेरा नाम रामेश्वर यादव है।"
```

A PDF can only be generated for a session the server has already seen (i.e. after at
least one `/api/voice` or `/api/check-doc` turn).

---

## Repository layout

```text
jansetu-ai/
├── main.py                  # single-command launcher (python main.py)
├── server.py                # FastAPI app + session store + uploads
├── schemas.py               # shared Pydantic v2 contracts
├── static/                  # index.html, style.css, app.js, favicon.svg (no build step)
├── rag/retriever.py         # hybrid FAISS + BM25 + RRF
├── core/agent.py            # dialect normalization, slot filling, eligibility
├── voice/speech.py          # ASR (Sarvam/Groq) + edge-tts TTS
├── vision/doc_guard.py      # blur gate + extraction + name match
├── forms/pdf_builder.py     # bilingual pre-filled PDF
├── data/schemes/            # 5 curated UP scheme Markdown files
├── assets/fonts/            # Noto Sans Devanagari + Noto Sans (bundled)
├── outputs/                 # generated audio, PDFs, uploads (gitignored)
├── requirements.txt         # pinned pip install set (from uv.lock)
├── pyproject.toml / uv.lock # uv-managed dependencies
├── .env.example             # environment template
└── PLAN.md                  # original build plan / team tracker
```

---

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `UnicodeEncodeError` / garbled Hindi in the terminal | Use Windows Terminal or run `chcp 65001`; the app's self-tests already reconfigure stdout to UTF-8. |
| First run hangs at `Loading weights` | The embedding model is downloading (~470 MB). Wait, or pre-populate `data/models/paraphrase-multilingual-MiniLM-L12-v2/`. |
| Model download fails behind a proxy | Set `HF_HUB_DISABLE_XET=1`, `HF_HUB_DOWNLOAD_TIMEOUT=180`, `HF_HUB_ETAG_TIMEOUT=60` before running. |
| No audio reply | `edge-tts` needs internet. The Hindi text reply still appears; check the server log for `TTS failed`. |
| Mic button does nothing | Browsers require `localhost`/HTTPS and microphone permission. Typing always works. |
| PDF generation fails with "No Devanagari-capable TTF" | Bundled fonts are in `assets/fonts/`; verify both files exist, or let the builder fall back to a system font. |
| `ModuleNotFoundError` | Run through `uv run ...` (after `uv sync`) or activate the venv where `pip install -r requirements.txt` was run. |
| Uploaded document "passes" a name mismatch in mock mode | Mock extraction cannot read real photos. Upload the generated `outputs/test_docs/cert_mismatch.jpg` (filename contains `mismatch`), or set `MOCK_MODE=false` with a vision key. |

---

## Notes

This is an offline-first **proof of concept**. Sessions are kept in memory, uploads are
stored unencrypted under `outputs/`, and generated drafts are explicitly marked
`(PoC Draft)` — they are not official government submissions. Sample documents use
fictional data.
