"""
vision/doc_guard.py — Module 4: Document Pre-Check Guard.

Two gates run before a citizen's document is accepted:

    Stage 1 (local, always on): OpenCV Laplacian variance rejects blurry photos.
    Stage 2 (vision model):     extract the name/income and compare the name with
                                the application using RapidFuzz.

Vision providers, in priority order:
    MOCK_MODE / no key -> deterministic mock fields (offline, testable)
    GEMINI_API_KEY     -> Gemini 2.5 Flash REST (inline base64 image)
    GROQ_API_KEY       -> Groq Vision (llama-4-scout) in JSON mode

Blur detection never depends on an API key, so the offline demo still demonstrates
the guard's most visible behaviour.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Works for both `uv run python -m vision.doc_guard` and `uv run python vision/doc_guard.py`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from rapidfuzz import fuzz

from schemas import DocCheckResult, DocumentType

load_dotenv(ROOT / ".env")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

NAME_MATCH_THRESHOLD = 80.0
DEFAULT_BLUR_THRESHOLD = 100.0

BLURRY_WARNING = "यह फोटो धुंधली है। कृपया अच्छी रोशनी में साफ़ फोटो दोबारा अपलोड करें।"
SUCCESS_WARNING = "दस्तावेज़ साफ़ है और जानकारी मेल खाती है। सत्यापन सफल रहा।"

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

EXTRACTION_PROMPT = (
    "You are reading a scanned Indian government document. Extract only what is "
    "printed on it. Return strict JSON: "
    '{"extracted_name": string|null, "extracted_income_inr": integer|null}. '
    "extracted_name must be the applicant/holder name exactly as printed. "
    "extracted_income_inr must be the annual income in rupees as a plain integer "
    "(no commas, no currency symbol), or null if absent. Never guess."
)


# --------------------------------------------------------------------- config


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def mock_mode() -> bool:
    return _flag("MOCK_MODE", default=True)


def _gemini_key() -> str | None:
    return (os.getenv("GEMINI_API_KEY") or "").strip() or None


def _groq_key() -> str | None:
    return (os.getenv("GROQ_API_KEY") or "").strip() or None


def vision_backend() -> str:
    """Which extraction path ``check_document`` will use: ``mock``, ``gemini`` or ``groq``."""
    gemini, groq = _gemini_key(), _groq_key()
    if mock_mode() or not (gemini or groq):
        return "mock"
    return "gemini" if gemini else "groq"


# ---------------------------------------------------------------- Stage 1: blur


def compute_blur_score(image_path: str | Path) -> float:
    """Variance of the Laplacian; lower means blurrier."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        # cv2.imread cannot open non-ASCII paths on Windows; imdecode can.
        buffer = np.fromfile(str(path), dtype=np.uint8)
        gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------- Stage 2: OCR


def _mime_type(path: Path) -> str:
    return IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/jpeg")


def _parse_fields(content: str) -> tuple[str | None, int | None]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)

    name = data.get("extracted_name")
    name = str(name).strip() if name else None

    income = data.get("extracted_income_inr")
    if isinstance(income, str):
        digits = re.sub(r"[^\d]", "", income)
        income = int(digits) if digits else None
    elif isinstance(income, (int, float)):
        income = int(income)
    else:
        income = None

    return name, income


def _mock_fields(
    path: Path, expected_name: str | None, expected_income_inr: int | None
) -> tuple[str | None, int | None]:
    """Deterministic stand-in for OCR, keyed off the filename."""
    if "mismatch" in path.stem.lower():
        return "Suresh Kumar", 75000
    return expected_name or "Rameshwar Yadav", (
        expected_income_inr if expected_income_inr is not None else 36000
    )


def _extract_gemini(path: Path) -> tuple[str | None, int | None]:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": _mime_type(path),
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0},
    }
    response = httpx.post(
        f"{GEMINI_URL}?key={_gemini_key()}",
        json=payload,
        timeout=60.0,
    )
    response.raise_for_status()
    content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_fields(content)


def _extract_groq(path: Path) -> tuple[str | None, int | None]:
    from groq import Groq  # imported lazily so the module loads without it

    data_url = (
        f"data:{_mime_type(path)};base64,"
        f"{base64.b64encode(path.read_bytes()).decode('ascii')}"
    )
    client = Groq(api_key=_groq_key() or "")
    completion = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return _parse_fields(completion.choices[0].message.content or "")


def _extract_fields(
    backend: str,
    path: Path,
    expected_name: str | None,
    expected_income_inr: int | None,
) -> tuple[str | None, int | None]:
    if backend == "mock":
        return _mock_fields(path, expected_name, expected_income_inr)
    if backend == "gemini":
        return _extract_gemini(path)
    return _extract_groq(path)


# ------------------------------------------------------------------- pipeline


def check_document(
    image_path: str | Path,
    document_type: DocumentType = "income_certificate",
    expected_name: str | None = None,
    expected_income_inr: int | None = None,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> DocCheckResult:
    """Run the blur gate, then the extraction/mismatch gate."""
    path = Path(image_path)

    # Stage 1 — local blur gate (runs even in MOCK_MODE, and short-circuits).
    blur_score = compute_blur_score(path)
    if blur_score < blur_threshold:
        return DocCheckResult(
            document_type=document_type,
            is_blurry=True,
            blur_score=blur_score,
            is_valid=False,
            hindi_warning=BLURRY_WARNING,
        )

    # Stage 2 — extract fields and compare the name with the application.
    backend = vision_backend()
    try:
        extracted_name, extracted_income = _extract_fields(
            backend, path, expected_name, expected_income_inr
        )
    except Exception as exc:  # never let a provider outage pass a document through
        print(f"[doc_guard] {backend} extraction failed ({exc})", file=sys.stderr)
        return DocCheckResult(
            document_type=document_type,
            is_blurry=False,
            blur_score=blur_score,
            is_valid=False,
            hindi_warning=(
                "दस्तावेज़ पढ़ा नहीं जा सका। कृपया साफ़ फोटो दोबारा अपलोड करें या "
                "थोड़ी देर बाद प्रयास करें।"
            ),
        )

    name_match_score: float | None = None
    if expected_name and extracted_name:
        name_match_score = float(fuzz.token_sort_ratio(expected_name, extracted_name))

    if name_match_score is not None and name_match_score < NAME_MATCH_THRESHOLD:
        warning = (
            f"दस्तावेज़ पर नाम '{extracted_name}' आवेदन के नाम '{expected_name}' से "
            "मेल नहीं खाता। कृपया सही दस्तावेज़ अपलोड करें।"
        )
        return DocCheckResult(
            document_type=document_type,
            is_blurry=False,
            blur_score=blur_score,
            extracted_name=extracted_name,
            extracted_income_inr=extracted_income,
            name_match_score=name_match_score,
            is_valid=False,
            hindi_warning=warning,
        )

    return DocCheckResult(
        document_type=document_type,
        is_blurry=False,
        blur_score=blur_score,
        extracted_name=extracted_name,
        extracted_income_inr=extracted_income,
        name_match_score=name_match_score,
        is_valid=True,
        hindi_warning=SUCCESS_WARNING,
    )


# ---------------------------------------------------------------- CLI self-test

OUTPUT_DIR = ROOT / "outputs" / "test_docs"
CERT_ID = "UP-INC-2026-0042"  # fictional, sample-only
VALID_NAME = "Rameshwar Yadav"
VALID_INCOME = 36000

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int):
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _render_certificate(
    path: Path, name: str, income: int, *, blur: bool = False, radius: float = 9.0
) -> Path:
    """Draw a synthetic UP income certificate (fictional data only)."""
    from PIL import Image, ImageDraw, ImageFilter

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)

    title, label, value, small = _font(32), _font(19), _font(23), _font(15)

    draw.rectangle([20, 20, 880, 580], outline="black", width=4)
    draw.rectangle([30, 30, 870, 570], outline="black", width=1)
    draw.text((290, 52), "UTTAR PRADESH e-DISTRICT", font=title, fill="black")
    draw.text((300, 96), "INCOME CERTIFICATE", font=title, fill="black")
    draw.line([(40, 150), (860, 150)], fill="black", width=2)

    rows = (
        ("Certificate ID:", CERT_ID),
        ("Applicant Name:", name),
        ("Annual Income:", f"Rs. {income}"),
        ("District:", "Ballia"),
        ("Issuing Authority:", "Tehsildar, Ballia"),
    )
    y = 180
    for label_text, value_text in rows:
        draw.text((60, y), label_text, font=label, fill="black")
        draw.text((250, y - 3), value_text, font=value, fill="black")
        y += 55

    draw.line([(40, 470), (860, 470)], fill="black", width=1)
    draw.text(
        (60, 495),
        "Synthetic sample generated for testing. Not a real certificate.",
        font=small,
        fill="black",
    )
    # A dense block gives the sharpness metric something to measure, like a stamp.
    for offset in range(0, 60, 6):
        draw.line([(720, 480 + offset), (820, 480 + offset)], fill="black", width=2)
        draw.line([(720 + offset, 480), (720 + offset, 545)], fill="black", width=2)

    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    image.save(path, quality=90)
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    blurry = _render_certificate(OUTPUT_DIR / "cert_blurry.jpg", VALID_NAME, VALID_INCOME, blur=True)
    mismatch = _render_certificate(OUTPUT_DIR / "cert_mismatch.jpg", "Suresh Kumar", 75000)
    valid = _render_certificate(OUTPUT_DIR / "cert_valid.jpg", VALID_NAME, VALID_INCOME)

    print(f"Vision backend: {vision_backend()}\n")

    blurry_result = check_document(
        blurry, expected_name=VALID_NAME, expected_income_inr=VALID_INCOME
    )
    mismatch_result = check_document(
        mismatch, expected_name=VALID_NAME, expected_income_inr=VALID_INCOME
    )
    valid_result = check_document(
        valid, expected_name=VALID_NAME, expected_income_inr=VALID_INCOME
    )

    for label, result in (
        ("cert_blurry.jpg", blurry_result),
        ("cert_mismatch.jpg", mismatch_result),
        ("cert_valid.jpg", valid_result),
    ):
        print(f"=== {label}  blur_score={result.blur_score:.2f} ===")
        print(result.model_dump_json(indent=2))
        print()

    assert blurry_result.is_blurry and not blurry_result.is_valid, "blurry doc not rejected"
    assert not mismatch_result.is_blurry and not mismatch_result.is_valid, (
        "name mismatch not rejected"
    )
    assert not valid_result.is_blurry and valid_result.is_valid, "valid doc rejected"
    assert mismatch_result.name_match_score is not None
    print("OK: blurry rejected, stale name rejected, valid document accepted")
