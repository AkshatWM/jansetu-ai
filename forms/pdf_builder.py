"""
forms/pdf_builder.py — Module 5: Pre-Filled UP e-District application draft.

Renders verified :class:`FormSlots` (plus the eligibility verdict from Module 3)
into a one-page A4 PDF that a citizen can carry to a CSC / Tehsil office.

Devanagari needs real text shaping, so the document is built with fpdf2 +
``set_text_shaping(True)`` (uharfbuzz) and an embedded Devanagari TTF.

NotoSansDevanagari ships without Latin letters, and the form is bilingual, so a
Latin font is registered as an fpdf2 *fallback*: every ASCII character is drawn
from it while Devanagari keeps the shaped font. Without this, names such as
"Rameshwar Yadav" and the application ID are silently dropped from the page.
"""

from __future__ import annotations

import json
import secrets
import string
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Works for both `uv run python -m forms.pdf_builder` and `uv run python forms/pdf_builder.py`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

from schemas import EligibilityResult, FormSlots, PDFGenerationResponse

# Devanagari-capable TTF, in the documented fallback order.
FONT_CANDIDATES: tuple[Path, ...] = (
    ROOT / "assets" / "fonts" / "NotoSansDevanagari-Regular.ttf",
    Path("C:/Windows/Fonts/Nirmala.ttf"),
    Path("C:/Windows/Fonts/mangal.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
)

# Latin companion for the bilingual labels. Nirmala/Mangal already contain Latin;
# Noto Sans Devanagari does not, so a fallback is registered when needed.
LATIN_FONT_CANDIDATES: tuple[Path, ...] = (
    ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf",
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)

FONT_FAMILY = "Devanagari"
LATIN_FAMILY = "Latin"
BRAND_COLOR = (21, 87, 141)
LIGHT_FILL = (238, 244, 250)
MUTED = (105, 105, 105)
DANGER = (160, 40, 40)
DASH = "\u2014"  # em dash, shown for empty fields
RUPEE = "\u20b9"  # ₹, present in Noto Sans Devanagari

# A pathological free-text value must not push the 1-page form into the footer.
# Only the rendered cell is shortened; the companion JSON keeps the full text.
TRUNCATE_LIMITS: dict[str, int] = {
    "address": 140,
    "grievance_summary": 320,
}

SCHEME_NAMES: dict[str, tuple[str, str]] = {
    "old_age_pension": ("Old Age Pension", "वृद्धावस्था पेंशन योजना"),
    "income_certificate": ("Income Certificate", "आय प्रमाण पत्र"),
    "caste_certificate": ("Caste Certificate", "जाति प्रमाण पत्र"),
    "pm_kisan": ("PM-KISAN Samman Nidhi", "पीएम किसान सम्मान निधि"),
    "jansunwai": ("Jansunwai Grievance", "जनसुनवाई शिकायत"),
    "unknown": ("General Application", "सामान्य आवेदन"),
}

FIELD_LABELS: dict[str, str] = {
    "applicant_name": "आवेदक का नाम / Applicant Name",
    "father_or_husband_name": "पिता या पति का नाम / Father or Husband",
    "age": "उम्र / Age",
    "annual_income_inr": "वार्षिक पारिवारिक आय / Annual Income",
    "district": "जिला / District",
    "address": "पता / Address",
    "grievance_summary": "शिकायत का विवरण / Grievance Summary",
}

ELIGIBILITY_STATUS: dict[str | None, tuple[str, str]] = {
    True: ("Eligible", "पात्र"),
    False: ("Not Eligible", "अपात्र"),
    None: ("Verification Pending", "सत्यापन जारी"),
}


# ---------------------------------------------------------------------- setup


def _first_existing(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def find_devanagari_font() -> Path:
    """First available Devanagari TTF, in the documented fallback order."""
    font = _first_existing(FONT_CANDIDATES)
    if font is not None:
        return font
    tried = "\n  ".join(str(path) for path in FONT_CANDIDATES)
    raise FileNotFoundError(
        "No Devanagari-capable TTF found. Place NotoSansDevanagari-Regular.ttf in "
        f"assets/fonts/ (tried:\n  {tried})"
    )


def find_latin_font() -> Path | None:
    """Latin companion font, or None when the Devanagari font already has Latin."""
    return _first_existing(LATIN_FONT_CANDIDATES)


def _resolve_output_dir(output_dir: str | Path) -> Path:
    """Relative paths are anchored at the repo root, not the caller's cwd."""
    path = Path(output_dir)
    return path if path.is_absolute() else ROOT / path


def _application_id() -> str:
    suffix = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    return f"UP-JS-{datetime.now():%Y}-{suffix}"


def _pdf_url(path: Path) -> str:
    try:
        return "/" + path.relative_to(ROOT).as_posix()
    except ValueError:
        return f"/outputs/pdfs/{path.name}"


def _money(amount: int | float) -> str:
    """Grouped rupees with the ₹ sign, matching the eligibility reasons."""
    return f"{RUPEE}{int(amount):,}"


def _format_value(field: str, value: object) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return DASH
    if field == "age":
        return f"{value} वर्ष"
    if field == "annual_income_inr":
        return _money(value)  # type: ignore[arg-type]
    text = str(value).strip()
    limit = TRUNCATE_LIMITS.get(field)
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text


def _new_pdf() -> FPDF:
    """A4 page with the Devanagari font and an optional Latin fallback."""
    devanagari = find_devanagari_font()

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(12, 12, 12)

    pdf.add_font(FONT_FAMILY, "", str(devanagari))

    latin = find_latin_font()
    if latin is not None:
        pdf.add_font(LATIN_FAMILY, "", str(latin))
        pdf.set_fallback_fonts([LATIN_FAMILY])

    pdf.set_text_shaping(True)  # Devanagari matras / half-letters need real shaping
    pdf.add_page()
    return pdf


# ------------------------------------------------------------------- building


def _header(pdf: FPDF, slots: FormSlots, application_id: str) -> None:
    pdf.set_fill_color(*BRAND_COLOR)
    pdf.rect(12, 12, 186, 21, style="F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font(FONT_FAMILY, "", 14.5)
    pdf.set_xy(16, 14.5)
    pdf.cell(120, 7, "उत्तर प्रदेश ई-डिस्ट्रिक्ट आवेदन प्रारूप")

    pdf.set_font(FONT_FAMILY, "", 8.5)
    pdf.set_xy(16, 23)
    pdf.cell(120, 5, "Uttar Pradesh e-District Application Draft")

    pdf.set_font(FONT_FAMILY, "", 7.5)
    pdf.set_xy(124, 15.5)
    pdf.cell(70, 4.5, f"Application ID: {application_id}", align="R")
    pdf.set_xy(124, 20.5)
    pdf.cell(70, 4.5, f"Date: {datetime.now():%d-%m-%Y}", align="R")

    pdf.set_text_color(0, 0, 0)

    scheme_en, scheme_hi = SCHEME_NAMES.get(slots.scheme_type, SCHEME_NAMES["unknown"])
    pdf.set_xy(12, 36)
    pdf.set_font(FONT_FAMILY, "", 11.5)
    pdf.cell(186, 6.5, f"{scheme_hi}  ({scheme_en})", align="C")

    pdf.set_draw_color(*BRAND_COLOR)
    pdf.set_line_width(0.5)
    pdf.line(12, 44, 198, 44)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.2)
    pdf.set_fill_color(255, 255, 255)  # table cells inherit fill_color


def _details_table(pdf: FPDF, slots: FormSlots) -> None:
    fields = [
        "applicant_name",
        "father_or_husband_name",
        "age",
        "annual_income_inr",
        "district",
        "address",
    ]
    if slots.scheme_type == "jansunwai":
        fields.append("grievance_summary")

    pdf.set_xy(12, 48)
    pdf.set_font(FONT_FAMILY, "", 8.5)

    with pdf.table(
        col_widths=(70, 116),
        text_align=("LEFT", "LEFT"),
        line_height=5.4,
        borders_layout="ALL",
        first_row_as_headings=True,
        headings_style=FontFace(fill_color=LIGHT_FILL),
        padding=1.1,
    ) as table:
        header = table.row()
        header.cell("विवरण / Field")
        header.cell("मान / Value")
        for field in fields:
            row = table.row()
            row.cell(FIELD_LABELS[field])
            row.cell(_format_value(field, getattr(slots, field)))


def _status_block(pdf: FPDF, eligibility: EligibilityResult) -> None:
    pdf.ln(4)
    pdf.set_fill_color(*LIGHT_FILL)
    pdf.set_font(FONT_FAMILY, "", 10)
    pdf.cell(
        186,
        7.5,
        "  पात्रता एवं दस्तावेज़ / Eligibility & Documents",
        fill=True,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(1.5)

    status_en, status_hi = ELIGIBILITY_STATUS[eligibility.is_eligible]
    pdf.set_font(FONT_FAMILY, "", 9.5)
    pdf.cell(40, 6, "स्थिति / Status:")
    pdf.set_font(FONT_FAMILY, "", 10)
    pdf.set_text_color(*(BRAND_COLOR if eligibility.is_eligible else DANGER))
    pdf.cell(
        0,
        6,
        f"{status_hi} ({status_en})",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_text_color(0, 0, 0)

    if eligibility.reasons:
        pdf.ln(1)
        pdf.set_font(FONT_FAMILY, "", 8.5)
        pdf.cell(0, 4.5, "कारण / Reasons:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for reason in eligibility.reasons:
            pdf.set_x(14)
            pdf.multi_cell(
                178,
                4.6,
                f"-  {reason}",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

    if eligibility.required_documents:
        pdf.ln(1.5)
        pdf.set_font(FONT_FAMILY, "", 8.5)
        pdf.cell(
            0,
            4.5,
            "आवश्यक दस्तावेज़ / Required documents (checklist):",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        for document in eligibility.required_documents:
            y = pdf.get_y()
            pdf.rect(14, y + 0.9, 3.2, 3.2)
            pdf.set_x(20)
            pdf.multi_cell(
                172,
                4.6,
                document,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

    if eligibility.source_citations:
        pdf.ln(1.5)
        pdf.set_font(FONT_FAMILY, "", 7)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(
            186,
            3.8,
            "स्रोत / Sources: " + "; ".join(eligibility.source_citations),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_text_color(0, 0, 0)


def _declaration(pdf: FPDF, slots: FormSlots) -> None:
    pdf.ln(5)
    pdf.set_font(FONT_FAMILY, "", 8.5)
    pdf.multi_cell(
        186,
        4.4,
        "घोषणा / Declaration: मैं घोषणा करता/करती हूँ कि ऊपर दी गई सभी जानकारी सत्य एवं सही है। "
        "यदि कोई जानकारी असत्य पाई गई तो मैं स्वयं उत्तरदायी रहूँगा/रहूँगी। "
        "I declare that the information given above is true and correct to the best of my knowledge.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf.ln(3)
    box_x, box_y = 136, pdf.get_y()
    pdf.rect(box_x, box_y, 62, 17)

    pdf.set_xy(box_x, box_y + 18)
    pdf.set_font(FONT_FAMILY, "", 7.5)
    pdf.cell(62, 4, "हस्ताक्षर / Signature of Applicant", align="C")

    pdf.set_xy(12, box_y + 1)
    pdf.set_font(FONT_FAMILY, "", 7.5)
    place = slots.district.strip() if slots.district and slots.district.strip() else DASH
    pdf.cell(0, 4.5, f"Place: {place}")
    pdf.set_xy(12, box_y + 6)
    pdf.cell(0, 4.5, "Submitted to: e-District / Tehsil office, Uttar Pradesh")


def _footer(pdf: FPDF) -> None:
    # The footer deliberately sits below the auto page-break trigger; without this
    # the still-empty second page would be created just to hold it.
    pdf.set_auto_page_break(auto=False)
    pdf.set_y(-13)
    pdf.set_font(FONT_FAMILY, "", 7)
    pdf.set_text_color(*MUTED)
    pdf.cell(
        186,
        4.5,
        f"Generated by JanSetu AI {DASH} Vernacular Citizen Services Assistant (PoC Draft)",
        align="C",
    )
    pdf.set_text_color(0, 0, 0)


# --------------------------------------------------------------------- public


def generate_application_pdf(
    slots: FormSlots,
    eligibility: EligibilityResult | None = None,
    output_dir: str | Path = "outputs/pdfs",
) -> PDFGenerationResponse:
    """Render the pre-filled draft PDF (+ a JSON snapshot) and describe the result."""
    target_dir = _resolve_output_dir(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    application_id = _application_id()
    filename = f"UP_{slots.scheme_type}_{application_id}.pdf"
    pdf_path = target_dir / filename

    pdf = _new_pdf()
    _header(pdf, slots, application_id)
    _details_table(pdf, slots)
    if eligibility is not None:
        _status_block(pdf, eligibility)
    _declaration(pdf, slots)
    _footer(pdf)

    pdf.output(str(pdf_path))

    snapshot = target_dir / f"{application_id}.json"
    snapshot.write_text(
        json.dumps(slots.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return PDFGenerationResponse(
        application_id=application_id,
        filename=filename,
        pdf_url=_pdf_url(pdf_path),
        slots_snapshot=slots,
    )


# ---------------------------------------------------------------- CLI self-test

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Devanagari font: {find_devanagari_font()}")
    print(f"Latin fallback : {find_latin_font() or '(font already covers Latin)'}\n")

    demo_slots = FormSlots(
        scheme_type="old_age_pension",
        applicant_name="Rameshwar Yadav",
        father_or_husband_name="Harihar Yadav",
        age=60,
        annual_income_inr=36000,
        district="बलिया",
        address="ग्राम रामपुर, बलिया",
    )
    demo_eligibility = EligibilityResult(
        is_eligible=True,
        reasons=[
            "उम्र 60 वर्ष है, न्यूनतम आयु 60 वर्ष पूरी होती है।",
            "वार्षिक आय ₹36,000 है, जो ₹46,080 की सीमा के भीतर है।",
        ],
        required_documents=[
            "आधार कार्ड / Aadhaar Card",
            "आयु प्रमाण पत्र / Age Proof",
            "आय प्रमाण पत्र / Income Certificate",
            "बैंक पासबुक / Bank Passbook",
        ],
        source_citations=["old_age_pension.md#Eligibility"],
    )

    response = generate_application_pdf(demo_slots, demo_eligibility)
    pdf_file = ROOT / response.pdf_url.lstrip("/")
    json_file = pdf_file.parent / f"{response.application_id}.json"

    print(response.model_dump_json(indent=2))
    print(f"\nPDF : {pdf_file}  ({pdf_file.stat().st_size:,} bytes)")
    print(f"JSON: {json_file}  ({json_file.stat().st_size:,} bytes)")

    assert pdf_file.stat().st_size > 0, "PDF is empty"
    assert json_file.is_file(), "companion JSON missing"
    assert pdf_file.read_bytes().startswith(b"%PDF"), "not a PDF file"
    print("OK: pre-filled draft PDF generated")
