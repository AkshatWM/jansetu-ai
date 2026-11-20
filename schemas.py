"""
schemas.py — Shared Pydantic v2 I/O contracts for JanSetu AI.
All modules (RAG, Voice, Agent, Vision, PDF, FastAPI) import their models from here.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

SchemeType = Literal[
    "old_age_pension",
    "income_certificate",
    "caste_certificate",
    "pm_kisan",
    "jansunwai",
    "unknown",
]

DocumentType = Literal[
    "aadhaar",
    "income_certificate",
    "bank_passbook",
    "other",
]


# --- Module 1: Grounded RAG ---

class RetrievedChunk(BaseModel):
    source_file: str = Field(..., description="e.g., old_age_pension.md")
    section: str = Field(default="", description="Markdown heading of the chunk")
    content: str
    score: float = Field(default=0.0, description="RRF / hybrid relevance score")


class EligibilityResult(BaseModel):
    is_eligible: Optional[bool] = Field(
        default=None,
        description="True/False once key slots are known, None if more info is needed",
    )
    reasons: list[str] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    source_citations: list[str] = Field(
        default_factory=list,
        description="List of scheme filenames/sections backing this decision",
    )


# --- Module 2: Voice I/O ---

class CitizenQuery(BaseModel):
    session_id: str = Field(default="demo-session")
    raw_text: str = Field(..., description="ASR transcript or typed citizen message")
    dialect_hint: Literal["hindi", "awadhi", "bhojpuri", "hinglish", "unknown"] = "hindi"


# --- Module 3: Agentic Slot-Filler ---

class FormSlots(BaseModel):
    scheme_type: SchemeType = "unknown"
    applicant_name: Optional[str] = None
    father_or_husband_name: Optional[str] = None
    age: Optional[int] = None
    annual_income_inr: Optional[int] = None
    district: Optional[str] = None
    address: Optional[str] = None
    grievance_summary: Optional[str] = Field(
        default=None, description="Used only when scheme_type == 'jansunwai'"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields still required before generating the PDF",
    )


class AgentTurnResponse(BaseModel):
    """Returned by POST /api/voice (and text fallback)."""
    transcript: str
    normalized_hindi: str = Field(
        ..., description="Dialect-normalized standard Hindi understanding"
    )
    slots: FormSlots
    eligibility: EligibilityResult
    rag_citations: list[RetrievedChunk] = Field(default_factory=list)
    bot_reply_hindi: str = Field(
        ..., description="Short, polite Hindi reply / next question for the citizen"
    )
    tts_audio_url: Optional[str] = Field(
        default=None, description="Relative URL to generated Hindi MP3 reply"
    )
    ready_for_docs: bool = Field(
        default=False, description="True once basic eligibility/slots are collected"
    )
    ready_for_pdf: bool = Field(
        default=False, description="True once all required slots are filled"
    )


# --- Module 4: Document Pre-Check Guard ---

class DocCheckResult(BaseModel):
    """Returned by POST /api/check-doc."""
    document_type: DocumentType = "other"
    is_blurry: bool
    blur_score: float = Field(
        ..., description="OpenCV Laplacian variance (lower = blurrier)"
    )
    extracted_name: Optional[str] = None
    extracted_income_inr: Optional[int] = None
    name_match_score: Optional[float] = Field(
        default=None, description="0-100 RapidFuzz similarity against FormSlots.applicant_name"
    )
    is_valid: bool = Field(
        ..., description="True only if image is sharp AND key fields match"
    )
    hindi_warning: str = Field(
        ..., description="Actionable Hindi feedback (or confirmation if valid)"
    )


# --- Module 5: PDF Form Generation ---

class PDFGenerationResponse(BaseModel):
    """Returned by POST /api/generate-pdf."""
    application_id: str
    filename: str
    pdf_url: str
    slots_snapshot: FormSlots


# --- Quick Local Self-Test ---

if __name__ == "__main__":
    sample = AgentTurnResponse(
        transcript="हमका बुढ़ापा पेंशन खातिर आवेदन करे के बा, हमार नाम रामेश्वर यादव ह।",
        normalized_hindi="मुझे वृद्धावस्था पेंशन के लिए आवेदन करना है, मेरा नाम रामेश्वर यादव है।",
        slots=FormSlots(
            scheme_type="old_age_pension",
            applicant_name="Rameshwar Yadav",
            missing_fields=["age", "annual_income_inr", "district"],
        ),
        eligibility=EligibilityResult(
            is_eligible=None,
            reasons=["उम्र (60+ वर्ष) और वार्षिक आय की जानकारी आवश्यक है।"],
            required_documents=["आधार कार्ड", "आय प्रमाण पत्र", "बैंक पासबुक"],
            source_citations=["old_age_pension.md#eligibility"],
        ),
        bot_reply_hindi="नमस्ते रामेश्वर जी! वृद्धावस्था पेंशन के लिए कृपया अपनी उम्र, सालाना आय और ज़िला बताएं।",
    )
    print(sample.model_dump_json(indent=2))
