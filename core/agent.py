"""
core/agent.py — Module 3: Dialect normalization, slot filling and eligibility.

    process_turn(CitizenQuery, FormSlots | None) -> AgentTurnResponse

Pipeline
    1. Retrieve the top-3 grounded RAG chunks for the citizen's message.
    2. Merge the LLM/rule output onto the slots collected in earlier turns.
    3. Normalize Awadhi/Bhojpuri/Hinglish to standard Hindi, fill slots, judge
       eligibility strictly from the retrieved chunks, and ask for the next
       missing detail.

Providers, in priority order (all use strict JSON mode):
    MOCK_MODE / no key -> deterministic rule engine (offline, always available)
    GROQ_API_KEY       -> Groq, openai/gpt-oss-120b
    DEEPSEEK_API_KEY   -> DeepSeek, deepseek-chat
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Works for both `uv run python -m core.agent` and `uv run python core/agent.py`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from dotenv import load_dotenv

from rag.retriever import SchemeRetriever
from schemas import (
    AgentTurnResponse,
    CitizenQuery,
    EligibilityResult,
    FormSlots,
    RetrievedChunk,
)

load_dotenv(ROOT / ".env")

GROQ_URL_MODEL = "openai/gpt-oss-120b"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
TOP_K = 3

# Fields each scheme needs before a draft PDF can be generated. Parsed from the
# RAG "Required Form Slots" section when available, else these defaults.
REQUIRED_FIELDS: dict[str, list[str]] = {
    "old_age_pension": [
        "applicant_name",
        "father_or_husband_name",
        "age",
        "annual_income_inr",
        "district",
        "address",
    ],
    "income_certificate": [
        "applicant_name",
        "father_or_husband_name",
        "annual_income_inr",
        "district",
        "address",
    ],
    "caste_certificate": [
        "applicant_name",
        "father_or_husband_name",
        "district",
        "address",
    ],
    "pm_kisan": ["applicant_name", "father_or_husband_name", "district", "address"],
    "jansunwai": ["applicant_name", "district", "address", "grievance_summary"],
    "unknown": ["applicant_name", "district"],
}

# Slots needed to judge eligibility (and therefore to start the document step).
KEY_ELIGIBILITY_SLOTS: dict[str, tuple[str, ...]] = {
    "old_age_pension": ("age", "annual_income_inr"),
    "income_certificate": ("annual_income_inr",),
    "caste_certificate": ("district",),
    "pm_kisan": ("district",),
    "jansunwai": ("grievance_summary",),
    "unknown": (),
}

SCHEME_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("पेंशन", "old_age_pension"),
    ("pension", "old_age_pension"),
    ("आय प्रमाण", "income_certificate"),
    ("इनकम", "income_certificate"),
    ("जाति", "caste_certificate"),
    ("किसान", "pm_kisan"),
    ("खतौनी", "pm_kisan"),
    ("शिकायत", "jansunwai"),
    ("जनसुनवाई", "jansunwai"),
)

# Used to keep eligibility grounded when a follow-up turn ("my income is 36000")
# no longer retrieves the scheme's own documents on its own merit.
SCHEME_QUERY_HINT: dict[str, str] = {
    "old_age_pension": "वृद्धावस्था पेंशन पात्रता आय सीमा",
    "income_certificate": "आय प्रमाण पत्र पात्रता शुल्क",
    "caste_certificate": "जाति प्रमाण पत्र पात्रता",
    "pm_kisan": "पीएम किसान पात्रता खतौनी",
    "jansunwai": "जनसुनवाई शिकायत दर्ज करें",
}

BHOJPURI_TO_HINDI: tuple[tuple[str, str], ...] = (
    (r"करे के बा", "करना है"),
    (r"करे के बाटे", "करना है"),
    (r"हमका", "मुझे"),
    (r"हमनी", "हम"),
    (r"हमार", "मेरा"),
    (r"रउरा", "आप"),
    (r"रउआ", "आप"),
    (r"बरिस", "वर्ष"),
    (r"खातिर", "के लिए"),
    (r"कइल", "किया"),
    (r"भइल", "हुआ"),
    (r"दिहल", "दिया"),
    (r"बाटे", "है"),
    (r"बा", "है"),
)

HINDI_NUMBERS: dict[str, int] = {
    "इकसठ": 61,
    "बासठ": 62,
    "तिरसठ": 63,
    "पैंसठ": 65,
    "अड़सठ": 68,
    "साठ": 60,
    "पचास": 50,
    "पचपन": 55,
    "सत्तर": 70,
    "पचहत्तर": 75,
    "अस्सी": 80,
}

# Devanagari letters/matras. The danda (U+0964/0965) is deliberately excluded so a
# word boundary check still matches at end of sentence: "बा।" -> "है।".
_DEVANAGARI = r"\u0900-\u0963\u0966-\u097F"


# --------------------------------------------------------------------- config


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def mock_mode() -> bool:
    return _flag("MOCK_MODE", default=True)


def _groq_key() -> str | None:
    return (os.getenv("GROQ_API_KEY") or "").strip() or None


def _deepseek_key() -> str | None:
    return (os.getenv("DEEPSEEK_API_KEY") or "").strip() or None


def llm_backend() -> str:
    """Which brain ``process_turn`` will use: ``mock``, ``groq`` or ``deepseek``."""
    groq, deepseek = _groq_key(), _deepseek_key()
    if mock_mode() or not (groq or deepseek):
        return "mock"
    return "groq" if groq else "deepseek"


# ---------------------------------------------------------- text normalization


def normalize_hindi(text: str) -> str:
    """Map common Bhojpuri/Awadhi forms to standard Hindi (word-boundary safe)."""
    out = text
    for pattern, replacement in BHOJPURI_TO_HINDI:
        out = re.sub(
            rf"(?<![{_DEVANAGARI}]){pattern}(?![{_DEVANAGARI}])",
            replacement,
            out,
        )
    return out


def _strip_tail(value: str) -> str:
    """Drop trailing verb particles such as 'है'/'ह' and stray punctuation."""
    value = value.strip(" ,।;")
    value = re.sub(rf"\s+(?:है|हैं|ह|हा|बा|बाटे|हई|था|थी)$", "", value)
    return value.strip(" ,।;")


def _after_keyword(text: str, pattern: str, stop: str = r",।;\n") -> str | None:
    match = re.search(rf"{pattern}\s+([^{stop}]+)", text)
    if not match:
        return None
    value = _strip_tail(match.group(1))
    return value or None


# -------------------------------------------------------------- rule engine


def _extract_scheme(text: str) -> str:
    lowered = text.lower()
    for keyword, scheme in SCHEME_KEYWORDS:
        if keyword in lowered:
            return scheme
    return "unknown"


def _extract_age(text: str) -> int | None:
    match = re.search(rf"(?:उमर|उम्र|आयु)\s*(?:है\s*)?([{_DEVANAGARI}]+|\d+)", text)
    if not match:
        return None
    raw = match.group(1)
    if raw.isdigit():
        return int(raw)
    return HINDI_NUMBERS.get(raw)


def _extract_income(text: str) -> int | None:
    match = re.search(r"(?:आय|आमदनी)\s*(?:है\s*)?(\d[\d,]*)", text)
    if not match:
        match = re.search(r"(\d[\d,]*)\s*(?:रुपये|रुपए|रु)", text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _extract_district(text: str) -> str | None:
    match = re.search(rf"([{_DEVANAGARI}]+)\s+जिला", text) or re.search(
        rf"जिला\s+([{_DEVANAGARI}]+)", text
    )
    return match.group(1) if match else None


def _extract_slots(text: str, base: FormSlots) -> FormSlots:
    """Regex/keyword extraction used only by the offline rule engine."""
    slots = base.model_copy(deep=True)

    scheme = _extract_scheme(text)
    if scheme != "unknown":
        slots.scheme_type = scheme  # type: ignore[assignment]

    # Father/husband name first, so it is not mistaken for the applicant's name.
    father = _after_keyword(text, rf"(?:पिता|पति)\s*(?:का|की|के)?\s*नाम")
    if father:
        slots.father_or_husband_name = father
        text = re.sub(rf"(?:पिता|पति)\s*(?:का|की|के)?\s*नाम[^,।;]*", "", text)

    name = _after_keyword(text, r"नाम")
    if name:
        slots.applicant_name = name

    age = _extract_age(text)
    if age is not None:
        slots.age = age

    income = _extract_income(text)
    if income is not None:
        slots.annual_income_inr = income

    district = _extract_district(text)
    if district:
        slots.district = district

    address = _after_keyword(text, r"पता", stop=r"।\n")
    if address:
        slots.address = address

    return slots


def _required_fields(scheme: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Prefer the scheme's own 'Required Form Slots' chunk over the built-in table."""
    for chunk in chunks:
        if chunk.section.strip().lower() == "required form slots":
            fields = [f.strip() for f in chunk.content.replace("\n", ",").split(",")]
            return [f for f in fields if f]
    return REQUIRED_FIELDS.get(scheme, REQUIRED_FIELDS["unknown"])


def _parse_documents(content: str) -> list[str]:
    return [
        line.lstrip("-*• ").strip()
        for line in content.splitlines()
        if line.strip().startswith(("-", "*", "•"))
    ]


def _required_documents(
    retriever: SchemeRetriever, scheme: str, chunks: list[RetrievedChunk]
) -> list[str]:
    """Document list from the retrieved context, else the scheme's own section."""
    for chunk in chunks:
        if chunk.section.strip().lower() == "required documents":
            docs = _parse_documents(chunk.content)
            if docs:
                return docs

    for raw in retriever.chunks:
        if (
            raw["source_file"] == f"{scheme}.md"
            and raw["section"].strip().lower() == "required documents"
        ):
            docs = _parse_documents(raw["content"])
            if docs:
                return docs

    return []


def _eligibility_chunks(
    retriever: SchemeRetriever, scheme: str, chunks: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Scheme-owned chunks used to judge eligibility, deduplicated by section.

    A follow-up turn ("my income is 36000") often stops retrieving the scheme's
    own documents, so we re-retrieve the scheme when its Eligibility rule is
    missing rather than judging eligibility from unrelated documents.
    """
    if scheme == "unknown":
        return []

    filename = f"{scheme}.md"
    own = [c for c in chunks if c.source_file == filename]
    if not any(c.section.strip().lower() == "eligibility" for c in own):
        hint = SCHEME_QUERY_HINT.get(scheme, scheme)
        own += [c for c in retriever.search(hint, top_k=TOP_K) if c.source_file == filename]

    unique: dict[tuple[str, str], RetrievedChunk] = {}
    for chunk in own:
        unique[(chunk.source_file, chunk.section)] = chunk
    return list(unique.values())


def _evaluate_eligibility(
    slots: FormSlots, chunks: list[RetrievedChunk], retriever: SchemeRetriever
) -> EligibilityResult:
    """Deterministic eligibility check grounded in the retrieved RAG text."""
    scheme = slots.scheme_type
    grounding = [c for c in chunks if c.source_file == f"{scheme}.md"]
    citations = [f"{c.source_file}#{c.section}" for c in grounding]
    documents = _required_documents(retriever, scheme, grounding)

    eligibility_text = next(
        (c.content for c in grounding if c.section.strip().lower() == "eligibility"),
        "",
    )

    if scheme == "old_age_pension":
        age_match = re.search(r"(\d+)\s*(?:years|वर्ष)", eligibility_text)
        min_age = int(age_match.group(1)) if age_match else 60

        caps = [
            int(value.replace(",", ""))
            for value in re.findall(r"₹\s*([\d,]+)", eligibility_text)
        ]
        income_cap = min(caps) if caps else 46080  # rural cap is the tighter one

        if slots.age is None or slots.annual_income_inr is None:
            return EligibilityResult(
                is_eligible=None,
                reasons=["पात्रता जाँचने के लिए उम्र और वार्षिक आय की जानकारी आवश्यक है।"],
                required_documents=documents,
                source_citations=citations,
            )

        reasons: list[str] = []
        eligible = True
        if slots.age >= min_age:
            reasons.append(f"उम्र {slots.age} वर्ष है, न्यूनतम आयु {min_age} वर्ष पूरी होती है।")
        else:
            eligible = False
            reasons.append(f"आयु {slots.age} वर्ष है, जबकि न्यूनतम आयु {min_age} वर्ष होनी चाहिए।")

        if slots.annual_income_inr <= income_cap:
            reasons.append(
                f"वार्षिक आय ₹{slots.annual_income_inr:,} है, जो ₹{income_cap:,} की सीमा के भीतर है।"
            )
        else:
            eligible = False
            reasons.append(
                f"वार्षिक आय ₹{slots.annual_income_inr:,} है, जो ₹{income_cap:,} की सीमा से अधिक है।"
            )

        return EligibilityResult(
            is_eligible=eligible,
            reasons=reasons,
            required_documents=documents,
            source_citations=citations,
        )

    if scheme == "unknown":
        return EligibilityResult(
            is_eligible=None,
            reasons=["योजना की पहचान नहीं हो सकी, कृपया बताएँ कि आप कौन-सी योजना चाहते हैं।"],
            required_documents=[],
            source_citations=[],
        )

    # Other schemes: state what is known, never invent a rule.
    return EligibilityResult(
        is_eligible=None,
        reasons=["आवेदन के लिए आवश्यक जानकारी एकत्र की जा रही है।"],
        required_documents=documents,
        source_citations=citations,
    )


def _missing_fields(slots: FormSlots, required: list[str]) -> list[str]:
    missing: list[str] = []
    for field in required:
        value = getattr(slots, field, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


_FIELD_LABELS = {
    "applicant_name": "अपना पूरा नाम",
    "father_or_husband_name": "पिता या पति का नाम",
    "age": "अपनी उम्र",
    "annual_income_inr": "परिवार की सालाना आय",
    "district": "अपना ज़िला",
    "address": "अपना पूरा पता",
    "grievance_summary": "अपनी शिकायत का विवरण",
}


def _next_question(missing: list[str]) -> str:
    if not missing:
        return "धन्यवाद! आपकी सभी जानकारी पूरी है। कृपया आधार कार्ड और अन्य दस्तावेज़ों की साफ़ फोटो अपलोड करें।"
    labels = [_FIELD_LABELS.get(f, f) for f in missing[:2]]
    return f"धन्यवाद! अब कृपया {' और '.join(labels)} बताएँ।"


# ------------------------------------------------------------------ prompting

SYSTEM_PROMPT = f"""आप जनसेतु AI हैं — उत्तर प्रदेश के नागरिक सेवा सहायक।
आपको नागरिक का संदेश, पिछली बातचीत के भरे हुए slots और RAG संदर्भ मिलेगा।

नियम:
1. normalized_hindi: अवधी/भोजपुरी/हिंग्लिश को मानक हिंदी में बदलें।
2. slots: संदेश से जानकारी निकालकर भरें/अपडेट करें। पहले से भरे गैर-शून्य मान कभी न मिटाएँ।
3. eligibility: मूल्यांकन केवल RAG संदर्भ से करें; अपने से कोई नियम न गढ़ें।
   हर निर्णय के लिए source_citations में "filename#Section" दें। जानकारी अपर्याप्त हो तो is_eligible=null रखें।
4. slots.missing_fields: required_fields में से वे फ़ील्ड जो अभी खाली हैं।
5. bot_reply_hindi: 1-2 वाक्य, विनम्र हिंदी; अगली 1-2 अधूरी जानकारी माँगें। सब भर जाने पर दस्तावेज़ अपलोड करने को कहें।

केवल यह JSON लौटाएँ:
{{
  "normalized_hindi": "...",
  "slots": {{
    "scheme_type": "old_age_pension|income_certificate|caste_certificate|pm_kisan|jansunwai|unknown",
    "applicant_name": null, "father_or_husband_name": null, "age": null,
    "annual_income_inr": null, "district": null, "address": null,
    "grievance_summary": null, "missing_fields": []
  }},
  "eligibility": {{
    "is_eligible": null, "reasons": [], "required_documents": [], "source_citations": []
  }},
  "bot_reply_hindi": "..."
}}"""


def _build_messages(
    query: CitizenQuery,
    slots: FormSlots,
    chunks: list[RetrievedChunk],
    required: list[str],
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{c.source_file}#{c.section}]\n{c.content}" for c in chunks
    ) or "(कोई संदर्भ नहीं मिला)"

    user_prompt = (
        f"RAG संदर्भ:\n{context}\n\n"
        f"पहले से भरे slots:\n{slots.model_dump_json(indent=2)}\n\n"
        f"required_fields: {', '.join(required)}\n\n"
        f"नागरिक का संदेश: {query.raw_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------------------------------------------------- agent


class JanSetuAgent:
    """Slot-filling + eligibility agent backed by the Module 1 retriever."""

    def __init__(self, retriever: SchemeRetriever | None = None) -> None:
        self.retriever = retriever or SchemeRetriever()

    # -- public ------------------------------------------------------------

    def process_turn(
        self,
        query: CitizenQuery,
        current_slots: FormSlots | None = None,
    ) -> AgentTurnResponse:
        base = current_slots.model_copy(deep=True) if current_slots else FormSlots()
        chunks = self.retriever.search(query.raw_text, top_k=TOP_K)
        required = _required_fields(base.scheme_type, chunks)

        backend = llm_backend()
        if backend == "mock":
            return self._rule_turn(query, base, chunks, required)

        try:
            payload = self._call_llm(backend, query, base, chunks, required)
        except Exception as exc:  # keep the demo alive if the provider misbehaves
            print(f"[agent] {backend} failed ({exc}); falling back to rules", file=sys.stderr)
            return self._rule_turn(query, base, chunks, required)

        return self._assemble(query, base, chunks, required, payload)

    # -- rule engine -------------------------------------------------------

    def _rule_turn(
        self,
        query: CitizenQuery,
        base: FormSlots,
        chunks: list[RetrievedChunk],
        required: list[str],
    ) -> AgentTurnResponse:
        normalized = normalize_hindi(query.raw_text)
        extracted = _extract_slots(normalized, base)
        slots = self._finalize_slots(base, extracted, required)

        grounding = _eligibility_chunks(self.retriever, slots.scheme_type, chunks)
        eligibility = _evaluate_eligibility(slots, grounding, self.retriever)

        payload = {
            "normalized_hindi": normalized,
            "slots": slots.model_dump(),
            "eligibility": eligibility.model_dump(),
            "bot_reply_hindi": self._reply(slots, eligibility),
        }
        return self._assemble(query, base, chunks, required, payload)

    def _reply(self, slots: FormSlots, eligibility: EligibilityResult) -> str:
        if eligibility.is_eligible is False:
            return (
                "आपकी जानकारी के अनुसार आप इस योजना की पात्रता में नहीं आते। "
                "अन्य योजनाओं के बारे में जानना चाहें तो बताइए।"
            )
        return _next_question(slots.missing_fields)

    # -- LLM providers -----------------------------------------------------

    def _call_llm(
        self,
        backend: str,
        query: CitizenQuery,
        base: FormSlots,
        chunks: list[RetrievedChunk],
        required: list[str],
    ) -> dict:
        messages = _build_messages(query, base, chunks, required)
        if backend == "groq":
            content = self._call_groq(messages)
        else:
            content = self._call_deepseek(messages)

        payload = json.loads(content)
        if "slots" not in payload:
            raise ValueError("LLM response missing 'slots'")
        return payload

    def _call_groq(self, messages: list[dict[str, str]]) -> str:
        from groq import Groq

        client = Groq(api_key=_groq_key() or "")
        completion = client.chat.completions.create(
            model=GROQ_URL_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return completion.choices[0].message.content or ""

    def _call_deepseek(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {_deepseek_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    # -- assembly ----------------------------------------------------------

    def _finalize_slots(
        self, base: FormSlots, patch: FormSlots, required: list[str]
    ) -> FormSlots:
        """Merge without ever erasing a previously collected value."""
        merged = base.model_copy(deep=True)
        for field in FormSlots.model_fields:
            if field == "missing_fields":
                continue
            new_value = getattr(patch, field)
            if field == "scheme_type" and new_value == "unknown":
                continue
            if new_value is not None and new_value != "":
                setattr(merged, field, new_value)
        merged.missing_fields = _missing_fields(merged, required)
        return merged

    def _assemble(
        self,
        query: CitizenQuery,
        base: FormSlots,
        chunks: list[RetrievedChunk],
        required: list[str],
        payload: dict,
    ) -> AgentTurnResponse:
        try:
            patch = FormSlots.model_validate(payload.get("slots") or {})
        except Exception as exc:
            print(f"[agent] invalid slots in response ({exc}); keeping previous", file=sys.stderr)
            patch = FormSlots()

        slots = self._finalize_slots(base, patch, required)

        try:
            eligibility = EligibilityResult.model_validate(payload.get("eligibility") or {})
        except Exception as exc:
            print(f"[agent] invalid eligibility ({exc}); falling back", file=sys.stderr)
            eligibility = _evaluate_eligibility(slots, chunks, self.retriever)

        if not eligibility.source_citations:
            eligibility.source_citations = list(
                dict.fromkeys(
                    f"{c.source_file}#{c.section}"
                    for c in _eligibility_chunks(self.retriever, slots.scheme_type, chunks)
                )
            )

        return AgentTurnResponse(
            transcript=query.raw_text,
            normalized_hindi=payload.get("normalized_hindi") or normalize_hindi(query.raw_text),
            slots=slots,
            eligibility=eligibility,
            rag_citations=chunks,
            bot_reply_hindi=payload.get("bot_reply_hindi") or _next_question(slots.missing_fields),
            tts_audio_url=None,
            ready_for_docs=self._ready_for_docs(slots),
            ready_for_pdf=slots.scheme_type != "unknown" and not slots.missing_fields,
        )

    @staticmethod
    def _ready_for_docs(slots: FormSlots) -> bool:
        if slots.scheme_type == "unknown":
            return False
        keys = KEY_ELIGIBILITY_SLOTS.get(slots.scheme_type, ())
        return all(getattr(slots, key) is not None for key in keys)


@lru_cache(maxsize=1)
def get_agent() -> JanSetuAgent:
    """Process-wide singleton so FastAPI reuses the retriever and its index."""
    return JanSetuAgent()


# ---------------------------------------------------------------- CLI self-test

TURN_1 = (
    "हमका बुढ़ापा पेंशन खातिर आवेदन करे के बा, हमार नाम रामेश्वर यादव ह, "
    "उमर साठ बरिस बा, हमार गाँव बलिया जिला में बा।"
)
TURN_2 = (
    "मेरा पिता का नाम हरिहर यादव है, सालाना आय 36000 रुपये है और "
    "पता ग्राम रामपुर, बलिया है।"
)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    agent = JanSetuAgent()
    print(f"LLM backend: {llm_backend()}\n")

    first = agent.process_turn(CitizenQuery(raw_text=TURN_1))
    print("=== TURN 1 ===")
    print(first.model_dump_json(indent=2))

    second = agent.process_turn(CitizenQuery(raw_text=TURN_2), current_slots=first.slots)
    print("\n=== TURN 2 ===")
    print(second.model_dump_json(indent=2))

    print("\n=== CHECKS ===")
    preserved = [
        field
        for field in ("scheme_type", "applicant_name", "age", "district")
        if getattr(second.slots, field) == getattr(first.slots, field)
    ]
    print(f"preserved from turn 1: {preserved}")
    print(f"ready_for_pdf: {second.ready_for_pdf}")
    print(f"missing_fields: {second.slots.missing_fields}")
    assert first.slots.applicant_name == second.slots.applicant_name, "name was lost"
    assert first.slots.age == second.slots.age, "age was lost"
    assert first.slots.district == second.slots.district, "district was lost"
    assert second.ready_for_pdf, "slots should be complete after turn 2"
    print("OK: turn 2 preserved turn 1 slots and set ready_for_pdf=True")
