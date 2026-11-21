"""
rag/retriever.py — Module 1: Hybrid grounded retrieval over curated UP scheme docs.

Pipeline
    1. Load every ``data/schemes/*.md`` file and chunk it on ``## `` headings.
    2. Dense index   : sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
                       -> normalized embeddings -> faiss.IndexFlatIP (cosine similarity).
    3. Sparse index  : rank_bm25.BM25Okapi over lowercased word tokens.
    4. Fusion        : Reciprocal Rank Fusion (RRF, k=60) of both rankings.

Only local models are used (no LangChain, no API keys), so ``MOCK_MODE`` is irrelevant here.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Works for both `uv run python -m rag.retriever` and `uv run python rag/retriever.py`:
# the latter puts rag/ (not the repo root) on sys.path, so `import schemas` would fail.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from schemas import RetrievedChunk

DATA_DIR = ROOT / "data" / "schemes"
MODEL_DIR = ROOT / "data" / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Devanagari must be spelled out explicitly: Python's \w excludes combining marks
# (matras such as ि ी ु), so "\w+" would shred "बिजली" into single consonants.
_TOKEN = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; Devanagari words stay intact."""
    return _TOKEN.findall(text.lower())


# Roman-Hindi ("Hinglish") is how many citizens actually type: "bijli bill ki
# shikayat kaha kare". BM25 cannot match that against देवनागरी or English text, so
# expand known domain words before scoring. Dense retrieval is left untouched —
# this is a lexical bridge only, and it is trivial to extend for new schemes.
QUERY_EXPANSIONS: dict[str, str] = {
    "bijli": "बिजली electricity",
    "shikayat": "शिकायत grievance complaint",
    "pension": "पेंशन pension",
    "kisan": "किसान farmer",
    "kheti": "खेती farming land",
    "jati": "जाति caste",
    "jaati": "जाति caste",
    "aay": "आय income",
    "income": "आय",
    "certificate": "प्रमाण पत्र",
    "praman": "प्रमाण पत्र certificate",
    "khatauni": "खतौनी land",
    "zila": "जिला district",
    "jila": "जिला district",
    "pani": "पानी water",
    "paani": "पानी water",
    "rashan": "राशन ration",
    "ration": "राशन",
    "aadhar": "आधार aadhaar",
}


def expand_query(text: str) -> str:
    """Append canonical Devanagari/English terms for Roman-Hindi keywords."""
    extra = [QUERY_EXPANSIONS[tok] for tok in tokenize(text) if tok in QUERY_EXPANSIONS]
    return f"{text} {' '.join(extra)}" if extra else text


def chunk_markdown(path: Path) -> list[dict]:
    """Split one Markdown file on ``## `` headings.

    ``index_text`` prepends the document's H1 title because section names such as
    "Eligibility" are identical across all five scheme files; without the title the
    dense and BM25 rankings cannot tell them apart.
    """
    raw = path.read_text(encoding="utf-8")
    title_match = _H1.search(raw)
    title = title_match.group(1).strip() if title_match else path.stem

    headings = list(_H2.finditer(raw))
    chunks: list[dict] = []
    for i, match in enumerate(headings):
        section = match.group(1).strip()
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
        body = raw[start:end].strip()
        if not body:
            continue
        chunks.append(
            {
                "source_file": path.name,
                "section": section,
                "content": body,
                "index_text": f"{title}\n{section}\n{body}",
            }
        )
    return chunks


def resolve_model_source(model_name: str = EMBEDDING_MODEL) -> str:
    """Prefer a vendored copy in ``data/models/`` so demos work offline."""
    if model_name == EMBEDDING_MODEL and (MODEL_DIR / "modules.json").exists():
        return str(MODEL_DIR)
    return model_name


class SchemeRetriever:
    """Hybrid FAISS + BM25 retriever with Reciprocal Rank Fusion."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        model_name: str = EMBEDDING_MODEL,
        rrf_k: int = RRF_K,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.rrf_k = rrf_k
        self.chunks: list[dict] = []

        for path in sorted(self.data_dir.glob("*.md")):
            self.chunks.extend(chunk_markdown(path))

        if not self.chunks:
            raise FileNotFoundError(
                f"No scheme Markdown files found in {self.data_dir}. "
                "Expected files such as old_age_pension.md."
            )

        corpus = [c["index_text"] for c in self.chunks]

        # --- Sparse: BM25 over lowercased word tokens ---
        self.bm25 = BM25Okapi([tokenize(text) for text in corpus])

        # --- Dense: normalized embeddings in an inner-product index ---
        self.model = SentenceTransformer(resolve_model_source(model_name))
        embeddings = self.model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    # ------------------------------------------------------------------ helpers

    def _dense_ranking(self, query: str, limit: int) -> list[int]:
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        distances, ids = self.index.search(vector, limit)
        return [int(i) for i in ids[0] if i != -1]

    def _sparse_ranking(self, query: str, limit: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(expand_query(query)))
        # Drop zero-score chunks: with no lexical overlap, argsort would hand RRF an
        # arbitrary ranking, letting an uninformative sparse side swamp the dense results.
        ranked = [int(i) for i in np.argsort(scores)[::-1] if scores[i] > 0]
        return ranked[:limit]

    def _rrf_fuse(self, rankings: list[list[int]]) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) with rank starting at 1."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)

    # -------------------------------------------------------------------- public

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return the ``top_k`` most relevant chunks, fused from dense + sparse rankings."""
        if not query.strip() or top_k <= 0:
            return []

        pool = min(len(self.chunks), max(top_k * 5, 10))
        rankings = [self._dense_ranking(query, pool), self._sparse_ranking(query, pool)]

        results: list[RetrievedChunk] = []
        for chunk_id, score in self._rrf_fuse(rankings)[:top_k]:
            chunk = self.chunks[chunk_id]
            results.append(
                RetrievedChunk(
                    source_file=chunk["source_file"],
                    section=chunk["section"],
                    content=chunk["content"],
                    score=round(score, 6),
                )
            )
        return results


@lru_cache(maxsize=1)
def get_retriever() -> SchemeRetriever:
    """Process-wide singleton so FastAPI loads the embedding model only once."""
    return SchemeRetriever()


def retrieve(query: str, top_k: int = 3) -> list[RetrievedChunk]:
    """Convenience wrapper used by core/agent.py."""
    return get_retriever().search(query, top_k=top_k)


# ---------------------------------------------------------------- CLI self-test

if __name__ == "__main__":
    # Windows consoles default to cp1252, which cannot encode Devanagari.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    retriever = SchemeRetriever()
    print(f"Indexed {len(retriever.chunks)} chunks from {retriever.data_dir}\n")

    test_queries = [
        "बुढ़ापा पेंशन के लिए कितनी आय होनी चाहिए?",
        "income certificate fee and time in UP",
        "bijli bill ki shikayat kaha kare",
    ]

    for query in test_queries:
        print(f"QUERY: {query}")
        for rank, hit in enumerate(retriever.search(query, top_k=3), start=1):
            print(f"  {rank}. {hit.source_file}#{hit.section}  (score={hit.score:.6f})")
        print()
