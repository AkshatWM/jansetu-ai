"""Module 1: Grounded RAG package.

Public surface (lazily imported so ``python -m rag.retriever`` stays warning-free):
    from rag import SchemeRetriever, get_retriever, retrieve
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from rag.retriever import SchemeRetriever, get_retriever, retrieve

_LAZY = {"SchemeRetriever", "get_retriever", "retrieve"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from rag import retriever as _retriever

        return getattr(_retriever, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY)


__all__ = sorted(_LAZY)
