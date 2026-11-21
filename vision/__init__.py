"""Module 4: Document Pre-Check Guard (blur gate + name/field mismatch gate).

Public surface (lazily imported, so ``python vision/doc_guard.py`` stays warning-free):
    from vision import check_document, compute_blur_score, vision_backend
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from vision.doc_guard import check_document, compute_blur_score, vision_backend

_LAZY = {"check_document", "compute_blur_score", "vision_backend"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from vision import doc_guard as _doc_guard

        return getattr(_doc_guard, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY)


__all__ = sorted(_LAZY)
