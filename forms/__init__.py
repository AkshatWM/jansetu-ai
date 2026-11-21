"""Module 5: Pre-Filled PDF Form Generator.

Public surface (lazily imported, so ``python forms/pdf_builder.py`` stays warning-free):
    from forms import generate_application_pdf, find_devanagari_font, find_latin_font
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from forms.pdf_builder import (
        find_devanagari_font,
        find_latin_font,
        generate_application_pdf,
    )

_LAZY = {"find_devanagari_font", "find_latin_font", "generate_application_pdf"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from forms import pdf_builder as _pdf_builder

        return getattr(_pdf_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY)


__all__ = sorted(_LAZY)
