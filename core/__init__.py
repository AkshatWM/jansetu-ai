"""Module 3: Agentic Slot-Filler & Eligibility Engine for JanSetu AI.

Public surface (lazily imported, so ``python core/agent.py`` stays warning-free):
    from core import JanSetuAgent, get_agent
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from core.agent import JanSetuAgent, get_agent

_LAZY = {"JanSetuAgent", "get_agent"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from core import agent as _agent

        return getattr(_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY)


__all__ = sorted(_LAZY)
