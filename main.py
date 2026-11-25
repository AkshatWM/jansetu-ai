"""
JanSetu AI — single-command development entry point.

    uv run python main.py

Starts the FastAPI app on http://127.0.0.1:8000 (override with HOST / PORT).
For auto-reload while editing the UI or backend use:

    uv run uvicorn server:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    # Windows consoles default to cp1252, which cannot encode Devanagari logs.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from server import run

    run()


if __name__ == "__main__":
    main()
