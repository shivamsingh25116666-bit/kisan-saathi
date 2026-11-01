"""
app/main.py
──────────────────────────────────────────────────────────────────────────────
Application entry point. Configures structured logging, then starts Uvicorn.
"""

from __future__ import annotations

import logging
import os

import uvicorn

from app.core.config import get_settings

settings = get_settings()


def _configure_logging() -> None:
    """Set up structured logging with appropriate level from config."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def main() -> None:
    _configure_logging()
    os.makedirs(settings.tmp_dir, exist_ok=True)
    os.makedirs(settings.audit_log_dir, exist_ok=True)

    uvicorn.run(
        "app.api.webhook:app",
        host="0.0.0.0",
        port=8000,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()

