"""
main.py
──────────────────────────────────────────────────────────────────────────────
Root execution entry point for Kisan-Agent FastAPI Server.

USAGE:
  python main.py
  OR
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import sys
import uvicorn

# Ensure UTF-8 output across Windows CMD / PowerShell consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.api.webhook import app, twilio_whatsapp_webhook
from app.core.config import get_settings

__all__ = ["app", "twilio_whatsapp_webhook"]

settings = get_settings()


def configure_logging() -> None:
    """Set up structured logging format for development and production."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence verbose low-level network logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    os.makedirs(settings.tmp_dir, exist_ok=True)
    os.makedirs(settings.audit_log_dir, exist_ok=True)

    print(f"""
    ===================================================================
      🌾 KISAN-AGENT: AUTONOMOUS FARMER SUBSIDY EXECUTION ENGINE 🌾
      Problem Statement 3: Autonomous Public Service Execution Agent
    ===================================================================
      Port:        {settings.port}
      Environment: {settings.environment}
      Model:       {settings.gemini_model}
      Webhook URL: http://localhost:{settings.port}/webhook/twilio
    ===================================================================
    """)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

