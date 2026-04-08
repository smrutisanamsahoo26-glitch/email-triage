from __future__ import annotations

"""
FastAPI server for `email_triage_env`.

This server exposes:
- OpenEnv-standard endpoints (via create_fastapi_app)
- Hackathon endpoints:
  - GET /tasks
  - GET /grader
  - GET /baseline
"""

import os
import logging
from time import time
from typing import Any, Dict, List

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, Response

from openenv.core.env_server import create_fastapi_app

from env import EmailTriagerEnvironment, get_last_score, get_last_breakdown
from models import EmailTriageAction, EmailTriageObservation
from tasks import TASKS
from baseline import run_baseline


# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI App via OpenEnv
# ---------------------------
app = create_fastapi_app(
    env=lambda: EmailTriagerEnvironment(),
    action_cls=EmailTriageAction,
    observation_cls=EmailTriageObservation,
)

# ---------------------------
# Rate Limiting (for baseline)
# ---------------------------
_last_baseline_call = 0


# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------
# Tasks Endpoint (FIXED ✅)
# ---------------------------
@app.get("/tasks")
def list_tasks():
    return [t.public_view() for t in TASKS]


# ---------------------------
# Grader Endpoint
# ---------------------------
@app.get("/grader")
def get_last_grader_score() -> Dict[str, Any]:
    return {
        "last_score": get_last_score(),
        "breakdown": get_last_breakdown(),
        "explanation": "Score computed using category, priority, tone, intent, and completeness metrics."
    }


# ---------------------------
# Baseline Endpoint
# ---------------------------
@app.get("/baseline")
def run_baseline_endpoint(seed: int | None = None) -> Dict[str, Any]:
    global _last_baseline_call

    now = time()
    if now - _last_baseline_call < 2:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")

    _last_baseline_call = now

    try:
        logger.info("Running baseline evaluation")

        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise HTTPException(
                status_code=400,
                detail="OPENAI_API_KEY not set."
            )

        resolved_seed = int(seed) if seed is not None else int(os.environ.get("BASELINE_SEED", "42"))

        results = run_baseline(seed=resolved_seed)

        return {
            "status": "success",
            "seed": resolved_seed,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Baseline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Web UI
# ---------------------------
@app.get("/web")
def web_home() -> HTMLResponse:
    html = """
    <html>
      <body style="font-family: sans-serif; margin: 2rem;">
        <h2>email_triage_env 🚀</h2>
        <ul>
          <li>/reset</li>
          <li>/step</li>
          <li>/state</li>
          <li>/tasks</li>
          <li>/grader</li>
          <li>/baseline</li>
        </ul>
        <a href="/docs">Docs</a>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


# ---------------------------
# Root Endpoint
# ---------------------------
@app.get("/")
def root():
    return {
        "name": "email_triage_env",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# ---------------------------
# Favicon Fix
# ---------------------------
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ---------------------------
# Entrypoint
# ---------------------------
def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    logger.info(f"Starting server on port {port}")

    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()