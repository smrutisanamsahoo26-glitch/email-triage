"""
FastAPI server for `email_triage_env`.

This server exposes:
- OpenEnv-standard endpoints (via `openenv.core.env_server.create_fastapi_app`)
- Hackathon endpoints:
  - GET /tasks
  - GET /grader
  - GET /baseline
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, Response

from openenv.core.env_server import create_fastapi_app

from env import EmailTriagerEnvironment
from models import EmailTriageAction, EmailTriageObservation
from tasks import TASKS
from env import get_last_score
from env import get_last_breakdown
from baseline import run_baseline


app = create_fastapi_app(
    env=lambda: EmailTriagerEnvironment(),
    action_cls=EmailTriageAction,
    observation_cls=EmailTriageObservation,
)


@app.get("/")
def root() -> HTMLResponse:
    """Root endpoint redirects to API documentation."""
    html = """
    <html>
      <head>
        <title>email_triage_env</title>
        <meta http-equiv="refresh" content="0; url=/docs" />
      </head>
      <body>
        <p>Redirecting to <a href="/docs">API documentation</a>...</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/tasks")
def list_tasks() -> List[Dict[str, Any]]:
    """Return all task definitions (public view)."""
    return [t.public_view() for t in TASKS]


@app.get("/grader")
def get_last_grader_score() -> Dict[str, Any]:
    """Return the last rubric-derived score for the latest episode_id."""
    return {"last_score": get_last_score(), "last_breakdown": get_last_breakdown()}


@app.get("/baseline")
def run_baseline_endpoint(seed: int | None = None) -> Dict[str, Any]:
    """
    Run the OpenAI-based baseline script and return scores.
    
    Requires OPENAI_API_KEY environment variable to be set.
    """
    try:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise HTTPException(
                status_code=400,
                detail="OPENAI_API_KEY environment variable is not set. Please configure your API key."
            )
        resolved_seed = int(seed) if seed is not None else int(os.environ.get("BASELINE_SEED", "42"))
        return run_baseline(seed=resolved_seed)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid seed value: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline execution failed: {str(e)}")


def main() -> None:
    """
    Entry point required by `openenv validate`.
    """
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


@app.get("/web")
def web_home() -> HTMLResponse:
    """
    Minimal web endpoint to avoid 404s in UIs that expect `/web`.

    For richer OpenEnv web UI, use OpenEnv's optional web interface.
    """
    html = """
    <html>
      <head><title>email_triage_env</title></head>
      <body style="font-family: sans-serif; margin: 2rem;">
        <h2>email_triage_env</h2>
        <p>Use the API endpoints:</p>
        <ul>
          <li><code>POST /reset</code></li>
          <li><code>POST /step</code></li>
          <li><code>GET /state</code></li>
          <li><code>GET /tasks</code></li>
          <li><code>GET /grader</code></li>
          <li><code>GET /baseline</code></li>
        </ul>
        <p>Docs: <a href="/docs">/docs</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/favicon.ico")
def favicon() -> Response:
    # Prevent noisy browser console errors for missing favicon.
    return Response(status_code=204)


if __name__ == "__main__":
    main()