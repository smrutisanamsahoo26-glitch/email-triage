"""
FastAPI server for `email_triage_env` with integrated Gradio UI.

This server exposes:
- Gradio UI at root (/)
- OpenEnv-standard endpoints (via `openenv.core.env_server.create_fastapi_app`)
- Hackathon endpoints: /tasks, /grader, /baseline
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import gradio as gr
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, Response

from openenv.core.env_server import create_fastapi_app

from env import EmailTriagerEnvironment
from models import EmailTriageAction, EmailTriageObservation
from tasks import TASKS
from env import get_last_score
from env import get_last_breakdown
from baseline import run_baseline


# Create the FastAPI app with OpenEnv endpoints
fastapi_app = create_fastapi_app(
    env=lambda: EmailTriagerEnvironment(),
    action_cls=EmailTriageAction,
    observation_cls=EmailTriageObservation,
)


# ============================================================================
# Gradio UI Logic - Calls local FastAPI endpoints
# ============================================================================

def analyze_email_ui(email: str) -> tuple[str, str, str, str]:
    """
    Gradio interface function that:
    1. Resets the environment
    2. Sends the action to /step
    3. Returns category, priority, response, score
    """
    import requests
    
    try:
        if not email.strip():
            return ("Error", "-", "Please enter an email", "-")

        # Use local FastAPI endpoints
        base_url = "http://127.0.0.1:7860"
        
        # Step 1: Reset
        reset_res = requests.post(f"{base_url}/reset", json={"seed": 42})
        if reset_res.status_code != 200:
            return ("Error", "-", f"Reset failed: {reset_res.text}", "-")
        
        # Step 2: Smart action generation (simple heuristic)
        email_lower = email.lower()
        if "refund" in email_lower or "charged" in email_lower:
            action = {"category": "billing", "priority": 4, "response": "We apologize for the billing issue. We will process your refund immediately."}
        elif "password" in email_lower or "reset" in email_lower:
            action = {"category": "technical", "priority": 3, "response": "Please try resetting your password again and check your spam folder."}
        elif "blocked" in email_lower or "can't access" in email_lower:
            action = {"category": "technical", "priority": 5, "response": "We'll help restore your access. Try resetting your password first."}
        else:
            action = {"category": "query", "priority": 2, "response": "Thank you for your email. We will review your request and respond shortly."}
        
        # Step 3: Call /step with action
        step_res = requests.post(f"{base_url}/step", json={"action": action})
        if step_res.status_code != 200:
            return ("Error", "-", f"Step failed: {step_res.text}", "-")
        
        result = step_res.json()
        reward = result.get("reward", "N/A")
        
        return (
            action["category"],
            str(action["priority"]),
            action["response"],
            f"{reward:.2f}" if isinstance(reward, float) else str(reward)
        )
    
    except Exception as e:
        return ("Error", "-", f"Exception: {str(e)}", "-")


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Email Triage AI") as gradio_app:
    gr.Markdown("# 📧 Email Triage AI")
    gr.Markdown("Classify emails, assign priority, and generate responses instantly.")
    
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(
                label="Email",
                lines=8,
                placeholder="Paste customer email here..."
            )
            analyze_btn = gr.Button("Analyze Email", size="lg")
        
        with gr.Column():
            category_output = gr.Textbox(label="Category", interactive=False)
            priority_output = gr.Textbox(label="Priority", interactive=False)
            response_output = gr.Textbox(label="Generated Response", interactive=False, lines=4)
            score_output = gr.Textbox(label="Score", interactive=False)
    
    analyze_btn.click(
        fn=analyze_email_ui,
        inputs=email_input,
        outputs=[category_output, priority_output, response_output, score_output]
    )


# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")


# Add custom endpoints to the underlying FastAPI app
@fastapi_app.get("/tasks")
def list_tasks() -> List[Dict[str, Any]]:
    """Return all task definitions (public view)."""
    return [t.public_view() for t in TASKS]


@fastapi_app.get("/grader")
def get_last_grader_score() -> Dict[str, Any]:
    """Return the last rubric-derived score for the latest episode_id."""
    return {"last_score": get_last_score(), "last_breakdown": get_last_breakdown()}


@fastapi_app.get("/baseline")
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


if __name__ == "__main__":
    main()