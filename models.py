"""
Pydantic contracts for the `email_triage_env` OpenEnv environment.

OpenEnv uses Pydantic models for:
- Actions sent by the agent
- Observations returned by the environment
- Internal state accessible via the `/state` endpoint
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field, field_validator

EmailCategory = Literal["complaint", "query", "spam", "technical", "billing"]


class Reward(BaseModel):
    """Standalone reward model for task-grading helpers."""

    score: float = Field(..., ge=0.0, le=1.0, description="Reward score between 0.0 and 1.0")


class EmailTriageAction(Action):
    """
    Agent action for one email triage step.

    The agent must choose a `category`, a `priority` (1-5),
    and a `response` email draft.
    """

    category: EmailCategory = Field(
        ...,
        description="Email category: complaint, query, spam, technical, or billing"
    )
    priority: int = Field(
        ...,
        ge=1,
        le=5,
        description="Priority level from 1 (lowest) to 5 (highest)"
    )
    response: str = Field(
        ...,
        description="Drafted email response text",
        min_length=1
    )
    
    @field_validator('response')
    @classmethod
    def response_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Response cannot be empty")
        return v


class EmailTriageObservation(Observation):
    """
    Environment observation returned after each action.

    Note: OpenEnv extracts `reward` and `done` from the Observation base class.
    Any additional "info" is included as fields here so it is available in
    the JSON payload under `observation`.
    """

    # Required by the hackathon spec
    email_text: str = Field(..., description="The email text to classify and respond to")
    sender: str = Field(..., description="Email sender address or identifier")
    history: List[str] = Field(
        default_factory=list,
        description="Conversation history for context"
    )

    # Extra info fields for debugging and "info" requirements
    episode_id: str = Field(..., description="Episode ID for tracking")
    task_id: str = Field(..., description="Current task identifier")
    grader_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed score breakdown from grader"
    )


class EmailTriageState(State):
    """Internal state persisted across `/reset` and `/step` via `episode_id`."""

    current_task_index: int = Field(
        default=0,
        ge=0,
        description="Current task index in the episode"
    )
    scores: List[float] = Field(
        default_factory=list,
        description="List of scores for completed tasks"
    )
    last_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Last computed score"
    )
    current_task_id: str = Field(
        default="task_1",
        description="ID of the current task ('task_1', 'task_2', 'task_3', or 'done')"
    )

