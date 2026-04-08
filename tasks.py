from __future__ import annotations

"""
Task definitions for the OpenEnv environment.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple

from grader import grade_task_1, grade_task_2, grade_task_3
from models import EmailTriageAction, EmailTriageObservation, EmailTriageState, Reward


TaskDifficulty = Literal["easy", "medium", "hard"]


# ---------------------------
# Task Definition
# ---------------------------
@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: TaskDifficulty
    input_observation: Dict[str, Any]
    expected: Dict[str, Any]
    required_keywords: List[str]
    required_response_elements: List[str]
    context_keywords: List[str]
    grader_fn: Callable[
        [EmailTriageAction, EmailTriageObservation, EmailTriageState],
        Tuple[Reward, Dict[str, float]],
    ]

    def public_view(self) -> Dict[str, Any]:
        """Public safe version for API exposure."""
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "input_observation": self.input_observation,
            "expected": self.expected,
            "required_keywords": self.required_keywords,
            "required_response_elements": self.required_response_elements,
            "grader": "deterministic rubric (keywords + tone + completeness)",
        }


# ---------------------------
# Helper
# ---------------------------
def _history() -> List[str]:
    return []


# ---------------------------
# Tasks List
# ---------------------------
TASKS: List[TaskDefinition] = [

    # ---------------------------
    # TASK 1 (Easy - Billing)
    # ---------------------------
    TaskDefinition(
        task_id="task_1",
        difficulty="easy",
        input_observation={
            "email_text": (
                "From: Customer\n"
                "Subject: Charged twice — I need a refund\n\n"
                "Hi Support,\n\n"
                "I was charged twice for my last order (Order #18492). "
                "Please reverse the duplicate charge and confirm the refund.\n\n"
                "Thanks,\nAvery"
            ),
            "sender": "avery@example.com",
            "history": _history(),
        },
        expected={
            "category": "billing",
            "priority": 4,
            "tone": "polite",
            "response_elements": [
                "reverse the duplicate charge",
                "confirm the refund",
                "order #18492",
            ],
        },
        required_keywords=[
            "charged twice",
            "refund",
            "order #18492",
        ],
        required_response_elements=[
            "reverse the duplicate charge",
            "confirm the refund",
            "order #18492",
        ],
        context_keywords=[
            "charged",
            "refund",
            "order",
        ],
        grader_fn=grade_task_1,
    ),

    # ---------------------------
    # TASK 2 (Medium - Technical)
    # ---------------------------
    TaskDefinition(
        task_id="task_2",
        difficulty="medium",
        input_observation={
            "email_text": (
                "From: User\n"
                "Subject: Password reset link not working\n\n"
                "Hello,\n\n"
                "I requested a password reset, but the reset link fails. "
                "When I open it, I get an error page.\n\n"
                "Regards,\nNoah"
            ),
            "sender": "noah@example.com",
            "history": _history(),
        },
        expected={
            "category": "technical",
            "priority": 3,
            "tone": "polite",
            "response_elements": [
                "check spam",
                "use latest reset link",
                "try different browser",
                "clear cache",
            ],
        },
        required_keywords=[
            "password reset",
            "reset link",
            "error",
        ],
        required_response_elements=[
            "check spam",
            "use latest reset link",
            "try different browser",
            "clear cache",
        ],
        context_keywords=[
            "password",
            "reset",
            "error",
        ],
        grader_fn=grade_task_2,
    ),

    # ---------------------------
    # TASK 3 (Hard - Multi-Issue)
    # ---------------------------
    TaskDefinition(
        task_id="task_3",
        difficulty="hard",
        input_observation={
            "email_text": (
                "From: Customer\n"
                "Subject: Multiple issues with account\n\n"
                "Hi Team,\n\n"
                "I can't access my account (blocked after login). "
                "Also, I see an unexpected charge. "
                "And I'm receiving promotional emails I didn't subscribe to.\n\n"
                "Please help.\n\n"
                "Thanks,\nMia"
            ),
            "sender": "mia@example.com",
            "history": _history(),
        },
        expected={
            "category": "technical",
            "priority": 5,
            "tone": "polite",
            "response_elements": [
                "reset password",
                "verify email",
                "unsubscribe",
                "review charge",
                "next steps",
            ],
        },
        required_keywords=[
            "account",
            "blocked",
            "unexpected charge",
            "promotional emails",
        ],
        required_response_elements=[
            "reset password",
            "verify email",
            "unsubscribe",
            "review charge",
            "next steps",
        ],
        context_keywords=[
            "access",
            "blocked",
            "charge",
            "unsubscribe",
        ],
        grader_fn=grade_task_3,
    ),
]