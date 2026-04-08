"""
Deterministic grading logic for the email triage tasks.

This module implements:
- tone detection helper
- keyword/intent matching helpers
- per-task deterministic rubric scoring with partial credit
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

from models import EmailTriageAction, EmailTriageObservation, EmailTriageState, Reward


_RE_MULTISPACE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower().strip()
    text = _RE_MULTISPACE.sub(" ", text)
    return text


def tone_polite_score(response: str) -> float:
    """
    Deterministic polite-tone score in [0, 1].

    Signals:
    - presence of common politeness markers
    - presence of a greeting/sign-off
    - presence of apology/acknowledgement
    """

    resp = _normalize_text(response)
    if not resp:
        return 0.0

    polite_markers = [
        "please",
        "thanks",
        "thank you",
        "sincerely",
        "regards",
        "kindly",
        "could you",
        "we can",
        "we'll",
        "we would",
        "i appreciate",
        "thank-you",
    ]
    apology_markers = ["sorry", "apologize", "apologies", "we apologize"]
    greeting_markers = ["hello", "hi ", "dear ", "good morning", "good afternoon", "greetings"]
    signoff_markers = ["sincerely", "regards", "best", "thanks,", "thank you,"]

    rude_markers = ["idiot", "stupid", "jerk", "unacceptable!!!", "shut up"]

    if any(marker in resp for marker in rude_markers):
        return 0.0

    marker_count = sum(1 for m in polite_markers if m in resp)
    apology_count = sum(1 for m in apology_markers if m in resp)
    greeting_count = sum(1 for m in greeting_markers if m in resp)
    signoff_count = sum(1 for m in signoff_markers if m in resp)

    # Smooth partial credit.
    score = 0.0
    score += min(0.6, marker_count * 0.15)
    score += min(0.25, apology_count * 0.20)
    score += min(0.15, greeting_count * 0.10)
    score += min(0.15, signoff_count * 0.10)

    return max(0.0, min(1.0, score))


def _keyword_match_fraction(text: str, keywords: List[str]) -> float:
    """
    Fraction of keywords that appear as substrings in `text` (after normalization).
    """
    t = _normalize_text(text)
    if not keywords:
        return 0.0
    hits = 0
    for kw in keywords:
        kw_norm = _normalize_text(kw)
        if kw_norm and kw_norm in t:
            hits += 1
    return hits / len(keywords)


def _multi_step_element_fraction(text: str, required_elements: List[str]) -> float:
    """
    Multi-step scoring: each required element contributes equally.
    """
    return _keyword_match_fraction(text, required_elements)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _empty_response_penalty(response: str) -> float:
    return 0.6 if not (response or "").strip() else 0.0


def _irrelevant_penalty(response: str, context_keywords: List[str]) -> float:
    """
    Penalize responses that don't overlap with the email's context.

    This is intentionally deterministic and conservative: only a *small* penalty.
    """
    resp = _normalize_text(response)
    if not resp:
        return 0.0
    overlap = _keyword_match_fraction(resp, context_keywords)
    return 0.15 if overlap <= 0.05 else 0.0


def _score_breakdown(
    *,
    category_correct: float,
    priority_correct: float,
    tone: float,
    keyword: float,
) -> Dict[str, float]:
    return {
        "category_correct": category_correct,
        "priority_correct": priority_correct,
        "tone_polite": tone,
        "keyword_intent": keyword,
    }


def _priority_partial_score(action_priority: int, expected_priority: int) -> float:
    """
    Partial credit based on distance. Exact match => 1.0.
    Off by 1 => 0.7, off by 2 => 0.4, off >=3 => 0.0
    """
    diff = abs(int(action_priority) - int(expected_priority))
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7
    if diff == 2:
        return 0.4
    return 0.0


def _generic_rubric_grade(
    *,
    action: EmailTriageAction,
    expected_category: str,
    expected_priority: int,
    required_keywords: List[str],
    required_response_elements: List[str],
    context_keywords: List[str],
    response_keyword_weight: float = 0.5,
) -> Tuple[Reward, Dict[str, float]]:
    """
    Compute rubric score using fixed weights:
    - category correctness: 0.3
    - priority correctness: 0.2
    - tone detection: 0.2
    - keyword/intent matching: 0.3
    """
    response = action.response or ""

    if not response.strip():
        breakdown = _score_breakdown(
            category_correct=1.0 if action.category == expected_category else 0.0,
            priority_correct=_priority_partial_score(action.priority, expected_priority),
            tone=0.0,
            keyword=0.0,
        )
        return Reward(score=0.0), breakdown

    category_correct = 1.0 if action.category == expected_category else 0.0
    priority_correct = _priority_partial_score(action.priority, expected_priority)
    tone = tone_polite_score(response)

    kw_score = _keyword_match_fraction(response, required_keywords)
    elem_score = _multi_step_element_fraction(response, required_response_elements)
    keyword_intent = _clamp01(response_keyword_weight * kw_score + (1.0 - response_keyword_weight) * elem_score)

    base_score = (
        0.3 * category_correct
        + 0.2 * priority_correct
        + 0.2 * tone
        + 0.3 * keyword_intent
    )

    # Deterministic penalties (still allow partial credit).
    penalty_empty = _empty_response_penalty(response)
    penalty_irrelevant = _irrelevant_penalty(response, context_keywords)

    final_score = _clamp01(base_score - penalty_empty - penalty_irrelevant)
    breakdown = _score_breakdown(
        category_correct=category_correct,
        priority_correct=priority_correct,
        tone=tone,
        keyword=keyword_intent,
    )
    return Reward(score=final_score), breakdown


def grade_task_1(
    action: EmailTriageAction, observation: EmailTriageObservation, state: EmailTriageState
) -> Tuple[Reward, Dict[str, float]]:
    expected_category = "billing"
    expected_priority = 4
    required_keywords = ["charged twice", "refund", "order #18492"]
    required_response_elements = [
        "reverse the duplicate charge",
        "confirm the refund",
        "order #18492",
    ]
    context_keywords = ["charged", "refund", "order"]
    return _generic_rubric_grade(
        action=action,
        expected_category=expected_category,
        expected_priority=expected_priority,
        required_keywords=required_keywords,
        required_response_elements=required_response_elements,
        context_keywords=context_keywords,
        response_keyword_weight=0.6,
    )


def grade_task_2(
    action: EmailTriageAction, observation: EmailTriageObservation, state: EmailTriageState
) -> Tuple[Reward, Dict[str, float]]:
    expected_category = "technical"
    expected_priority = 3
    required_keywords = ["password reset", "reset link", "verification email", "try again"]
    required_response_elements = [
        "check spam",
        "use the latest reset link",
        "try a different browser",
        "clear cache",
    ]
    context_keywords = ["password reset", "reset link", "error"]
    return _generic_rubric_grade(
        action=action,
        expected_category=expected_category,
        expected_priority=expected_priority,
        required_keywords=required_keywords,
        required_response_elements=required_response_elements,
        context_keywords=context_keywords,
        response_keyword_weight=0.55,
    )


def grade_task_3(
    action: EmailTriageAction, observation: EmailTriageObservation, state: EmailTriageState
) -> Tuple[Reward, Dict[str, float]]:
    expected_category = "technical"
    expected_priority = 5
    required_keywords = ["access", "account blocked", "unexpected charge", "promotional emails"]
    required_response_elements = [
        "reset password",
        "verify email",
        "unsubscribe",
        "review the charge",
        "next steps",
    ]
    context_keywords = ["can't access", "blocked", "unexpected charge", "promotional", "unsubscribe"]
    return _generic_rubric_grade(
        action=action,
        expected_category=expected_category,
        expected_priority=expected_priority,
        required_keywords=required_keywords,
        required_response_elements=required_response_elements,
        context_keywords=context_keywords,
        response_keyword_weight=0.45,
    )

