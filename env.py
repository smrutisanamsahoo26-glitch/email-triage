"""
Environment logic for the `email_triage_env` OpenEnv environment.

Implements the OpenEnv simulation interface:
- reset(seed, episode_id) -> Observation
- step(action, episode_id) -> Observation
- state -> State

HTTP `/reset` and `/step` in OpenEnv may create fresh Environment instances,
so we persist episode/task progress in an in-memory store keyed by `episode_id`.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from models import EmailTriageAction, EmailTriageObservation, EmailTriageState, Reward
from tasks import TASKS, TaskDefinition


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_response_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join((text or "").strip().split()).lower()


def _word_set_similarity(a: str, b: str) -> float:
    """
    Deterministic similarity in [0, 1] using Jaccard overlap of word sets.
    """
    a_norm = _normalize_response_text(a)
    b_norm = _normalize_response_text(b)
    if not a_norm or not b_norm:
        return 0.0
    a_set = set(a_norm.split())
    b_set = set(b_norm.split())
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


@dataclass
class EpisodeData:
    episode_id: str
    seed: Optional[int] = None
    current_task_index: int = 0
    scores: List[float] = field(default_factory=list)
    last_score: Optional[float] = None
    last_breakdown: Dict[str, float] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    agent_responses: List[str] = field(default_factory=list)
    agent_categories: List[str] = field(default_factory=list)

    def current_task(self) -> TaskDefinition:
        return TASKS[self.current_task_index]

    def is_done(self) -> bool:
        return self.current_task_index >= len(TASKS)


_EPISODES: Dict[str, EpisodeData] = {}
_LATEST_EPISODE_ID: Optional[str] = None
_LOCK = threading.Lock()

def get_latest_episode_id() -> Optional[str]:
    with _LOCK:
        return _LATEST_EPISODE_ID


def get_last_score() -> Optional[float]:
    with _LOCK:
        if not _LATEST_EPISODE_ID:
            return None
        ep = _EPISODES.get(_LATEST_EPISODE_ID)
        return None if ep is None else ep.last_score


def get_last_breakdown() -> Dict[str, float]:
    with _LOCK:
        if not _LATEST_EPISODE_ID:
            return {}
        ep = _EPISODES.get(_LATEST_EPISODE_ID)
        return {} if ep is None else dict(ep.last_breakdown)


def get_latest_state() -> EmailTriageState:
    with _LOCK:
        if not _LATEST_EPISODE_ID:
            return EmailTriageState(
                episode_id=None,
                step_count=0,
                current_task_index=0,
                scores=[],
                last_score=None,
                current_task_id="task_1"  # Default to first task
            )
        episode = _EPISODES.get(_LATEST_EPISODE_ID)
        if episode is None:
            return EmailTriageState(
                episode_id=None,
                step_count=0,
                current_task_index=0,
                scores=[],
                last_score=None,
                current_task_id="task_1"
            )
        # Determine current task ID
        if episode.is_done():
            task_id = "done"
        elif episode.current_task_index < len(TASKS):
            task_id = TASKS[episode.current_task_index].task_id
        else:
            task_id = "done"
        
        return EmailTriageState(
            episode_id=episode.episode_id,
            step_count=len(episode.scores),
            current_task_index=episode.current_task_index,
            scores=list(episode.scores),
            last_score=episode.last_score,
            current_task_id=task_id,
        )


class EmailTriagerEnvironment(Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]):
    """
    AI Email Triage & Response System.

    Each episode consists of three graded steps (Task1 -> Task2 -> Task3).
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__(transform=None, rubric=None)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        global _LATEST_EPISODE_ID

        with _LOCK:
            _id = episode_id or str(uuid.uuid4())
            _LATEST_EPISODE_ID = _id

            episode = EpisodeData(
                episode_id=_id,
                seed=seed,
                current_task_index=0,
                scores=[],
                last_score=None,
            )

            # Initialize history with the task's email context.
            t0 = TASKS[0]
            email_text = str(t0.input_observation["email_text"])
            sender = str(t0.input_observation["sender"])
            episode.history = [f"From: {sender}\n{email_text}"]

            _EPISODES[_id] = episode

        return self._build_observation(episode=episode, reward=None, done=False, breakdown={})

    def step(
        self,
        action: EmailTriageAction,
        timeout_s: Optional[float] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        global _LATEST_EPISODE_ID
        _episode_id = episode_id or _LATEST_EPISODE_ID
        if not _episode_id:
            raise ValueError("Missing episode_id. Call /reset first (or pass episode_id to /step).")

        with _LOCK:
            episode = _EPISODES.get(_episode_id)
            if episode is None:
                raise ValueError(f"Unknown episode_id: {_episode_id}")

            if episode.is_done():
                # Already terminal; return terminal observation.
                return self._build_observation(
                    episode=episode,
                    reward=episode.last_score,
                    done=True,
                    breakdown={"error": "Episode already completed. Cannot accept more actions."},
                )

            # Validate action
            if not isinstance(action, EmailTriageAction):
                raise ValueError(f"Invalid action type: {type(action)}")
            
            if not action.category:
                raise ValueError("Action category cannot be empty")
            
            if not (action.response or "").strip():
                raise ValueError("Action response cannot be empty")
            
            if action.priority < 1 or action.priority > 5:
                raise ValueError(f"Priority must be between 1 and 5, got {action.priority}")

            task = episode.current_task()

            # Grade with deterministic rubric.
            reward_obj, breakdown = task.grader_fn(action, observation=self._build_current_task_observation(episode), state=self.state_for_episode(episode))
            rubric_score = reward_obj.score

            # Looping penalty: penalize near-identical repeated responses.
            loop_penalty = 0.0
            if episode.agent_responses:
                sims = [_word_set_similarity(prev, action.response) for prev in episode.agent_responses]
                max_sim = max(sims) if sims else 0.0
                # Penalize similarity above a threshold.
                if max_sim >= 0.85:
                    loop_penalty = 0.15
                elif max_sim >= 0.75:
                    loop_penalty = 0.08

            # Reward shaping with progress signals (not binary).
            prev_mean = sum(episode.scores) / len(episode.scores) if episode.scores else 0.0
            progress_factor = (prev_mean + rubric_score) / 2.0

            # Additional penalties for core failure modes.
            missing_response_penalty = 0.0
            if not (action.response or "").strip():
                missing_response_penalty = 0.2

            wrong_category_penalty = 0.0
            if action.category != task.expected["category"]:
                wrong_category_penalty = 0.1

            shaped = _clamp01(0.65 * rubric_score + 0.25 * progress_factor + 0.10 * rubric_score - loop_penalty)
            shaped = _clamp01(shaped - missing_response_penalty - wrong_category_penalty)

            # Update episode trajectory.
            episode.scores.append(shaped)
            episode.last_score = shaped
            episode.last_breakdown = dict(breakdown)
            episode.agent_responses.append(action.response)
            episode.agent_categories.append(action.category)

            # Append agent message to history (for loop detection + debugging).
            episode.history.append(f"Agent[{task.task_id}] category={action.category} priority={action.priority}\nResponse: {action.response}")

            # Advance to next task.
            episode.current_task_index += 1
            done = episode.current_task_index >= len(TASKS)

            next_breakdown = dict(breakdown)
            next_breakdown.update(
                {
                    "loop_penalty": loop_penalty,
                    "missing_response_penalty": missing_response_penalty,
                    "wrong_category_penalty": wrong_category_penalty,
                    "progress_factor": progress_factor,
                    "rubric_score": rubric_score,
                }
            )

            return self._build_observation(
                episode=episode,
                reward=shaped,
                done=done,
                breakdown=next_breakdown,
            )

    @property
    def state(self) -> EmailTriageState:
        """
        State for the latest episode_id (used by OpenEnv's `/state` endpoint).
        """
        # Note: OpenEnv instantiates a fresh Environment for `/state`, so we
        # return the latest episode persisted in our module-level store.
        return get_latest_state()

    def state_for_episode(self, episode: EpisodeData) -> EmailTriageState:
        """Build state for a specific episode."""
        if episode.is_done():
            task_id = "done"
        elif episode.current_task_index < len(TASKS):
            task_id = TASKS[episode.current_task_index].task_id
        else:
            task_id = "done"
        
        return EmailTriageState(
            episode_id=episode.episode_id,
            step_count=len(episode.scores),
            current_task_index=episode.current_task_index,
            scores=list(episode.scores),
            last_score=episode.last_score,
            current_task_id=task_id,
        )

    def _build_current_task_observation(self, episode: EpisodeData) -> EmailTriageObservation:
        # Used for graders; should reflect the current task context.
        idx = min(episode.current_task_index, len(TASKS) - 1)
        task = TASKS[idx]
        email_text = str(task.input_observation["email_text"])
        sender = str(task.input_observation["sender"])
        return EmailTriageObservation(
            done=False,
            reward=None,
            email_text=email_text,
            sender=sender,
            history=list(episode.history),
            episode_id=episode.episode_id,
            task_id=task.task_id,
            grader_breakdown={},
        )

    def _build_observation(
        self,
        episode: EpisodeData,
        reward: Optional[float],
        done: bool,
        breakdown: Dict[str, float],
    ) -> EmailTriageObservation:
        if episode.is_done():
            last_task = TASKS[-1]
            email_text = str(last_task.input_observation["email_text"])
            sender = str(last_task.input_observation["sender"])
            task_id = "done"
        else:
            task = TASKS[episode.current_task_index]
            email_text = str(task.input_observation["email_text"])
            sender = str(task.input_observation["sender"])
            task_id = task.task_id

        return EmailTriageObservation(
            done=done,
            reward=reward,
            email_text=email_text,
            sender=sender,
            history=list(episode.history),
            episode_id=episode.episode_id,
            task_id=task_id,
            grader_breakdown=breakdown,
        )

