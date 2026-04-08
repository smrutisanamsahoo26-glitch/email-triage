"""
Inference module for email_triage_env.

Provides a standard interface for running inference on the email triage environment.
This is the entry point for evaluation systems and external APIs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from env import EmailTriagerEnvironment
from models import EmailTriageAction, EmailTriageObservation, EmailTriageState


class EmailTriageInference:
    """
    Inference wrapper for the email triage environment.
    
    Provides methods to:
    - Initialize an episode
    - Run inference (action generation + grading)
    - Get state and scores
    """
    
    def __init__(self, seed: Optional[int] = None, episode_id: Optional[str] = None):
        """Initialize inference session."""
        self.env = EmailTriagerEnvironment()
        self.seed = seed or 42
        self.episode_id = episode_id
        self.observation = None
        self.state = None
        self._reset()
    
    def _reset(self) -> None:
        """Reset the environment for a new episode."""
        self.observation = self.env.reset(seed=self.seed, episode_id=self.episode_id)
        self.state = self.env.state
    
    def run_step(
        self,
        action: EmailTriageAction | Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single step of inference.
        
        Args:
            action: EmailTriageAction or dict with keys: category, priority, response
        
        Returns:
            Dict with keys: observation, reward, done, state
        """
        # Convert dict to EmailTriageAction if needed
        if isinstance(action, dict):
            action = EmailTriageAction(**action)
        
        # Step the environment
        observation = self.env.step(action)
        state = self.env.state
        
        return {
            "observation": observation,
            "reward": observation.reward if hasattr(observation, "reward") else 0.0,
            "done": observation.done if hasattr(observation, "done") else False,
            "state": state,
        }
    
    def get_state(self) -> EmailTriageState:
        """Get current environment state."""
        return self.env.state
    
    def get_observation(self) -> EmailTriageObservation:
        """Get current observation."""
        return self.observation


def infer(
    action: Dict[str, Any],
    seed: int = 42,
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single inference step.
    
    This is the main entry point for external evaluation systems.
    
    Args:
        action: Dict with keys {category, priority, response}
        seed: Random seed for reproducibility
        episode_id: Optional episode ID for tracking
    
    Returns:
        Dict with keys: observation, reward, done, state
    
    Example:
        >>> action = {
        ...     "category": "billing",
        ...     "priority": 4,
        ...     "response": "We will process your refund immediately."
        ... }
        >>> result = infer(action, seed=42)
        >>> print(f"Reward: {result['reward']}")
    """
    inference = EmailTriageInference(seed=seed, episode_id=episode_id)
    return inference.run_step(action)


def infer_batch(
    actions: list[Dict[str, Any]],
    seed: int = 42,
    episode_id: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """
    Run multiple inference steps (batch processing).
    
    Args:
        actions: List of action dicts
        seed: Random seed for reproducibility
        episode_id: Optional episode ID for tracking
    
    Returns:
        List of result dicts
    """
    inference = EmailTriageInference(seed=seed, episode_id=episode_id)
    results = []
    
    for action in actions:
        result = inference.run_step(action)
        results.append(result)
    
    return results


def infer_smart(
    email_text: str,
    seed: int = 42,
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run smart inference: generates action from email text automatically.
    
    Uses simple heuristics to classify the email and generate a response.
    
    Args:
        email_text: Customer email text
        seed: Random seed for reproducibility
        episode_id: Optional episode ID for tracking
    
    Returns:
        Dict with keys: action, observation, reward, done, state
    """
    # Simple heuristic-based action generation
    email_lower = email_text.lower()
    
    if "refund" in email_lower or "charged" in email_lower:
        action = {
            "category": "billing",
            "priority": 4,
            "response": "We apologize for the billing issue. We will process your refund immediately and confirm it within 2-3 business days.",
        }
    elif "password" in email_lower or "reset" in email_lower:
        action = {
            "category": "technical",
            "priority": 3,
            "response": "Please try resetting your password again. Ensure you check your spam folder for the reset link and clear your browser cache.",
        }
    elif "blocked" in email_lower or "can't access" in email_lower or "locked" in email_lower:
        action = {
            "category": "technical",
            "priority": 5,
            "response": "We understand your frustration. Please reset your password first, then verify your email. If the issue persists, contact our support team.",
        }
    elif "unsubscribe" in email_lower or "opt out" in email_lower or "marketing" in email_lower:
        action = {
            "category": "query",
            "priority": 2,
            "response": "You can unsubscribe from our marketing emails at any time using the link in the footer of our emails or by updating your preferences.",
        }
    else:
        action = {
            "category": "query",
            "priority": 2,
            "response": "Thank you for reaching out. We will review your message and get back to you as soon as possible.",
        }
    
    # Run inference with generated action
    result = infer(action, seed=seed, episode_id=episode_id)
    
    return {
        "action": action,
        **result,
    }


if __name__ == "__main__":
    # Example usage
    print("📧 Email Triage Inference Module")
    print("=" * 50)
    
    # Test 1: Single action inference
    action = {
        "category": "billing",
        "priority": 4,
        "response": "We apologize for the billing issue and will process your refund."
    }
    result = infer(action, seed=42)
    print(f"\n✅ Single Action Inference")
    print(f"   Reward: {result['reward']:.2f}")
    print(f"   Done: {result['done']}")
    
    # Test 2: Smart inference from email text
    email = "I was charged twice for my order. Please refund me immediately."
    result = infer_smart(email, seed=42)
    print(f"\n✅ Smart Inference from Email")
    print(f"   Action Category: {result['action']['category']}")
    print(f"   Priority: {result['action']['priority']}")
    print(f"   Reward: {result['reward']:.2f}")
    print(f"   Response: {result['action']['response']}")
