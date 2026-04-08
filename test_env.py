#!/usr/bin/env python
"""
Quick test of the environment to check for issues.
This script tests the core environment methods directly.
"""

from env import EmailTriagerEnvironment
from models import EmailTriageAction
import json

def test_environment():
    """Test key environment functions."""
    env = EmailTriagerEnvironment()
    
    # Test reset
    print("=" * 60)
    print("Testing reset()...")
    obs = env.reset()
    print(f"✓ Reset successful")
    print(f"  Episode ID: {obs.episode_id}")
    print(f"  Task ID: {obs.task_id}")
    print(f"  Email sender: {obs.sender}")
    print(f"  History length: {len(obs.history)}")
    print(f"  Done: {obs.done}")
    print(f"  Reward: {obs.reward}")
    
    # Test step
    print("\n" + "=" * 60)
    print("Testing step()...")
    action = EmailTriageAction(
        category="billing",
        priority=4,
        response="Hi, we will reverse the duplicate charge and confirm the refund for Order #18492. Thank you."
    )
    obs2 = env.step(action, episode_id=obs.episode_id)
    print(f"✓ Step 1 successful")
    print(f"  Task ID: {obs2.task_id}")
    print(f"  Reward: {obs2.reward}")
    print(f"  Done: {obs2.done}")
    print(f"  Grader breakdown: {obs2.grader_breakdown}")
    
    # Test step 2
    print("\n" + "=" * 60)
    print("Testing step 2...")
    action2 = EmailTriageAction(
        category="technical",
        priority=3,
        response="Hello! Please check your spam folder for the reset link. If you still don't see it, try using a different browser and clear your cache. Let us know if this helps."
    )
    obs3 = env.step(action2, episode_id=obs.episode_id)
    print(f"✓ Step 2 successful")
    print(f"  Task ID: {obs3.task_id}")
    print(f"  Reward: {obs3.reward}")
    print(f"  Done: {obs3.done}")
    print(f"  Grader breakdown: {obs3.grader_breakdown}")
    
    # Test step 3
    print("\n" + "=" * 60)
    print("Testing step 3...")
    action3 = EmailTriageAction(
        category="technical",
        priority=5,
        response="Hi Mia, we're here to help! To resolve your account access issue, please reset your password and verify your email address. We'll also unsubscribe you from promotional emails and review the unexpected charge. Let us know the next steps you'd like to take."
    )
    obs4 = env.step(action3, episode_id=obs.episode_id)
    print(f"✓ Step 3 successful")
    print(f"  Task ID: {obs4.task_id}")
    print(f"  Reward: {obs4.reward}")
    print(f"  Done: {obs4.done}")
    print(f"  Grader breakdown: {obs4.grader_breakdown}")
    
    # Test state
    print("\n" + "=" * 60)
    print("Testing state property...")
    state = env.state
    print(f"✓ State retrieved")
    print(f"  Episode ID: {state.episode_id}")
    print(f"  Step count: {state.step_count}")
    print(f"  Current task index: {state.current_task_index}")
    print(f"  Scores: {state.scores}")
    print(f"  Last score: {state.last_score}")
    print(f"  Current task ID: {state.current_task_id}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_environment()
