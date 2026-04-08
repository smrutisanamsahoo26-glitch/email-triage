#!/usr/bin/env python
"""
Final validation and comprehensive test of the email_triage_env.
Tests all requirements and conformance to OpenEnv specification.
"""

import sys
from env import EmailTriagerEnvironment, TASKS
from models import EmailTriageAction, EmailTriageObservation, EmailTriageState, Reward
from tasks import TaskDefinition
import json

def test_openenv_spec():
    """Test core OpenEnv specification compliance."""
    print("\n" + "=" * 70)
    print("TESTING OPENENV SPEC COMPLIANCE")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Environment has required methods
    print("\n[1] Checking required methods...")
    env = EmailTriagerEnvironment()
    required_methods = ['reset', 'step', 'state']
    for method in required_methods:
        if hasattr(env, method):
            print(f"  ✓ {method} exists")
            tests_passed += 1
        else:
            print(f"  ✗ {method} missing!")
            tests_failed += 1
    
    # Test 2: Pydantic models exist
    print("\n[2] Checking Pydantic models...")
    models = [
        ('Observation', EmailTriageObservation),
        ('Action', EmailTriageAction),
        ('Reward', Reward),
        ('State', EmailTriageState),
    ]
    for name, model_cls in models:
        try:
            if name == 'Observation':
                obj = model_cls(done=False, reward=None, email_text="test", sender="test@test.com", 
                               history=[], episode_id="test", task_id="task_1")
            elif name == 'Action':
                obj = model_cls(category="billing", priority=3, response="test")
            elif name == 'Reward':
                obj = model_cls(score=0.5)
            elif name == 'State':
                obj = model_cls(episode_id="test")
            print(f"  ✓ {name} model works")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ {name} model error: {e}")
            tests_failed += 1
    
    # Test 3: Task system
    print("\n[3] Testing task system...")
    if len(TASKS) >= 3:
        print(f"  ✓ {len(TASKS)} tasks configured")
        tests_passed += 1
        
        difficulties = {"easy": False, "medium": False, "hard": False}
        for task in TASKS:
            if task.difficulty in difficulties:
                difficulties[task.difficulty] = True
        
        if all(difficulties.values()):
            print(f"  ✓ All difficulty levels present (easy, medium, hard)")
            tests_passed += 1
        else:
            print(f"  ✗ Missing difficulty levels: {[k for k,v in difficulties.items() if not v]}")
            tests_failed += 1
    else:
        print(f"  ✗ Only {len(TASKS)} tasks, need at least 3")
        tests_failed += 1
    
    # Test 4: Task structure
    print("\n[4] Checking task structure...")
    for task in TASKS:
        required_fields = ['task_id', 'difficulty', 'input_observation', 'expected', 'grader_fn']
        missing = [f for f in required_fields if not hasattr(task, f)]
        if not missing:
            print(f"  ✓ {task.task_id} has all required fields")
            tests_passed += 1
        else:
            print(f"  ✗ {task.task_id} missing: {missing}")
            tests_failed += 1
    
    # Test 5: Observation/Action requirements
    print("\n[5] Checking observation/action requirements...")
    required_obs_fields = ['email_text', 'sender', 'history']
    required_action_fields = ['category', 'priority', 'response']
    
    task = TASKS[0]
    inp = task.input_observation
    for field in required_obs_fields:
        if field in inp:
            print(f"  ✓ Observation has '{field}'")
            tests_passed += 1
        else:
            print(f"  ✗ Observation missing '{field}'")
            tests_failed += 1
    
    # Test 6: Grader system
    print("\n[6] Testing grader system...")
    try:
        action = EmailTriageAction(category="billing", priority=4, response="Test response")
        dummy_obs = EmailTriageObservation(
            done=False, reward=None, email_text="test", sender="test@test.com",
            history=[], episode_id="test", task_id="task_1"
        )
        dummy_state = EmailTriageState(episode_id="test")
        
        reward_obj, breakdown = TASKS[0].grader_fn(action, dummy_obs, dummy_state)
        
        if isinstance(reward_obj.score, float) and 0.0 <= reward_obj.score <= 1.0:
            print(f"  ✓ Grader returns score in [0, 1]: {reward_obj.score:.4f}")
            tests_passed += 1
        else:
            print(f"  ✗ Invalid score: {reward_obj.score}")
            tests_failed += 1
        
        if isinstance(breakdown, dict) and len(breakdown) > 0:
            print(f"  ✓ Grader returns breakdown with {len(breakdown)} components")
            tests_passed += 1
        else:
            print(f"  ✗ Grader breakdown invalid")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Grader error: {e}")
        tests_failed += 2
    
    # Test 7: Reward contains partial credit
    print("\n[7] Testing reward partial credit...")
    scores = []
    for task in TASKS:
        action = EmailTriageAction(category=task.expected["category"], priority=task.expected["priority"], response="Test response")
        dummy_obs = EmailTriageObservation(
            done=False, reward=None, email_text="test", sender="test@test.com",
            history=[], episode_id="test", task_id=task.task_id
        )
        dummy_state = EmailTriageState(episode_id="test")
        reward_obj, _ = task.grader_fn(action, dummy_obs, dummy_state)
        scores.append(reward_obj.score)
    
    if any(0 < s < 1 for s in scores):
        print(f"  ✓ Scores contain partial credit (not all 0 or 1)")
        tests_passed += 1
    else:
        print(f"  ✗ No partial credit scores: {scores}")
        tests_failed += 1
    
    # Test 8: Full episode test
    print("\n[8] Testing full episode workflow...")
    try:
        env = EmailTriagerEnvironment()
        obs1 = env.reset()
        print(f"  ✓ Reset successful")
        tests_passed += 1
        
        if obs1.done == False:
            print(f"  ✓ Initial observation has done=False")
            tests_passed += 1
        else:
            print(f"  ✗ Initial observation has done=True")
            tests_failed += 1
        
        state1 = env.state
        if state1.current_task_id in ["task_1", "task_2", "task_3"]:
            print(f"  ✓ State shows current task: {state1.current_task_id}")
            tests_passed += 1
        else:
            print(f"  ✗ Invalid current_task_id: {state1.current_task_id}")
            tests_failed += 1
        
        episode_id = obs1.episode_id
        
        # Full workflow
        action1 = EmailTriageAction(category="billing", priority=4, response="We'll reverse the duplicate charge.")
        obs2 = env.step(action1, episode_id=episode_id)
        print(f"  ✓ Step 1 successful (reward: {obs2.reward:.4f})")
        tests_passed += 1
        
        action2 = EmailTriageAction(category="technical", priority=3, response="Try checking spam and use latest reset link.")
        obs3 = env.step(action2, episode_id=episode_id)
        print(f"  ✓ Step 2 successful (reward: {obs3.reward:.4f})")
        tests_passed += 1
        
        action3 = EmailTriageAction(category="technical", priority=5, response="Reset password, verify email, and unsubscribe from promotions.")
        obs4 = env.step(action3, episode_id=episode_id)
        print(f"  ✓ Step 3 successful (reward: {obs4.reward:.4f})")
        tests_passed += 1
        
        if obs4.done == True:
            print(f"  ✓ Episode marked done after 3 steps")
            tests_passed += 1
        else:
            print(f"  ✗ Episode not marked done")
            tests_failed += 1
        
        state_final = env.state
        if len(state_final.scores) == 3:
            print(f"  ✓ Final state has 3 scores: {[f'{s:.3f}' for s in state_final.scores]}")
            tests_passed += 1
        else:
            print(f"  ✗ Unexpected scores count: {len(state_final.scores)}")
            tests_failed += 1
        
    except Exception as e:
        print(f"  ✗ Episode workflow error: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 7
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TESTS PASSED: {tests_passed}")
    print(f"TESTS FAILED: {tests_failed}")
    print(f"TOTAL: {tests_passed + tests_failed}")
    print("=" * 70)
    
    return tests_failed == 0

if __name__ == "__main__":
    success = test_openenv_spec()
    sys.exit(0 if success else 1)
