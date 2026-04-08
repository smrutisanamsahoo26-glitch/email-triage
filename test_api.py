#!/usr/bin/env python
"""
Comprehensive API endpoint testing.
"""

import requests
import json
import time

BASE_URL = "http://localhost:7860"

def test_endpoints():
    """Test all endpoints."""
    print("=" * 70)
    print("TESTING ALL ENDPOINTS")
    print("=" * 70)
    
    # Test 1: GET /tasks
    print("\n[1] Testing GET /tasks")
    try:
        r = requests.get(f"{BASE_URL}/tasks", timeout=5)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Tasks retrieved: {len(data)} tasks")
            for task in data:
                print(f"    - {task['task_id']}: {task['difficulty']}")
        else:
            print(f"  ✗ Error: {r.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    # Test 2: POST /reset
    print("\n[2] Testing POST /reset")
    try:
        r = requests.post(f"{BASE_URL}/reset", json={}, timeout=5)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            episode_id = data.get("observation", {}).get("episode_id")
            print(f"  ✓ Reset successful")
            print(f"    Episode ID: {episode_id}")
            print(f"    Task: {data.get('observation', {}).get('task_id')}")
            print(f"    Reward: {data.get('reward')}")
            print(f"    Done: {data.get('done')}")
        else:
            print(f"  ✗ Error: {r.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return
    
    # Save episode_id for later tests
    saved_episode_id = episode_id
    
    # Test 3: GET /state
    print("\n[3] Testing GET /state")
    try:
        r = requests.get(f"{BASE_URL}/state", timeout=5)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ State retrieved")
            print(f"    Episode ID: {data.get('episode_id')}")
            print(f"    Step count: {data.get('step_count')}")
            print(f"    Current task: {data.get('current_task_id')}")
            print(f"    Last score: {data.get('last_score')}")
        else:
            print(f"  ✗ Error: {r.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    # Test 4: POST /step with valid action
    print("\n[4] Testing POST /step (valid action)")
    try:
        action = {
            "category": "billing",
            "priority": 4,
            "response": "We will reverse the duplicate charge for Order #18492."
        }
        r = requests.post(
            f"{BASE_URL}/step",
            json={"action": action, "episode_id": saved_episode_id},
            timeout=5
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Step successful")
            print(f"    Reward: {data.get('reward')}")
            print(f"    Done: {data.get('done')}")
            print(f"    Task ID: {data.get('observation', {}).get('task_id')}")
        else:
            print(f"  ✗ Error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    # Test 5: GET /grader
    print("\n[5] Testing GET /grader")
    try:
        r = requests.get(f"{BASE_URL}/grader", timeout=5)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Grader info retrieved")
            print(f"    Last score: {data.get('last_score')}")
            print(f"    Breakdown keys: {list(data.get('last_breakdown', {}).keys())}")
        else:
            print(f"  ✗ Error: {r.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    # Test 6: GET /baseline (only if OPENAI_API_KEY is set)
    print("\n[6] Testing GET /baseline")
    import os
    if os.environ.get("OPENAI_API_KEY"):
        try:
            r = requests.get(f"{BASE_URL}/baseline?seed=42", timeout=30)
            print(f"  Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"  ✓ Baseline run successful")
                print(f"    Average score: {data.get('average_score'):.4f}")
                print(f"    Tasks: {len(data.get('tasks', []))}")
            else:
                print(f"  ✗ Error: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")
    else:
        print("  ⊘ Skipped (OPENAI_API_KEY not set)")
    
    # Test 7: GET /web
    print("\n[7] Testing GET /web")
    try:
        r = requests.get(f"{BASE_URL}/web", timeout=5)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  ✓ Web page retrieved")
            print(f"    Content length: {len(r.text)} bytes")
        else:
            print(f"  ✗ Error: {r.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    # Test 8: GET /docs (OpenAPI docs)
    print("\n[8] Testing GET /docs (OpenAPI documentation)")
    try:
        r = requests.get(f"{BASE_URL}/docs", timeout=5)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  ✓ OpenAPI docs available")
        else:
            print(f"  ✗ Error: {r.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    print("\n" + "=" * 70)
    print("ENDPOINT TESTS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    print("Waiting for server to be ready...")
    time.sleep(1)
    test_endpoints()
