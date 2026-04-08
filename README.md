# Email Triage & Response OpenEnv Environment

This repository contains a production-ready OpenEnv environment for an AI Email Triage and Response System built for hackathons and real-world prototyping.

## Real-world motivation

Customer support teams process a high volume of email traffic. A triage-and-respond workflow can:

- categorize incoming emails (billing, technical, query, complaint, spam),
- estimate urgency (priority 1 to 5),
- draft an appropriate first response that is polite and actionable.

This environment simulates a realistic multi-step workflow where the agent must classify emails and generate high-quality responses for increasing difficulty tasks.

## Environment

Environment name: `email_triage_env`

OpenEnv runtime contract (core methods):

- `reset()` starts a new episode
- `step(action)` applies an agent action and returns the next observation plus a reward signal
- `state()` exposes episode metadata via the `/state` endpoint

## Observation, Action, Reward

### Observation

The observation includes the hackathon-required fields:

- `email_text` (`str`)
- `sender` (`str`)
- `history` (`list[str]`)

Additional fields are included for grading/debugging:

- `episode_id` (`str`)
- `task_id` (`str`)
- `grader_breakdown` (`dict[str, float]`)

### Action

The agent action includes the hackathon-required fields:

- `category` (`str`): one of `complaint`, `query`, `spam`, `technical`, `billing`
- `priority` (`int`): integer in the range `1` to `5`
- `response` (`str`): drafted response email text

### Reward

Reward is a float in `[0.0, 1.0]` with partial credit.

Scoring includes:

- category correctness (0.3)
- priority correctness (0.2)
- polite tone detection (0.2)
- keyword/intent matching (0.3)

The environment also applies additional shaping:

- missing/empty response penalties
- looping behavior penalties (repeated near-identical responses)
- progress signals across Task 1 -> Task 2 -> Task 3

## Tasks

Each episode contains three tasks that increase in difficulty.

### Task 1 (Easy): Classification

- Input: a billing/duplicate-charge email
- Expected: `category="billing"`, priority `4`
- Grader: deterministic rubric-based scoring for category, priority, tone, and keyword/intent overlap.

### Task 2 (Medium): Classification + Response Drafting

- Input: a technical email about a password reset link failing
- Expected: `category="technical"`, priority `3`
- Grader: requires response elements such as using a reset link, checking spam, trying a different browser, and clearing cache.

### Task 3 (Hard): Multi-intent Response

- Input: an email that includes account access trouble plus an unexpected charge plus unwanted promotional emails
- Expected: `category="technical"`, priority `5`
- Grader: checks for polite tone and multi-step elements that resolve:
  - account access / blocked sign-in issue
  - verification guidance
  - unsubscribe/notification opt-out messaging
  - billing-charge follow-up wording

## API usage

The server runs on port `7860` and provides OpenEnv-standard endpoints plus hackathon endpoints:

- `GET /tasks`: list tasks
- `GET /grader`: last score for the latest episode
- `GET /baseline`: run the OpenAI baseline inference script and return scores

Core OpenEnv-standard endpoints:

- `POST /reset`: start episode and return initial observation
- `POST /step`: apply action and return observation, reward, done
- `GET /state`: return internal episode state (latest episode persisted in memory)

### Example (cURL)

Reset:

```bash
curl -s http://localhost:7860/reset | jq
```

Step (example JSON action):

```bash
curl -s http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "category": "billing",
      "priority": 4,
      "response": "Hi Support, please reverse the duplicate charge and confirm the refund for Order #18492. Thank you."
    },
    "episode_id": "OPTIONAL_USE_FROM_RESET_RESPONSE"
  }' | jq
```

Note: OpenEnv's step request schema allows extra fields, so `episode_id` can be provided to ensure the server updates the correct episode.

Tasks:

```bash
curl -s http://localhost:7860/tasks | jq
```

Last grader score:

```bash
curl -s http://localhost:7860/grader | jq
```

## Setup instructions

### Local (Docker)

Build:

```bash
docker build -t email-triage-env .
```

Run:

```bash
docker run -p 7860:7860 email-triage-env
```

### Local (Python)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Testing

Test environment directly:

```bash
python test_env.py
```

Test API endpoints:

```bash
python test_api.py
```

## Baseline results

The `baseline.py` script runs all three tasks using the OpenAI API and grades each generated action with the same deterministic graders used by the environment.

To run:

```bash
export OPENAI_API_KEY=your-api-key-here
python baseline.py
```

Or via server (requires API key set):

```bash
curl -s http://localhost:7860/baseline?seed=42 | jq
```

The script prints `Average score:` followed by per-task scores. Exact numeric results depend on the OpenAI model output for the provided inputs, while the scoring itself is deterministic.

**Note**: You must set a valid `OPENAI_API_KEY` environment variable for the baseline to work. The environment variable should contain your OpenAI API key.

## Hugging Face Space compatibility notes

- The container listens on port `7860`.
- `openenv.yaml` sets `port: 7860` and points to `server.app:app`.
- HF Spaces typically provide HTTP access; the environment exposes `POST /reset` and `POST /step` as required by the OpenEnv simulation HTTP API.

## Error handling and troubleshooting

### "Episode not found" errors

- Always call `/reset` before `/step`
- Use the `episode_id` returned from `/reset` for subsequent `/step` calls

### "Invalid action" errors

- Ensure `category` is one of: `complaint`, `query`, `spam`, `technical`, `billing`
- Ensure `priority` is an integer between 1 and 5
- Ensure `response` is a non-empty string

### Baseline endpoint returns 400

- Verify that `OPENAI_API_KEY` environment variable is set
- Verify that the API key is valid and not a placeholder value

### Complete workflow example

```bash
# 1. Reset the environment
RESET=$(curl -s -X POST http://localhost:7860/reset)
EPISODE_ID=$(echo $RESET | jq -r '.observation.episode_id')

# 2. Get first task details
curl -s http://localhost:7860/tasks | jq '.[0]'

# 3. Perform first action
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d "{
    \"action\": {
      \"category\": \"billing\",
      \"priority\": 4,
      \"response\": \"We'll reverse the duplicate charge for Order #18492. Thank you for your patience.\"
    },
    \"episode_id\": \"$EPISODE_ID\"
  }" | jq

# 4. Check state
curl -s http://localhost:7860/state | jq

# 5. Check last score
curl -s http://localhost:7860/grader | jq
```
