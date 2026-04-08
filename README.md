---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# Email Triage Environment 🚀

FastAPI + OpenEnv project for email classification, prioritization, and response generation.

## Features

- Email categorization (billing, technical, etc.)
- Priority detection
- Automated response suggestions
- Evaluation via baseline + grader

## Endpoints

- `/docs` → API documentation
- `/tasks` → Available tasks
- `/grader` → Scoring breakdown
- `/baseline` → Run evaluation
