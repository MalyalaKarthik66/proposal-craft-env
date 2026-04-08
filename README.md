---
title: ProposalCraftEnv
emoji: "🚀"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ProposalCraftEnv

## Environment Description
ProposalCraftEnv is an OpenEnv-style reinforcement learning environment for iterative proposal writing. An agent writes and revises proposal sections based on source material and receives deterministic rewards for quality, completeness, and formatting compliance. The environment models real-world drafting workflows for abstracts, project proposals, and full research/grant proposals.

## Action Space
The environment accepts an Action object with the following fields:

| Field | Type | Description |
|---|---|---|
| `section_name` | `str` | Section being edited (for example: `abstract`, `methodology`) |
| `content` | `str` | Proposed text content for the section |
| `action_type` | `"write" \| "revise" \| "finalize"` | Write a new section, revise an existing section, or finalize the episode |

## Observation Space
Each reset/step returns an Observation object:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Current task identifier |
| `task_description` | `str` | Human-readable task name |
| `source_material` | `str` | Background context used for writing |
| `sections_required` | `list[str]` | Ordered sections that must be completed |
| `sections_completed` | `list[str]` | Sections currently written |
| `current_draft` | `dict[str, str]` | Current section-to-content mapping |
| `feedback` | `str` | Feedback from the latest scoring step |
| `step_count` | `int` | Number of actions taken |
| `score_so_far` | `float` | Running reward total |

## Task Descriptions
| Task ID | Difficulty | Sections | Max Steps |
|---|---|---|---|
| `easy_abstract` | Easy | `abstract` | 5 |
| `medium_proposal` | Medium | `introduction`, `methodology`, `timeline` | 10 |
| `hard_full_proposal` | Hard | `abstract`, `introduction`, `problem_statement`, `methodology`, `expected_outcomes`, `budget_justification` | 20 |

## Reward Function
Rewards are deterministic and bounded in `[0.0, 1.0]` per step output:

- Section score components:
  - Keyword coverage (weight 0.35)
  - Word-count compliance with partial/penalty scoring (weight 0.30)
  - Minimum sentence structure (weight 0.20)
  - Coherence checks (weight 0.15)
- Write action reward:
  - `section_score * (1 / total_required_sections)`
- Revise action reward:
  - Same as write reward with a 30% penalty (`* 0.7`)
- Finalize action reward:
  - Full task score computed as average section score

## Setup Instructions
1. Create and activate a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_hf_token_here
set PORT=7860
```

4. Run the API server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

5. Run tests:

```bash
pytest -q
```

6. Run baseline inference:

```bash
python inference.py
```

## API Usage
Reset to a task:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy_abstract"}'
```

Step with an action:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "section_name": "abstract",
    "content": "Students use this app to improve study outcomes with machine learning recommendations.",
    "action_type": "write"
  }'
```

Inspect state:

```bash
curl http://localhost:7860/state
```

## Baseline Scores
Populate this table after running `python inference.py`.

| Task ID | Total Reward | Final Score | Notes |
|---|---|---|---|
| `easy_abstract` | TBD | TBD | TBD |
| `medium_proposal` | TBD | TBD | TBD |
| `hard_full_proposal` | TBD | TBD | TBD |

## Docker Instructions
Build image:

```bash
docker build -t proposal-craft-env .
```

Run container:

```bash
docker run -p 7860:7860 proposal-craft-env
```
