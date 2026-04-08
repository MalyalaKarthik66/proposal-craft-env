# ProposalCraftEnv — Complete Implementation Guide for GitHub Copilot

> **Purpose:** This document is the single source of truth for GitHub Copilot to build the
> entire `ProposalCraftEnv` project. Before writing any code for a file, Copilot MUST read
> the section for that file in this document and implement exactly what is described.
> Do NOT deviate from the algorithms, models, or structure defined here.

---

## 1. Project Overview

**Environment name:** `ProposalCraftEnv`
**Domain:** Real-world proposal/document drafting
**Description:** An RL environment where an AI agent iteratively drafts a structured
proposal (research, GSoC, project grant, etc.) section-by-section. The agent receives
observations (current draft state, sections remaining, feedback) and takes actions
(write or revise a section). It gets reward for quality, completeness, and adherence
to formatting constraints.

**Why this is real-world:** Proposal writing is something every researcher, student,
and professional does. Training an agent to draft proposals well has immediate value.

---

## 2. Folder & File Structure

```
proposal_craft_env/
│
├── openenv.yaml                  # OpenEnv metadata spec
├── Dockerfile                    # Container for HF Spaces deployment
├── requirements.txt              # Python dependencies
├── inference.py                  # Baseline inference script (ROOT LEVEL, mandatory)
├── README.md                     # Full documentation
├── .env.example                  # Example environment variables
│
├── env/
│   ├── __init__.py
│   ├── environment.py            # Main ProposalCraftEnv class
│   ├── models.py                 # Pydantic models: Observation, Action, Reward
│   ├── tasks.py                  # Task definitions (easy, medium, hard)
│   └── graders.py                # Deterministic grading functions per task
│
├── server/
│   ├── __init__.py
│   └── app.py                    # FastAPI server exposing step/reset/state endpoints
│
└── tests/
    ├── __init__.py
    ├── test_environment.py       # Unit tests for env logic
    ├── test_graders.py           # Unit tests for grader determinism
    └── test_server.py            # Integration tests for API endpoints
```

---

## 3. Environment Variables Required

These MUST be read from the system environment (os.getenv). Never hardcode.

| Variable        | Purpose                                          |
|-----------------|--------------------------------------------------|
| `API_BASE_URL`  | Base URL for the LLM API (OpenAI-compatible)     |
| `MODEL_NAME`    | Model identifier string for inference            |
| `HF_TOKEN`      | Hugging Face access token for deployment         |
| `PORT`          | Server port (default: 7860 for HF Spaces)        |

---

## 4. File-by-File Implementation Guide

---

### 4.1 `openenv.yaml`

**Purpose:** OpenEnv metadata. Must pass `openenv validate`.

**Algorithm / Content:**
```
name: proposal-craft-env
version: "1.0.0"
description: >
  An OpenEnv environment for iterative proposal/document drafting.
  An AI agent writes structured proposals section-by-section,
  receiving rewards for quality, completeness, and format adherence.
author: <your-hf-username>
tags:
  - real-world
  - nlp
  - document-generation
  - proposal-writing
tasks:
  - id: easy_abstract
    name: "Write a Single Abstract"
    difficulty: easy
    max_steps: 5
  - id: medium_proposal
    name: "Write a 3-Section Project Proposal"
    difficulty: medium
    max_steps: 10
  - id: hard_full_proposal
    name: "Write a Full 6-Section Research Proposal"
    difficulty: hard
    max_steps: 20
action_space:
  type: object
  fields:
    - name: section_name
      type: string
    - name: content
      type: string
    - name: action_type
      type: string
      enum: [write, revise, finalize]
observation_space:
  type: object
  fields:
    - name: task_id
      type: string
    - name: task_description
      type: string
    - name: sections_required
      type: list[string]
    - name: sections_completed
      type: list[string]
    - name: current_draft
      type: dict[string, string]
    - name: feedback
      type: string
    - name: step_count
      type: integer
    - name: score_so_far
      type: float
```

---

### 4.2 `env/models.py`

**Purpose:** Define all Pydantic v2 typed models used across the environment.

**Models to implement:**

#### `Action` model
Fields:
- `section_name: str` — which section is being acted on (e.g. "abstract", "introduction")
- `content: str` — the text content the agent is writing or revising
- `action_type: Literal["write", "revise", "finalize"]` — type of action
  - "write": first time writing a section
  - "revise": rewriting an already-written section (penalized slightly)
  - "finalize": signals agent is done; triggers final grading

#### `Observation` model
Fields:
- `task_id: str`
- `task_description: str` — human-readable description of what to write
- `source_material: str` — background context given to the agent (e.g. a project summary)
- `sections_required: list[str]` — ordered list of sections agent must complete
- `sections_completed: list[str]` — which sections have been written so far
- `current_draft: dict[str, str]` — map of section_name → content written so far
- `feedback: str` — feedback string from last grader call (empty on first step)
- `step_count: int`
- `score_so_far: float` — running reward total so far (0.0 to 1.0)

#### `Reward` model
Fields:
- `value: float` — reward for this step (0.0 to 1.0)
- `breakdown: dict[str, float]` — sub-scores by criterion
- `feedback: str` — explanation of reward

#### `StepResult` model
Fields:
- `observation: Observation`
- `reward: Reward`
- `done: bool`
- `info: dict`

**Implementation notes:**
- Use `pydantic.BaseModel` with `model_config = ConfigDict(extra="forbid")`
- All fields must have type annotations
- Add docstrings to each model class

---

### 4.3 `env/tasks.py`

**Purpose:** Define the 3 tasks as Python dataclass or dict configs. Each task has:
- id, name, difficulty, max_steps
- source_material (the context given to the agent)
- sections_required (ordered list)
- rubric per section (keywords required, word count range, quality indicators)

#### Task 1 — `easy_abstract`
- **Goal:** Write one section: an abstract for a simple project
- **max_steps:** 5
- **source_material:** A 3-sentence description of a software project
  (e.g.: "We are building a web app that helps students track their study schedules.
  The app uses machine learning to suggest optimal study times based on past performance.
  Target users are undergraduate students.")
- **sections_required:** `["abstract"]`
- **Rubric for "abstract":**
  - required_keywords: ["students", "machine learning", "study", "app", "performance"]
  - min_words: 50, max_words: 200
  - must_have_sentences: 3 (minimum number of sentences)

#### Task 2 — `medium_proposal`
- **Goal:** Write 3 sections for a project proposal
- **max_steps:** 10
- **source_material:** A paragraph describing a data science project
  (e.g.: "DataViz Pro is an open-source Python library for creating interactive dashboards.
  It integrates with pandas and supports real-time data streaming.
  The project needs funding to hire a developer and for server costs.")
- **sections_required:** `["introduction", "methodology", "timeline"]`
- **Rubric:**
  - "introduction": keywords=["open-source", "Python", "dashboards", "data"], min=80, max=300
  - "methodology": keywords=["implementation", "pandas", "streaming", "development"], min=100, max=400
  - "timeline": keywords=["phase", "month", "milestone", "deliverable"], min=60, max=250

#### Task 3 — `hard_full_proposal`
- **Goal:** Write 6 sections for a full research/grant proposal
- **max_steps:** 20
- **source_material:** A detailed 5-sentence research project description
  (e.g.: "We propose developing an AI-based early warning system for crop disease detection
  using satellite imagery and deep learning. The system targets smallholder farmers in
  Sub-Saharan Africa. We will partner with local agricultural NGOs for data collection.
  The model will use transfer learning from existing plant disease datasets.
  Expected outcomes include a mobile-friendly API deployed within 18 months.")
- **sections_required:** `["abstract", "introduction", "problem_statement", "methodology", "expected_outcomes", "budget_justification"]`
- **Rubric per section:**
  - "abstract": keywords=["AI", "crop", "disease", "satellite", "deep learning"], min=100, max=250
  - "introduction": keywords=["farmers", "Africa", "agriculture", "detection"], min=150, max=400
  - "problem_statement": keywords=["problem", "challenge", "impact", "current"], min=120, max=350
  - "methodology": keywords=["model", "data", "transfer learning", "training", "validation"], min=200, max=500
  - "expected_outcomes": keywords=["outcome", "impact", "farmers", "accuracy", "deployment"], min=100, max=300
  - "budget_justification": keywords=["cost", "budget", "personnel", "equipment", "justification"], min=80, max=250

**Implementation:** Store all tasks in a dict `TASKS: dict[str, TaskConfig]` where
`TaskConfig` is a dataclass with fields: id, name, difficulty, max_steps,
source_material, sections_required, rubric (dict mapping section_name → SectionRubric).
`SectionRubric` is a dataclass with: required_keywords, min_words, max_words, must_have_sentences (default=2).

---

### 4.4 `env/graders.py`

**Purpose:** Deterministic, reproducible scoring functions. No randomness. No LLM calls.
All graders must return a float in [0.0, 1.0].

#### `grade_section(content: str, rubric: SectionRubric) -> tuple[float, dict, str]`

**Algorithm:**
```
Initialize score = 0.0, breakdown = {}, feedback_parts = []

1. KEYWORD SCORE (weight 0.35):
   matched = count of required_keywords found (case-insensitive) in content
   keyword_score = matched / len(required_keywords)
   score += 0.35 * keyword_score
   breakdown["keyword_coverage"] = keyword_score
   feedback_parts.append(f"Keyword coverage: {matched}/{len(required_keywords)}")

2. WORD COUNT SCORE (weight 0.30):
   word_count = len(content.split())
   if word_count < min_words:
       wc_score = word_count / min_words  (partial credit)
   elif word_count > max_words:
       wc_score = max(0.0, 1.0 - (word_count - max_words) / max_words)  (penalty)
   else:
       wc_score = 1.0
   score += 0.30 * wc_score
   breakdown["word_count"] = wc_score

3. SENTENCE COUNT SCORE (weight 0.20):
   sentences = count of '.' or '!' or '?' in content
   if sentences >= must_have_sentences:
       sent_score = 1.0
   else:
       sent_score = sentences / must_have_sentences
   score += 0.20 * sent_score
   breakdown["sentence_structure"] = sent_score

4. COHERENCE SCORE (weight 0.15):
   — Check content starts with a capital letter: +0.05
   — Check content does not start with bullet points ("- " or "* " or "•"): +0.05
   — Check content has no more than 3 consecutive newlines: +0.05
   coherence_score = sum of above checks / 0.15 (max 1.0)
   score += 0.15 * coherence_score
   breakdown["coherence"] = coherence_score

Return (round(min(score, 1.0), 4), breakdown, "; ".join(feedback_parts))
```

#### `grade_task(task_id: str, draft: dict[str, str], task_config: TaskConfig) -> tuple[float, dict]`

**Algorithm:**
```
section_scores = {}
for section in task_config.sections_required:
    if section in draft and draft[section].strip():
        score, breakdown, _ = grade_section(draft[section], task_config.rubric[section])
    else:
        score = 0.0
        breakdown = {}
    section_scores[section] = score

total = sum(section_scores.values()) / len(task_config.sections_required)
return (round(total, 4), section_scores)
```

#### `compute_step_reward(action, observation, task_config, prev_score) -> Reward`

**Algorithm:**
```
If action.action_type == "finalize":
    final_score, section_scores = grade_task(...)
    value = final_score
    feedback = f"Final score: {final_score}. Section scores: {section_scores}"
    return Reward(value=value, breakdown=section_scores, feedback=feedback)

If action.action_type == "write":
    section_score, breakdown, feedback_str = grade_section(action.content, rubric)
    step_reward = section_score * (1.0 / len(task_config.sections_required))
    # Partial progress signal

If action.action_type == "revise":
    section_score, breakdown, feedback_str = grade_section(action.content, rubric)
    step_reward = section_score * (1.0 / len(task_config.sections_required)) * 0.7
    # Revision penalty: 30% less reward than writing fresh

value = round(step_reward, 4)
return Reward(value=value, breakdown=breakdown, feedback=feedback_str)
```

---

### 4.5 `env/environment.py`

**Purpose:** Main environment class implementing the OpenEnv interface.

**Class: `ProposalCraftEnv`**

**State to maintain (internal, not exposed):**
- `_current_task: TaskConfig | None`
- `_current_task_id: str | None`
- `_draft: dict[str, str]` — current working draft
- `_step_count: int`
- `_total_reward: float`
- `_done: bool`
- `_last_feedback: str`

**Methods:**

#### `reset(task_id: str | None = None) -> Observation`
**Algorithm:**
```
If task_id is None:
    Pick a random task_id from TASKS keys
Else:
    Validate task_id exists in TASKS, raise ValueError if not

Set _current_task = TASKS[task_id]
Set _current_task_id = task_id
Set _draft = {}
Set _step_count = 0
Set _total_reward = 0.0
Set _done = False
Set _last_feedback = "Start writing. Focus on the required sections."

Return Observation(
    task_id=task_id,
    task_description=_current_task.name,
    source_material=_current_task.source_material,
    sections_required=_current_task.sections_required,
    sections_completed=[],
    current_draft={},
    feedback=_last_feedback,
    step_count=0,
    score_so_far=0.0
)
```

#### `step(action: Action) -> StepResult`
**Algorithm:**
```
If _done:
    Raise RuntimeError("Episode is done. Call reset() first.")

If _current_task is None:
    Raise RuntimeError("No task loaded. Call reset() first.")

_step_count += 1

Validate action.section_name:
    If action.action_type != "finalize" AND action.section_name not in task.sections_required:
        Return StepResult with Reward(value=0.0, feedback="Invalid section name"), done=False

If action.action_type == "write":
    If action.section_name in _draft:
        # Treat as revise silently
        action = Action(section_name=..., content=..., action_type="revise")
    _draft[action.section_name] = action.content

If action.action_type == "revise":
    _draft[action.section_name] = action.content

reward = compute_step_reward(action, current obs, _current_task, _total_reward)
_total_reward += reward.value
_last_feedback = reward.feedback

# Check done conditions:
done = False
If action.action_type == "finalize":
    done = True
Elif _step_count >= _current_task.max_steps:
    done = True
    _last_feedback += " | Max steps reached."

_done = done
sections_completed = list(_draft.keys())

obs = Observation(
    task_id=_current_task_id,
    task_description=_current_task.name,
    source_material=_current_task.source_material,
    sections_required=_current_task.sections_required,
    sections_completed=sections_completed,
    current_draft=dict(_draft),
    feedback=_last_feedback,
    step_count=_step_count,
    score_so_far=round(_total_reward, 4)
)

return StepResult(observation=obs, reward=reward, done=done, info={"step": _step_count})
```

#### `state() -> dict`
**Algorithm:**
```
Return a plain dict with all current state:
{
    "task_id": _current_task_id,
    "step_count": _step_count,
    "total_reward": _total_reward,
    "done": _done,
    "sections_completed": list(_draft.keys()),
    "sections_remaining": [s for s in task.sections_required if s not in _draft],
    "draft_word_counts": {k: len(v.split()) for k, v in _draft.items()},
    "last_feedback": _last_feedback
}
```

---

### 4.6 `server/app.py`

**Purpose:** FastAPI server that wraps the environment and exposes HTTP endpoints.
This is what HF Spaces will serve.

**Endpoints to implement:**

| Method | Path        | Body / Params           | Returns              |
|--------|-------------|-------------------------|----------------------|
| POST   | `/reset`    | JSON: `{task_id?: str}` | `Observation` JSON   |
| POST   | `/step`     | JSON: `Action` model    | `StepResult` JSON    |
| GET    | `/state`    | —                       | state dict JSON      |
| GET    | `/health`   | —                       | `{"status": "ok"}`   |
| GET    | `/tasks`    | —                       | list of task configs |

**Algorithm:**
```
Create FastAPI app instance
Create single global ProposalCraftEnv instance (env = ProposalCraftEnv())

POST /reset:
    Parse optional task_id from body
    Call env.reset(task_id)
    Return observation as JSON

POST /step:
    Parse Action from body (validate with Pydantic)
    Call env.step(action)
    Return StepResult as JSON

GET /state:
    Return env.state() as JSON

GET /health:
    Return {"status": "ok", "version": "1.0.0"}

GET /tasks:
    Return list of {id, name, difficulty, max_steps} for all tasks

Run with uvicorn on host="0.0.0.0", port=int(os.getenv("PORT", 7860))
```

**CORS:** Add CORS middleware allowing all origins (for HF Spaces compatibility).

---

### 4.7 `requirements.txt`

List these exact packages (with minimum versions):
```
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
openai>=1.0.0
python-dotenv>=1.0.0
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
openenv-core>=0.1.0
```

---

### 4.8 `Dockerfile`

**Algorithm / Structure:**
```
Base image: python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
ENV PORT=7860
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

### 4.9 `inference.py` (ROOT LEVEL — MANDATORY FORMAT)

**Purpose:** Baseline inference script. Runs an LLM agent against all 3 tasks using the
OpenAI client. Must complete in under 20 minutes. Uses exact [START]/[STEP]/[END] log format.

**Variables read from environment:**
- `API_BASE_URL` — LLM base URL
- `MODEL_NAME` — model name
- `HF_TOKEN` — not used in inference directly but must be present

**Algorithm:**
```
Import: openai, json, time, os, requests (for calling local env server)

Initialize OpenAI client:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

Start local env server in subprocess OR import env directly:
    Import ProposalCraftEnv directly (no subprocess needed for local run)
    env = ProposalCraftEnv()

For each task_id in ["easy_abstract", "medium_proposal", "hard_full_proposal"]:

    obs = env.reset(task_id=task_id)

    Print [START] block:
    {
        "[START]": {
            "task_id": task_id,
            "task_description": obs.task_description,
            "sections_required": obs.sections_required,
            "timestamp": current ISO timestamp
        }
    }

    done = False
    step_num = 0
    total_reward = 0.0

    While not done and step_num < task.max_steps:
        step_num += 1

        Build LLM prompt:
            system = "You are a proposal writing assistant. Write concise, professional proposal sections."
            user = f"""
            Task: {obs.task_description}
            Source material: {obs.source_material}
            Sections required: {obs.sections_required}
            Sections completed so far: {obs.sections_completed}
            Current feedback: {obs.feedback}

            Write the next section. Respond ONLY as valid JSON:
            {{
                "section_name": "<next section name from sections_required not yet completed>",
                "content": "<your section text here>",
                "action_type": "write"
            }}
            If all sections are done, use action_type "finalize" with section_name "".
            """

        Call LLM:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=800
            )
            raw = response.choices[0].message.content

        Parse JSON from raw (strip markdown fences if present)
        Create Action from parsed JSON

        result = env.step(action)
        obs = result.observation
        done = result.done
        total_reward += result.reward.value

        Print [STEP] block:
        {
            "[STEP]": {
                "task_id": task_id,
                "step": step_num,
                "action_type": action.action_type,
                "section_name": action.section_name,
                "reward": result.reward.value,
                "total_reward": round(total_reward, 4),
                "done": done,
                "feedback": result.reward.feedback
            }
        }

        Sleep 1 second between steps (rate limit safety)

    Final score, section_breakdown = grade_task(task_id, obs.current_draft, TASKS[task_id])

    Print [END] block:
    {
        "[END]": {
            "task_id": task_id,
            "total_steps": step_num,
            "total_reward": round(total_reward, 4),
            "final_score": final_score,
            "section_breakdown": section_breakdown,
            "sections_completed": obs.sections_completed,
            "status": "done" if done else "max_steps_reached"
        }
    }

Print summary of all 3 tasks at end.
```

**CRITICAL:** All prints must use `print(json.dumps(...))` so output is valid JSON lines.
The [START], [STEP], [END] keys must appear EXACTLY as shown.

---

### 4.10 `tests/test_environment.py`

**Tests to implement:**

1. `test_reset_returns_valid_observation` — call reset() for each task, assert Observation fields
2. `test_reset_random_task` — call reset() with no args, assert a valid task is loaded
3. `test_step_write_action` — reset easy task, step with write action, assert reward > 0
4. `test_step_finalize_ends_episode` — reset, write all sections, finalize, assert done=True
5. `test_step_after_done_raises` — after done, call step again, assert RuntimeError
6. `test_max_steps_ends_episode` — exceed max_steps, assert done=True
7. `test_invalid_task_raises` — reset with bad task_id, assert ValueError
8. `test_state_returns_dict` — call state(), assert it's a dict with expected keys

---

### 4.11 `tests/test_graders.py`

**Tests to implement:**

1. `test_grade_section_perfect_score` — content with all keywords, right word count → score close to 1.0
2. `test_grade_section_zero_keywords` — content with no keywords → score < 0.5
3. `test_grade_section_too_short` — content below min_words → word_count score < 1.0
4. `test_grade_section_too_long` — content above max_words → word_count score < 1.0
5. `test_grade_task_all_sections` — complete draft for easy task → final score calculated
6. `test_grade_task_missing_section` — draft missing one section → score < 1.0
7. `test_grader_is_deterministic` — call grade_section twice with same input, assert identical output

---

### 4.12 `tests/test_server.py`

**Tests to implement (use httpx TestClient):**

1. `test_health_endpoint` — GET /health returns 200 and {"status": "ok"}
2. `test_reset_endpoint` — POST /reset with task_id returns valid observation JSON
3. `test_step_endpoint` — POST /reset then POST /step, assert reward in response
4. `test_state_endpoint` — GET /state returns dict with expected keys
5. `test_tasks_endpoint` — GET /tasks returns list of 3 tasks

---

### 4.13 `README.md`

**Sections to include:**

1. **Environment Description** — what ProposalCraftEnv is and why it matters
2. **Action Space** — describe Action model fields
3. **Observation Space** — describe Observation model fields
4. **Task Descriptions** — table of 3 tasks with difficulty, sections, max_steps
5. **Reward Function** — explain partial progress, keyword coverage, word count scoring
6. **Setup Instructions** — pip install, env vars, local run command
7. **API Usage** — curl examples for /reset, /step, /state
8. **Baseline Scores** — placeholder table to fill after running inference.py
9. **Docker Instructions** — build and run commands

---

### 4.14 `.env.example`

```
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your_hf_token_here
PORT=7860
```

---

## 5. Implementation Constraints & Rules

1. **No LLM calls in the environment itself.** The env is purely deterministic Python.
   LLM calls happen ONLY in `inference.py`.

2. **All reward values MUST be in [0.0, 1.0].** Use `min(value, 1.0)` and `max(value, 0.0)`.

3. **Graders are deterministic.** Same input always gives same output. No randomness.

4. **Pydantic v2 syntax.** Use `model_config = ConfigDict(...)`, not `class Config:`.

5. **FastAPI endpoint bodies** must use Pydantic models for validation, not raw dicts.

6. **The inference script** must import from `env/` directly (no HTTP calls to self).

7. **openenv.yaml** must be in the ROOT directory of the project.

8. **inference.py** must be in the ROOT directory of the project.

9. **Port 7860** is the default for HF Spaces. Do not change this.

10. **Python 3.11** is the target version for compatibility.

---

## 6. Step-by-Step Build Order for Copilot

Follow this exact order to avoid import errors:

```
Step 1: Create folder structure (mkdir commands)
Step 2: Write requirements.txt
Step 3: Write openenv.yaml
Step 4: Write env/models.py  (no internal imports, pure Pydantic)
Step 5: Write env/tasks.py   (imports models.py only)
Step 6: Write env/graders.py (imports models.py and tasks.py)
Step 7: Write env/environment.py (imports all above)
Step 8: Write server/app.py  (imports environment.py)
Step 9: Write all tests/     (imports from env/ and server/)
Step 10: Write inference.py  (imports env/ directly)
Step 11: Write Dockerfile
Step 12: Write README.md
Step 13: Write .env.example
```

---

## 7. Deployment Notes for HF Spaces

HUGGING FACE ACCESS TOKEN: <set-in-HF-secrets>

- The HF Space must be of type **"Docker"** (not Gradio/Streamlit).
- The `Dockerfile` must expose port 7860.
- Set HF Space environment secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
- After `openenv push`, the Space URL will be:
  `https://huggingface.co/spaces/<username>/proposal-craft-env`
- Test with: `curl https://<space-url>/health` → should return `{"status":"ok"}`

---

## 8. Validation Checklist (Before Submitting)

- [ ] `openenv validate` passes
- [ ] `docker build -t proposal-craft-env .` succeeds
- [ ] `docker run -p 7860:7860 proposal-craft-env` starts without errors
- [ ] `curl localhost:7860/health` returns 200
- [ ] `curl -X POST localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy_abstract"}'` returns observation
- [ ] All pytest tests pass
- [ ] `python inference.py` completes in under 20 minutes and produces [START]/[STEP]/[END] logs
- [ ] HF Space URL responds to `/health` and `/reset`
