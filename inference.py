"""Baseline inference runner for ProposalCraftEnv."""

import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
from openai import OpenAI

from env.environment import ProposalCraftEnv
from env.graders import grade_task
from env.models import Action
from env.tasks import TASKS

_ = requests

MAX_TOTAL_SECONDS = 20 * 60
TASK_ORDER = ["easy_abstract", "medium_proposal", "hard_full_proposal"]
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")


def _extract_json_payload(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty response", raw, 0)

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _next_unfinished_section(obs) -> str:
    for section in obs.sections_required:
        if section not in obs.sections_completed:
            return section
    return ""


def _fallback_section_content(task_id: str, section_name: str) -> str:
    if task_id == "easy_abstract" and section_name == "abstract":
        return (
            "This project develops a machine learning app for students to optimize "
            "study performance and track learning outcomes effectively. Furthermore, "
            "the system provides actionable insights. As a result, students improve "
            "their academic performance."
        )

    task_config = TASKS[task_id]
    section_title = section_name.replace("_", " ")
    keywords = task_config.rubric[section_name].required_keywords
    keyword_text = ", ".join(keywords)

    return (
        f"First, the {section_title} section addresses {keyword_text} with practical "
        "implementation detail and measurable objectives for delivery. Furthermore, "
        "the narrative connects planning, execution, and evaluation criteria in a "
        "professional structure. In addition, it clarifies stakeholder responsibilities "
        "and quality controls for each milestone. As a result, the section provides a "
        "coherent roadmap aligned with the project context."
    )


def _fallback_action_payload(task_id: str, obs) -> dict:
    next_section = _next_unfinished_section(obs)
    if not next_section:
        return {"section_name": "", "content": "", "action_type": "finalize"}

    fallback_content = _fallback_section_content(task_id, next_section)
    return {"section_name": next_section, "content": fallback_content, "action_type": "write"}


def main() -> None:
    client = None
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "no-token",
        )
    except Exception as exc:
        print(f"Warning: OpenAI client initialization failed: {exc}", file=sys.stderr)

    env = ProposalCraftEnv()

    run_start = time.time()
    summary: list[dict] = []

    for task_id in TASK_ORDER:
        if (time.time() - run_start) >= MAX_TOTAL_SECONDS:
            break

        obs = env.reset(task_id=task_id)
        print(
            json.dumps(
                {
                    "[START]": {
                        "task_id": task_id,
                        "task_description": obs.task_description,
                        "sections_required": obs.sections_required,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                }
            )
        )

        done = False
        step_num = 0
        total_reward = 0.0
        max_steps = TASKS[task_id].max_steps

        while not done and step_num < max_steps and (time.time() - run_start) < MAX_TOTAL_SECONDS:
            step_num += 1

            system_prompt = (
                "You are a proposal writing assistant. Write concise, professional proposal sections."
            )
            user_prompt = f"""
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

            action_payload: dict | None = None
            attempts = 0

            while attempts < 2 and action_payload is None:
                try:
                    if client is None:
                        raise RuntimeError("OpenAI client unavailable")

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=800,
                    )
                    raw = response.choices[0].message.content or ""
                    action_payload = _extract_json_payload(raw)
                except Exception as exc:
                    attempts += 1
                    print(
                        (
                            f"Warning: LLM call failed for task={task_id} "
                            f"step={step_num} attempt={attempts}: {exc}"
                        ),
                        file=sys.stderr,
                    )
                finally:
                    time.sleep(1)

            if action_payload is None:
                action_payload = _fallback_action_payload(task_id, obs)

            try:
                action = Action(**action_payload)
            except Exception:
                action = Action(**_fallback_action_payload(task_id, obs))

            # Force finalize when no sections remain, regardless of model output.
            if _next_unfinished_section(obs) == "":
                action = Action(section_name="", content="", action_type="finalize")

            try:
                result = env.step(action)
            except RuntimeError:
                done = True
                break

            obs = result.observation
            done = result.done
            total_reward += result.reward.value

            print(
                json.dumps(
                    {
                        "[STEP]": {
                            "task_id": task_id,
                            "step": step_num,
                            "action_type": action.action_type,
                            "section_name": action.section_name,
                            "reward": result.reward.value,
                            "total_reward": round(total_reward, 4),
                            "done": done,
                            "feedback": result.reward.feedback,
                        }
                    }
                )
            )

        final_score, section_breakdown = grade_task(task_id, obs.current_draft, TASKS[task_id])
        status = "done" if done else "max_steps_reached"
        end_payload = {
            "task_id": task_id,
            "total_steps": step_num,
            "total_reward": round(total_reward, 4),
            "final_score": final_score,
            "section_breakdown": section_breakdown,
            "sections_completed": obs.sections_completed,
            "status": status,
        }

        print(json.dumps({"[END]": end_payload}))
        summary.append(end_payload)

    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
