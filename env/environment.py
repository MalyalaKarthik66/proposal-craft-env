"""Main ProposalCraftEnv environment implementation."""

import random

from env.graders import compute_step_reward
from env.models import Action, Observation, Reward, StepResult
from env.tasks import TASKS, TaskConfig


class ProposalCraftEnv:
    """OpenEnv-style environment for iterative proposal writing."""

    def __init__(self) -> None:
        self._current_task: TaskConfig | None = None
        self._current_task_id: str | None = None
        self._draft: dict[str, str] = {}
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._last_feedback: str = ""

    def reset(self, task_id: str | None = None) -> Observation:
        """Reset the environment and load a specific or random task."""

        if task_id is None:
            task_id = random.choice(list(TASKS.keys()))
        elif task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        self._current_task = TASKS[task_id]
        self._current_task_id = task_id
        self._draft = {}
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._last_feedback = "Start writing. Focus on the required sections."

        return Observation(
            task_id=task_id,
            task_description=self._current_task.name,
            source_material=self._current_task.source_material,
            sections_required=self._current_task.sections_required,
            sections_completed=[],
            current_draft={},
            feedback=self._last_feedback,
            step_count=0,
            score_so_far=0.0,
        )

    def step(self, action: Action) -> StepResult:
        """Apply one action and return observation, reward, and done state."""

        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        if self._current_task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        self._step_count += 1

        if (
            action.action_type != "finalize"
            and action.section_name not in self._current_task.sections_required
        ):
            reward = Reward(value=0.0, breakdown={}, feedback="Invalid section name")
            self._last_feedback = reward.feedback
            obs = Observation(
                task_id=self._current_task_id or "",
                task_description=self._current_task.name,
                source_material=self._current_task.source_material,
                sections_required=self._current_task.sections_required,
                sections_completed=list(self._draft.keys()),
                current_draft=dict(self._draft),
                feedback=self._last_feedback,
                step_count=self._step_count,
                score_so_far=round(self._total_reward, 4),
            )
            return StepResult(observation=obs, reward=reward, done=False, info={"step": self._step_count})

        if action.action_type == "write":
            if action.section_name in self._draft:
                action = Action(
                    section_name=action.section_name,
                    content=action.content,
                    action_type="revise",
                )
            self._draft[action.section_name] = action.content

        if action.action_type == "revise":
            self._draft[action.section_name] = action.content

        current_obs = Observation(
            task_id=self._current_task_id or "",
            task_description=self._current_task.name,
            source_material=self._current_task.source_material,
            sections_required=self._current_task.sections_required,
            sections_completed=list(self._draft.keys()),
            current_draft=dict(self._draft),
            feedback=self._last_feedback,
            step_count=self._step_count,
            score_so_far=round(self._total_reward, 4),
        )

        reward = compute_step_reward(action, current_obs, self._current_task, self._total_reward)
        self._total_reward += reward.value
        self._last_feedback = reward.feedback

        done = False
        if action.action_type == "finalize":
            done = True
        elif self._step_count >= self._current_task.max_steps:
            done = True
            self._last_feedback += " | Max steps reached."

        self._done = done
        sections_completed = list(self._draft.keys())

        obs = Observation(
            task_id=self._current_task_id or "",
            task_description=self._current_task.name,
            source_material=self._current_task.source_material,
            sections_required=self._current_task.sections_required,
            sections_completed=sections_completed,
            current_draft=dict(self._draft),
            feedback=self._last_feedback,
            step_count=self._step_count,
            score_so_far=round(self._total_reward, 4),
        )

        return StepResult(observation=obs, reward=reward, done=done, info={"step": self._step_count})

    def state(self) -> dict:
        """Return a plain dictionary describing the current environment state."""

        sections_remaining: list[str] = []
        if self._current_task is not None:
            sections_remaining = [
                section
                for section in self._current_task.sections_required
                if section not in self._draft
            ]

        return {
            "task_id": self._current_task_id,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "done": self._done,
            "sections_completed": list(self._draft.keys()),
            "sections_remaining": sections_remaining,
            "draft_word_counts": {key: len(value.split()) for key, value in self._draft.items()},
            "last_feedback": self._last_feedback,
        }
