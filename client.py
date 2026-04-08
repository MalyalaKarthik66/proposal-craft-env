"""Root-level client wrapper for OpenEnv CLI compatibility."""

from env.environment import ProposalCraftEnv
from env.models import Action, Observation, StepResult


class Client:
    """Thin adapter exposing reset/step/state over ProposalCraftEnv."""

    def __init__(self) -> None:
        self._env = ProposalCraftEnv()

    def reset(self, task_id: str | None = None) -> Observation:
        return self._env.reset(task_id=task_id)

    def step(self, action: Action | dict) -> StepResult:
        if isinstance(action, dict):
            action = Action(**action)
        return self._env.step(action)

    def state(self) -> dict:
        return self._env.state()
