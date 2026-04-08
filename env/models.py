"""Pydantic models for ProposalCraftEnv actions, observations, and rewards."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Action(BaseModel):
    """Agent action describing section updates or episode finalization."""

    model_config = ConfigDict(extra="forbid")

    section_name: str
    content: str
    action_type: Literal["write", "revise", "finalize"]


class Observation(BaseModel):
    """Environment observation returned after reset and each step."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_description: str
    source_material: str
    sections_required: list[str]
    sections_completed: list[str]
    current_draft: dict[str, str]
    feedback: str
    step_count: int
    score_so_far: float


class Reward(BaseModel):
    """Step-level reward with deterministic scoring breakdown and feedback."""

    model_config = ConfigDict(extra="forbid")

    value: float
    breakdown: dict[str, float]
    feedback: str


class StepResult(BaseModel):
    """Result envelope containing observation, reward, and termination flag."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Reward
    done: bool
    info: dict
