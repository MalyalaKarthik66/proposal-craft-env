"""Unit tests for ProposalCraftEnv environment behavior."""

import pytest

from env.environment import ProposalCraftEnv
from env.models import Action, Observation
from env.tasks import TASKS


EASY_CONTENT = (
    "Students use this app to plan their study workflow with machine learning guidance "
    "that adapts to performance trends over time. The app helps study scheduling by "
    "analyzing performance and recommending better sessions for students. This proposal "
    "describes measurable improvements in study outcomes and app usability."
)


@pytest.mark.parametrize("task_id", ["easy_abstract", "medium_proposal", "hard_full_proposal"])
def test_reset_returns_valid_observation(task_id: str) -> None:
    env = ProposalCraftEnv()
    obs = env.reset(task_id=task_id)

    assert isinstance(obs, Observation)
    assert obs.task_id == task_id
    assert obs.sections_required == TASKS[task_id].sections_required
    assert obs.sections_completed == []
    assert obs.current_draft == {}
    assert obs.step_count == 0


def test_reset_random_task() -> None:
    env = ProposalCraftEnv()
    obs = env.reset()

    assert obs.task_id in TASKS
    assert obs.task_description == TASKS[obs.task_id].name


def test_step_write_action() -> None:
    env = ProposalCraftEnv()
    env.reset(task_id="easy_abstract")

    result = env.step(
        Action(section_name="abstract", content=EASY_CONTENT, action_type="write")
    )

    assert result.reward.value > 0
    assert result.observation.current_draft["abstract"] == EASY_CONTENT
    assert result.done is False


def test_step_finalize_ends_episode() -> None:
    env = ProposalCraftEnv()
    env.reset(task_id="easy_abstract")

    env.step(Action(section_name="abstract", content=EASY_CONTENT, action_type="write"))
    result = env.step(Action(section_name="", content="", action_type="finalize"))

    assert result.done is True


def test_step_after_done_raises() -> None:
    env = ProposalCraftEnv()
    env.reset(task_id="easy_abstract")
    env.step(Action(section_name="", content="", action_type="finalize"))

    with pytest.raises(RuntimeError):
        env.step(Action(section_name="", content="", action_type="finalize"))


def test_max_steps_ends_episode() -> None:
    env = ProposalCraftEnv()
    env.reset(task_id="easy_abstract")

    result = None
    for _ in range(TASKS["easy_abstract"].max_steps):
        result = env.step(
            Action(section_name="abstract", content=EASY_CONTENT, action_type="write")
        )

    assert result is not None
    assert result.done is True
    assert "Max steps reached." in result.observation.feedback


def test_invalid_task_raises() -> None:
    env = ProposalCraftEnv()

    with pytest.raises(ValueError):
        env.reset(task_id="not_a_real_task")


def test_state_returns_dict() -> None:
    env = ProposalCraftEnv()
    env.reset(task_id="easy_abstract")
    state = env.state()

    assert isinstance(state, dict)
    assert "task_id" in state
    assert "step_count" in state
    assert "total_reward" in state
    assert "done" in state
    assert "sections_completed" in state
    assert "sections_remaining" in state
    assert "draft_word_counts" in state
    assert "last_feedback" in state
