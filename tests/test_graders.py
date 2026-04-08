"""Unit tests for deterministic grading functions."""

from env.graders import grade_section, grade_task
from env.tasks import TASKS, SectionRubric


def test_grade_section_perfect_score() -> None:
    rubric = SectionRubric(
        required_keywords=["students", "machine learning", "study", "app", "performance"],
        min_words=50,
        max_words=200,
        must_have_sentences=3,
    )
    content = (
        "Students rely on this app to improve study planning with machine learning and "
        "clear tracking of performance over time. The app supports daily study decisions "
        "and helps students adapt routines based on performance signals. This study "
        "proposal explains measurable outcomes, practical deployment, and sustained "
        "student benefits."
    )

    score, breakdown, _ = grade_section(content, rubric)

    assert score >= 0.95
    assert breakdown["keyword_coverage"] == 1.0


def test_grade_section_zero_keywords() -> None:
    rubric = TASKS["easy_abstract"].rubric["abstract"]
    content = "Unrelated cooking notes mention recipes, spices, and kitchen tools only."

    score, _, _ = grade_section(content, rubric)

    assert score < 0.5


def test_grade_section_too_short() -> None:
    rubric = TASKS["medium_proposal"].rubric["timeline"]
    content = "Phase one starts now."

    _, breakdown, _ = grade_section(content, rubric)

    assert breakdown["word_count"] < 1.0


def test_grade_section_too_long() -> None:
    rubric = TASKS["easy_abstract"].rubric["abstract"]
    repeated = "students machine learning study app performance "
    content = "Intro sentence. " + (repeated * 260) + "Final sentence."

    _, breakdown, _ = grade_section(content, rubric)

    assert breakdown["word_count"] < 1.0


def test_grade_task_all_sections() -> None:
    task = TASKS["easy_abstract"]
    draft = {
        "abstract": (
            "Students benefit from a machine learning app that optimizes study schedules "
            "and links recommendations to performance trends. The app helps students "
            "improve study focus through adaptive suggestions. This abstract summarizes "
            "project goals, implementation value, and performance impact."
        )
    }

    total, section_scores = grade_task(task.id, draft, task)

    assert "abstract" in section_scores
    assert 0.0 <= total <= 1.0


def test_grade_task_missing_section() -> None:
    task = TASKS["medium_proposal"]
    draft = {
        "introduction": (
            "This open-source Python project builds dashboards for data work. "
            "It explains why dashboards matter for teams and how data workflows improve."
        ),
        "methodology": (
            "Implementation will use pandas and streaming pipelines during development. "
            "The development process includes testing, integration, and deployment."
        ),
    }

    total, _ = grade_task(task.id, draft, task)

    assert total < 1.0


def test_grader_is_deterministic() -> None:
    rubric = TASKS["hard_full_proposal"].rubric["abstract"]
    content = (
        "AI methods for crop disease detection from satellite imagery and deep learning "
        "will support early intervention. The abstract states scope, data assumptions, "
        "and measurable outcomes for deployment readiness."
    )

    result_one = grade_section(content, rubric)
    result_two = grade_section(content, rubric)

    assert result_one == result_two
