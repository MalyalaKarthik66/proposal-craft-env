"""Task configuration and rubric definitions for ProposalCraftEnv."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionRubric:
    """Rubric constraints and keyword requirements for a single section."""

    required_keywords: list[str]
    min_words: int
    max_words: int
    must_have_sentences: int = 2


@dataclass(frozen=True)
class TaskConfig:
    """Configuration describing one proposal-writing task."""

    id: str
    name: str
    difficulty: str
    max_steps: int
    source_material: str
    sections_required: list[str]
    rubric: dict[str, SectionRubric]


TASKS: dict[str, TaskConfig] = {
    "easy_abstract": TaskConfig(
        id="easy_abstract",
        name="Write a Single Abstract",
        difficulty="easy",
        max_steps=5,
        source_material=(
            "We are building a web app that helps students track their study schedules. "
            "The app uses machine learning to suggest optimal study times based on past "
            "performance. Target users are undergraduate students."
        ),
        sections_required=["abstract"],
        rubric={
            "abstract": SectionRubric(
                required_keywords=[
                    "students",
                    "machine learning",
                    "study",
                    "app",
                    "performance",
                ],
                min_words=50,
                max_words=200,
                must_have_sentences=3,
            )
        },
    ),
    "medium_proposal": TaskConfig(
        id="medium_proposal",
        name="Write a 3-Section Project Proposal",
        difficulty="medium",
        max_steps=10,
        source_material=(
            "DataViz Pro is an open-source Python library for creating interactive "
            "dashboards. It integrates with pandas and supports real-time data "
            "streaming. The project needs funding to hire a developer and for server "
            "costs."
        ),
        sections_required=["introduction", "methodology", "timeline"],
        rubric={
            "introduction": SectionRubric(
                required_keywords=["open-source", "Python", "dashboards", "data"],
                min_words=80,
                max_words=300,
            ),
            "methodology": SectionRubric(
                required_keywords=["implementation", "pandas", "streaming", "development"],
                min_words=100,
                max_words=400,
            ),
            "timeline": SectionRubric(
                required_keywords=["phase", "month", "milestone", "deliverable"],
                min_words=60,
                max_words=250,
            ),
        },
    ),
    "hard_full_proposal": TaskConfig(
        id="hard_full_proposal",
        name="Write a Full 6-Section Research Proposal",
        difficulty="hard",
        max_steps=20,
        source_material=(
            "We propose developing an AI-based early warning system for crop disease "
            "detection using satellite imagery and deep learning. The system targets "
            "smallholder farmers in Sub-Saharan Africa. We will partner with local "
            "agricultural NGOs for data collection. The model will use transfer "
            "learning from existing plant disease datasets. Expected outcomes include a "
            "mobile-friendly API deployed within 18 months."
        ),
        sections_required=[
            "abstract",
            "introduction",
            "problem_statement",
            "methodology",
            "expected_outcomes",
            "budget_justification",
        ],
        rubric={
            "abstract": SectionRubric(
                required_keywords=["AI", "crop", "disease", "satellite", "deep learning"],
                min_words=100,
                max_words=250,
            ),
            "introduction": SectionRubric(
                required_keywords=["farmers", "Africa", "agriculture", "detection"],
                min_words=150,
                max_words=400,
            ),
            "problem_statement": SectionRubric(
                required_keywords=["problem", "challenge", "impact", "current"],
                min_words=120,
                max_words=350,
            ),
            "methodology": SectionRubric(
                required_keywords=["model", "data", "transfer learning", "training", "validation"],
                min_words=200,
                max_words=500,
            ),
            "expected_outcomes": SectionRubric(
                required_keywords=["outcome", "impact", "farmers", "accuracy", "deployment"],
                min_words=100,
                max_words=300,
            ),
            "budget_justification": SectionRubric(
                required_keywords=["cost", "budget", "personnel", "equipment", "justification"],
                min_words=80,
                max_words=250,
            ),
        },
    ),
}
