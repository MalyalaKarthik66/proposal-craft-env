"""Deterministic grading functions for ProposalCraftEnv."""

import math
from collections import Counter

from env.models import Action, Observation, Reward
from env.tasks import SectionRubric, TaskConfig


def grade_section(content: str, rubric: SectionRubric) -> tuple[float, dict, str]:
    """Score a single section using keyword, length, sentence, and coherence checks."""

    score = 0.0
    breakdown: dict[str, float] = {}
    feedback_parts: list[str] = []

    matched = 0
    content_lower = content.lower()
    for keyword in rubric.required_keywords:
        if keyword.lower() in content_lower:
            matched += 1
    keyword_score = matched / len(rubric.required_keywords)
    score += 0.35 * keyword_score
    breakdown["keyword_coverage"] = keyword_score
    feedback_parts.append(f"Keyword coverage: {matched}/{len(rubric.required_keywords)}")

    word_count = len(content.split())
    if word_count < rubric.min_words:
        wc_score = word_count / rubric.min_words
    elif word_count > rubric.max_words:
        wc_score = max(0.0, 1.0 - (word_count - rubric.max_words) / rubric.max_words)
    else:
        wc_score = 1.0
    score += 0.30 * wc_score
    breakdown["word_count"] = wc_score

    sentences = content.count(".") + content.count("!") + content.count("?")
    if sentences >= rubric.must_have_sentences:
        sent_score = 1.0
    else:
        sent_score = sentences / rubric.must_have_sentences
    score += 0.20 * sent_score
    breakdown["sentence_structure"] = sent_score

    coherence_points = 0.0
    stripped = content.lstrip()
    if stripped and stripped[0].isupper():
        coherence_points += 0.05
    if not stripped.startswith("- ") and not stripped.startswith("* ") and not stripped.startswith("•"):
        coherence_points += 0.05
    if "\n\n\n\n" not in content:
        coherence_points += 0.05

    coherence_score = min(1.0, coherence_points / 0.15)
    score += 0.15 * coherence_score
    breakdown["coherence"] = coherence_score

    words = content.lower().split()
    found_positions: dict[str, int] = {}
    for keyword in rubric.required_keywords:
        keyword_lower = keyword.lower()
        for index, word in enumerate(words):
            if keyword_lower in word:
                found_positions[keyword] = index
                break

    proximity_bonus = 0.0
    if len(found_positions) >= 2:
        positions = sorted(found_positions.values())
        for index in range(len(positions) - 1):
            if positions[index + 1] - positions[index] <= 50:
                proximity_bonus = 0.05
                break
    score += proximity_bonus
    breakdown["proximity_bonus"] = proximity_bonus

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "of",
        "to",
        "for",
        "and",
        "but",
        "or",
        "with",
        "that",
        "this",
        "it",
        "at",
        "by",
        "from",
    }
    filtered = [word.lower() for word in content.split() if word.lower() not in stopwords]
    counts = Counter(filtered)
    threshold = math.ceil(len(content.split()) / 10)
    repetition_penalty = 0.0
    if counts and counts.most_common(1)[0][1] > threshold:
        repetition_penalty = -0.1
    score += repetition_penalty
    breakdown["repetition_penalty"] = repetition_penalty

    transition_terms = [
        "furthermore",
        "however",
        "therefore",
        "in addition",
        "as a result",
        "consequently",
        "in conclusion",
        "first",
        "second",
        "finally",
        "additionally",
    ]
    transition_bonus = 0.0
    for term in transition_terms:
        if term in content_lower:
            transition_bonus = 0.05
            break
    score += transition_bonus
    breakdown["transition_bonus"] = transition_bonus

    if proximity_bonus > 0.0:
        feedback_parts.append("Proximity bonus applied")
    if repetition_penalty < 0.0:
        feedback_parts.append("Repetition penalty applied")
    if transition_bonus > 0.0:
        feedback_parts.append("Transition bonus applied")

    final_score = round(max(0.0, min(score, 1.0)), 4)
    return final_score, breakdown, "; ".join(feedback_parts)


def grade_task(task_id: str, draft: dict[str, str], task_config: TaskConfig) -> tuple[float, dict]:
    """Score the full task draft as the mean score across required sections."""

    section_scores: dict[str, float] = {}
    for section in task_config.sections_required:
        if section in draft and draft[section].strip():
            score, _, _ = grade_section(draft[section], task_config.rubric[section])
        else:
            score = 0.0
        section_scores[section] = score

    total = sum(section_scores.values()) / len(task_config.sections_required)
    return round(max(0.0, min(total, 1.0)), 4), section_scores


def compute_step_reward(
    action: Action,
    observation: Observation,
    task_config: TaskConfig,
    prev_score: float,
) -> Reward:
    """Compute reward for one action according to task rubrics and action type."""

    _ = observation
    _ = prev_score

    if action.action_type == "finalize":
        final_score, section_scores = grade_task(task_config.id, observation.current_draft, task_config)
        feedback = f"Final score: {final_score}. Section scores: {section_scores}"
        return Reward(value=final_score, breakdown=section_scores, feedback=feedback)

    rubric = task_config.rubric[action.section_name]
    section_score, breakdown, feedback_str = grade_section(action.content, rubric)

    if action.action_type == "write":
        step_reward = section_score * (1.0 / len(task_config.sections_required))
    else:
        step_reward = section_score * (1.0 / len(task_config.sections_required)) * 0.7

    value = round(max(0.0, min(step_reward, 1.0)), 4)
    return Reward(value=value, breakdown=breakdown, feedback=feedback_str)
