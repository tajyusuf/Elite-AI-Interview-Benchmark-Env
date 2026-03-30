from __future__ import annotations

import re
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "can",
    "did",
    "do",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "why",
    "would",
    "you",
    "your",
}

SKILL_NORMALIZATION = {
    "pythn": "python",
    "kubrnetes": "kubernetes",
    "prompt engg": "prompt engineering",
}

GENERIC_QUESTIONS = {
    "tell me about yourself",
    "what are your strengths",
    "why should we hire you",
    "what motivates you",
}

DEPTH_CUES = {
    "architecture",
    "challenge",
    "decision",
    "design",
    "evaluate",
    "evaluation",
    "failure",
    "latency",
    "measure",
    "metric",
    "metrics",
    "monitoring",
    "ownership",
    "production",
    "quality",
    "reliability",
    "root",
    "scale",
    "scalability",
    "tradeoff",
    "tradeoffs",
    "validate",
    "validation",
    "why",
    "how",
}

TECHNICAL_SIGNAL_CUES = {
    "alternative",
    "bottleneck",
    "constraint",
    "constraints",
    "drawback",
    "latency",
    "load",
    "metric",
    "metrics",
    "rollback",
    "scalability",
    "signal",
    "throughput",
    "timeout",
}

VAGUE_QUESTION_PHRASES = {
    "what would you improve",
    "what tradeoff did you face",
    "what failure case",
}

FOLLOW_UP_CUES = {"clarify", "earlier", "follow", "following", "mentioned", "previously"}
VAGUE_RESPONSE_CUES = {"unclear", "vague", "missing", "not fully documented", "no exact", "maybe", "happier"}
NONSENSE_PATTERNS = (r"^[^a-zA-Z0-9]*$", r"^(?:hi|hello|ok|hmm|test)\??$")

STYLE_MULTIPLIERS = {
    "friendly": {
        "clarity_bonus": 1.15,
        "depth_bonus": 1.0,
        "repetition_penalty": 0.9,
        "focus_penalty": 0.9,
        "generic_penalty": 0.9,
        "irrelevant_penalty": 0.9,
        "vague_penalty": 0.95,
    },
    "strict": {
        "clarity_bonus": 1.0,
        "depth_bonus": 1.0,
        "repetition_penalty": 1.15,
        "focus_penalty": 1.15,
        "generic_penalty": 1.1,
        "irrelevant_penalty": 1.15,
        "vague_penalty": 1.1,
    },
    "technical": {
        "clarity_bonus": 1.0,
        "depth_bonus": 1.18,
        "repetition_penalty": 1.0,
        "focus_penalty": 1.0,
        "generic_penalty": 1.0,
        "irrelevant_penalty": 1.0,
        "vague_penalty": 1.05,
    },
    "aggressive": {
        "clarity_bonus": 0.95,
        "depth_bonus": 1.0,
        "repetition_penalty": 1.2,
        "focus_penalty": 1.2,
        "generic_penalty": 1.15,
        "irrelevant_penalty": 1.15,
        "vague_penalty": 1.2,
    },
}


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [token for token in cleaned.split() if token and token not in STOPWORDS]


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _normalize_question(text: str) -> str:
    return " ".join(sorted(_token_set(text)))


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _normalize_skill(skill: str) -> str:
    return SKILL_NORMALIZATION.get(skill.strip().lower(), skill.strip().lower())


def _normalized_topic(topic: str) -> str:
    return _normalize_skill(_normalize_phrase(topic))


def _build_topic_registry(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profile = task["candidate_profile"]
    expectations = task["hidden_evaluation_expectations"]
    registry: dict[str, dict[str, Any]] = {}

    for skill in profile.get("skills", []):
        if not skill.strip():
            continue
        normalized = _normalize_skill(skill)
        registry[normalized] = {
            "tokens": set(_tokenize(normalized)),
            "weight": 1.0,
            "kind": "skill",
        }

    for project in profile.get("projects", []):
        normalized = _normalized_topic(project["name"])
        registry[normalized] = {
            "tokens": set(_tokenize(normalized)),
            "weight": 1.25,
            "kind": "project",
        }

    for topic in expectations.get("important_topics", []):
        normalized = _normalized_topic(topic)
        registry.setdefault(
            normalized,
            {
                "tokens": set(_tokenize(normalized)),
                "weight": 1.1,
                "kind": "important_topic",
            },
        )

    role = _normalize_phrase(str(profile.get("target_role", "")))
    if role:
        registry[role] = {"tokens": set(_tokenize(role)), "weight": 0.8, "kind": "role"}

    experience = _normalize_phrase(str(profile.get("years_experience", "")))
    if experience:
        registry["experience"] = {
            "tokens": set(_tokenize(experience)) | {"experience", "years", "year"},
            "weight": 0.9,
            "kind": "experience",
        }

    return registry


def _topic_match_strength(question_tokens: set[str], normalized_question: str, topic: str, meta: dict[str, Any]) -> tuple[float, str] | None:
    tokens = meta["tokens"]
    if not tokens:
        return None
    if topic in normalized_question:
        return meta["weight"], "exact"

    overlap = len(tokens & question_tokens)
    if overlap == len(tokens) and overlap > 0:
        return meta["weight"] * 0.92, "complete_token"
    if overlap > 0:
        partial_ratio = overlap / len(tokens)
        if partial_ratio >= 0.5:
            return meta["weight"] * (0.45 + 0.35 * partial_ratio), "partial"
        if len(tokens) == 1:
            return meta["weight"] * 0.55, "single_token"
    return None


def _score_relevance(question: str, task: dict[str, Any]) -> tuple[float, set[str], list[str]]:
    registry = _build_topic_registry(task)
    question_tokens = _token_set(question)
    normalized_question = _normalize_phrase(question)
    matched_topics: set[str] = set()
    match_reasons: list[str] = []
    weighted_hits = 0.0

    for topic, meta in registry.items():
        match = _topic_match_strength(question_tokens, normalized_question, topic, meta)
        if not match:
            continue
        strength, reason = match
        weighted_hits += strength
        matched_topics.add(topic)
        match_reasons.append(f"{meta['kind']}:{topic}:{reason}")

    if task["difficulty"] in {"hard", "edge"} and any(
        cue in question.lower() for cue in {"clarify", "conflict", "metric", "metrics", "ownership", "missing"}
    ):
        weighted_hits += 0.45
        match_reasons.append("difficulty_bonus:clarification")

    if any(token in question.lower() for token in {"experience", "years", "role"}):
        weighted_hits += 0.25
        matched_topics.add("experience")
        match_reasons.append("experience_probe")

    strong_match_bonus = 0.0
    if len(matched_topics) >= 2:
        strong_match_bonus += 0.18
        match_reasons.append("multi_match_bonus")
    if any("project:" in reason and reason.endswith(":exact") for reason in match_reasons):
        strong_match_bonus += 0.16
        match_reasons.append("exact_project_bonus")
    if any("skill:" in reason and reason.endswith(":exact") for reason in match_reasons):
        strong_match_bonus += 0.1
        match_reasons.append("exact_skill_bonus")

    relevance = min(0.3, round(0.06 * weighted_hits + strong_match_bonus * 0.2, 4))
    return relevance, matched_topics, match_reasons


def _score_depth(question: str) -> float:
    tokens = _token_set(question)
    cue_hits = len(tokens & DEPTH_CUES)
    signal_hits = len(tokens & TECHNICAL_SIGNAL_CUES)
    multi_clause = question.count(",") + question.count(";") > 0 or " and " in question.lower()
    depth = 0.05 * min(cue_hits, 4)
    depth += 0.03 * min(signal_hits, 3)
    if len(tokens) >= 8:
        depth += 0.05
    if len(tokens) >= 14:
        depth += 0.03
    if multi_clause:
        depth += 0.04
    return min(0.3, round(depth, 4))


def _recent_topics(history: list[dict[str, Any]]) -> set[str]:
    recent: set[str] = set()
    for item in history[-2:]:
        recent.update(item.get("matched_topics", []))
    return recent


def _last_response_is_vague(history: list[dict[str, Any]]) -> bool:
    if not history:
        return False
    last_response = str(history[-1].get("candidate_response", "")).lower()
    return any(cue in last_response for cue in VAGUE_RESPONSE_CUES)


def _score_sequence(question: str, history: list[dict[str, Any]], task: dict[str, Any], matched_topics: set[str]) -> tuple[float, float]:
    question_lower = question.lower()
    if not history:
        opening_targets = set(task["hidden_evaluation_expectations"].get("priority_topics", []))
        if any(target in question_lower for target in opening_targets):
            return 0.18, 0.0
        if matched_topics:
            return 0.14, 0.0
        return 0.08, 0.04

    recent_topics = _recent_topics(history)
    follow_up = any(cue in question_lower for cue in FOLLOW_UP_CUES)
    vague_follow_up = _last_response_is_vague(history) and any(
        cue in question_lower for cue in {"clarify", "metric", "metrics", "ownership", "challenge", "example"}
    )

    if matched_topics & recent_topics:
        return 0.2 if vague_follow_up or follow_up else 0.17, 0.0
    if vague_follow_up:
        return 0.18, 0.0
    if matched_topics:
        return 0.11, 0.03
    return 0.04, 0.08


def _score_coverage(
    matched_topics: set[str],
    covered_topics: set[str],
    history: list[dict[str, Any]],
    task: dict[str, Any],
) -> tuple[float, set[str], float]:
    expectations = {_normalized_topic(topic) for topic in task["hidden_evaluation_expectations"].get("important_topics", [])}
    expected_topics = expectations | {
        _normalized_topic(skill) for skill in task["candidate_profile"].get("skills", []) if skill.strip()
    } | {
        _normalized_topic(project["name"]) for project in task["candidate_profile"].get("projects", [])
    }
    new_topics = matched_topics - covered_topics
    important_new = new_topics & expectations
    uncovered_topics = expected_topics - covered_topics
    repeated_focus_penalty = 0.0

    if not new_topics and matched_topics:
        repeated_focus_penalty = 0.05
        if history and set(history[-1].get("matched_topics", [])) == matched_topics:
            repeated_focus_penalty = 0.09

    repeated_focus_count = 0
    for item in history[-2:]:
        if matched_topics and set(item.get("matched_topics", [])) == matched_topics:
            repeated_focus_count += 1

    if not new_topics and matched_topics:
        repeated_focus_penalty = 0.05
        if repeated_focus_count >= 1:
            repeated_focus_penalty = 0.12
        if repeated_focus_count >= 2:
            repeated_focus_penalty = 0.22
        if uncovered_topics:
            repeated_focus_penalty = max(repeated_focus_penalty, 0.22)

    topic_asked_count = 0
    for item in history:
        if matched_topics and set(item.get("matched_topics", [])) == matched_topics:
            topic_asked_count += 1
    if topic_asked_count > 2:
        repeated_focus_penalty = max(repeated_focus_penalty, 0.26)

    no_progress_streak = 0
    for item in reversed(history[-2:]):
        if item.get("new_topics"):
            break
        no_progress_streak += 1
    if not new_topics and no_progress_streak >= 2:
        repeated_focus_penalty = max(repeated_focus_penalty, 0.22)

    if important_new:
        coverage = min(0.1, 0.06 + 0.025 * len(important_new))
        return coverage, new_topics, repeated_focus_penalty
    if new_topics:
        return min(0.08, 0.04 + 0.02 * len(new_topics)), new_topics, repeated_focus_penalty
    return 0.0, set(), repeated_focus_penalty


def _score_clarity(question: str) -> float:
    word_count = len(question.split())
    clarity = 0.0
    if question.endswith("?"):
        clarity += 0.04
    if 6 <= word_count <= 24:
        clarity += 0.04
    if not re.search(r"[^\w\s,\-\?']", question):
        clarity += 0.02
    elif word_count <= 18:
        clarity += 0.01
    return min(0.1, round(clarity, 4))


def _empty_or_nonsense_penalty(question: str) -> float:
    if not question.strip():
        return 0.3
    stripped = question.strip().lower()
    if any(re.match(pattern, stripped) for pattern in NONSENSE_PATTERNS):
        return 0.25
    if len(_token_set(question)) <= 1:
        return 0.2
    return 0.0


def _repetition_penalty(question: str, history: list[dict[str, Any]]) -> float:
    normalized = _normalize_question(question)
    question_tokens = set(normalized.split())
    for item in history:
        previous = item.get("normalized_question", "")
        previous_tokens = set(previous.split())
        if normalized == previous and normalized:
            return 0.3
        if previous_tokens and question_tokens:
            overlap = len(question_tokens & previous_tokens) / max(len(question_tokens), len(previous_tokens))
            if overlap >= 0.8:
                return 0.22
            if overlap >= 0.6:
                return 0.14
    return 0.0


def _generic_penalty(question: str, relevance: float) -> float:
    normalized = question.lower().strip().rstrip("?")
    if normalized in GENERIC_QUESTIONS:
        return 0.25
    if any(phrase in normalized for phrase in VAGUE_QUESTION_PHRASES):
        return 0.18
    if relevance <= 0.02 and len(_token_set(question)) < 5:
        return 0.16
    return 0.0


def _irrelevant_penalty(relevance: float, matched_topics: set[str], question: str, task: dict[str, Any]) -> float:
    if matched_topics or relevance >= 0.08:
        return 0.0
    normalized = question.lower().strip().rstrip("?")
    if normalized in GENERIC_QUESTIONS:
        return 0.25
    if task["difficulty"] in {"hard", "edge"}:
        return 0.22
    return 0.16


def _shallow_penalty(depth: float, question: str) -> float:
    if depth >= 0.16:
        return 0.0
    if len(_token_set(question)) <= 5:
        return 0.2
    return 0.1


def _build_reasoning(
    components: dict[str, float],
    penalties: dict[str, float],
    matched_topics: set[str],
    new_topics: set[str],
) -> str:
    positives: list[str] = []
    negatives: list[str] = []

    if matched_topics:
        positives.append(f"matched {', '.join(sorted(matched_topics)[:3])}")
    if new_topics:
        positives.append(f"expanded coverage into {', '.join(sorted(new_topics)[:2])}")
    if components["depth_of_question"] >= 0.2:
        positives.append("asked for technical depth")
    if components["logical_sequence"] >= 0.16:
        positives.append("followed the prior context well")
    if components["clarity_of_question"] >= 0.08:
        positives.append("was clearly phrased")

    if penalties["repetition"] >= 0.14:
        negatives.append("repeated earlier wording")
    if penalties["repeated_focus"] >= 0.05:
        negatives.append("stayed on an already-covered topic")
    if penalties["context_jump"] >= 0.05:
        negatives.append("jumped away from recent context")
    if penalties["generic_question"] >= 0.12:
        negatives.append("was too generic")
    if penalties["empty_or_nonsense"] >= 0.2:
        negatives.append("looked empty or nonsensical")

    if not positives:
        positives.append("provided limited interview signal")
    if not negatives:
        negatives.append("avoided major penalties")

    return f"Positive: {'; '.join(positives)}. Negative: {'; '.join(negatives)}."


def grade_question(
    question: str,
    history: list[dict[str, Any]],
    covered_topics: set[str],
    task: dict[str, Any],
    interviewer_style: str,
) -> dict[str, Any]:
    style = STYLE_MULTIPLIERS[interviewer_style]

    relevance, matched_topics, relevance_reasons = _score_relevance(question, task)
    depth = min(0.3, round(_score_depth(question) * style["depth_bonus"], 4))
    logical_sequence, context_jump_penalty = _score_sequence(question, history, task, matched_topics)
    coverage, new_topics, repeated_focus_penalty = _score_coverage(matched_topics, covered_topics, history, task)
    clarity = min(0.1, round(_score_clarity(question) * style["clarity_bonus"], 4))

    empty_penalty = _empty_or_nonsense_penalty(question)
    repetition_penalty = min(0.3, round(_repetition_penalty(question, history) * style["repetition_penalty"], 4))
    generic_penalty = min(0.25, round(_generic_penalty(question, relevance) * style["generic_penalty"], 4))
    irrelevant_penalty = min(
        0.3,
        round(_irrelevant_penalty(relevance, matched_topics, question, task) * style["irrelevant_penalty"], 4),
    )
    shallow_penalty = min(0.2, round(_shallow_penalty(depth, question), 4))
    focus_penalty = min(0.3, round(repeated_focus_penalty * style["focus_penalty"], 4))
    jump_penalty = min(0.12, round(context_jump_penalty, 4))
    vague_penalty = 0.0

    if task["difficulty"] in {"hard", "edge"} and _last_response_is_vague(history) and not any(
        cue in question.lower() for cue in {"clarify", "metric", "metrics", "ownership", "example", "specific"}
    ):
        vague_penalty = min(0.14, round(0.08 * style["vague_penalty"], 4))

    positive_total = relevance + depth + logical_sequence + coverage + clarity
    negative_total = (
        repetition_penalty
        + generic_penalty
        + irrelevant_penalty
        + shallow_penalty
        + jump_penalty
        + focus_penalty
        + vague_penalty
        + empty_penalty
    )
    reward = max(0.0, min(1.0, round(positive_total - negative_total, 4)))

    normalized_question = _normalize_question(question)
    components = {
        "relevance_to_profile": round(relevance, 4),
        "depth_of_question": round(depth, 4),
        "logical_sequence": round(logical_sequence, 4),
        "coverage_of_topics": round(coverage, 4),
        "clarity_of_question": round(clarity, 4),
    }
    penalties = {
        "repetition": repetition_penalty,
        "irrelevant_question": irrelevant_penalty,
        "generic_question": generic_penalty,
        "shallow_question": shallow_penalty,
        "context_jump": jump_penalty,
        "repeated_focus": focus_penalty,
        "vague_follow_up_miss": vague_penalty,
        "empty_or_nonsense": round(empty_penalty, 4),
    }

    return {
        "reward": reward,
        "matched_topics": sorted(matched_topics),
        "new_topics": sorted(new_topics),
        "breakdown": {
            **components,
            "penalties": penalties,
        },
        "normalized_question": normalized_question,
        "reasoning": _build_reasoning(components, penalties, matched_topics, new_topics),
        "relevance_reasons": relevance_reasons,
    }
