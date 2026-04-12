from __future__ import annotations

import re
from typing import Any


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?%?\b")
_EPSILON = 0.01
_REASONING_TERMS = {
    "because",
    "tradeoff",
    "approach",
    "design",
    "constraint",
    "therefore",
    "decision",
    "rationale",
    "reason",
    "impact",
}

_SYNONYMS = {
    "retraining": {"periodic", "refresh", "retrain", "retrained", "schedule"},
    "monitoring": {"observability", "alerts", "alerting", "metrics"},
    "tradeoff": {"compromise", "balance", "latency", "cost"},
    "evaluation": {"validation", "testing", "benchmark"},
    "precision": {"accuracy", "exactness"},
    "recall": {"coverage", "sensitivity"},
    "latency": {"delay", "response"},
    "throughput": {"qps", "capacity"},
    "architecture": {"design", "system", "topology"},
    "queue": {"buffer", "backlog"},
    "fallback": {"degrade", "backup"},
    "mitigation": {"remediation", "fix"},
}


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _contains_numbers(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


def _phrase_present(tokens: set[str], phrase: str) -> bool:
    words = phrase.lower().split()
    if not words:
        return False
    return all(word in tokens for word in words)


def _expand_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        for root, syns in _SYNONYMS.items():
            if token == root or token in syns:
                expanded.add(root)
                expanded.update(syns)
    return expanded


def _fuzzy_match(token: str, candidates: set[str]) -> bool:
    if token in candidates:
        return True
    if len(token) >= 5:
        prefix = token[:4]
        for cand in candidates:
            if cand.startswith(prefix) or prefix in cand:
                return True
    return False


def _coverage_facts(answer_tokens: set[str], facts: list[str]) -> list[str]:
    covered: list[str] = []
    expanded = _expand_tokens(answer_tokens)
    for fact in facts:
        fact_words = fact.lower().split()
        if _phrase_present(expanded, fact):
            covered.append(fact)
            continue
        if fact_words and all(_fuzzy_match(word, expanded) for word in fact_words):
            covered.append(fact)
    return covered


def _keyword_match_ratio(answer_tokens: set[str], keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    matched = 0
    expanded = _expand_tokens(answer_tokens)
    for kw in keywords:
        kw_words = kw.lower().split()
        if _phrase_present(expanded, kw):
            matched += 1
            continue
        if kw_words and all(_fuzzy_match(word, expanded) for word in kw_words):
            matched += 1
    return matched / len(keywords)


def _generic_penalty(answer: str, tokens: set[str], signal_count: int, fact_ratio: float) -> float:
    generic_phrases = [
        "my experience",
        "used in production",
        "worked on",
        "responsible for",
        "handled",
        "involved in",
        "we used",
        "i used",
        "project work",
    ]
    if not any(phrase in answer for phrase in generic_phrases):
        return 0.0
    if signal_count <= 1 and fact_ratio < 0.5:
        return 0.35
    if signal_count <= 1 and fact_ratio < 0.67:
        return 0.25
    return 0.0


def _depth_score(answer: str, tokens: set[str]) -> tuple[float, int, bool]:
    design_terms = {
        "architecture",
        "pipeline",
        "queue",
        "throughput",
        "latency",
        "monitoring",
        "rollback",
        "circuit",
        "fallback",
        "retraining",
        "drift",
        "features",
        "baseline",
        "evaluation",
    }
    reasoning_terms = set(_REASONING_TERMS) | {
        "bottleneck",
        "due",
        "chose",
        "alternative",
        "compared",
    }
    expanded = _expand_tokens(tokens)
    has_numbers = _contains_numbers(answer)
    has_design = any(term in expanded for term in design_terms)
    has_reasoning = any(term in expanded for term in reasoning_terms)
    signal_count = int(has_numbers) + int(has_design) + int(has_reasoning)
    if signal_count >= 3:
        depth = 0.92
    elif signal_count == 2:
        depth = 0.72
    elif signal_count == 1:
        depth = 0.42
    else:
        depth = 0.2
    return depth, signal_count, has_reasoning


def _clarity_score(word_count: int) -> float:
    if word_count < 8:
        return 0.3
    if word_count < 16:
        return 0.5
    if word_count < 30:
        return 0.7
    return 0.85


def _is_irrelevant(relevance: float, prompt_overlap: float, word_count: int) -> bool:
    if word_count == 0:
        return True
    if relevance < 0.05 and prompt_overlap < 0.05 and word_count >= 3:
        return True
    return False


def _prompt_overlap(answer_tokens: set[str], prompt: str) -> float:
    prompt_tokens = _expand_tokens(_token_set(prompt))
    if not prompt_tokens:
        return 0.0
    expanded = _expand_tokens(answer_tokens)
    return len(expanded.intersection(prompt_tokens)) / len(prompt_tokens)


def _band_for_score(
    relevance: float,
    correctness: float,
    depth: float,
    clarity: float,
    total_penalty: float,
    fact_ratio: float,
    word_count: int,
    has_numbers: bool,
) -> str:
    if relevance >= 0.7 and depth >= 0.7 and correctness >= 0.6 and clarity >= 0.6 and total_penalty == 0.0 and fact_ratio >= 0.67:
        return "excellent"
    if relevance >= 0.7 and depth >= 0.7 and correctness >= 0.6 and (has_numbers or word_count >= 20):
        return "good"
    if relevance >= 0.3:
        return "average"
    return "weak"


def grade_answer(answer: str, task: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    normalized = _normalize(answer or "")
    if not normalized:
        return {
            "score": _EPSILON,
            "breakdown": {
                "relevance": 0.0,
                "correctness": 0.0,
                "depth": 0.0,
                "clarity": 0.0,
            },
            "reason": "Invalid input",
            "penalties": {"invalid": 0.4},
            "covered_facts": [],
        }

    tokens = _token_set(normalized)
    required_facts = task.get("required_facts", [])
    keywords = task.get("rubric_keywords", [])
    forbidden = task.get("forbidden_claims", [])

    covered = _coverage_facts(tokens, required_facts)
    fact_ratio = len(covered) / len(required_facts) if required_facts else 0.0
    keyword_ratio = _keyword_match_ratio(tokens, keywords)

    relevance = min(1.0, 0.5 * fact_ratio + 0.5 * keyword_ratio)
    correctness = fact_ratio

    if (fact_ratio > 0.0) or (keyword_ratio > 0.0):
        relevance = max(0.2, relevance)
    if fact_ratio > 0.0:
        correctness = max(0.2, correctness)
    penalties: dict[str, float] = {}

    forbidden_hits = [phrase for phrase in forbidden if phrase in normalized]
    if forbidden_hits:
        correctness = max(0.05, correctness * 0.4)
        penalties["forbidden_claim"] = 0.25

    depth, signal_count, has_reasoning = _depth_score(normalized, tokens)
    if not has_reasoning:
        depth = max(0.2, depth * 0.85)
    if has_reasoning or signal_count >= 2:
        depth = max(0.3, depth)

    word_count = len(tokens)
    clarity = _clarity_score(word_count)

    prompt_overlap = _prompt_overlap(tokens, task.get("prompt", ""))
    if _is_irrelevant(relevance, prompt_overlap, word_count):
        return {
            "score": _EPSILON,
            "breakdown": {
                "relevance": relevance,
                "correctness": correctness,
                "depth": depth,
                "clarity": clarity,
            },
            "reason": "Irrelevant response",
            "penalties": {"irrelevant": 0.5},
            "covered_facts": covered,
        }

    generic_pen = _generic_penalty(normalized, tokens, signal_count, fact_ratio)
    if generic_pen > 0.0:
        penalties["generic"] = generic_pen

    if word_count < 10:
        clarity = max(0.2, clarity - 0.1)
        penalties["short_answer"] = 0.1

    if history:
        past_contents = {
            _normalize(entry.get("content", "")) for entry in history if entry.get("content")
        }
        if normalized in past_contents:
            penalties["repetition"] = 0.2

    if task.get("difficulty") == "hard" and fact_ratio > 0.0:
        correctness = max(0.6, correctness)
        if has_reasoning:
            depth = max(0.5, depth)

    base_score = 0.35 * relevance + 0.30 * correctness + 0.20 * depth + 0.15 * clarity

    total_penalty = sum(penalties.values())
    penalty_factor = min(0.5, total_penalty)
    score = base_score * (1 - penalty_factor)

    band = _band_for_score(relevance, correctness, depth, clarity, total_penalty, fact_ratio, word_count, _contains_numbers(normalized))
    if "generic" in penalties and signal_count <= 1:
        band = "weak"
    if band == "weak":
        score = min(score, 0.3)
        score = max(score, 0.1)
    elif band == "average":
        score = min(max(score, 0.4), 0.6)
    elif band == "good":
        score = min(max(score, 0.7), 0.85)
    else:
        score = max(score, 0.9)
        score = min(score, 1.0)

    if "generic" in penalties:
        score = min(score, 0.35)

    if total_penalty > 0.0 and score > 0.85:
        score = 0.85

    if history:
        prev_rewards = [entry.get("reward") for entry in history if isinstance(entry.get("reward"), (int, float))]
        if prev_rewards:
            last_reward = prev_rewards[-1]
            if score > last_reward + 0.1:
                score = min(1.0, score + 0.03)
            if score < 0.35 and last_reward < 0.35:
                penalties.setdefault("repeated_weak", 0.15)
                score = max(0.1, score * 0.8)

    score = max(_EPSILON, min(1.0 - _EPSILON, score))

    reason_parts = []
    if covered:
        reason_parts.append(f"Covered {len(covered)} required facts.")
    if relevance >= 0.6:
        reason_parts.append("Relevant response.")
    elif relevance >= 0.3:
        reason_parts.append("Partially relevant response.")
    else:
        reason_parts.append("Low relevance response.")
    if depth >= 0.7:
        reason_parts.append("Strong technical depth.")
    elif depth >= 0.4:
        reason_parts.append("Moderate technical depth.")
    else:
        reason_parts.append("Shallow technical depth.")
    if penalties:
        reason_parts.append(f"Penalties applied: {', '.join(penalties.keys())}.")

    return {
        "score": round(score, 4),
        "breakdown": {
            "relevance": round(relevance, 4),
            "correctness": round(correctness, 4),
            "depth": round(depth, 4),
            "clarity": round(clarity, 4),
        },
        "reason": " ".join(reason_parts),
        "penalties": penalties,
        "covered_facts": covered,
    }


def internal_tests() -> dict[str, float]:
    task = {
        "prompt": "Describe your system design and monitoring tradeoffs.",
        "required_facts": ["monitoring", "tradeoff"],
        "rubric_keywords": ["metrics", "architecture", "latency"],
        "forbidden_claims": ["guaranteed perfect"],
    }
    strong = (
        "We built an architecture with a queue and monitoring. We tracked p95 latency and "
        "accuracy metrics, and chose a tradeoff to reduce cost by 18% while keeping recall stable."
    )
    strong_plus = (
        "Our design used a queue-based architecture with monitoring and alerting. Because p95 "
        "latency spiked under load, we added a fallback and a retraining schedule every 2 weeks, "
        "trading off cost for stability. Metrics showed a 12% accuracy lift."
    )
    average = "We used monitoring and had a tradeoff between speed and accuracy in deployment."
    weak = "My experience comes from projects where I used monitoring in production."
    garbage = "asdf qwer zxcv"
    return {
        "strong": grade_answer(strong, task)["score"],
        "strong_plus": grade_answer(strong_plus, task)["score"],
        "average": grade_answer(average, task)["score"],
        "weak": grade_answer(weak, task)["score"],
        "garbage": grade_answer(garbage, task)["score"],
    }
