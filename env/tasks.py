from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


def _build_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "easy_python_api",
            "difficulty": "easy",
            "max_turns": 4,
            "candidate_profile": {
                "name": "Aarav Singh",
                "years_experience": 2,
                "target_role": "Junior AI Engineer",
                "skills": ["Python"],
                "projects": [
                    {
                        "name": "Resume Parser API",
                        "summary": "Built a Python API that extracts entities from resumes for internal recruiters.",
                        "impact": "Reduced manual screening time by 40%.",
                    }
                ],
                "notes": "Strong coding fundamentals, limited production scale exposure.",
            },
            "hidden_evaluation_expectations": {
                "important_topics": [
                    "python",
                    "resume parser api",
                    "entity extraction",
                    "testing",
                    "deployment",
                ],
                "required_follow_ups": ["impact", "testing", "design choices"],
                "vague_topics": [],
                "inconsistencies": [],
                "priority_topics": ["python", "project", "impact"],
            },
        },
        {
            "task_id": "medium_mle_platform",
            "difficulty": "medium",
            "max_turns": 5,
            "candidate_profile": {
                "name": "Mina Thomas",
                "years_experience": 5,
                "target_role": "Machine Learning Engineer",
                "skills": ["Python", "PyTorch", "SQL", "MLOps"],
                "projects": [
                    {
                        "name": "Demand Forecasting Pipeline",
                        "summary": "Trained PyTorch models and scheduled retraining jobs for retail forecasting.",
                        "impact": "Improved weekly forecast accuracy by 14%.",
                    },
                    {
                        "name": "Feature Store Migration",
                        "summary": "Moved offline features into a managed feature store and rewrote serving queries in SQL.",
                        "impact": "Cut online feature latency from 220ms to 80ms.",
                    },
                ],
                "notes": "Good platform exposure across experimentation and deployment.",
            },
            "hidden_evaluation_expectations": {
                "important_topics": [
                    "pytorch",
                    "mlops",
                    "sql",
                    "demand forecasting pipeline",
                    "feature store migration",
                    "retraining",
                    "latency",
                ],
                "required_follow_ups": ["tradeoffs", "monitoring", "cross-project comparison"],
                "vague_topics": [],
                "inconsistencies": [],
                "priority_topics": ["project", "mlops", "impact", "tradeoff"],
            },
        },
        {
            "task_id": "hard_vague_research_to_prod",
            "difficulty": "hard",
            "max_turns": 5,
            "candidate_profile": {
                "name": "N. K. Rao",
                "years_experience": "6? maybe 4 in industry + 3 research",
                "target_role": "Senior Applied AI Engineer",
                "skills": ["LLMs", "Python", "Distributed systems"],
                "projects": [
                    {
                        "name": "Assistant Platform",
                        "summary": "Led parts of an assistant stack for enterprise users.",
                        "impact": "Made it more robust and users were happier.",
                    },
                    {
                        "name": "Ranking Research",
                        "summary": "Worked on retrieval and ranking ideas before productionization.",
                        "impact": "Results were promising but not fully documented.",
                    },
                ],
                "notes": "Profile mixes research and production claims; some metrics and ownership details are missing.",
            },
            "hidden_evaluation_expectations": {
                "important_topics": [
                    "assistant platform",
                    "ranking research",
                    "ownership",
                    "evaluation",
                    "metrics",
                    "productionization",
                    "distributed systems",
                ],
                "required_follow_ups": ["missing metrics", "ownership", "production challenges"],
                "vague_topics": [
                    "made it more robust",
                    "users were happier",
                    "promising results",
                ],
                "inconsistencies": ["years_experience"],
                "priority_topics": ["metrics", "ownership", "followup", "production"],
            },
        },
        {
            "task_id": "edge_noisy_conflicting",
            "difficulty": "edge",
            "max_turns": 5,
            "candidate_profile": {
                "name": "Riya / Riyaa Patel",
                "years_experience": "3 and also 7??",
                "target_role": "AI Systems Engineer",
                "skills": ["pythn", "kubrnetes", "", "prompt engg"],
                "projects": [
                    {
                        "name": "chatbot??",
                        "summary": "made bot 4 support maybe with rag i think",
                        "impact": "good csat, no exact num",
                    },
                    {
                        "name": "infra cleanup",
                        "summary": "fixed deploy stuff, faster build maybe 2x? docs missing",
                        "impact": "unclear",
                    },
                ],
                "notes": "Noisy profile with typos, missing skill clarity, and conflicting experience numbers.",
            },
            "hidden_evaluation_expectations": {
                "important_topics": [
                    "python",
                    "kubernetes",
                    "prompt engineering",
                    "chatbot",
                    "rag",
                    "infra cleanup",
                    "experience conflict",
                ],
                "required_follow_ups": ["clarify skills", "clarify metrics", "resolve conflicts"],
                "vague_topics": [
                    "good csat",
                    "faster build maybe 2x",
                    "made bot 4 support maybe",
                ],
                "inconsistencies": ["years_experience", "name", "skills"],
                "priority_topics": ["clarify", "metrics", "ownership", "reliability"],
            },
        },
    ]


TASKS: list[dict[str, Any]] = _build_tasks()


def get_task_by_index(index: int) -> dict[str, Any]:
    return deepcopy(TASKS[index % len(TASKS)])


def list_tasks() -> list[dict[str, Any]]:
    return [deepcopy(task) for task in TASKS]


def _normalized_skill(skill: str) -> str:
    mapping = {
        "pythn": "python",
        "kubrnetes": "kubernetes",
        "prompt engg": "prompt engineering",
    }
    return mapping.get(skill.strip().lower(), skill.strip().lower())


def normalize_topic_name(topic: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", topic.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    return _normalized_skill(cleaned)


def normalize_topic_collection(topics: list[str] | set[str]) -> list[str]:
    normalized = {normalize_topic_name(topic) for topic in topics if normalize_topic_name(topic)}
    return sorted(normalized)


def generate_candidate_response(
    task: dict[str, Any],
    question: str,
    covered_topics: set[str],
    history: list[dict[str, Any]],
) -> str:
    question_lower = question.lower()
    profile = task["candidate_profile"]
    expectations = task["hidden_evaluation_expectations"]
    difficulty = task["difficulty"]
    turn_number = len(history) + 1

    def response_state() -> str:
        if difficulty == "easy":
            return "strong"
        if difficulty == "medium":
            return "strong" if turn_number <= 2 else "partial"
        if difficulty == "hard":
            return "partial" if turn_number <= 2 else "strong"
        return "contradictory" if turn_number == 1 else "partial"

    answer_style = response_state()

    if "experience" in question_lower or "year" in question_lower:
        if difficulty == "edge" and turn_number == 1:
            return f"The profile is inconsistent there. One version says {profile['years_experience']}, but the cleaner answer is that I need to separate overlapping internships from full-time work."
        return f"I should clarify that first. My profile mixes research and industry work, and the clearest split is {profile['years_experience']}."

    if any(token in question_lower for token in {"skill", "python", "kubernetes", "prompt engineering", "prompt engg"}):
        normalized = [_normalized_skill(skill) for skill in profile.get("skills", []) if skill]
        if normalized:
            if answer_style == "strong":
                return f"The skill names in the profile are noisy. The clean list is: {', '.join(normalized)}. My strongest production examples came from the listed projects."
            return f"The skill names in the profile are noisy. The clean list is: {', '.join(normalized)}."

    if "name" in question_lower:
        return f"The correct name is {profile['name'].split('/')[0].strip()}."

    if any(token in question_lower for token in {"experience", "year", "ownership", "research", "industry"}):
        if "years_experience" in expectations["inconsistencies"]:
            return f"The experience numbers are inconsistent in the profile. The nuance is {profile['years_experience']}."

    for project in profile.get("projects", []):
        name_lower = project["name"].lower()
        project_tokens = [token for token in re.split(r"[^a-z0-9]+", name_lower) if token]
        if name_lower in question_lower or any(token in question_lower for token in project_tokens):
            if "metric" in question_lower or "impact" in question_lower:
                if answer_style == "strong":
                    return f"{project['impact']} I used that metric because it reflected the business outcome more reliably than raw model accuracy."
                if answer_style == "partial":
                    return f"{project['impact']} I do not have every supporting metric in the profile, but that was the main directional result."
                return f"{project['impact']} although the exact validation trail is incomplete in the profile."
            if "challenge" in question_lower or "trade" in question_lower or "why" in question_lower:
                if answer_style == "strong":
                    return f"In {project['name']}, the main challenge was balancing delivery speed with reliability. I owned the decision to favor simpler rollout paths first, then tightened reliability once usage patterns were clear."
                if answer_style == "partial":
                    return f"In {project['name']}, the main challenge was balancing delivery speed with reliability. Some of the tradeoff details are not fully captured in the profile."
                return f"In {project['name']}, the challenge was messy. We changed things a few times, and I would need to reconstruct the exact tradeoffs."
            if any(token in question_lower for token in {"ownership", "own", "responsible"}):
                if answer_style == "strong":
                    return f"For {project['name']}, I owned implementation and rollout coordination. I was directly responsible for the design choices that affected reliability and delivery speed."
                return f"My ownership on {project['name']} was meaningful, although the profile summarizes it too vaguely to show the full split."
            return project["summary"] if answer_style == "strong" else f"{project['summary']} The profile is intentionally compact, so some implementation details are omitted."

    for skill in profile.get("skills", []):
        normalized = _normalized_skill(skill)
        if normalized and normalized in question_lower:
            if answer_style == "strong":
                return f"My strongest experience with {normalized} came from the projects on the profile, where I used it in a hands-on and production-facing way."
            return f"My strongest experience with {normalized} came from the projects on the profile, though the exact examples need follow-up."

    if "metric" in question_lower or "measure" in question_lower or "evaluate" in question_lower:
        if expectations["vague_topics"]:
            if turn_number <= 2:
                return "The profile leaves metrics vague on purpose; I would explain evaluation criteria, reliability indicators, and user impact if asked during a real interview."
            return "The vague profile summary hides the specifics, but I would ground success in user-facing outcomes, quality checks, and operational reliability rather than a single vanity metric."
        return "I tracked measurable business and model outcomes, including quality improvements and operational performance."

    if "follow" in question_lower or "earlier" in question_lower:
        if difficulty in {"hard", "edge"}:
            return "Following up is helpful here because the profile intentionally leaves ambiguity that a strong interviewer should resolve step by step."
        return "Following up is useful here because it gets from summary-level claims into concrete implementation detail."

    if any(token in question_lower for token in {"ownership", "production", "reliability"}):
        if answer_style == "strong":
            return "I focused on rollout reliability, debugging, and the production decisions that reduced downstream surprises after launch."
        if answer_style == "partial":
            return "I worked on production-facing concerns, but the profile only hints at the ownership boundaries and operating constraints."
        return "I was involved, but the exact split between my work and the team around me is one of the profile ambiguities that should be clarified."

    normalized_covered_topics = {normalize_topic_name(topic) for topic in covered_topics}
    uncovered = [
        topic for topic in expectations["important_topics"] if normalize_topic_name(topic) not in normalized_covered_topics
    ]
    if uncovered:
        lead = uncovered[0]
        if difficulty == "edge":
            return f"A strong next discussion area would be {lead}, because the current profile text is noisy and that area still needs clarification."
        return f"A strong next discussion area would be {lead}, because it has not been explored yet."

    if difficulty == "hard":
        return "I can go deeper on design decisions, evaluation, or ownership depending on which ambiguity you want to resolve next."
    if difficulty == "edge":
        return "I can clarify the noisy parts of the profile, especially skills, impact, and ownership."
    return "I can expand on design decisions, metrics, or project ownership depending on what you want to probe next."
