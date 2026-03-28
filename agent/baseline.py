from __future__ import annotations

import re

from env.models import Action, Observation


def _normalize_skill(skill: str) -> str:
    mapping = {
        "pythn": "Python",
        "kubrnetes": "Kubernetes",
        "prompt engg": "prompt engineering",
    }
    return mapping.get(skill.strip().lower(), skill.strip())


def _normalize_topic(topic: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", topic.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    mapped = {
        "pythn": "python",
        "kubrnetes": "kubernetes",
        "prompt engg": "prompt engineering",
    }
    return mapped.get(cleaned, cleaned)


class BaselineInterviewAgent:
    """Rule-based interviewer that adapts to coverage gaps and prior answers."""

    def act(self, observation: Observation) -> Action:
        profile = observation.candidate_profile
        history = observation.history
        turn_number = len(history) + 1
        asked_questions = {entry["question"].lower() for entry in history}
        asked_topic_counts = self._asked_topic_counts(history)
        skills = [_normalize_skill(skill) for skill in profile.get("skills", []) if skill.strip()]
        projects = profile.get("projects", [])
        notes = str(profile.get("notes", "")).lower()
        years = str(profile.get("years_experience", ""))
        covered_topics = {
            _normalize_topic(topic)
            for entry in history
            for topic in entry.get("matched_topics", [])
        }
        uncovered_topics = self._derive_uncovered_topics(observation, covered_topics)
        last_response = str(history[-1].get("candidate_response", "")).lower() if history else ""
        last_depth = float(history[-1].get("breakdown", {}).get("depth_of_question", 0.0)) if history else 0.0
        needs_clarification = any(token in last_response for token in {"vague", "unclear", "missing", "not fully documented", "need follow-up"})

        def choose(*questions: str) -> Action:
            for question in questions:
                if not question:
                    continue
                if question.lower() not in asked_questions:
                    return Action(question=question)
            fallback = next((question for question in reversed(questions) if question), "Can you give one concrete example with a measurable outcome?")
            return Action(question=fallback)

        def project_name(index: int) -> str | None:
            if index < len(projects):
                return projects[index]["name"]
            return None

        def topic_question_variants(topic: str, display_name: str, kind: str) -> tuple[str, ...]:
            if kind == "skill":
                templates = (
                    f"What was the hardest technical decision you made while using {display_name}, and how did you validate it?",
                    f"What tradeoff did you face when using {display_name}, and why did you choose that path?",
                    f"What failure case or debugging issue exposed the limits of your {display_name} approach, and how did you respond?",
                    f"If you had to optimize your {display_name} workflow today, what would you change first and why?",
                )
            elif kind == "project":
                templates = (
                    f"Walk me through {display_name}, focusing on the most important technical decision and its measurable impact.",
                    f"In {display_name}, what tradeoff mattered most between speed, quality, and reliability?",
                    f"In {display_name}, what failure mode or operational issue taught you the most about the system?",
                    f"If you revisited {display_name} now, what optimization would create the biggest engineering improvement?",
                )
            else:
                templates = (
                    f"You have not covered {display_name} yet. What specific technical decision best demonstrates your strength in that area?",
                    f"You have not covered {display_name} yet. What tradeoff or constraint most shaped your approach there?",
                    f"You have not covered {display_name} yet. What failure case, reliability issue, or blind spot did you have to work through?",
                    f"You have not covered {display_name} yet. If you had to optimize that area now, what would you improve first?",
                )

            start_index = (turn_number - 1) % len(templates)
            ordered = templates[start_index:] + templates[:start_index]
            return ordered

        if not history and not uncovered_topics:
            if projects:
                return choose(
                    f"Can you walk me through the architecture, design decisions, and measurable impact of your project {projects[0]['name']}?"
                )
            return choose("Can you describe your strongest skill and the project where you used it most deeply?")

        covered_text = " ".join(
            f"{entry.get('question', '')} {entry.get('candidate_response', '')}".lower() for entry in history
        )

        if needs_clarification and not uncovered_topics and last_depth < 0.2:
            primary = project_name(0)
            return choose(
                "Your last answer was still broad. Can you give one specific example with a concrete metric, decision, or ownership boundary?",
                f"Staying on {primary}, what exact metric moved, what changed in the system, and what part did you personally own?" if primary else "",
            )

        if uncovered_topics:
            prioritized_topics = self._prioritize_topics(uncovered_topics, projects, skills)
            selectable_topics = [topic for topic in prioritized_topics if asked_topic_counts.get(topic, 0) == 0]
            top_topic = selectable_topics[0] if selectable_topics else prioritized_topics[0]
            assert top_topic in uncovered_topics, "Selected topic must come from uncovered topics."
            if top_topic in {skill.lower() for skill in skills}:
                display_skill = next((skill for skill in skills if skill.lower() == top_topic), top_topic)
                return choose(*topic_question_variants(top_topic, display_skill, "skill"))
            if top_topic in {_normalize_topic(project['name']) for project in projects}:
                project_label = next(
                    (project["name"] for project in projects if _normalize_topic(project["name"]) == top_topic),
                    top_topic,
                )
                return choose(*topic_question_variants(top_topic, project_label, "project"))
            return choose(*topic_question_variants(top_topic, top_topic, "system"))

        if projects:
            primary_project = projects[0]["name"]
            secondary_project = project_name(1)
            secondary_question = (
                f"Following up on {primary_project}, what failure modes, monitoring signals, or scaling bottlenecks mattered most once the system was in production?"
            )
            redesign_question = (
                f"Following up on {primary_project}, what would you redesign today to improve robustness, scalability, or maintainability?"
            )
            comparison_question = (
                f"Across {primary_project} and {secondary_project}, where did your strongest engineering judgment show up, and what evidence convinced you that the decision was correct?"
                if secondary_project
                else ""
            )
            return choose(secondary_question, redesign_question, comparison_question)

        return choose(
            "Looking across your experience, what is the strongest example of a system you improved through careful technical reasoning and measurable validation?",
            "Across your background, where did your interview-worthy impact come most from: system design, experimentation, or operations discipline?",
            "What is the best example of a vague problem statement that you turned into a concrete technical plan and measurable outcome?"
        )

    def _asked_topic_counts(self, history: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in history:
            for topic in entry.get("matched_topics", []):
                normalized = _normalize_topic(topic)
                counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    def _derive_uncovered_topics(self, observation: Observation, covered_topics: set[str]) -> set[str]:
        profile = observation.candidate_profile
        topics = {_normalize_topic(skill) for skill in profile.get("skills", []) if skill.strip()}
        topics.update(_normalize_topic(project["name"]) for project in profile.get("projects", []))
        notes_blob = " ".join(
            [str(profile.get("notes", ""))]
            + [project.get("summary", "") for project in profile.get("projects", [])]
            + [project.get("impact", "") for project in profile.get("projects", [])]
        ).lower()
        system_topics = self._infer_system_topics(observation, notes_blob)
        topics.update(system_topics)
        return {topic for topic in topics if topic and topic not in covered_topics}

    def _prioritize_topics(self, uncovered_topics: set[str], projects: list[dict], skills: list[str]) -> list[str]:
        project_topics = { _normalize_topic(project["name"]) for project in projects }
        skill_topics = { skill.lower() for skill in skills }
        system_topics = uncovered_topics - project_topics - skill_topics
        ordered = sorted(skill_topics & uncovered_topics) + sorted(project_topics & uncovered_topics) + sorted(system_topics)
        return ordered

    def _infer_system_topics(self, observation: Observation, notes_blob: str) -> list[str]:
        inferred: list[str] = []
        profile = observation.candidate_profile
        skills = {_normalize_topic(skill) for skill in profile.get("skills", []) if skill.strip()}
        project_text = " ".join(project.get("summary", "") for project in profile.get("projects", []))
        combined = f"{notes_blob} {project_text}".lower()

        if "extract" in combined or "entities" in combined:
            inferred.append("entity extraction")
        if observation.difficulty == "easy":
            inferred.extend(["testing", "deployment"])
        if "mlops" in skills or observation.difficulty == "medium":
            inferred.extend(["retraining", "latency"])
        if observation.difficulty == "hard":
            inferred.extend(["metrics", "evaluation", "ownership", "productionization"])
        if observation.difficulty == "edge":
            inferred.extend(["rag", "experience conflict", "reliability"])

        return sorted({_normalize_topic(topic) for topic in inferred if _normalize_topic(topic)})
