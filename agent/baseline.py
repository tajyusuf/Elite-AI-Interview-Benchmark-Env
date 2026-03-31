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


def _display_topic(topic: str) -> str:
    return _normalize_topic(topic).replace("_", " ")


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
        last_question = str(history[-1].get("question", "")).lower() if history else ""
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

        def previous_topic() -> str | None:
            if not history:
                return None
            matched = history[-1].get("matched_topics", [])
            return matched[0] if matched else None

        def previous_question_type() -> str | None:
            if "metric" in last_question or "signal" in last_question:
                return "metric"
            if "failure" in last_question or "root cause" in last_question or "debug" in last_question or "unexpectedly" in last_question:
                return "failure"
            if "optimiz" in last_question or "bottleneck" in last_question or "headroom" in last_question or "latency" in last_question:
                return "optimization"
            if "reject" in last_question or "alternative" in last_question or "drawback" in last_question or "avoid" in last_question:
                return "tradeoff"
            if "architecture" in last_question or "design" in last_question or "constraint" in last_question:
                return "architecture"
            return None

        def recent_question_signatures() -> set[str]:
            signatures: set[str] = set()
            for entry in history[-3:]:
                normalized_question = _normalize_topic(str(entry.get("question", "")))
                if not normalized_question:
                    continue
                signature = " ".join(normalized_question.split()[:5])
                signatures.add(signature)
            return signatures

        def transition_prefix(current_topic: str) -> str:
            prev_topic = previous_topic()
            if not prev_topic:
                return ""
            if _normalize_topic(prev_topic) == _normalize_topic(current_topic):
                same_topic_prefixes = (
                    "In that system, ",
                    "Given that design, ",
                    "From your previous answer, ",
                )
                return same_topic_prefixes[(turn_number - 1) % len(same_topic_prefixes)]
            transition_prefixes = (
                f"Earlier you mentioned {prev_topic}; ",
                f"In that system tied to {prev_topic}, ",
                f"Given that design around {prev_topic}, ",
                f"From your previous answer about {prev_topic}, ",
            )
            return transition_prefixes[(turn_number - 1) % len(transition_prefixes)]

        def ordered_question_types(kind: str, topic_seen_count: int) -> list[str]:
            by_difficulty = {
                "easy": ["architecture", "metric", "optimization", "failure"],
                "medium": ["architecture", "tradeoff", "metric", "failure", "optimization"],
                "hard": ["architecture", "metric", "tradeoff", "failure", "optimization"],
                "edge": ["failure", "metric", "architecture", "tradeoff", "optimization"],
            }
            if topic_seen_count > 0:
                base = ["failure", "metric", "optimization", "tradeoff", "architecture"]
            else:
                base = by_difficulty.get(observation.difficulty, by_difficulty["medium"])

            if kind == "system" and "architecture" in base:
                base = [question_type for question_type in base if question_type != "architecture"] + ["architecture"]

            last_type = previous_question_type()
            if last_type in base:
                base = [question_type for question_type in base if question_type != last_type] + [last_type]
            start_index = (turn_number - 1) % len(base)
            ordered = base[start_index:] + base[:start_index]
            if last_type and len(ordered) > 1 and ordered[0] == last_type:
                ordered = ordered[1:] + ordered[:1]
            return ordered

        def render_question(question_type: str, topic: str, display_name: str, kind: str, topic_seen_count: int) -> str:
            prefix = transition_prefix(topic)
            clean_name = _display_topic(display_name)
            hard_suffix = ""
            if observation.difficulty == "hard":
                hard_suffixes = {
                    "architecture": " under distributed service coordination",
                    "tradeoff": " once production scale and coordination overhead were visible",
                    "failure": " when the system faced scaling pressure in production",
                    "metric": " while production bottlenecks and scale constraints were visible",
                    "optimization": " under production bottlenecks or scaling pressure",
                }
                hard_suffix = hard_suffixes[question_type]
            if kind == "project":
                stems = {
                    "architecture": f"{prefix}which design constraint inside {clean_name}{hard_suffix} forced the most important architectural decision, and how did you work around it?",
                    "tradeoff": f"{prefix}which alternative architecture for {clean_name}{hard_suffix} did you reject, and which drawback or operational risk made that path unacceptable?",
                    "failure": f"{prefix}which failure pattern inside {clean_name}{hard_suffix} exposed the root cause fastest, and what system constraint shaped the fix?",
                    "metric": f"{prefix}which metric, operating signal, or production indicator for {clean_name}{hard_suffix} most influenced your decision, and why did you trust it over the competing signals?",
                    "optimization": f"{prefix}which scaling limit, latency hotspot, or throughput constraint inside {clean_name}{hard_suffix} created the most pressure, and what concrete system change relieved it?",
                }
            elif kind == "skill":
                stems = {
                    "architecture": f"{prefix}which design constraint most influenced how you used {clean_name}{hard_suffix}, and what architecture or implementation decision followed from it?",
                    "tradeoff": f"{prefix}which {clean_name} strategy{hard_suffix} did you deliberately avoid, and which technical drawback or system constraint ruled it out?",
                    "failure": f"{prefix}when your {clean_name} solution hit a failure mode{hard_suffix}, which signal exposed the root cause and what constraint shaped the recovery?",
                    "metric": f"{prefix}which metric, operating signal, or quality indicator guided your key {clean_name} decision{hard_suffix}, and why was it more trustworthy than the alternatives?",
                    "optimization": f"{prefix}which operating constraint, scaling issue, or latency tradeoff limited your {clean_name} approach{hard_suffix}, and what concrete change delivered the largest gain?",
                }
            else:
                stems = {
                    "architecture": f"{prefix}which system-level constraint most shaped your work in {clean_name}{hard_suffix}, and how did you design around that limitation?",
                    "tradeoff": f"{prefix}which alternative approach in {clean_name}{hard_suffix} did you reject, and which specific drawback or operational constraint pushed you away from it?",
                    "failure": f"{prefix}which production risk or failure pattern in {clean_name}{hard_suffix} exposed the root cause most clearly, and what constrained the fix?",
                    "metric": f"{prefix}which metric, operational signal, or reliability indicator in {clean_name}{hard_suffix} most influenced your decision, and why did it matter more than the others?",
                    "optimization": f"{prefix}which scaling limit, bottleneck, or latency tradeoff in {clean_name}{hard_suffix} created the most pressure, and what concrete change relieved it?",
                }
            return stems[question_type][0].upper() + stems[question_type][1:]

        def topic_question_variants(topic: str, display_name: str, kind: str) -> tuple[str, ...]:
            topic_seen_count = asked_topic_counts.get(_normalize_topic(topic), 0)
            banned_fragments = (
                "which architectural bottleneck",
                "what broke first",
                "what did you do",
            )
            recent_signatures = recent_question_signatures()
            variants = [
                render_question(question_type, topic, display_name, kind, topic_seen_count)
                for question_type in ordered_question_types(kind, topic_seen_count)
            ]
            filtered: list[str] = []
            for question in variants:
                lowered = question.lower()
                if any(fragment in lowered for fragment in banned_fragments):
                    continue
                signature = " ".join(_normalize_topic(question).split()[:5])
                if signature in recent_signatures:
                    continue
                filtered.append(question)
            if filtered:
                return tuple(filtered)
            return tuple(
                question for question in variants
                if not any(fragment in question.lower() for fragment in banned_fragments)
            )

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
