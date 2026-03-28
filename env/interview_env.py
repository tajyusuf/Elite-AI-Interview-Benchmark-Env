from __future__ import annotations

from typing import Any

from env.grader import grade_question
from env.models import Action, Observation, State
from env.tasks import (
    generate_candidate_response,
    get_task_by_index,
    list_tasks,
    normalize_topic_collection,
    normalize_topic_name,
)


class EliteAIInterviewEvaluationEnv:
    """Deterministic multi-turn interview evaluation environment."""

    STYLES = ["friendly", "strict", "technical", "aggressive"]
    SUCCESS_THRESHOLD = 0.75
    MIN_TURNS = 2
    ALLOWED_UNCOVERED_THRESHOLD = 2

    def __init__(self, task_index: int = 0, style_index: int = 0) -> None:
        self._initial_task_index = task_index
        self._initial_style_index = style_index
        self._episode_index = 0
        self._task: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._covered_topics: set[str] = set()
        self._uncovered_topics: set[str] = set()
        self._state = State(turn=0, step_count=0, done=False, cumulative_score=0.0)
        self._current_style = self.STYLES[style_index % len(self.STYLES)]

    def reset(self) -> Observation:
        task_index = self._initial_task_index + self._episode_index
        style_index = self._initial_style_index + self._episode_index
        self._task = get_task_by_index(task_index)
        self._current_style = self.STYLES[style_index % len(self.STYLES)]
        self._history = []
        self._covered_topics = set()
        self._uncovered_topics = self._derive_expected_topics(self._task)
        self._state = State(
            turn=0,
            step_count=0,
            done=False,
            difficulty=self._task["difficulty"],
            cumulative_score=0.0,
            covered_topics=[],
            uncovered_topics=sorted(self._uncovered_topics),
        )
        self._episode_index += 1
        return self._observation()

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._task is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._state.done:
            raise RuntimeError("Episode is complete. Call reset() to start a new episode.")

        parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
        question = parsed_action.question.strip()
        grade = grade_question(
            question=question,
            history=self._history,
            covered_topics=self._covered_topics,
            task=self._task,
            interviewer_style=self._current_style,
        )

        matched_topics = set(normalize_topic_collection(grade["matched_topics"]))
        new_topics = set(normalize_topic_collection(grade["new_topics"]))
        candidate_response = generate_candidate_response(
            self._task,
            question,
            self._covered_topics | matched_topics,
            self._history,
        )

        prior_uncovered_topics = set(self._uncovered_topics)
        self._covered_topics.update(new_topics)
        self._uncovered_topics = self._derive_expected_topics(self._task) - self._covered_topics
        adjusted_reward, reward_notes = self._apply_reward_adjustments(
            grade["reward"],
            new_topics,
            prior_uncovered_topics,
        )

        self._state.step_count += 1
        self._state.turn = self._state.step_count
        self._state.cumulative_score = round(self._state.cumulative_score + adjusted_reward, 4)
        self._state.normalized_turn_scores.append(adjusted_reward)
        self._state.average_score = round(self._state.cumulative_score / self._state.step_count, 4)
        self._state.covered_topics = sorted(self._covered_topics)
        self._state.uncovered_topics = sorted(self._uncovered_topics)
        self._state.difficulty = self._task["difficulty"]
        coverage_ready = len(self._uncovered_topics) <= self.ALLOWED_UNCOVERED_THRESHOLD
        all_key_topics_covered = not self._uncovered_topics
        success_condition = all_key_topics_covered or (
            adjusted_reward >= self.SUCCESS_THRESHOLD and coverage_ready
        )
        self._state.done = (
            all_key_topics_covered
            or (self._state.step_count >= self.MIN_TURNS and success_condition and all_key_topics_covered)
        )

        history_entry = {
            "turn": self._state.step_count,
            "question": question,
            "candidate_response": candidate_response,
            "reward": adjusted_reward,
            "matched_topics": sorted(matched_topics),
            "new_topics": sorted(new_topics),
            "breakdown": grade["breakdown"],
            "normalized_question": grade["normalized_question"],
            "reasoning": self._merge_reasoning(
                grade["reasoning"],
                reward_notes,
                success_condition,
                coverage_ready,
            ),
        }
        self._history.append(history_entry)

        info = {
            "task_id": self._task["task_id"],
            "difficulty": self._task["difficulty"],
            "interviewer_style": self._current_style,
            "max_turns": self._task["max_turns"],
            "success_threshold": self.SUCCESS_THRESHOLD,
            "min_turns": self.MIN_TURNS,
            "allowed_uncovered_threshold": self.ALLOWED_UNCOVERED_THRESHOLD,
            "reward": adjusted_reward,
            "relevance": grade["breakdown"]["relevance_to_profile"],
            "depth": grade["breakdown"]["depth_of_question"],
            "sequence": grade["breakdown"]["logical_sequence"],
            "coverage": grade["breakdown"]["coverage_of_topics"],
            "clarity": grade["breakdown"]["clarity_of_question"],
            "penalties": grade["breakdown"]["penalties"],
            "score_breakdown": grade["breakdown"],
            "matched_topics": sorted(matched_topics),
            "new_topics": sorted(new_topics),
            "covered_topics": sorted(self._covered_topics),
            "uncovered_topics": sorted(self._uncovered_topics),
            "candidate_response": candidate_response,
            "hidden_expectations": self._task["hidden_evaluation_expectations"],
            "reasoning": self._merge_reasoning(
                grade["reasoning"],
                reward_notes,
                success_condition,
                coverage_ready,
            ),
            "relevance_reasons": grade["relevance_reasons"],
            "average_score": self._state.average_score,
            "normalized_turn_scores": list(self._state.normalized_turn_scores),
            "state": self.state().model_dump(),
        }
        return self._observation(), adjusted_reward, self._state.done, info

    def state(self) -> State:
        return self._state.model_copy(deep=True)

    def available_tasks(self) -> list[dict[str, Any]]:
        return list_tasks()

    def _observation(self) -> Observation:
        if self._task is None:
            raise RuntimeError("Environment is not initialized.")
        return Observation(
            candidate_profile=self._task["candidate_profile"],
            current_turn=self._state.step_count,
            history=[{key: value for key, value in item.items() if key != "normalized_question"} for item in self._history],
            interviewer_style=self._current_style,
            difficulty=self._task["difficulty"],
        )

    def _derive_expected_topics(self, task: dict[str, Any]) -> set[str]:
        topics = set(normalize_topic_collection(task["hidden_evaluation_expectations"].get("important_topics", [])))
        profile = task["candidate_profile"]
        topics.update(normalize_topic_collection([skill for skill in profile.get("skills", []) if skill.strip()]))
        topics.update(normalize_topic_collection([project["name"] for project in profile.get("projects", [])]))
        return {normalize_topic_name(topic) for topic in topics if normalize_topic_name(topic)}

    def _apply_reward_adjustments(
        self,
        reward: float,
        new_topics: set[str],
        prior_uncovered_topics: set[str],
    ) -> tuple[float, list[str]]:
        if self._task is None:
            return reward, []

        notes: list[str] = []
        adjusted = reward
        difficulty_bonus = {
            "easy": 0.0,
            "medium": 0.015,
            "hard": 0.04,
            "edge": 0.045,
        }.get(self._task["difficulty"], 0.0)

        important_new_topics = prior_uncovered_topics & new_topics

        if new_topics:
            coverage_bonus = min(0.07, 0.025 * len(new_topics)) + difficulty_bonus
            if important_new_topics:
                coverage_bonus = min(0.09, coverage_bonus + 0.015 * len(important_new_topics))
            adjusted += coverage_bonus
            notes.append(f"coverage bonus applied for new topics: {', '.join(sorted(new_topics))}")

        no_progress_streak = 0
        for item in reversed(self._history[-2:]):
            if item.get("new_topics"):
                break
            no_progress_streak += 1
        if not new_topics and no_progress_streak >= 2:
            adjusted -= 0.22
            notes.append("heavy no-progress penalty applied after two consecutive turns without new topic coverage")

        if not self._uncovered_topics:
            adjusted += 0.3
            notes.append("full coverage achieved; terminal completion bonus applied")
            return max(0.0, min(1.0, round(adjusted, 4))), notes

        if self._state.step_count >= 2 and self._uncovered_topics:
            missing_penalty = min(0.1, 0.02 * len(self._uncovered_topics))
            adjusted -= missing_penalty
            notes.append(f"uncovered-topic penalty applied for remaining topics: {', '.join(sorted(self._uncovered_topics)[:3])}")

        return max(0.0, min(1.0, round(adjusted, 4))), notes

    def _merge_reasoning(
        self,
        base_reasoning: str,
        reward_notes: list[str],
        success_condition: bool,
        coverage_ready: bool,
    ) -> str:
        notes = list(reward_notes)
        if not coverage_ready:
            notes.append("early completion blocked because too many important topics remain uncovered")
        elif not success_condition and self._state.step_count < self.MIN_TURNS:
            notes.append("early completion blocked until minimum interview length is reached")

        if not notes:
            return base_reasoning
        return f"{base_reasoning} Adjustments: {'; '.join(notes)}."
