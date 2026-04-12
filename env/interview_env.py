from __future__ import annotations

from typing import Any

from env.grader import grade_answer
from env.models import Action, Observation, State
from env.tasks import get_hint, get_task_by_index, list_tasks


class EliteAIInterviewEvaluationEnv:
    """Deterministic multi-step evaluation environment."""

    def __init__(self, task_index: int = 0) -> None:
        self._initial_task_index = task_index
        self._episode_index = 0
        self._task: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._state = State(step_count=0, done=False, cumulative_score=0.0)
        self._draft_answer = ""

    def reset(self) -> Observation:
        task_index = self._initial_task_index + self._episode_index
        self._task = get_task_by_index(task_index)
        self._history = []
        self._draft_answer = ""
        self._state = State(
            step_count=0,
            done=False,
            difficulty=self._task["difficulty"],
            cumulative_score=0.0,
            progress=0.0,
            history=[],
        )
        self._episode_index += 1
        return self._observation(last_feedback="")

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        try:
            if self._task is None:
                raise RuntimeError("Environment must be reset before calling step().")
            if self._state.done:
                raise RuntimeError("Episode is complete. Call reset() to start a new episode.")

            parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
            action_type = (parsed_action.action_type or "answer").strip().lower()
            content = parsed_action.content or parsed_action.question or ""
            if content and self._history:
                last_content = str(self._history[-1].get("content", "")).strip().lower()
                if content.strip().lower() == last_content:
                    content = f"{content} (clarified)"

            self._state.step_count += 1
            max_steps = int(self._task.get("max_steps", 5))
            reward = 0.0
            info: dict[str, Any] = {}
            last_feedback = ""

            if action_type == "request_hint":
                hint = get_hint(self._task, self._state.step_count)
                reward = 0.0
                last_feedback = hint
                info = self._format_info(
                    score=0.0,
                    breakdown={
                        "relevance": 0.0,
                        "correctness": 0.0,
                        "depth": 0.0,
                        "clarity": 0.0,
                    },
                    reason="Hint requested",
                    penalties={"hint": 0.0},
                    covered_facts=[],
                )
            elif action_type in {"answer", "refine"}:
                if action_type == "refine" and self._draft_answer:
                    self._draft_answer = f"{self._draft_answer} {content}".strip()
                else:
                    self._draft_answer = content.strip()

                graded = grade_answer(self._draft_answer, self._task, history=list(self._history))
                reward = float(graded["score"])
                progress = self._progress_from_coverage(graded.get("covered_facts", []))
                self._state.progress = progress
                last_feedback = self._follow_up(graded["score"], graded["reason"])
                info = self._format_info(
                    score=reward,
                    breakdown=graded["breakdown"],
                    reason=graded["reason"],
                    penalties=graded["penalties"],
                    covered_facts=graded.get("covered_facts", []),
                )
            elif action_type == "submit_final":
                final_answer = content.strip() or self._draft_answer
                graded = grade_answer(final_answer, self._task, history=list(self._history))
                reward = float(graded["score"])
                progress = self._progress_from_coverage(graded.get("covered_facts", []))
                self._state.progress = progress
                last_feedback = self._follow_up(graded["score"], graded["reason"])
                info = self._format_info(
                    score=reward,
                    breakdown=graded["breakdown"],
                    reason=graded["reason"],
                    penalties=graded["penalties"],
                    covered_facts=graded.get("covered_facts", []),
                )
                self._state.done = True
            else:
                reward = 0.0
                last_feedback = "Invalid input"
                info = self._format_info(
                    score=0.0,
                    breakdown={
                        "relevance": 0.0,
                        "correctness": 0.0,
                        "depth": 0.0,
                        "clarity": 0.0,
                    },
                    reason="Invalid input",
                    penalties={"invalid_action": 0.4},
                    covered_facts=[],
                )

            self._state.cumulative_score = round(self._state.cumulative_score + reward, 4)
            self._state.history = list(self._history) + [
                {
                    "step": self._state.step_count,
                    "action_type": action_type,
                    "content": content,
                    "reward": reward,
                    "feedback": last_feedback,
                    "progress": self._state.progress,
                }
            ]
            self._history = list(self._state.history)

            if self._state.step_count >= max_steps:
                self._state.done = True

            return self._observation(last_feedback=last_feedback), reward, self._state.done, info
        except Exception:
            fallback = self._format_info(
                score=0.0,
                breakdown={
                    "relevance": 0.0,
                    "correctness": 0.0,
                    "depth": 0.0,
                    "clarity": 0.0,
                },
                reason="Invalid input",
                penalties={"error": 0.4},
                covered_facts=[],
            )
            self._state.done = True
            return self._observation(last_feedback="Invalid input"), 0.0, True, fallback

    def state(self) -> State:
        return self._state.model_copy(deep=True)

    def available_tasks(self) -> list[dict[str, Any]]:
        return list_tasks()

    def _progress_from_coverage(self, covered_facts: list[str]) -> float:
        required = self._task.get("required_facts", []) if self._task else []
        if not required:
            return 0.0
        return round(min(1.0, len(set(covered_facts)) / len(required)), 4)

    def _observation(self, last_feedback: str) -> Observation:
        if self._task is None:
            raise RuntimeError("Environment is not initialized.")
        return Observation(
            candidate_profile={
                "summary": self._task["prompt"],
                "objective": self._task["objective"],
            },
            current_turn=self._state.step_count,
            current_task={
                "task_id": self._task["task_id"],
                "difficulty": self._task["difficulty"],
                "objective": self._task["objective"],
                "prompt": self._task["prompt"],
                "max_steps": self._task["max_steps"],
            },
            step_count=self._state.step_count,
            progress=self._state.progress,
            history=list(self._history),
            last_feedback=last_feedback,
            interviewer_style="evaluator",
            difficulty=self._task["difficulty"],
        )

    def _format_info(
        self,
        score: float,
        breakdown: dict[str, float],
        reason: str,
        penalties: dict[str, float],
        covered_facts: list[str],
    ) -> dict[str, Any]:
        return {
            "score": score,
            "breakdown": breakdown,
            "reason": reason,
            "penalties": penalties,
            "covered_facts": covered_facts,
        }

    def _follow_up(self, score: float, reason: str) -> str:
        if score >= 0.75:
            return f"{reason} Provide a deeper system-level detail or constraint."
        if score <= 0.35:
            return f"{reason} Add a simple concrete example or metric."
        return f"{reason} Please elaborate with one more specific detail."
