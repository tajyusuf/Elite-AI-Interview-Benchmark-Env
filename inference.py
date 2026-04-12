from __future__ import annotations

import json
import os

from agent.baseline import BaselineInterviewAgent
from env.interview_env import EliteAIInterviewEvaluationEnv


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
MAX_STEPS = 32
EPSILON = 0.0001


def build_messages(observation) -> list[dict[str, str]]:
    history_lines = []
    for entry in observation.history[-4:]:
        history_lines.append(f"Question: {entry.get('question', '')}")
        history_lines.append(f"Response: {entry.get('candidate_response', '')}")

    history_text = "\n".join(history_lines) if history_lines else "No prior turns."

    return [
        {
            "role": "system",
            "content": (
                "You are an AI interviewer. "
                "Return exactly one concise interview question. "
                "Be deterministic, specific, and avoid repetition."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Difficulty: {observation.difficulty}\n"
                f"Style: {observation.interviewer_style}\n"
                f"Turn: {observation.current_turn + 1}\n"
                f"Candidate profile:\n{json.dumps(observation.candidate_profile, ensure_ascii=True)}\n"
                f"Recent history:\n{history_text}\n"
                "Ask the next best interview question as plain text only."
            ),
        },
    ]


def _build_client():
    try:
        from openai import OpenAI
    except Exception:
        return None

    base_url = API_BASE_URL
    api_key = os.getenv("HF_TOKEN")
    if not base_url or not api_key or not MODEL_NAME:
        return None

    try:
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    except Exception:
        return None


def _log_start(task_name: str, difficulty: str, style: str) -> None:
    print(f"[START] task={task_name} difficulty={difficulty} style={style}", flush=True)


def _log_step(task_name: str, step: int, reward: float) -> None:
    print(f"[STEP] task={task_name} step={step} reward={reward:.4f}", flush=True)


def _log_end(task_name: str, score: float, steps: int) -> None:
    print(f"[END] task={task_name} score={score:.4f} steps={steps}", flush=True)


def generate_question(observation, client, baseline_agent: BaselineInterviewAgent) -> str:
    if client is None:
        return baseline_agent.act(observation).question

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=build_messages(observation),
            temperature=0,
            max_tokens=80,
        )
        question = (completion.choices[0].message.content or "").strip()
        if not question:
            return baseline_agent.act(observation).question
        return question.splitlines()[0].strip()
    except Exception:
        return baseline_agent.act(observation).question


def run_task(task_index: int, client, baseline_agent: BaselineInterviewAgent) -> dict[str, object]:
    env = EliteAIInterviewEvaluationEnv(task_index=task_index)
    observation = env.reset()
    done = False
    steps = 0
    task_name = f"task_{task_index}"
    _log_start(task_name, observation.difficulty, observation.interviewer_style)

    while not done and steps < MAX_STEPS:
        question = generate_question(observation, client, baseline_agent)
        observation, reward, done, info = env.step({"action_type": "answer", "content": question})
        steps += 1
        _log_step(task_name, int(observation.current_turn), float(reward))

    state = env.state().model_dump()
    average_score = float(state.get("average_score", 0.0))
    bounded_score = max(EPSILON, min(1.0 - EPSILON, average_score))
    _log_end(task_name, bounded_score, int(state.get("turn", steps)))
    return {
        "difficulty": observation.difficulty,
        "score": bounded_score,
        "turns": int(state.get("turn", steps)),
    }


def safe_run_task(task_index: int, client, baseline_agent: BaselineInterviewAgent) -> dict[str, object]:
    try:
        return run_task(task_index, client, baseline_agent)
    except Exception:
        env = EliteAIInterviewEvaluationEnv(task_index=task_index)
        observation = env.reset()
        state = env.state().model_dump()
        average_score = float(state.get("average_score", 0.0))
        bounded_score = max(EPSILON, min(1.0 - EPSILON, average_score))
        task_name = f"task_{task_index}"
        _log_start(task_name, observation.difficulty, observation.interviewer_style)
        _log_end(task_name, bounded_score, int(state.get("turn", 0)))
        return {
            "difficulty": observation.difficulty,
            "score": bounded_score,
            "turns": int(state.get("turn", 0)),
        }


def main() -> None:
    client = _build_client()
    baseline_agent = BaselineInterviewAgent()
    results = [safe_run_task(task_index, client, baseline_agent) for task_index in range(3)]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
