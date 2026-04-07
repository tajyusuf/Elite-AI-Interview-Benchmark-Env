from __future__ import annotations

import json
import os

from agent.baseline import BaselineInterviewAgent
from env.interview_env import EliteAIInterviewEvaluationEnv


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
MAX_STEPS = 32


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


def _log_event(stage: str, payload: dict[str, object]) -> None:
    print(stage)
    print(json.dumps(payload, ensure_ascii=True))


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
    env = EliteAIInterviewEvaluationEnv(task_index=task_index, style_index=task_index)
    observation = env.reset()
    done = False
    steps = 0

    _log_event(
        "STEP",
        {
            "event": "task_started",
            "task_index": task_index,
            "difficulty": observation.difficulty,
            "style": observation.interviewer_style,
        },
    )

    while not done and steps < MAX_STEPS:
        question = generate_question(observation, client, baseline_agent)
        observation, reward, done, info = env.step({"question": question})
        steps += 1
        _log_event(
            "STEP",
            {
                "event": "turn_completed",
                "task_index": task_index,
                "difficulty": observation.difficulty,
                "turn": int(observation.current_turn),
                "reward": float(reward),
                "done": bool(done),
                "covered_topics": list(info.get("covered_topics", [])),
                "uncovered_topics": list(info.get("uncovered_topics", [])),
            },
        )

    state = env.state().model_dump()
    return {
        "difficulty": observation.difficulty,
        "score": float(state["cumulative_score"]),
        "turns": int(state["turn"]),
    }


def safe_run_task(task_index: int, client, baseline_agent: BaselineInterviewAgent) -> dict[str, object]:
    try:
        return run_task(task_index, client, baseline_agent)
    except Exception:
        env = EliteAIInterviewEvaluationEnv(task_index=task_index, style_index=task_index)
        observation = env.reset()
        state = env.state().model_dump()
        return {
            "difficulty": observation.difficulty,
            "score": float(state["cumulative_score"]),
            "turns": int(state["turn"]),
        }


def main() -> None:
    client = _build_client()
    baseline_agent = BaselineInterviewAgent()
    _log_event(
        "START",
        {
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "hf_token_present": bool(os.getenv("HF_TOKEN")),
            "using_openai_client": client is not None,
        },
    )
    results = [safe_run_task(task_index, client, baseline_agent) for task_index in range(3)]
    _log_event("END", {"results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
