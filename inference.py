from __future__ import annotations

import json
import os

from openai import OpenAI

from env.interview_env import EliteAIInterviewEvaluationEnv


client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN"),
)

MODEL_NAME = os.getenv("MODEL_NAME")
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


def generate_question(observation) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=build_messages(observation),
        temperature=0,
        max_tokens=80,
    )
    question = (completion.choices[0].message.content or "").strip()
    if not question:
        question = "What is the most important technical decision in your work, and how did you validate it?"
    return question.splitlines()[0].strip()


def run_task(task_index: int) -> dict[str, object]:
    env = EliteAIInterviewEvaluationEnv(task_index=task_index, style_index=task_index)
    observation = env.reset()
    done = False
    last_info: dict[str, object] = {}
    steps = 0

    while not done and steps < MAX_STEPS:
        question = generate_question(observation)
        observation, reward, done, info = env.step({"question": question})
        last_info = info
        steps += 1

    state = env.state().model_dump()
    return {
        "difficulty": observation.difficulty,
        "score": float(state["cumulative_score"]),
        "turns": int(state["turn"]),
    }


def main() -> None:
    results = [run_task(task_index) for task_index in range(3)]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
