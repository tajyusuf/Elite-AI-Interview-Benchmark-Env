from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.interview_env import EliteAIInterviewEvaluationEnv
from env.models import Action


app = FastAPI(title="Elite AI Interview Evaluation Environment")
env = EliteAIInterviewEvaluationEnv()


class StepRequest(BaseModel):
    action: str


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> dict[str, Any]:
    observation = env.reset()
    return {"observation": _to_jsonable(observation)}


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    question = request.action.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Action question must be a non-empty string.")

    observation, reward, done, info = env.step(Action(question=question))
    return {
        "observation": _to_jsonable(observation),
        "reward": float(reward),
        "done": bool(done),
        "info": _to_jsonable(info),
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return _to_jsonable(env.state())


@app.post("/state")
def state_post() -> dict[str, Any]:
    return _to_jsonable(env.state())


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()

