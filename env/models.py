from __future__ import annotations

import os
import sys
from typing import Any

VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from pydantic import BaseModel, Field


class Observation(BaseModel):
    candidate_profile: dict[str, Any] = Field(default_factory=dict)
    current_turn: int = 0
    current_task: dict[str, Any]
    step_count: int
    progress: float = 0.0
    history: list[dict[str, Any]] = Field(default_factory=list)
    last_feedback: str = ""
    interviewer_style: str = ""
    difficulty: str


class Action(BaseModel):
    action_type: str = "answer"
    content: str | None = None
    question: str | None = None


class State(BaseModel):
    step_count: int
    done: bool
    difficulty: str = ""
    cumulative_score: float
    progress: float = 0.0
    history: list[dict[str, Any]] = Field(default_factory=list)
