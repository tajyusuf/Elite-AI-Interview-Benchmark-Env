from __future__ import annotations

import os
import sys
from typing import Any

VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from pydantic import BaseModel, Field


class Observation(BaseModel):
    candidate_profile: dict[str, Any]
    current_turn: int
    history: list[dict[str, Any]] = Field(default_factory=list)
    interviewer_style: str
    difficulty: str


class Action(BaseModel):
    question: str


class State(BaseModel):
    turn: int
    done: bool
    difficulty: str = ""
    cumulative_score: float
    covered_topics: list[str] = Field(default_factory=list)
    uncovered_topics: list[str] = Field(default_factory=list)
    average_score: float = 0.0
    normalized_turn_scores: list[float] = Field(default_factory=list)
    step_count: int = 0
