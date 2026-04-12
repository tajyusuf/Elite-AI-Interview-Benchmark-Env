from __future__ import annotations

from copy import deepcopy
from typing import Any


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "easy_resume_parser",
        "difficulty": "easy",
        "max_steps": 5,
        "objective": "Explain the core design and evaluation approach for a resume parser API.",
        "prompt": (
            "You built a Resume Parser API that extracts entities (name, email, skills, experience) "
            "for internal recruiters. Provide a concise explanation of how it works and how you "
            "validated its accuracy."
        ),
        "required_facts": [
            "entity extraction",
            "precision/recall",
            "evaluation dataset",
        ],
        "rubric_keywords": [
            "pipeline",
            "tokenization",
            "regex",
            "model",
            "evaluation",
            "precision",
            "recall",
            "f1",
            "validation",
        ],
        "forbidden_claims": [
            "real-time distributed training",
            "self-driving",
        ],
        "hints": [
            "Mention the extraction pipeline and at least one evaluation metric.",
            "Include how you validated the model (dataset or test set).",
        ],
    },
    {
        "task_id": "medium_forecasting",
        "difficulty": "medium",
        "max_steps": 6,
        "objective": "Describe a demand-forecasting system with tradeoffs and monitoring.",
        "prompt": (
            "You implemented a demand-forecasting pipeline for retail. Explain the model choice, "
            "the retraining strategy, and the tradeoffs you made in deployment."
        ),
        "required_facts": [
            "retraining schedule",
            "monitoring/alerting",
            "tradeoff",
        ],
        "rubric_keywords": [
            "baseline",
            "features",
            "drift",
            "monitoring",
            "latency",
            "retraining",
            "tradeoff",
            "metrics",
        ],
        "forbidden_claims": [
            "guaranteed perfect forecasts",
            "zero latency at infinite scale",
        ],
        "hints": [
            "Describe how you detect drift or degradation.",
            "Explain at least one tradeoff (accuracy vs latency, cost vs freshness).",
        ],
    },
    {
        "task_id": "hard_assistant_platform",
        "difficulty": "hard",
        "max_steps": 7,
        "objective": "Explain a production assistant platform with constraints and failure handling.",
        "prompt": (
            "You helped build an enterprise assistant platform. Explain the architecture, the "
            "constraints you faced at scale, and one failure mode you had to mitigate."
        ),
        "required_facts": [
            "scalability constraint",
            "failure mode",
            "mitigation strategy",
        ],
        "rubric_keywords": [
            "architecture",
            "queue",
            "latency",
            "throughput",
            "rollback",
            "circuit breaker",
            "fallback",
            "monitoring",
        ],
        "forbidden_claims": [
            "zero failures in production",
            "infinite throughput",
        ],
        "hints": [
            "Include at least one failure mode and how you mitigated it.",
            "Mention a scaling constraint or bottleneck you had to address.",
        ],
    },
]


def get_task_by_index(index: int) -> dict[str, Any]:
    return deepcopy(TASKS[index % len(TASKS)])


def list_tasks() -> list[dict[str, Any]]:
    return [deepcopy(task) for task in TASKS]


def get_hint(task: dict[str, Any], step_count: int) -> str:
    hints = task.get("hints", [])
    if not hints:
        return "Focus on concrete details, metrics, and constraints."
    return hints[(step_count - 1) % len(hints)]
