---
title: AI Interview Eval Env
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Elite AI Interview Evaluation Environment

## Overview

This project is a production-grade, deterministic environment for benchmarking AI interviewer agents. It simulates realistic hiring interviews where the agent must ask relevant, deep, and well-sequenced questions against candidate profiles of varying quality.

The system is designed for evaluation, not model training. It follows a strict OpenEnv-style interaction model with `reset()`, `step(action)`, and `state()` and returns auditable reward breakdowns without any LLM dependency.

## Real-World Relevance

Modern hiring systems are increasingly using AI interviewer agents for screening and structured interviews. In practice, useful interviewers must:

- Ask questions tied to the candidate's actual profile
- Dig into metrics, ownership, and technical tradeoffs
- Follow up when answers or profiles are vague
- Avoid repetition and irrelevant topic jumps

This environment turns those expectations into a deterministic benchmark.

It is directly relevant to:

- HR tech vendors benchmarking AI interviewers before customer deployment
- Applied AI teams comparing prompting, planning, or agent stacks
- Evaluation researchers who need a controlled multi-turn environment instead of one-shot judging

## Why This Environment Is Unique

Most interview benchmarks only check whether an agent asks a vaguely reasonable question. This environment instead evaluates whether the interviewer can manage a realistic interview loop:

- Start from imperfect profile data
- Ask progressively better questions over multiple turns
- Detect and pursue ambiguity
- Balance coverage with continuity
- Produce high signal without repeating itself

Because the grader is deterministic and explanation-first, it can evaluate any interviewing agent, whether that agent is rule-based, tool-using, prompted, fine-tuned, or part of a larger orchestration stack.

## Project Structure

```text
env/
  __init__.py
  grader.py
  interview_env.py
  models.py
  tasks.py
agent/
  __init__.py
  baseline.py
app.py
openenv.yaml
Dockerfile
requirements.txt
README.md
```

## OpenEnv Interface

The environment implements:

- `reset() -> Observation`
- `step(action) -> (Observation, reward: float, done: bool, info: dict)`
- `state() -> State`

### Observation Space

The `Observation` Pydantic model contains:

- `candidate_profile: dict`
- `current_turn: int`
- `history: list[dict]`
- `interviewer_style: str`
- `difficulty: str`

### Action Space

The `Action` Pydantic model contains:

- `question: str`

### State Space

The `State` Pydantic model contains:

- `step_count: int`
- `done: bool`
- `cumulative_score: float`

## Task Design

Each scenario includes a visible `candidate_profile` and hidden evaluation expectations.

### Easy

- Single-skill candidate
- One clear Python project
- Best for validating relevance, clarity, and project-depth basics

### Medium

- Multi-skill candidate
- Two ML and platform-oriented projects
- Tests sequencing, cross-project comparison, and coverage

### Hard

- Vague project summaries
- Mixed research and production claims
- Missing metrics and unclear ownership
- Requires follow-up questions to achieve high reward

### Edge Case

- Typos and missing skill clarity
- Conflicting experience values
- Noisy project descriptions
- Tests clarification behavior and robustness to profile quality issues

## Multi-Turn Interview Flow

Each episode runs for 4 or 5 turns depending on the task.

At each turn:

1. The agent asks a question.
2. The environment grades the question deterministically.
3. The environment generates a deterministic candidate response.
4. The turn is appended to history.

Future rewards depend on prior turns, topic coverage, and whether the question logically follows from what has already been explored.

The candidate simulator also changes its response style by difficulty and conversation state. Easier profiles produce stronger direct answers, while hard and edge tasks deliberately remain vague or partially contradictory until the interviewer asks sharper follow-ups.

## Reward Design

The final reward is clamped between `0.0` and `1.0`.

### Positive Components

- `relevance_to_profile` in `[0.0, 0.3]`
- `depth_of_question` in `[0.0, 0.3]`
- `logical_sequence` in `[0.0, 0.2]`
- `coverage_of_topics` in `[0.0, 0.1]`
- `clarity_of_question` in `[0.0, 0.1]`

### Penalties

- `repetition` up to `-0.3`
- `irrelevant_question` up to `-0.3`
- `shallow_question` up to `-0.2`

### Style Modifiers

- `friendly`: boosts clarity and softens repetition penalties
- `strict`: makes repetition, irrelevance, and shallow questions cost more
- `technical`: increases the value of deeper questioning
- `aggressive`: is harsher overall and slightly reduces clarity gain

Style assignment is deterministic by episode index so runs remain reproducible.

### Evaluation Design

The environment scores interviewer quality rather than candidate quality. The grader combines:

- Weighted relevance to skills, projects, role, and experience
- Technical depth cues such as metrics, tradeoffs, architecture, and validation
- Conversation intelligence, including follow-up quality and avoidance of context jumps
- Topic coverage with diminishing returns for repeatedly probing the same area
- Clarity and actionability of the question itself

It also exposes a short reasoning string and penalty breakdown in `info`, making it suitable for offline analysis, regression testing, and leaderboard-style evaluation.

## Coverage Tracking

The grader tracks:

- Previously asked questions
- Matched profile topics
- Newly covered topics
- Logical continuity with earlier turns
- Average score and normalized per-turn scores

Agents are rewarded for exploring uncovered skills, projects, and hidden priority areas. They are penalized for repeating questions or ignoring important profile details.

## Baseline Agent

The included baseline agent is rule-based and fully deterministic. It:

- Reads the candidate profile
- Starts with project or clarification questions
- Follows up on metrics, ownership, and missing details
- Moves to uncovered projects or skills
- Uses conversation history to avoid trivial repetition

## Example Interaction

```text
Turn 1 question:
Can you walk me through the architecture, design decisions, and measurable impact of your project Resume Parser API?

Candidate response:
Built a Python API that extracts entities from resumes for internal recruiters.

Reward breakdown:
relevance=0.3, depth=0.3, logical_sequence=0.2, coverage=0.1, clarity=0.1
penalties: repetition=0.0, irrelevant=0.0, shallow=0.0
```

## How to Run

### Local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Docker

```bash
docker build -t elite-interview-env .
docker run --rm elite-interview-env
```

## Determinism Guarantees

- No LLM is used in the grader or environment
- No stochastic grading logic is used
- Task selection and style assignment are deterministic
- Candidate responses come from fixed rules tied to the question and task data

## Minimal Usage Example

```python
from env.interview_env import EliteAIInterviewEvaluationEnv
from env.models import Action

env = EliteAIInterviewEvaluationEnv(task_index=0, style_index=0)
observation = env.reset()
observation, reward, done, info = env.step(
    Action(question="What metrics did you use to evaluate the system?")
)
```

`info` includes explainable scoring fields such as:

- `relevance`
- `depth`
- `sequence`
- `coverage`
- `clarity`
- `penalties`
- `covered_topics`
- `reasoning`
- `average_score`
- `normalized_turn_scores`
