---
title: AI Interview Eval Env
emoji: "🤖"
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# AI Interview Evaluation Environment (OpenEnv Compatible)

**A deterministic, multi-dimensional evaluation environment that simulates real-world technical interviews and benchmarks AI agents with structured intelligence scoring.**

## Problem

- Current AI interview systems are inconsistent and non-deterministic.
- Most benchmarks do not enforce topic coverage, depth, or reasoning quality.
- Reliable comparison across agents is hard without reproducible scoring.

## Solution

- This project is a deterministic OpenEnv-based interview environment.
- It simulates real technical interviews across Easy, Medium, Hard, and Edge scenarios.
- It enforces full topic coverage, contextual flow, and structured reasoning.
- It evaluates agents with a multi-factor reward system instead of shallow pass/fail logic.

## Key Features

- Deterministic evaluation with no grading randomness
- Multi-difficulty interviews: Easy -> Medium -> Hard -> Edge
- Topic coverage enforcement
- Context-aware question chaining
- Depth-aware adaptive questioning
- Repetition control and flow optimization
- Advanced reward system: relevance, depth, sequence, clarity, coverage
- Explainable scoring with detailed reasoning at every step

## How It Works

- `env.reset()` -> initializes the interview episode
- `env.step(action)` -> processes the next interviewer question
- the environment evaluates the question and updates coverage
- reward is calculated with a structured score breakdown
- the loop continues until full coverage is achieved

## Output Format

```python
(observation, reward, done, info)
```

`info` includes:
- `score_breakdown`
- `covered_topics`
- `uncovered_topics`
- `reasoning`

## Sample Output

```text
Turn 1 | Reward: 0.7550
Covered: [python] | Uncovered: [deployment, entity extraction, resume parser api, testing]
Reasoning: matched python, expanded coverage, and maintained technical depth.
```

## Project Structure

- `env/` -> core environment logic
- `agent/` -> rule-based baseline agent
- `app.py` -> entry point
- `Dockerfile` -> containerized deployment
- `requirements.txt` -> dependencies

## Run Instructions

### Local

```bash
python app.py
```

### Docker

```bash
docker build -t interview-env .
docker run interview-env
```

## Use Cases

- Evaluating LLM agents
- AI hiring systems
- Benchmarking interview performance
- Research in agent evaluation

## Why This Stands Out

- Not just Q&A: it is a structured evaluation system.
- Enforces intelligent interviewer behavior under constraints.
- Fully deterministic and reproducible.
- Simulates real-world interview conditions with explainable scoring.

## Future Scope

- Plug-in external agents
- UI dashboard
- Real-time analytics
