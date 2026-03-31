from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout

import gradio as gr

from agent.baseline import BaselineInterviewAgent
from env.interview_env import EliteAIInterviewEvaluationEnv

DEBUG_MODE = False
HEADER_LINE = "=" * 40
SECTION_LINE = "-" * 40


def _format_label_value(label: str, value: str) -> str:
    return f"{label:<10}: {value}"


def _format_topics(topics: list[str]) -> str:
    if not topics:
        return "[]"
    return "[" + ", ".join(topics) + "]"


def _format_penalties(penalties: dict) -> list[str]:
    visible = [(name, value) for name, value in penalties.items() if value > 0]
    if not visible:
        return ["  none"]
    return [f"  - {name}: {value:.4f}" for name, value in visible]


def _print_episode_header(observation) -> None:
    profile = observation.candidate_profile
    skills = ", ".join(skill for skill in profile.get("skills", []) if skill.strip()) or "N/A"
    print(HEADER_LINE)
    print(
        f"Interview | difficulty={observation.difficulty.upper()} | "
        f"style={observation.interviewer_style.upper()}"
    )
    print(HEADER_LINE)
    print(_format_label_value("Candidate", str(profile.get("name", "Unknown"))))
    print(_format_label_value("Role", str(profile.get("target_role", "Unknown"))))
    print(_format_label_value("Skills", skills))
    print()


def _print_turn_report(turn: int, question: str, reward: float, info: dict) -> None:
    breakdown = info["score_breakdown"]
    print(f"Turn {turn}")
    print(f"Q: {question}")
    print(f"Reward: {reward:.4f}")
    print()
    print("Breakdown:")
    print(f"  - relevance: {breakdown['relevance_to_profile']:.4f}")
    print(f"  - depth: {breakdown['depth_of_question']:.4f}")
    print(f"  - sequence: {breakdown['logical_sequence']:.4f}")
    print(f"  - coverage: {breakdown['coverage_of_topics']:.4f}")
    print(f"  - clarity: {breakdown['clarity_of_question']:.4f}")
    print()
    print("Penalties:")
    for line in _format_penalties(breakdown["penalties"]):
        print(line)
    print()
    print(f"Covered   : {_format_topics(info['covered_topics'])}")
    print(f"Uncovered : {_format_topics(info['uncovered_topics'])}")
    print(f"Reasoning : {info['reasoning']}")
    if DEBUG_MODE:
        print()
        print("Debug JSON:")
        print(
            json.dumps(
                {
                    "candidate_response": info["candidate_response"],
                    "matched_topics": info["matched_topics"],
                    "new_topics": info["new_topics"],
                    "relevance_reasons": info["relevance_reasons"],
                    "state": info["state"],
                },
                indent=2,
            )
        )
    print()


def _print_episode_summary(final_info: dict) -> None:
    state = final_info["state"]
    print(SECTION_LINE)
    print("EPISODE COMPLETE")
    print(SECTION_LINE)
    print(_format_label_value("difficulty", final_info["difficulty"]))
    print(_format_label_value("style", final_info["interviewer_style"]))
    print(_format_label_value("turns", str(state["turn"])))
    print(_format_label_value("score", f"{state['cumulative_score']:.4f}"))
    print(_format_label_value("average", f"{state['average_score']:.4f}"))
    print(_format_label_value("covered", _format_topics(final_info["covered_topics"])))
    print(_format_label_value("uncovered", _format_topics(final_info["uncovered_topics"])))
    print()


def _print_aggregate_results(summaries: list[dict]) -> None:
    print(HEADER_LINE)
    print("AGGREGATE RESULTS")
    print(HEADER_LINE)
    print(f"{'Difficulty':<12} {'Style':<12} {'Turns':<7} {'Score':<8}")
    print(SECTION_LINE)
    for summary in summaries:
        print(
            f"{summary['difficulty']:<12} "
            f"{summary['style']:<12} "
            f"{summary['turns']:<7} "
            f"{summary['cumulative_score']:<8.4f}"
        )


def run_episode(env: EliteAIInterviewEvaluationEnv, agent: BaselineInterviewAgent) -> dict:
    observation = env.reset()
    done = False
    final_info: dict = {}
    _print_episode_header(observation)

    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        final_info = info
        _print_turn_report(observation.current_turn, action.question, reward, info)

    _print_episode_summary(final_info)
    return {
        "task_id": final_info["task_id"],
        "difficulty": final_info["difficulty"],
        "style": final_info["interviewer_style"],
        "turns": final_info["state"]["turn"],
        "cumulative_score": final_info["state"]["cumulative_score"],
        "average_score": final_info["state"]["average_score"],
        "covered_topics": final_info["covered_topics"],
        "uncovered_topics": final_info["uncovered_topics"],
    }


def run_benchmark() -> tuple[str, list[dict]]:
    agent = BaselineInterviewAgent()
    summaries: list[dict] = []
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        for task_index in range(4):
            env = EliteAIInterviewEvaluationEnv(task_index=task_index, style_index=task_index)
            summary = run_episode(env, agent)
            summaries.append(summary)

        _print_aggregate_results(summaries)
        if DEBUG_MODE:
            print()
            print("Raw summaries:")
            print(json.dumps(summaries, indent=2))

    return buffer.getvalue(), summaries


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Elite AI Interview Evaluation Environment") as demo:
        gr.Markdown(
            "# Elite AI Interview Evaluation Environment\n"
            "Run the full deterministic benchmark across all interview scenarios."
        )
        run_button = gr.Button("Run Benchmark", variant="primary")
        report_output = gr.Textbox(label="Evaluation Report", lines=36)
        summary_output = gr.JSON(label="Aggregate Results")
        run_button.click(fn=run_benchmark, outputs=[report_output, summary_output])
    return demo


def main() -> None:
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
