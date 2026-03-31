from __future__ import annotations
import json
from llm import OpenAILLM
from agents import ReflectiveAgent, SimpleAgent, MultiAgentSystem, TreeOfThoughtAgent
from metrics import Timer, Metrics
from comparison_utils import (
    save_run_outputs,
    create_comparison_chart,
    build_markdown_comparison_table,
    build_pdf_ready_text,
)


def run_and_measure(runner):
    with Timer() as t:
        out = runner()
    m = Metrics(elapsed_ms=t.elapsed_ms, output_chars=len(out.final_text))
    return out, m


def main():
    task = "Create a 7-day study plan for a midterm covering API design and requirements engineering."

    llm = OpenAILLM(model="gpt-4.1-mini")

    reflective = ReflectiveAgent(llm)

    planner = SimpleAgent(llm, "Planner")
    critic = SimpleAgent(llm, "Critic")
    optimizer = SimpleAgent(llm, "Optimizer")
    multi = MultiAgentSystem(planner, critic, optimizer)

    tot = TreeOfThoughtAgent(llm, branches=3)

    r_out, r_m = run_and_measure(lambda: reflective.run(task))
    m_out, m_m = run_and_measure(lambda: multi.run(task))
    t_out, t_m = run_and_measure(lambda: tot.run(task))

    print("\n=== FINAL OUTPUTS ===\n")
    print("[Reflective]\n", r_out.final_text, "\n")
    print("[Multi-Agent]\n", m_out.final_text, "\n")
    print("[Tree-of-Thoughts]\n", t_out.final_text, "\n")

    print("\n=== METRICS ===")
    print(f"Reflective: {r_m.elapsed_ms} ms, {r_m.output_chars} chars, {r_out.llm_calls} calls")
    print(f"Multi-Agent: {m_m.elapsed_ms} ms, {m_m.output_chars} chars, {m_out.llm_calls} calls")
    print(f"ToT: {t_m.elapsed_ms} ms, {t_m.output_chars} chars, {t_out.llm_calls} calls")

    metrics_path = save_run_outputs(task, r_out, r_m, m_out, m_m, t_out, t_m)

    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    time_chart, chars_chart, calls_chart = create_comparison_chart(metrics_data)

    markdown_table = build_markdown_comparison_table(metrics_data)
    pdf_ready_text = build_pdf_ready_text(metrics_data)

    print("\n=== MARKDOWN COMPARISON TABLE ===\n")
    print(markdown_table)

    print("\n=== PDF READY TEXT ===\n")
    print(pdf_ready_text)

    print("\nSaved metrics to:", metrics_path)
    print("Saved charts:")
    print("-", time_chart)
    print("-", chars_chart)
    print("-", calls_chart)


if __name__ == "__main__":
    main()