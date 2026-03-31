from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas



def save_run_outputs(task, r_out, r_m, m_out, m_m, t_out, t_m):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    (out_dir / f"{timestamp}_reflective.txt").write_text(
        "\n\n".join(r_out.trace) + "\n\n=== FINAL ===\n" + r_out.final_text,
        encoding="utf-8"
    )
    (out_dir / f"{timestamp}_multi_agent.txt").write_text(
        "\n\n".join(m_out.trace) + "\n\n=== FINAL ===\n" + m_out.final_text,
        encoding="utf-8"
    )
    (out_dir / f"{timestamp}_tot.txt").write_text(
        "\n\n".join(t_out.trace) + "\n\n=== FINAL ===\n" + t_out.final_text,
        encoding="utf-8"
    )

    metrics_payload = {
        "task": task,
        "reflective": {
            "elapsed_ms": r_m.elapsed_ms,
            "output_chars": r_m.output_chars,
            "llm_calls": r_out.llm_calls,
        },
        "multi_agent": {
            "elapsed_ms": m_m.elapsed_ms,
            "output_chars": m_m.output_chars,
            "llm_calls": m_out.llm_calls,
        },
        "tree_of_thoughts": {
            "elapsed_ms": t_m.elapsed_ms,
            "output_chars": t_m.output_chars,
            "llm_calls": t_out.llm_calls,
        },
    }

    metrics_path = out_dir / f"{timestamp}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return metrics_path


def create_comparison_chart(metrics_data: dict):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    labels = ["Reflective", "Multi-Agent", "ToT"]
    times = [
        metrics_data["reflective"]["elapsed_ms"],
        metrics_data["multi_agent"]["elapsed_ms"],
        metrics_data["tree_of_thoughts"]["elapsed_ms"],
    ]
    chars = [
        metrics_data["reflective"]["output_chars"],
        metrics_data["multi_agent"]["output_chars"],
        metrics_data["tree_of_thoughts"]["output_chars"],
    ]
    calls = [
        metrics_data["reflective"]["llm_calls"],
        metrics_data["multi_agent"]["llm_calls"],
        metrics_data["tree_of_thoughts"]["llm_calls"],
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times)
    plt.title("Execution Time Comparison (ms)")
    plt.xlabel("Architecture")
    plt.ylabel("Milliseconds")
    time_path = out_dir / f"{timestamp}_time_comparison.png"
    plt.tight_layout()
    plt.savefig(time_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, chars)
    plt.title("Output Length Comparison (characters)")
    plt.xlabel("Architecture")
    plt.ylabel("Characters")
    chars_path = out_dir / f"{timestamp}_chars_comparison.png"
    plt.tight_layout()
    plt.savefig(chars_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, calls)
    plt.title("LLM Calls Comparison")
    plt.xlabel("Architecture")
    plt.ylabel("Number of Calls")
    calls_path = out_dir / f"{timestamp}_calls_comparison.png"
    plt.tight_layout()
    plt.savefig(calls_path)
    plt.close()

    return str(time_path), str(chars_path), str(calls_path)


def build_markdown_comparison_table(metrics_data: dict) -> str:
    table = f"""| Model | Execution Time (ms) | Output Length (chars) | LLM Calls |
|---|---:|---:|---:|
| Reflective | {metrics_data["reflective"]["elapsed_ms"]} | {metrics_data["reflective"]["output_chars"]} | {metrics_data["reflective"]["llm_calls"]} |
| Multi-Agent | {metrics_data["multi_agent"]["elapsed_ms"]} | {metrics_data["multi_agent"]["output_chars"]} | {metrics_data["multi_agent"]["llm_calls"]} |
| Tree-of-Thoughts | {metrics_data["tree_of_thoughts"]["elapsed_ms"]} | {metrics_data["tree_of_thoughts"]["output_chars"]} | {metrics_data["tree_of_thoughts"]["llm_calls"]} |
"""
    return table


def build_pdf_ready_text(metrics_data: dict) -> str:
    return f"""
Comparison Table

Reflective:
- Execution Time: {metrics_data["reflective"]["elapsed_ms"]} ms
- Output Length: {metrics_data["reflective"]["output_chars"]} chars
- LLM Calls: {metrics_data["reflective"]["llm_calls"]}

Multi-Agent:
- Execution Time: {metrics_data["multi_agent"]["elapsed_ms"]} ms
- Output Length: {metrics_data["multi_agent"]["output_chars"]} chars
- LLM Calls: {metrics_data["multi_agent"]["llm_calls"]}

Tree-of-Thoughts:
- Execution Time: {metrics_data["tree_of_thoughts"]["elapsed_ms"]} ms
- Output Length: {metrics_data["tree_of_thoughts"]["output_chars"]} chars
- LLM Calls: {metrics_data["tree_of_thoughts"]["llm_calls"]}
"""

def create_pdf_report(metrics_data: dict, task: str, output_path: str):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Agentic AI Comparison Report")

    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Task: {task}")

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Comparison Results")

    y -= 25
    c.setFont("Helvetica", 11)

    lines = [
        f"Reflective -> Time: {metrics_data['reflective']['elapsed_ms']} ms, "
        f"Output Length: {metrics_data['reflective']['output_chars']}, "
        f"LLM Calls: {metrics_data['reflective']['llm_calls']}",

        f"Multi-Agent -> Time: {metrics_data['multi_agent']['elapsed_ms']} ms, "
        f"Output Length: {metrics_data['multi_agent']['output_chars']}, "
        f"LLM Calls: {metrics_data['multi_agent']['llm_calls']}",

        f"Tree-of-Thoughts -> Time: {metrics_data['tree_of_thoughts']['elapsed_ms']} ms, "
        f"Output Length: {metrics_data['tree_of_thoughts']['output_chars']}, "
        f"LLM Calls: {metrics_data['tree_of_thoughts']['llm_calls']}",
    ]

    for line in lines:
        c.drawString(50, y, line)
        y -= 20

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Interpretation")

    y -= 25
    c.setFont("Helvetica", 11)

    interpretation = [
        "Reflective produced a detailed and iterative answer.",
        "Multi-Agent generated a role-based structured answer.",
        "Tree-of-Thoughts produced the shortest but fastest reasoning path.",
    ]

    for line in interpretation:
        c.drawString(50, y, line)
        y -= 20

    c.save()