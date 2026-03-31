from __future__ import annotations
import json
import pandas as pd
import streamlit as st

from llm import OpenAILLM
from agents import ReflectiveAgent, SimpleAgent, MultiAgentSystem, TreeOfThoughtAgent
from metrics import Timer
from comparison_utils import create_pdf_report

st.set_page_config(page_title="Agentic AI Comparison", layout="wide")

# ---------------------------
# Session state
# ---------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Toggle CSS
# ---------------------------
st.markdown("""
<style>

/* toggle container */
[data-testid="stToggle"] {
    display: flex;
    justify-content: flex-end;
}

/* toggle track */
[data-testid="stToggle"] label div {
    width: 70px !important;
    height: 36px !important;
    border-radius: 999px !important;
    background: #cbd5e1 !important;
    transition: all 0.25s ease !important;
}

/* toggle circle */
[data-testid="stToggle"] label div:before {
    width: 28px !important;
    height: 28px !important;
    border-radius: 50% !important;
    top: 4px !important;
    left: 5px !important;
    background: white !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

/* checked state (mavi) */
[data-testid="stToggle"] input:checked + div {
    background: linear-gradient(135deg,#3b82f6,#2563eb) !important;
}

/* checked circle move */
[data-testid="stToggle"] input:checked + div:before {
    transform: translateX(32px);
}

</style>
""", 
unsafe_allow_html=True
)

# ---------------------------
# Theme toggle
# ---------------------------
top_left, top_right = st.columns([6, 2])

with top_right:
    dark_mode = st.toggle("Dark Mode", value=st.session_state.theme_mode == "Dark")

if dark_mode:
    st.session_state.theme_mode = "Dark"
else:
    st.session_state.theme_mode = "Light"

# ---------------------------
# Theme CSS
# ---------------------------
if st.session_state.theme_mode == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #081225 0%, #0b1730 100%);
            color: #f8fafc;
        }

        header[data-testid="stHeader"] {
            background: #081225 !important;
        }

        .stAppToolbar {
            background: #081225 !important;
        }

        .block-container {
            padding-top: 2rem;
        }

        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #f8fafc !important;
        }

        .stTextArea textarea {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 14px !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.55rem 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.30) !important;
        }

        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981, #059669) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.55rem 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.30) !important;
        }

        div[data-testid="stMetric"] {
            background-color: #16243d !important;
            padding: 16px !important;
            border-radius: 16px !important;
            border: 1px solid #334155 !important;
        }

        div[data-testid="stExpander"] {
            background-color: #16243d !important;
            border-radius: 14px !important;
            border: 1px solid #334155 !important;
        }

        div[data-testid="stDataFrame"] {
            background-color: #16243d !important;
            border-radius: 14px !important;
            border: 1px solid #334155 !important;
        }

        .custom-card {
            background: rgba(22, 36, 61, 0.9);
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
            color: #0f172a;
        }

        header[data-testid="stHeader"] {
            background: #f8fbff !important;
        }

        .stAppToolbar {
            background: #f8fbff !important;
        }

        .block-container {
            padding-top: 2rem;
        }

        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #0f172a !important;
        }

        .stTextArea textarea {
            background-color: white !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 14px !important;
        }

        div[data-baseweb="select"] > div {
            background-color: white !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.55rem 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.20) !important;
        }

        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981, #059669) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.55rem 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.20) !important;
        }

        div[data-testid="stMetric"] {
            background-color: white !important;
            padding: 16px !important;
            border-radius: 16px !important;
            border: 1px solid #cbd5e1 !important;
        }

        div[data-testid="stExpander"] {
            background-color: white !important;
            border-radius: 14px !important;
            border: 1px solid #cbd5e1 !important;
        }

        div[data-testid="stDataFrame"] {
            background-color: white !important;
            border-radius: 14px !important;
            border: 1px solid #cbd5e1 !important;
        }

        .custom-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid #cbd5e1;
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div class="custom-card">
        <h1 style="margin-bottom:0.4rem;">🤖 Agentic AI Comparison Dashboard</h1>
        <p style="font-size:1.05rem; margin-bottom:0.3rem;">
            Compare Reflective, Multi-Agent and Tree-of-Thoughts reasoning strategies
        </p>
        <p style="opacity:0.8; margin-top:0.2rem;">
            Current Theme: <b>{}</b>
        </p>
    </div>
    """.format(st.session_state.theme_mode),
    unsafe_allow_html=True
)

# ---------------------------
# Inputs
# ---------------------------
task = st.text_area(
    "Enter a task",
    value="Create a 7-day study plan for a midterm covering API design and requirements engineering.",
    height=120
)

model_choice = st.selectbox(
    "Choose mode",
    ["Compare All", "Reflective Only", "Multi-Agent Only", "Tree-of-Thoughts Only"]
)

run_button = st.button("Run")

# ---------------------------
# Helper functions
# ---------------------------
def run_reflective(llm, task):
    reflective = ReflectiveAgent(llm)
    with Timer() as t:
        out = reflective.run(task)
    return out, t.elapsed_ms

def run_multi(llm, task):
    planner = SimpleAgent(llm, "Planner")
    critic = SimpleAgent(llm, "Critic")
    optimizer = SimpleAgent(llm, "Optimizer")
    multi = MultiAgentSystem(planner, critic, optimizer)
    with Timer() as t:
        out = multi.run(task)
    return out, t.elapsed_ms

def run_tot(llm, task):
    tot = TreeOfThoughtAgent(llm, branches=3)
    with Timer() as t:
        out = tot.run(task)
    return out, t.elapsed_ms

# ---------------------------
# Main run
# ---------------------------
if run_button:
    llm = OpenAILLM(model="gpt-4.1-mini")

    results = []
    outputs_for_pdf = {}

    if model_choice in ["Compare All", "Reflective Only"]:
        with st.spinner("Running Reflective..."):
            r_out, r_time = run_reflective(llm, task)
            results.append({
                "Model": "Reflective",
                "Execution Time (ms)": r_time,
                "Output Length": len(r_out.final_text),
                "LLM Calls": r_out.llm_calls
            })
            outputs_for_pdf["reflective"] = {
                "elapsed_ms": r_time,
                "output_chars": len(r_out.final_text),
                "llm_calls": r_out.llm_calls
            }

    if model_choice in ["Compare All", "Multi-Agent Only"]:
        with st.spinner("Running Multi-Agent..."):
            m_out, m_time = run_multi(llm, task)
            results.append({
                "Model": "Multi-Agent",
                "Execution Time (ms)": m_time,
                "Output Length": len(m_out.final_text),
                "LLM Calls": m_out.llm_calls
            })
            outputs_for_pdf["multi_agent"] = {
                "elapsed_ms": m_time,
                "output_chars": len(m_out.final_text),
                "llm_calls": m_out.llm_calls
            }

    if model_choice in ["Compare All", "Tree-of-Thoughts Only"]:
        with st.spinner("Running Tree-of-Thoughts..."):
            t_out, tot_time = run_tot(llm, task)
            results.append({
                "Model": "Tree-of-Thoughts",
                "Execution Time (ms)": tot_time,
                "Output Length": len(t_out.final_text),
                "LLM Calls": t_out.llm_calls
            })
            outputs_for_pdf["tree_of_thoughts"] = {
                "elapsed_ms": tot_time,
                "output_chars": len(t_out.final_text),
                "llm_calls": t_out.llm_calls
            }

    st.session_state.history.append(task)

    st.subheader("Outputs")

    if "r_out" in locals():
        with st.expander("Reflective Output", expanded=False):
            st.write(r_out.final_text)

    if "m_out" in locals():
        with st.expander("Multi-Agent Output", expanded=False):
            st.write(m_out.final_text)

    if "t_out" in locals():
        with st.expander("Tree-of-Thoughts Output", expanded=False):
            st.write(t_out.final_text)

    st.subheader("Comparison Metrics")

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    st.subheader("Model Overview")
    ov1, ov2, ov3 = st.columns(3)

    if "Reflective" in df["Model"].values:
        with ov1:
            st.metric(
                "Reflective",
                f"{df[df['Model'] == 'Reflective']['Execution Time (ms)'].values[0]} ms",
                "Iterative reasoning"
            )

    if "Multi-Agent" in df["Model"].values:
        with ov2:
            st.metric(
                "Multi-Agent",
                f"{df[df['Model'] == 'Multi-Agent']['Execution Time (ms)'].values[0]} ms",
                "Role-based collaboration"
            )

    if "Tree-of-Thoughts" in df["Model"].values:
        with ov3:
            st.metric(
                "Tree-of-Thoughts",
                f"{df[df['Model'] == 'Tree-of-Thoughts']['Execution Time (ms)'].values[0]} ms",
                "Branching reasoning"
            )

    fastest = df.loc[df["Execution Time (ms)"].idxmin()]
    longest = df.loc[df["Output Length"].idxmax()]

    st.subheader("Automatic Insights")
    c1, c2 = st.columns(2)

    with c1:
        st.success(f"⚡ Fastest Model: {fastest['Model']} ({fastest['Execution Time (ms)']} ms)")

    with c2:
        st.info(f"🧠 Most Detailed Model: {longest['Model']} ({longest['Output Length']} chars)")

    if not df.empty:
        st.subheader("Charts")

        chart_df = df.set_index("Model")[[
            "Execution Time (ms)",
            "Output Length",
            "LLM Calls"
        ]].copy()

        for col in chart_df.columns:
            max_val = chart_df[col].max()
            if max_val != 0:
                chart_df[col] = chart_df[col] / max_val

        st.bar_chart(chart_df)

    st.subheader("Overall Interpretation")
    st.write(
        f"Fastest model is **{fastest['Model']}**, while "
        f"**{longest['Model']}** produced the most detailed output. "
        "This shows the trade-off between reasoning depth and execution speed."
    )

    st.subheader("Downloads")

    json_data = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button(
        label="Download Metrics JSON",
        data=json_data,
        file_name="comparison_metrics.json",
        mime="application/json"
    )

    if model_choice == "Compare All" and len(outputs_for_pdf) == 3:
        pdf_path = "agentic_comparison_report.pdf"
        create_pdf_report(outputs_for_pdf, task, pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="agentic_comparison_report.pdf",
                mime="application/pdf"
            )

    st.subheader("Prompt History")
    for i, item in enumerate(reversed(st.session_state.history[-5:]), start=1):
        st.write(f"{i}. {item}")