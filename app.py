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

if "agent_histories" not in st.session_state:
    st.session_state.agent_histories = {
        "Reflective": [],
        "Multi-Agent": [],
        "Tree-of-Thoughts": []
    }

if "last_results" not in st.session_state:
    st.session_state.last_results = {}

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Reflective"

if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = None

if "outputs_for_pdf" not in st.session_state:
    st.session_state.outputs_for_pdf = {}

# ---------------------------
# Toggle CSS
# ---------------------------
st.markdown(
    """
    <style>
    [data-testid="stToggle"] {
        display: flex;
        justify-content: flex-end;
        margin-top: 0.2rem;
    }

    [data-testid="stToggle"] label p {
        font-weight: 600 !important;
    }

    .custom-card {
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Theme toggle
# ---------------------------
top_left, top_right = st.columns([6, 2])

with top_right:
    dark_mode = st.toggle("Dark Mode", value=st.session_state.theme_mode == "Dark")

st.session_state.theme_mode = "Dark" if dark_mode else "Light"

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

        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
        }

        p, label {
            color: #e5e7eb !important;
        }

        .stTextArea textarea {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 14px !important;
        }

        .stTextInput input {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
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
        }

        [data-testid="stToolbar"] * {
            color: #f8fafc !important;
            fill: #f8fafc !important;
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

        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
        }

        p, label {
            color: #1e293b !important;
        }

        .stTextArea textarea {
            background-color: white !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 14px !important;
        }

        .stTextInput input {
            background-color: white !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
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
            background: rgba(255,255,255,0.95);
            border: 1px solid #cbd5e1;
        }

        [data-testid="stToolbar"] * {
            color: #0f172a !important;
            fill: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Header
# ---------------------------
st.markdown(
    f"""
    <div class="custom-card">
        <h1 style="margin-bottom:0.4rem;">🤖 Agentic AI Comparison Dashboard</h1>
        <p style="font-size:1.05rem; margin-bottom:0.3rem;">
            Compare Reflective, Multi-Agent and Tree-of-Thoughts reasoning strategies
        </p>
        <p style="opacity:0.8; margin-top:0.2rem;">
            Current Theme: <b>{st.session_state.theme_mode}</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Helper functions
# ---------------------------
def get_reflective_agent(llm):
    return ReflectiveAgent(llm)

def get_multi_agent(llm):
    planner = SimpleAgent(llm, "Planner")
    critic = SimpleAgent(llm, "Critic")
    optimizer = SimpleAgent(llm, "Optimizer")
    return MultiAgentSystem(planner, critic, optimizer)

def get_tot_agent(llm):
    return TreeOfThoughtAgent(llm, branches=3)

def run_reflective(llm, task):
    reflective = get_reflective_agent(llm)
    with Timer() as t:
        out = reflective.run(task)
    return out, t.elapsed_ms

def run_multi(llm, task):
    multi = get_multi_agent(llm)
    with Timer() as t:
        out = multi.run(task)
    return out, t.elapsed_ms

def run_tot(llm, task):
    tot = get_tot_agent(llm)
    with Timer() as t:
        out = tot.run(task)
    return out, t.elapsed_ms

def build_agent_context(agent_name: str, new_user_message: str) -> str:
    history = st.session_state.agent_histories[agent_name]

    if not history:
        return new_user_message

    transcript_parts = []
    for item in history:
        transcript_parts.append(f"User: {item['user']}")
        transcript_parts.append(f"Assistant: {item['assistant']}")

    transcript_parts.append(f"User: {new_user_message}")
    transcript_parts.append("Continue the conversation consistently and answer the user's latest message.")
    return "\n".join(transcript_parts)

def continue_with_agent(llm, agent_name: str, user_message: str):
    prompt = build_agent_context(agent_name, user_message)

    if agent_name == "Reflective":
        out, elapsed = run_reflective(llm, prompt)
    elif agent_name == "Multi-Agent":
        out, elapsed = run_multi(llm, prompt)
    else:
        out, elapsed = run_tot(llm, prompt)

    st.session_state.agent_histories[agent_name].append({
        "user": user_message,
        "assistant": out.final_text
    })

    return out, elapsed

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

run_button = st.button("Run Comparison")

# ---------------------------
# Main comparison run
# ---------------------------
if run_button:
    llm = OpenAILLM(model="gpt-4.1-mini")

    results = []
    outputs_for_pdf = {}
    last_results = {}

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
            last_results["Reflective"] = r_out.final_text

            st.session_state.agent_histories["Reflective"] = [
                {"user": task, "assistant": r_out.final_text}
            ]

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
            last_results["Multi-Agent"] = m_out.final_text

            st.session_state.agent_histories["Multi-Agent"] = [
                {"user": task, "assistant": m_out.final_text}
            ]

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
            last_results["Tree-of-Thoughts"] = t_out.final_text

            st.session_state.agent_histories["Tree-of-Thoughts"] = [
                {"user": task, "assistant": t_out.final_text}
            ]

    st.session_state.history.append(task)
    st.session_state.last_results = last_results
    st.session_state.comparison_df = pd.DataFrame(results)
    st.session_state.outputs_for_pdf = outputs_for_pdf

# ---------------------------
# Show outputs and comparison if available
# ---------------------------
if st.session_state.last_results:
    st.subheader("Outputs")

    if "Reflective" in st.session_state.last_results:
        with st.expander("Reflective Output", expanded=False):
            if st.button("Use Reflective as Final"):
                st.session_state.final_answer = st.session_state.last_results["Reflective"]
            st.write(st.session_state.last_results["Reflective"])

    if "Multi-Agent" in st.session_state.last_results:
        with st.expander("Multi-Agent Output", expanded=False):
            st.write(st.session_state.last_results["Multi-Agent"])
            if st.button("Use Multi-Agent as Final"):
                st.session_state.final_answer = st.session_state.last_results["Multi-Agent"]

    if "Tree-of-Thoughts" in st.session_state.last_results:
        with st.expander("Tree-of-Thoughts Output", expanded=False):
            st.write(st.session_state.last_results["Tree-of-Thoughts"])
            if st.button("Use Tree-of-Thoughts as Final"):
                st.session_state.final_answer = st.session_state.last_results["Tree-of-Thoughts"]

if st.session_state.comparison_df is not None and not st.session_state.comparison_df.empty:
    df = st.session_state.comparison_df

    st.subheader("Comparison Metrics")
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
    json_data = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)
    st.download_button(
        label="Download Metrics JSON",
        data=json_data,
        file_name="comparison_metrics.json",
        mime="application/json"
    )

    if len(st.session_state.outputs_for_pdf) == 3:
        pdf_path = "agentic_comparison_report.pdf"
        create_pdf_report(st.session_state.outputs_for_pdf, task, pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="agentic_comparison_report.pdf",
                mime="application/pdf"
            )

# ---------------------------
# Continue with selected agent
# ---------------------------
st.subheader("Agent Conversations")

tabs = st.tabs(["Reflective", "Multi-Agent", "Tree-of-Thoughts"])

for i, agent_name in enumerate(["Reflective", "Multi-Agent", "Tree-of-Thoughts"]):
    with tabs[i]:
        history = st.session_state.agent_histories[agent_name]

        st.markdown(f"### {agent_name} Chat")

        if not history:
            st.info(f"{agent_name} için henüz konuşma yok.")
        else:
            for idx, turn in enumerate(history, start=1):
                with st.chat_message("user"):
                    st.write(turn["user"])

                with st.chat_message("assistant"):
                    assistant_text = turn["assistant"]

                    preview_limit = 300
                    if len(assistant_text) > preview_limit:
                        st.write(assistant_text[:preview_limit] + "...")
                        with st.expander("Devamını gör"):
                            st.write(assistant_text)
                    else:
                        st.write(assistant_text)

        msg = st.chat_input(f"{agent_name} ile konuşmaya devam et", key=f"chat_{agent_name}")

        if msg:
            llm = OpenAILLM(model="gpt-4.1-mini")
            out, elapsed = continue_with_agent(llm, agent_name, msg)
            st.success(f"{agent_name} {elapsed} ms içinde yanıt verdi.")
            st.rerun()
        if "final_answer" in st.session_state:
            st.subheader("Final Selected Answer")
            st.write(st.session_state.final_answer)            
# ---------------------------
# Prompt history
# ---------------------------
st.subheader("Prompt History")
for i, item in enumerate(reversed(st.session_state.history[-5:]), start=1):
    st.write(f"{i}. {item}")