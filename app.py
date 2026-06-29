import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import document_agent
import orchestrator
from utils.cosmos_logger import get_recent_logs

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "orion_ai.png"

st.set_page_config(
    page_title="Orion AI",
    page_icon=LOGO_PATH,
    layout="wide"
)

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "ingested_name" not in st.session_state:
    st.session_state.ingested_name = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

logo_col, title_col = st.columns([1, 4])

with logo_col:
    st.image(LOGO_PATH, width=170)

with title_col:
    st.title("Agentic AI Cloud Orchestration System")
    st.caption(
        "Multi-agent system · Groq LLaMA 3.3 70B · "
        "Azure Cosmos DB · RAG + ReAct + Critic"
    )

with st.sidebar:
    st.header("📄 Document Upload (RAG)")
    uploaded_file = st.file_uploader("Upload a PDF for document queries", type=["pdf"])
    if uploaded_file:
        if uploaded_file.name != st.session_state.ingested_name:
            with st.spinner("Ingesting PDF..."):
                n = document_agent.ingest_pdf(uploaded_file, source_name=uploaded_file.name)
            st.session_state.ingested = True
            st.session_state.ingested_name = uploaded_file.name
            st.success(f"✅ Ingested {n} chunks from **{uploaded_file.name}**")
        else:
            st.info(f"📎 **{uploaded_file.name}** already loaded")
    else:
        st.info("Upload a PDF to enable document Q&A tasks")

    st.divider()
    st.header("🗂️ Recent Audit Logs")
    if st.button("🔄 Refresh Logs"):
        logs = get_recent_logs(limit=8)
        if logs:
            for log in logs:
                with st.expander(f"{log.get('agent_name','?')} · Score: {log.get('critic_score','—')}"):
                    st.caption(f"🕐 {log.get('timestamp','')[:19]}")
                    st.caption(f"Workflow: `{log.get('workflow_id','')}`")
                    st.write(f"**Task:** {log.get('task','')[:80]}...")
        else:
            st.caption("No logs yet or Cosmos DB not connected.")

st.header("Enter Your Goal")
st.markdown("The orchestrator will break your goal into sub-tasks and assign them to specialized agents.")

example_goals = [
    "What is the current Bitcoin price and explain how blockchain works?",
    "Get the weather in Delhi and tell me what to wear for that weather",
    "What is the stock price of Apple and compare it to Tesla?"
]

selected_example = st.selectbox("Or choose an example goal:", ["(type your own below)"] + example_goals)
goal_input = st.text_area(
    "Your goal:",
    value=selected_example if selected_example != "(type your own below)" else "",
    height=80,
    placeholder="e.g. Get the Bitcoin price and explain what makes it volatile"
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("🚀 Run Agents", type="primary", use_container_width=True)
with col2:
    st.caption("⚡ Powered by Groq LPU · ~2s per agent · Critic auto-retries low quality outputs")

if run_button and goal_input.strip():
    with st.spinner("🧠 Orchestrating agents..."):
        result = orchestrator.run(goal_input.strip())
    st.session_state.last_result = result

if st.session_state.last_result:
    result = st.session_state.last_result

    if "error" in result:
        st.error(f"{result['error']}")
    else:
        st.divider()
        st.subheader("Final Response")
        st.info(result["final_response"])
        st.caption(f"Workflow ID: `{result['workflow_id']}`")

        st.subheader("🔍 Agent Task Breakdown")
        for i, r in enumerate(result["results"]):
            score = r["critic_score"]
            color = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
            retried_badge = " · 🔁 Retried" if r.get("retried") else ""
            label = f"Task {i+1} · {r['type'].upper()} Agent · {color} Score: {score}/10{retried_badge}"
            with st.expander(label, expanded=True):
                st.markdown(f"**Task:** {r['description']}")
                st.markdown(f"**Output:**\n\n{r['output']}")
                st.caption(f"Critic feedback: {r['critic_feedback']}")

elif run_button and not goal_input.strip():
    st.warning("Please enter a goal before running.")
