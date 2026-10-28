"""Visual chat page for the Weather + Calculator agent.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import logging

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agentic_playground import run_supervisor

logging.basicConfig(level=logging.INFO)


def get_api_key() -> str | None:
    """Read the key from local .env or the deployed app's Streamlit secrets."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None


def show_trace(trace: dict) -> None:
    """Show the agent's decisions and results so the workflow is visible."""
    with st.expander("See how the agent worked"):
        if trace["plan"]:
            st.markdown("**Plan**")
            st.write(trace["plan"])

        st.markdown(f"**Supervisor route:** `{trace['route']}`")
        st.markdown(f"**Specialists:** {', '.join(trace['specialists'])}")

        st.markdown("**Tool results**")
        if trace["observations"]:
            for observation in trace["observations"]:
                st.write(f"{observation['tool']}: {observation['result']}")
        else:
            st.write("No tools were used.")

        if trace["reflection"]:
            st.markdown("**Reflection check**")
            st.write("Draft:", trace["reflection"]["draft_answer"])
            st.write("Reviewed:", trace["reflection"]["reviewed_answer"])


st.set_page_config(page_title="Agentic AI Playground", page_icon="🌦️")
st.title("Weather + Calculator Agent")
st.caption("Explore planning, routing, specialist agents, tools, and reflection.")
st.info("Weather responses use sample data, not live forecasts.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for index, entry in enumerate(st.session_state.chat_history):
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

        if entry["role"] == "assistant" and "trace" in entry:
            show_trace(entry["trace"])
            if entry["trace"]["human_approval"] == "pending":
                approve_col, reject_col = st.columns(2)
                if approve_col.button("Approve answer", key=f"approve_{index}"):
                    entry["trace"]["human_approval"] = "approved"
                    st.rerun()
                if reject_col.button("Reject answer", key=f"reject_{index}"):
                    entry["trace"]["human_approval"] = "rejected"
                    st.rerun()
            else:
                st.caption(f"Human review: {entry['trace']['human_approval']}")

prompt = st.chat_input("Ask about weather or a calculation", max_chars=1000)

if prompt:
    api_key = get_api_key()
    if not api_key:
        st.error("Add GROQ_API_KEY to your .env file before using the agent.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Planning, routing, and running the agent...", expanded=True) as status:
                try:
                    result = run_supervisor(
                        prompt,
                        client=Groq(api_key=api_key),
                        plan_first=True,
                    )
                    status.update(label="Agent finished. Review the trace below.", state="complete")
                except Exception:
                    logging.exception("Agent request failed")
                    status.update(label="The agent could not complete this request.", state="error")
                    st.error("The agent hit an error. Check the app logs for details.")
                    result = None

            if result:
                trace = {
                    "plan": result["plan"],
                    "route": result["route"],
                    "specialists": result["specialists"],
                    "observations": result["observations"],
                    "reflection": result["reflection"],
                    "human_approval": "pending",
                }
                entry = {
                    "role": "assistant",
                    "content": result["answer"],
                    "trace": trace,
                }
                st.session_state.chat_history.append(entry)
                st.markdown(result["answer"])
                show_trace(trace)
                approve_col, reject_col = st.columns(2)
                if approve_col.button("Approve answer", key=f"approve_{len(st.session_state.chat_history)}"):
                    trace["human_approval"] = "approved"
                    st.rerun()
                if reject_col.button("Reject answer", key=f"reject_{len(st.session_state.chat_history)}"):
                    trace["human_approval"] = "rejected"
                    st.rerun()
                st.caption("Human review: pending")
