# streamlit_app/app.py

import streamlit as st
from api_client import start_diagnosis, continue_diagnosis
from config_frontend import frontend_settings

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Disease Diagnosis",
    layout="centered",
)

# -------------------------------------------------
# Environment banner
# -------------------------------------------------
if frontend_settings.app_env == "test":
    st.warning("⚠ Running in TEST mode (Fake LLM, no external APIs)")

# -------------------------------------------------
# UI Header
# -------------------------------------------------
st.title("🩺 Multi-Agent Disease Diagnosis Assistant")
st.caption("Powered by a multi-agent AI system")

# -------------------------------------------------
# Session state initialization
# -------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "status" not in st.session_state:
    st.session_state.status = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Helper: render chat history
# -------------------------------------------------
def render_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# -------------------------------------------------
# Start diagnosis
# -------------------------------------------------
if st.session_state.session_id is None:
    st.subheader("Describe your symptoms")

    initial_text = st.text_area(
        "Initial symptoms",
        placeholder="E.g. fever, cough, fatigue for 3 days...",
    )

    if st.button("Start Diagnosis"):
        if not initial_text.strip():
            st.warning("Please enter your symptoms.")
        else:
            with st.spinner("Analyzing symptoms..."):
                response = start_diagnosis(initial_text)

            st.session_state.session_id = response["session_id"]
            st.session_state.status = response["status"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["message"],
            })

            st.rerun()

# -------------------------------------------------
# Continue diagnosis
# -------------------------------------------------
else:
    render_chat()

    status = st.session_state.status

    if status == "awaiting_user_input":
        user_answer = st.chat_input("Your answer")

        if user_answer:
            st.session_state.messages.append({
                "role": "user",
                "content": user_answer,
            })

            with st.spinner("Processing..."):
                response = continue_diagnosis(
                    st.session_state.session_id,
                    user_answer,
                )

            st.session_state.status = response["status"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["message"],
            })

            st.rerun()

    elif status == "completed":
        st.success("Diagnosis completed")

        if response.get("data"):
            with st.expander("📊 Diagnosis Details"):
                st.json(response["data"])

        st.button("Start New Diagnosis", on_click=lambda: st.session_state.clear())
