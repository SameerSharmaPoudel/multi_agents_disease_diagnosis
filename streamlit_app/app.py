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
st.caption("Responses generated aren't to be taken as authentic medical advice !")

# -------------------------------------------------
# Session state initialization
# -------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "status" not in st.session_state:
    st.session_state.status = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

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

    # 🔹 OPTIONAL patient identifier
    patient_id = st.text_input(
        "Patient ID (optional – leave empty if this is your first visit)",
        placeholder="e.g. 9db81563-234d-4ec0-aef2-6c4e30a09e03",
    )

    initial_text = st.text_area(
        "Initial symptoms",
        placeholder="E.g. fever, cough, fatigue for 3 days...",
    )

    if st.button("Start Diagnosis"):
        if not initial_text.strip():
            st.warning("Please enter your symptoms.")
        else:
            with st.spinner("Analyzing symptoms..."):
                response = start_diagnosis(
                    initial_symptoms=initial_text,
                    patient_id=patient_id.strip() or None,  # 🔐 critical line
                )

            # 🔐 Persist backend response
            st.session_state.last_response = response
            st.session_state.session_id = response["session_id"]
            st.session_state.status = response["status"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("message", ""),
            })

            st.rerun()

# -------------------------------------------------
# Continue diagnosis
# -------------------------------------------------
else:
    render_chat()

    status = st.session_state.status
    response = st.session_state.last_response  # 🔑 ALWAYS defined

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

            # 🔐 Persist backend response
            st.session_state.last_response = response
            st.session_state.status = response["status"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("message", ""),
            })

            st.rerun()

    elif status == "completed":
        st.success("Diagnosis completed")

        data = response.get("data") if response else None
        if data:
            with st.expander("📊 Diagnosis Details"):
                st.json(data)

        if st.button("Start New Diagnosis"):
            st.session_state.clear()
            st.rerun()

    else:
        st.error("Unexpected state from backend")
        st.write(response)
