import requests
from typing import Dict, Any
from streamlit_app.config import settings


def start_diagnosis(initial_symptoms: str) -> Dict[str, Any]:
    response = requests.post(
        f"{settings.fastapi_base_url}/diagnosis/start",
        json={"initial_symptoms": initial_symptoms},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def continue_diagnosis(session_id: str, user_answer: str) -> Dict[str, Any]:
    response = requests.post(
        f"{settings.fastapi_base_url}/diagnosis/continue",
        json={
            "session_id": session_id,
            "user_answer": user_answer,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()