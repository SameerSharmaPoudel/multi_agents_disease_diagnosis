import requests
from typing import Dict, Any, Optional
from config_frontend import frontend_settings


def start_diagnosis(
    initial_symptoms: str,
    patient_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "initial_symptoms": initial_symptoms,
    }

    # 🔐 Only include patient_id if explicitly provided
    if patient_id:
        payload["patient_id"] = patient_id

    response = requests.post(
        f"{frontend_settings.fastapi_base_url}/diagnosis/start",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def continue_diagnosis(
    session_id: str,
    user_answer: str,
) -> Dict[str, Any]:
    response = requests.post(
        f"{frontend_settings.fastapi_base_url}/diagnosis/continue",
        json={
            "session_id": session_id,
            "user_answer": user_answer,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
