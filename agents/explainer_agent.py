# agents/explainer_agent.py
from typing import Dict, Any


class ExplainerAgent:
    """
    Option A–compliant ExplainerAgent:
    - Only LangChain-valid messages in state["messages"]
    - All internal metadata in state["debug"]
    - Uses patient_id (NOT session_id) for user-facing identity
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ranked = state.get("ranked_candidates", []) or []
        symptoms = state.get("symptoms", {}) or {}

        # ✅ Correct identifier
        patient_id = state.get("patient_id")

        # --------------------------------------------------
        # Debug (internal only)
        # --------------------------------------------------
        state.setdefault("debug", []).append({
            "agent": "explainer",
            "received_ranked_count": len(ranked),
            "patient_id": patient_id,
        })

        # --------------------------------------------------
        # CASE 1 — No diagnosis available
        # --------------------------------------------------
        if not ranked:
            text = "[explainer] No diagnosis available at this time."
            if patient_id:
                text += f" (Patient ID: {patient_id})"

            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": text,
            })

            # Expose patient_id for API / UI
            state["patient_id"] = patient_id
            return state

        # --------------------------------------------------
        # CASE 2 — Explain top-ranked diagnosis
        # --------------------------------------------------
        top = ranked[0]
        disease = top.get("disease")
        likelihood = top.get("likelihood")

        prompt = (
            f"You are a medical explainer agent.\n"
            f"Patient ID: {patient_id}\n\n"
            f"Explain concisely why '{disease}' is the most likely diagnosis, "
            f"given the symptoms: {symptoms}. "
            f"Provide next steps (tests, lifestyle advice, red flags) "
            f"in calm, non-alarming language. "
            f"End by reminding the user of their Patient ID."
        )

        try:
            resp = self.llm.invoke(prompt)
            explanation = getattr(resp, "content", resp)
        except Exception:
            explanation = (
                f"{disease} is the most likely diagnosis "
                f"(likelihood={likelihood:.2f}). "
                f"Please follow up with a clinician. "
                f"Your Patient ID is: {patient_id}."
            )

        # --------------------------------------------------
        # LC-compatible assistant message
        # --------------------------------------------------
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[explainer] {explanation}",
        })

        # --------------------------------------------------
        # Final structured outputs
        # --------------------------------------------------
        state["diagnosis_result"] = ranked
        state["explainer_output"] = explanation

        # 🔑 DO NOT rename or duplicate IDs
        state["patient_id"] = patient_id

        return state
