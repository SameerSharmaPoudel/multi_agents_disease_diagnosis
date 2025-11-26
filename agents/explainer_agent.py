# agents/explainer_agent.py
from typing import Dict, Any


class ExplainerAgent:
    """
    Option A–compliant ExplainerAgent:
    - Only LangChain-valid messages in state["messages"].
    - All internal metadata into state["debug"].
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ranked = state.get("ranked_candidates", [])
        symptoms = state.get("symptoms", {})
        session_id = state.get("session_id")

        # ------------------------------------------------------------------
        # Internal debug info only (NOT exposed to LC)
        # ------------------------------------------------------------------
        state.setdefault("debug", []).append({
            "agent": "explainer",
            "received_ranked_count": len(ranked),
            "session_id": session_id
        })

        # ------------------------------------------------------------------
        # CASE 1 — No diagnosis available
        # ------------------------------------------------------------------
        if not ranked:
            text = "[explainer] No diagnosis available."
            if session_id:
                text += f" (Session Patient ID: {session_id})"

            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": text
            })

            # propagate session id
            state["session_patient_id"] = session_id
            return state

        # ------------------------------------------------------------------
        # CASE 2 — Explain the top-ranked diagnosis
        # ------------------------------------------------------------------
        top = ranked[0]
        disease = top.get("disease")
        likelihood = top.get("likelihood")

        # Prompt to LLM
        prompt = (
            f"You are a medical explainer agent. The patient's session ID is: {session_id}.\n"
            f"Explain concisely why '{disease}' is the most likely diagnosis, "
            f"given the symptoms: {symptoms}. "
            f"Also provide clear next steps such as tests, lifestyle advice, "
            f"and red flags (non-alarming language). "
            f"End the explanation by reminding the user of their session patient ID."
        )

        # LLM generation with safe fallback
        try:
            resp = self.llm.invoke(prompt)
            explanation = getattr(resp, "content", resp)
        except Exception:
            explanation = (
                f"{disease} is most likely (likelihood={likelihood:.2f}). "
                f"Please follow up with a clinician. "
                f"Your session patient ID is: {session_id}."
            )

        # Structured LC-friendly assistant message
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[explainer] {explanation}"
        })

        # Structured final result fields
        state["diagnosis_result"] = ranked
        state["session_patient_id"] = session_id

        return state
