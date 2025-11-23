from typing import Dict, Any

class ExplainerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ranked = state.get("ranked_candidates", [])
        symptoms = state.get("symptoms", {})
        messages = state.get("messages", [])

        # NEW — read incoming session ID (may be None)
        session_id = state.get("session_id")

        if not ranked:
            explanation = "[explainer] No diagnosis available."
            if session_id:
                explanation += f" (Session Patient ID: {session_id})"

            state.setdefault("messages", []).append(explanation)
            state["session_patient_id"] = session_id
            return state

        top = ranked[0]
        disease = top.get("disease")
        likelihood = top.get("likelihood")

        # EXPLAINER PROMPT INCLUDING THE PATIENT SESSION ID
        prompt = (
            f"You are a medical explainer agent. The patient's session ID is: {session_id}.\n"
            f"Explain concisely why '{disease}' is the most likely diagnosis, "
            f"given the symptoms: {symptoms}. "
            f"Also provide clear next steps such as tests, lifestyle advice, "
            f"and red flags (non-alarming language). "
            f"End the explanation by reminding the user of their session patient ID."
        )

        try:
            resp = self.llm.invoke(prompt)
            explanation = getattr(resp, "content", resp)
        except Exception:
            explanation = (
                f"{disease} is most likely (likelihood={likelihood:.2f}). "
                f"Please follow up with a clinician. "
                f"Your session patient ID is: {session_id}."
            )

        # Append explanation to messages
        state.setdefault("messages", []).append(f"[explainer] {explanation}")

        # Store final structured results
        state["diagnosis_result"] = ranked

        # NEW — explicit structured field
        state["session_patient_id"] = session_id

        return state
