from typing import Dict, Any

class ExplainerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ranked = state.get("ranked_candidates", [])
        symptoms = state.get("symptoms", {})
        messages = state.get("messages", [])

        if not ranked:
            state.setdefault("messages", []).append("[explainer] No diagnosis available.")
            return state

        top = ranked[0]
        disease = top.get("disease")
        likelihood = top.get("likelihood")

        prompt = (
            f"Explain concisely why '{disease}' is the most likely diagnosis given symptoms {symptoms} "
            f"and provide clear next steps (suggest tests, red flags). Keep language non-alarming."
        )
        try:
            resp = self.llm.invoke(prompt)
            explanation = getattr(resp, "content", resp)
        except Exception:
            explanation = f"{disease} is most likely (likelihood={likelihood:.2f}). Consider follow-up with a clinician."

        state.setdefault("messages", []).append(f"[explainer] {explanation}")
        # final result for memory + UI
        state["diagnosis_result"] = ranked
        return state