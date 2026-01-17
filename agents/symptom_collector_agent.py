# agents/symptom_collector_agent.py
from typing import Dict, Any
import json

"""
SymptomCollectorAgent (history-aware, no questioning)

- Extracts symptom:value pairs from the last user message.
- Updates state['symptoms'] for the *current* session.
- Exposes historical symptoms from state['patient_history'] into state['historical_symptoms'].
- Stores internal debug logs in state['debug'] (not state['messages']).
"""

class SymptomCollectorAgent:
    def __init__(self, llm):
        self.llm = llm

    # --------------------------------------------------------
    # LLM-based extraction with fallback
    # --------------------------------------------------------
    def _extract_with_llm(self, free_text: str) -> Dict[str, str]:
        prompt = (
            "Extract symptoms and short values from the following user text.\n"
            "Return strict JSON (no explanation). Keys should be short snake_case symptom names.\n\n"
            f"Text: \"{free_text}\"\n\n"
            "Example output: {\"fever\":\"high\",\"cough\":\"present\",\"duration\":\"2 days\"}"
        )
        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", resp)
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # fallback token parsing
        out = {}
        tokens = free_text.lower().replace(",", " ").split()
        if "fever" in tokens:
            out["fever"] = "yes"
        if "cough" in tokens:
            out["cough"] = "yes"
        if "pain" in tokens:
            out["pain"] = "yes"
        return out

    # --------------------------------------------------------
    # MAIN RUN FUNCTION
    # --------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return state

        last = messages[-1]
        text = getattr(last, "content", last)
        extracted = self._extract_with_llm(str(text))

        # update current-session symptoms
        symptoms = dict(state.get("symptoms", {}))
        for k, v in extracted.items():
            symptoms[k] = v

        # normalize historical symptoms from memory
        hist = state.get("patient_history", {}) or {}
        known = hist.get("known_symptoms", {})
        normalized = {}

        for s, meta in known.items():
            if isinstance(meta, dict):
                normalized[s] = {
                    "value": meta.get("value"),
                    "chronic": bool(meta.get("chronic")),
                    "first_seen": meta.get("first_seen"),
                    "last_seen": meta.get("last_seen"),
                }
            else:
                normalized[s] = {
                    "value": meta,
                    "chronic": False,
                    "first_seen": None,
                    "last_seen": None,
                }

        state["symptoms"] = symptoms
        if normalized:
            state["historical_symptoms"] = normalized

        # --------------------------------------------------------
        # Put internal agent logs in debug channel (Option A)
        # --------------------------------------------------------
        state.setdefault("debug", []).append(
            {"agent": "collector", "extracted": extracted}
        )

        # --------------------------------------------------------
        # Only clean, LC-compatible messages go into messages[]
        # --------------------------------------------------------
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[collector] processed input"
        })

        state["_symptoms_collected"] = True

        return state
