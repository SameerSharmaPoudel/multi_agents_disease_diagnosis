# agents/symptom_collector_agent.py
"""
SymptomCollectorAgent (history-aware, no questioning)

Responsibilities:
- Accept free-text user input (state["messages"] last entry)
- Extract symptom:value pairs using an LLM prompt (lightweight extraction)
- Merge with long-term memory if patient_id is present by calling MemoryAgent helper (via state)
- Write result to state["symptoms"] (dict: symptom -> value)
- Does NOT ask follow-up questions (as requested)

Notes:
- This uses self.llm.invoke(prompt) returning a string-like response. If your LLM object uses different API (.run/.chat), adjust accordingly.
- For simple prototyping you can replace LLM extraction with rule-based parsing.
"""

from typing import Dict, Any
import json

class SymptomCollectorAgent:
    def __init__(self, llm):
        self.llm = llm

    def _extract_with_llm(self, free_text: str) -> Dict[str, str]:
        """
        Use an LLM to extract symptom:value pairs from free_text.
        The LLM should return JSON like: {"fever":"yes","cough":"mild","duration":"2 days"}
        """
        prompt = (
            "Extract symptoms and short values from the following user text.\n"
            "Return strict JSON (no explanation). Keys should be short snake_case symptom names.\n\n"
            f"Text: \"{free_text}\"\n\n"
            "Example output: {\"fever\":\"high\",\"cough\":\"present\",\"duration\":\"2 days\"}"
        )
        try:
            # adapt to your llm interface if needed
            resp = self.llm.invoke(prompt)
            # if resp is AIMessage or similar:
            content = getattr(resp, "content", resp)
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # Fallback: naive parser (very simple)
            pass

        # Very basic fallback parsing (imperfect)
        out = {}
        tokens = free_text.lower().replace(",", " ").split()
        if "fever" in tokens:
            out["fever"] = "yes"
        if "cough" in tokens:
            out["cough"] = "yes"
        if "pain" in tokens:
            out["pain"] = "yes"
        return out

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {**state, "messages": messages}

        last = messages[-1]
        # last might be an AIMessage/HumanMessage object or string
        text = getattr(last, "content", last)

        extracted = self._extract_with_llm(str(text))

        # Merge with existing session symptoms if present
        symptoms = state.get("symptoms", {}).copy()
        for k, v in extracted.items():
            symptoms[k] = v

        # If patient history items exist in state['patient_history'], we can mark chronics
        patient_history = state.get("patient_history", {})  # injected by MemoryAgent if available
        # Do not overwrite historical records, but annotate:
        for h_sym, h_val in patient_history.get("known_symptoms", {}).items():
            if h_sym not in symptoms:
                # keep historical info available for analyzer, but don't assume present
                # store in state under 'historical_symptoms' (non-authoritative)
                state.setdefault("historical_symptoms", {})[h_sym] = h_val

        state["symptoms"] = symptoms
        state.setdefault("messages", []).append(f"[collector] extracted {extracted}")
        return state