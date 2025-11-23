# agents/symptom_collector_agent.py
from typing import Dict, Any
import json
from datetime import datetime

"""
SymptomCollectorAgent (history-aware, no questioning)

- Extracts symptom:value pairs from the last user message (LLM extraction with fallback).
- Emits state['symptoms'] (only current session inputs).
- Exposes historical items from state['patient_history'] into state['historical_symptoms']
  in a normalized structure (value, chronic, first_seen, last_seen).
- Does NOT treat historical items as active symptoms.
"""

class SymptomCollectorAgent:
    def __init__(self, llm):
        self.llm = llm

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
        text = getattr(last, "content", last)
        extracted = self._extract_with_llm(str(text))

        # Current session symptoms: user-provided values override existing
        symptoms = dict(state.get("symptoms", {}))
        for k, v in extracted.items():
            symptoms[k] = v

        # Normalize patient_history (if provided by MemoryAgent) into historical_symptoms
        hist = state.get("patient_history", {}) or {}
        known = hist.get("known_symptoms", {})  # memory returns {sym: {value, first_seen, last_updated, chronic?}}
        normalized = {}
        for s, meta in known.items():
            # meta may be {"value":..., "first_seen":..., "last_updated":...}
            val = meta.get("value") if isinstance(meta, dict) else meta
            first = meta.get("first_seen") if isinstance(meta, dict) else None
            last_seen = meta.get("last_updated") if isinstance(meta, dict) else None
            chronic = bool(meta.get("chronic")) if isinstance(meta, dict) and "chronic" in meta else False
            normalized[s] = {
                "value": val,
                "chronic": chronic,
                "first_seen": first,
                "last_seen": last_seen
            }

        state["symptoms"] = symptoms
        # Only add historical_symptoms if any exist; Analyzer will decide relevance
        if normalized:
            state["historical_symptoms"] = normalized

        state.setdefault("messages", []).append({"agent": "collector", "content": f"extracted {extracted}"})
        return state
