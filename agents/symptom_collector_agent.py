# agents/symptom_collector_agent.py

from typing import Dict, Any
import json

"""
SymptomCollectorAgent (Option A)

Responsibilities:
- ALWAYS runs on every user turn (start + resume).
- Extracts symptoms from:
    1. Last user free-text message
    2. user_response (answer to pending question)
- Merges extracted symptoms into state['symptoms'].
- Loads historical symptoms from memory into state['historical_symptoms'].
- NEVER asks questions.
- NEVER controls flow or status.
"""

class SymptomCollectorAgent:
    def __init__(self, llm):
        self.llm = llm

    # --------------------------------------------------------
    # LLM-based extraction with fallback
    # --------------------------------------------------------
    def _extract_with_llm(self, free_text: str) -> Dict[str, str]:
        if not free_text:
            return {}

        prompt = (
            "Extract symptoms and short values from the following user text.\n"
            "Return strict JSON only.\n"
            "Keys must be snake_case symptom names.\n\n"
            f"Text: \"{free_text}\""
        )

        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", resp)
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # --------------------
        # Fallback heuristic
        # --------------------
        out = {}
        tokens = free_text.lower().replace(",", " ").split()

        KNOWN = {
            "fever": "yes",
            "cough": "yes",
            "headache": "yes",
            "fatigue": "yes",
            "pain": "yes",
            "breathlessness": "yes",
            "rash": "yes",
        }

        for k, v in KNOWN.items():
            if k in tokens:
                out[k] = v

        return out

    # --------------------------------------------------------
    # MAIN RUN FUNCTION
    # --------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = dict(state.get("symptoms", {}) or {})

        # ----------------------------------------------------
        # 1. Handle user_response (YES / NO / free text)
        # ----------------------------------------------------
        user_response = state.get("user_response")
        pending = state.get("pending_questions") or []

        if user_response and pending:
            q = pending[0]
            key = (
                q.replace("Do you have ", "")
                 .replace("?", "")
                 .strip()
                 .replace(" ", "_")
                 .lower()
            )

            val = str(user_response).strip().lower()

            if val.startswith("y"):
                symptoms[key] = "yes"
            elif val.startswith("n"):
                symptoms[key] = "no"
            else:
                symptoms[key] = val

            # clear handled input
            state["user_response"] = None
            state["pending_questions"] = []

        # ----------------------------------------------------
        # 2. Handle free-text user messages
        # ----------------------------------------------------
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            text = getattr(last, "content", last)

            extracted = self._extract_with_llm(str(text))
            for k, v in extracted.items():
                symptoms[k] = v
        else:
            extracted = {}

        # ----------------------------------------------------
        # 3. Load historical symptoms from memory
        # ----------------------------------------------------
        hist = state.get("patient_history", {}) or {}
        known = hist.get("known_symptoms", {}) or {}

        historical = {}
        for s, meta in known.items():
            if isinstance(meta, dict):
                historical[s] = {
                    "value": meta.get("value"),
                    "chronic": bool(meta.get("chronic")),
                    "first_seen": meta.get("first_seen"),
                    "last_seen": meta.get("last_seen"),
                }
            else:
                historical[s] = {
                    "value": meta,
                    "chronic": False,
                    "first_seen": None,
                    "last_seen": None,
                }

        # ----------------------------------------------------
        # 4. Persist state
        # ----------------------------------------------------
        state["symptoms"] = symptoms
        if historical:
            state["historical_symptoms"] = historical

        # Debug only (NOT messages)
        state.setdefault("debug", []).append({
            "agent": "collector",
            "extracted_from_text": extracted,
            "current_symptoms": list(symptoms.keys()),
        })

        return state
