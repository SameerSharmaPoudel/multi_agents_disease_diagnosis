# agents/symptom_analyzer_agent.py

from typing import Dict, Any
from datetime import datetime, timedelta
import logging

from langchain_core.messages import AIMessage
from rag.symptom_retriever import SymptomRetriever

log = logging.getLogger("SymptomAnalyzerAgent")


class SymptomAnalyzerAgent:
    """
    Analyzer:
    - Uses current symptoms + selectively relevant historical symptoms.
    - Produces candidate diseases and missing symptoms.
    - All LC-visible messages follow role/content schema.
    - Internal logs stored only in state['debug'] (Option A).
    """

    def __init__(
        self,
        index_dir: str = "./indices/symptoms_faiss",
        hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        retriever=None
    ):
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = SymptomRetriever(index_dir=index_dir, hf_model=hf_model)

        self.recent_days_threshold = 90

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def _is_recent(self, last_seen_iso: str) -> bool:
        if not last_seen_iso:
            return False
        try:
            ts = datetime.fromisoformat(last_seen_iso)
            return (datetime.utcnow() - ts) <= timedelta(days=self.recent_days_threshold)
        except Exception:
            return False

    def _select_relevant_history(
        self,
        current: Dict[str, Any],
        hist: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:

        if not hist:
            return {}

        relevant = {}
        current_keys_lower = {k.lower() for k in current.keys()}

        for key, meta in hist.items():
            value = meta.get("value")
            chronic = bool(meta.get("chronic"))
            last_seen = meta.get("last_seen")

            # Rule 1 — chronic always relevant
            if chronic:
                relevant[key] = value
                continue

            # Rule 2 — same symptom exists in current session
            if key.lower() in current_keys_lower:
                relevant[key] = value
                continue

            # Rule 3 — recent historical observation
            if last_seen and self._is_recent(last_seen):
                relevant[key] = value
                continue

        return relevant

    # -------------------------------------------------------------
    # MAIN EXECUTION
    # -------------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        symptoms = state.get("symptoms", {})
        messages = state.get("messages", [])

        # If no symptoms exist, respond politely
        if not isinstance(symptoms, dict) or not symptoms:
            state["messages"] = [
                *messages,
                {"role": "assistant", "content": "No symptoms found; please provide symptoms."}
            ]
            state["candidates"] = []
            state["missing_symptoms"] = []
            return state

        # Load historical symptoms if present
        historical = state.get("historical_symptoms", {}) or {}
        relevant_history = self._select_relevant_history(symptoms, historical)

        # ---------------------------------------------------------
        # Internal debug log only
        # ---------------------------------------------------------
        state.setdefault("debug", []).append({
            "agent": "analyzer",
            "relevant_history_used": list(relevant_history.keys())
        })
        log.info("Relevant history used: %s", list(relevant_history.keys()))

        # ---------------------------------------------------------
        # LC-visible message (assistant)
        # ---------------------------------------------------------
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[analyzer] relevant_history_used={list(relevant_history.keys())}"
        })

        # Combine current + relevant historical for query
        combined_query = dict(symptoms)
        for hk, hv in relevant_history.items():
            if hk not in combined_query:
                combined_query[hk] = hv

        # Retrieve diseases
        results = self.retriever.retrieve(
            symptoms_dict=combined_query,
            historical_symptoms=relevant_history,
            top_k=8,
            rerank_by_jaccard=True,
            history_weight=0.3
        )

        candidates = []
        missing_union = set()

        for r in results:
            candidates.append({
                "disease": r["disease"],
                "jaccard": r.get("jaccard"),
                "matched_symptoms": r.get("matched_symptoms", []),
                "missing_symptoms": r.get("missing_symptoms", []),
                "row_id": r.get("metadata", {}).get("row_id"),
                "vector_score": r.get("vector_score")
            })
            missing_union |= set(r.get("missing_symptoms", []))

        state["candidates"] = candidates
        state["missing_symptoms"] = sorted(list(missing_union))

        # ---------------------------------------------------------
        # append human-readable summary (assistant)
        # ---------------------------------------------------------
        if candidates:
            top_line = ", ".join([
                f"{c['disease']} (J={c['jaccard']:.2f})" for c in candidates[:3]
            ])
        else:
            top_line = "no candidates yet"

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[analyzer] top_candidates: {top_line}"
        })

        return state
