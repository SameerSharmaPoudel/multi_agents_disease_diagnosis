# agents/symptom_analyzer_agent.py

from typing import Dict, Any
from datetime import datetime, timedelta
import logging

from rag.symptom_retriever import SymptomRetriever

log = logging.getLogger("SymptomAnalyzerAgent")


class SymptomAnalyzerAgent:
    """
    Analyzer (Option A):
    - Uses current symptoms + selectively relevant historical symptoms.
    - Produces candidates + missing symptoms.
    - All LC-visible messages follow {"role": "...", "content": "..."}.
    - Internal logs stored only in state['debug'].
    - NEW: writes structured state['relevant_history'] as a list of dicts:
        [{"symptom": "...", "value": "...", "chronic": bool, "last_seen": "..."}]
    """

    def __init__(
        self,
        index_dir: str = "./indices/symptoms_faiss",
        hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        retriever=None,
        recent_days_threshold: int = 90,
        history_weight: float = 0.3,
        top_k: int = 8,
    ):
        self.retriever = retriever or SymptomRetriever(index_dir=index_dir, hf_model=hf_model)
        self.recent_days_threshold = recent_days_threshold
        self.history_weight = history_weight
        self.top_k = top_k

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
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return relevant history as dict of symptom -> meta dict.
        meta has at least: value, chronic, last_seen, first_seen.
        """
        if not hist:
            return {}

        relevant: Dict[str, Dict[str, Any]] = {}
        current_keys_lower = {k.lower() for k in current.keys()}

        for key, meta in hist.items():
            if not isinstance(meta, dict):
                meta = {"value": meta}

            value = meta.get("value")
            chronic = bool(meta.get("chronic"))
            last_seen = meta.get("last_seen")

            # Rule 1 — chronic always relevant
            if chronic:
                relevant[key] = meta
                continue

            # Rule 2 — same symptom exists in current session
            if key.lower() in current_keys_lower:
                relevant[key] = meta
                continue

            # Rule 3 — recent historical observation
            if last_seen and self._is_recent(str(last_seen)):
                relevant[key] = meta
                continue

        return relevant

    # -------------------------------------------------------------
    # MAIN EXECUTION
    # -------------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = state.get("symptoms", {}) or {}
        messages = state.get("messages", []) or []

        if not isinstance(symptoms, dict) or not symptoms:
            state["messages"] = [
                *messages,
                {"role": "assistant", "content": "No symptoms found; please provide symptoms."}
            ]
            state["candidates"] = []
            state["missing_symptoms"] = []
            state["relevant_history"] = []
            return state

        historical = state.get("historical_symptoms", {}) or {}

        # choose which history to use
        relevant_hist_meta = self._select_relevant_history(symptoms, historical)

        # NEW: write structured relevant_history list (diagnoser-friendly)
        relevant_history_list = []
        for k, meta in relevant_hist_meta.items():
            relevant_history_list.append({
                "symptom": k,
                "value": meta.get("value"),
                "chronic": bool(meta.get("chronic")),
                "last_seen": meta.get("last_seen"),
                "first_seen": meta.get("first_seen"),
            })
        state["relevant_history"] = relevant_history_list

        # Debug only
        state.setdefault("debug", []).append({
            "agent": "analyzer",
            "relevant_history_used": [x["symptom"] for x in relevant_history_list],
        })
        log.info("Relevant history used: %s", [x["symptom"] for x in relevant_history_list])

        # LC-visible message
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[analyzer] relevant_history_used={[x['symptom'] for x in relevant_history_list]}"
        })

        # Combine current + relevant historical for retrieval query
        combined_query = dict(symptoms)
        for item in relevant_history_list:
            hk = item["symptom"]
            hv = item.get("value")
            if hk not in combined_query and hv is not None:
                combined_query[hk] = hv

        # Prepare historical_symptoms payload for retriever (simple dict symptom -> value)
        historical_for_retriever = {
            item["symptom"]: item.get("value")
            for item in relevant_history_list
        }

        results = self.retriever.retrieve(
            symptoms_dict=combined_query,
            historical_symptoms=historical_for_retriever,
            top_k=self.top_k,
            rerank_by_jaccard=True,
            history_weight=self.history_weight,
        )

        candidates = []
        missing_union = set()

        for r in results or []:
            candidates.append({
                "disease": r.get("disease"),
                "jaccard": r.get("jaccard"),
                "matched_symptoms": r.get("matched_symptoms", []) or [],
                "missing_symptoms": r.get("missing_symptoms", []) or [],
                "row_id": (r.get("metadata") or {}).get("row_id"),
                "vector_score": r.get("vector_score"),
            })
            missing_union |= set(r.get("missing_symptoms", []) or [])

        state["candidates"] = candidates
        state["missing_symptoms"] = sorted(list(missing_union))

        # Summary message
        if candidates:
            def _fmt_j(c):
                j = c.get("jaccard")
                return f"{float(j):.2f}" if j is not None else "NA"

            top_line = ", ".join([f"{c['disease']} (J={_fmt_j(c)})" for c in candidates[:3]])
        else:
            top_line = "no candidates yet"

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[analyzer] top_candidates: {top_line}"
        })

        return state
