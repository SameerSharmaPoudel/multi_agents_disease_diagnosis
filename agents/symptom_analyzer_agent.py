# agents/symptom_analyzer_agent.py
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from rag.symptom_retriever import SymptomRetriever  # your retriever (updated below)
from datetime import datetime, timedelta
import logging

log = logging.getLogger("SymptomAnalyzerAgent")

class SymptomAnalyzerAgent:
    """
    Option B Analyzer:
    - Uses only current symptoms (state['symptoms']) by default.
    - Selectively uses historical items from state['historical_symptoms'] when relevant:
        * chronic flag
        * simple lexical relation (same symptom key)
        * recent time window relevance (configurable)
    - Does NOT overwrite state['symptoms']. Instead it creates a combined query for the retriever.
    """

    def __init__(self, index_dir: str = "./indices/symptoms_faiss", hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.retriever = SymptomRetriever(index_dir=index_dir, hf_model=hf_model)
        # Recency thresholds (example): treat history within these as possibly relevant
        self.recent_days_threshold = 90  # months/weeks can be tuned

    def _is_recent(self, last_seen_iso: str) -> bool:
        if not last_seen_iso:
            return False
        try:
            ts = datetime.fromisoformat(last_seen_iso)
            return (datetime.utcnow() - ts) <= timedelta(days=self.recent_days_threshold)
        except Exception:
            return False

    def _select_relevant_history(self, current: Dict[str, Any], hist: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Decide which historical symptoms to include for retrieval (not to mutate state['symptoms']).
        Returns a dict mapping symptom_key -> historical_value for relevant entries.
        Heuristics:
         - include if chronic (always)
         - include if same key appears in current
         - include if last_seen is recent (within threshold)
         - (future) include if ontology relation found
        """
        if not hist:
            return {}

        relevant = {}
        current_keys_lower = {k.lower() for k in (current or {}).keys()}

        for k, meta in hist.items():
            v = meta.get("value") if isinstance(meta, dict) else meta
            chronic = bool(meta.get("chronic")) if isinstance(meta, dict) else False
            last_seen = meta.get("last_seen") if isinstance(meta, dict) else None

            # 1) chronic always relevant
            if chronic:
                relevant[k] = v
                continue

            # 2) same symptom key appears in current -> relevant
            if k.lower() in current_keys_lower:
                relevant[k] = v
                continue

            # 3) recent occurrence (freshness)
            if last_seen and self._is_recent(last_seen):
                relevant[k] = v
                continue

            # otherwise ignore
        return relevant

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = state.get("symptoms", {})
        messages = state.get("messages", [])
        if not isinstance(symptoms, dict) or not symptoms:
            msg = AIMessage(content="No symptoms found; please provide symptoms.")
            state["messages"] = [*messages, msg]
            state["candidates"] = []
            state["missing_symptoms"] = []
            return state

        historical = state.get("historical_symptoms", {}) or {}
        relevant_history = self._select_relevant_history(symptoms, historical)

        # For observability
        state.setdefault("messages", []).append({"agent": "analyzer", "content": f"relevant_history_used={list(relevant_history.keys())}"})
        log.info("Relevant history used: %s", list(relevant_history.keys()))

        # Build combined query dict for retriever: current symptoms have precedence
        combined_for_query = dict(symptoms)
        # Only include relevant historical as "soft" context (keys not in current will be added)
        for hk, hv in relevant_history.items():
            if hk not in combined_for_query:
                combined_for_query[hk] = hv

        # Retrieve candidate diseases
        results = self.retriever.retrieve(
            symptoms_dict=combined_for_query,
            historical_symptoms=relevant_history,
            top_k=8,
            rerank_by_jaccard=True,
            history_weight=0.3  # small weight for history-only matches
        )

        # transform results
        candidates = []
        missing_union = set()
        for r in results:
            candidates.append({
                "disease": r["disease"],
                "jaccard": r.get("jaccard"),
                "matched_symptoms": r.get("matched_symptoms", []),
                "missing_symptoms": r.get("missing_symptoms", []),
                "row_id": r.get("metadata", {}).get("row_id"),
                "vector_score": r.get("vector_score", None)
            })
            missing_union |= set(r.get("missing_symptoms", []))

        state["candidates"] = candidates
        state["missing_symptoms"] = sorted(list(missing_union))
        top_line = ", ".join([f"{c['disease']} (J={c['jaccard']:.2f})" for c in candidates[:3]]) if candidates else "no candidates yet"
        state.setdefault("messages", []).append({"agent": "analyzer", "content": f"top_candidates: {top_line}"})
        return state
