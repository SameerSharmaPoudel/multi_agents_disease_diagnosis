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

    def _deduplicate_candidates(self, candidates):
        """
        Remove duplicate disease entries while preserving distinct diseases.
        A duplicate is defined as the same disease with very similar symptom matches.
        """
        if not candidates:
            return []
        
        # Group by disease name
        disease_groups = {}
        for c in candidates:
            disease = c["disease"]
            if disease not in disease_groups:
                disease_groups[disease] = []
            disease_groups[disease].append(c)
        
        # For each disease group, select the best representative
        deduplicated = []
        for disease, group in disease_groups.items():
            if len(group) == 1:
                # Only one entry for this disease, keep it
                deduplicated.append(group[0])
            else:
                # Multiple entries for same disease - need to deduplicate
                # Strategy: Keep the most representative one (highest jaccard score)
                group.sort(key=lambda x: x.get("jaccard", 0), reverse=True)
                
                # Check if entries are truly duplicates or just variants
                top_entry = group[0]
                matched_symptoms_top = set(top_entry.get("matched_symptoms", []))
                
                # Look for significantly different variants to keep
                variants_to_keep = [top_entry]  # Always keep the best match
                
                for other_entry in group[1:]:
                    matched_symptoms_other = set(other_entry.get("matched_symptoms", []))
                    
                    # Calculate symptom overlap
                    overlap = len(matched_symptoms_top & matched_symptoms_other)
                    total_unique = len(matched_symptoms_top | matched_symptoms_other)
                    
                    if total_unique > 0:
                        similarity = overlap / total_unique
                        # If less than 80% similar, keep as a variant
                        if similarity < 0.8:
                            variants_to_keep.append(other_entry)
                        # Otherwise, it's a duplicate - skip it
                
                deduplicated.extend(variants_to_keep)
        
        # Sort all by jaccard score
        deduplicated.sort(key=lambda x: x.get("jaccard", 0), reverse=True)
        return deduplicated

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

        # === IMPROVED DEDUPLICATION: Keep distinct diseases, remove true duplicates ===
        original_count = len(candidates)
        candidates = self._deduplicate_candidates(candidates)
        deduplicated_count = len(candidates)
        
        # Debug log for deduplication
        state.setdefault("debug", []).append({
            "agent": "analyzer_dedup",
            "original_count": original_count,
            "deduplicated_count": deduplicated_count,
            "unique_diseases": [c["disease"] for c in candidates],
            "message": f"Deduplicated {original_count} -> {deduplicated_count} candidates"
        })
        log.info("Candidate deduplication: %d -> %d candidates", 
                original_count, deduplicated_count)
        
        # If we still have too few candidates after deduplication, 
        # check if retriever is returning diverse results
        if deduplicated_count < 3 and original_count >= 5:
            log.warning(
                "Low diversity in retrieved diseases. "
                "Check knowledge base for disease variety."
            )

        state["candidates"] = candidates
        state["missing_symptoms"] = sorted(list(missing_union))

        # Summary message
        if candidates:
            def _fmt_j(c):
                j = c.get("jaccard")
                return f"{float(j):.2f}" if j is not None else "NA"

            # Show top 5 candidates for better visibility
            display_count = min(5, len(candidates))
            top_line = ", ".join([f"{c['disease']} (J={_fmt_j(c)})" for c in candidates[:display_count]])
        else:
            top_line = "no candidates yet"

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[analyzer] top_candidates: {top_line}"
        })

        return state