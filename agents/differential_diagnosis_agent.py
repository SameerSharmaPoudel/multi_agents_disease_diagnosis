# agents/differential_diagnosis_agent.py

from typing import Dict, List, Any
import numpy as np


class DifferentialDiagnosisAgent:
    """
    Differential diagnosis agent (Option A version):
    - All LC-visible messages use {"role": "assistant", "content": "..."}.
    - Internal signals go into state['debug'] only.
    - Uses historical symptoms (state['historical_symptoms'] OR state['relevant_history'])
      to adjust ranking & discriminator selection.
    """

    def __init__(self, llm=None, confidence_threshold: float = 0.8, history_weight: float = 0.15):
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.history_weight = history_weight

    # -----------------------------------------------------------
    # RANKING
    # -----------------------------------------------------------
    def _hybrid_rank(
        self,
        candidates: List[Dict[str, Any]],
        history_index: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        vecs = [c.get("vector_score") or 0.0 for c in candidates]
        min_v, max_v = min(vecs), max(vecs)

        def norm(v: float) -> float:
            return 0.0 if max_v - min_v == 0 else (v - min_v) / (max_v - min_v)

        scored = []
        for c in candidates:
            j = float(c.get("jaccard") or 0.0)
            vs = float(norm(float(c.get("vector_score") or 0.0)))

            # Historical bonus: sum weights for matched symptoms that are present in history_index
            hist_bonus = 0.0
            for s in c.get("matched_symptoms", []) or []:
                if s in history_index:
                    hist_bonus += float(history_index[s])

            score = 0.6 * j + 0.4 * vs + hist_bonus
            score = float(np.clip(score, 0.0, 1.0))

            scored.append({
                "disease": c.get("disease"),
                "likelihood": score,
                "reason": "hybrid(jaccard+vec+history)",
                "matched_symptoms": c.get("matched_symptoms", []) or [],
                "missing_symptoms": c.get("missing_symptoms", []) or [],
            })

        scored.sort(key=lambda x: x["likelihood"], reverse=True)
        return scored

    # -----------------------------------------------------------
    # DISCRIMINATOR SELECTION
    # -----------------------------------------------------------
    def _select_discriminators(
        self,
        candidates: List[Dict[str, Any]],
        known_symptoms: set,
        limit: int,
        history_index: Dict[str, float],
    ) -> List[str]:
        """
        History-aware discriminator selection:
        - Base weight from candidate jaccard
        - Bonus for symptoms that appear historically (especially chronic)
        """
        weights: Dict[str, float] = {}

        for c in candidates or []:
            base = float(c.get("jaccard") or 0.0)

            for symptom in c.get("missing_symptoms", []) or []:
                if symptom in known_symptoms:
                    continue

                w = base + 0.01
                if symptom in history_index:
                    w += float(history_index[symptom]) * 2.0

                weights[symptom] = weights.get(symptom, 0.0) + w

        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_items[:limit]]

    def _to_natural_question(self, symptom_key: str) -> str:
        return f"Do you have {symptom_key.replace('_', ' ')}?"

    # -----------------------------------------------------------
    # MAIN RUN
    # -----------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        candidates = state.get("candidates", []) or []
        symptoms = state.get("symptoms", {}) or {}

        # Build history_index from either:
        # 1) state['historical_symptoms'] (dict) OR
        # 2) state['relevant_history'] (list of dicts) produced by analyzer
        history_index: Dict[str, float] = {}

        history_raw = state.get("historical_symptoms", {}) or {}
        if isinstance(history_raw, dict) and history_raw:
            for key, meta in history_raw.items():
                if isinstance(meta, dict):
                    chronic = bool(meta.get("chronic"))
                    history_index[key] = self.history_weight * (2.0 if chronic else 1.0)

        relevant_history = state.get("relevant_history", []) or []
        if isinstance(relevant_history, list) and relevant_history:
            for entry in relevant_history:
                if not isinstance(entry, dict):
                    continue
                sym = entry.get("symptom")
                if not sym:
                    continue
                chronic = bool(entry.get("chronic"))
                history_index[sym] = self.history_weight * (2.0 if chronic else 1.0)

        # Internal debug only
        state.setdefault("debug", []).append({
            "agent": "diagnoser",
            "history_index_keys": list(history_index.keys()),
            "num_candidates": len(candidates),
        })

        # Rank with historical adjustments
        ranked = self._hybrid_rank(candidates, history_index)
        state["ranked_candidates"] = ranked

        top_likelihood = float(ranked[0]["likelihood"]) if ranked else 0.0
        state["top_likelihood"] = top_likelihood
        state["uncertainty"] = 1.0 - top_likelihood

        # If no candidates, still define pending_questions
        if not ranked:
            state["pending_questions"] = []
            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": "[diagnoser] no ranked candidates available yet"
            })
            return state

        # CONFIDENT CASE
        if top_likelihood >= self.confidence_threshold:
            state["pending_questions"] = []
            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": f"[diagnoser] confident top {ranked[0]['disease']} ({top_likelihood:.2f})"
            })
            return state

        # NEED FOLLOW-UP QUESTIONS
        missing = state.get("missing_symptoms", []) or []

        known_present = {
            k for k, v in symptoms.items()
            if str(v).strip().lower() not in {"no", "none", "absent", "0", "false", ""}
        }

        unknown_missing = [s for s in missing if s not in known_present]
        n_missing = len(unknown_missing)

        if n_missing == 0:
            # No obvious follow-ups, but not confident: keep pending empty
            state["pending_questions"] = []
            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": "[diagnoser] low confidence but no missing symptoms to ask about"
            })
            return state

        # Batch size policy
        if n_missing <= 3:
            batch_size = n_missing
        elif 4 <= n_missing <= 10:
            batch_size = 3
        else:
            batch_size = 5

        if n_missing > batch_size:
            discriminators = self._select_discriminators(
                candidates=candidates,
                known_symptoms=known_present,
                limit=batch_size,
                history_index=history_index,
            )
        else:
            discriminators = unknown_missing[:batch_size]

        questions = [self._to_natural_question(s) for s in discriminators]
        state["pending_questions"] = questions

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[diagnoser] pending_questions: {questions}"
        })
        return state
