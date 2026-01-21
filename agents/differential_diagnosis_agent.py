# agents/differential_diagnosis_agent.py

from typing import Dict, List, Any
import numpy as np


class DifferentialDiagnosisAgent:
    """
    Differential diagnosis agent (Option A compliant):

    - NO status control (GraphBuilder decides)
    - YES/NO answers are both treated as resolved
    - pending_questions only contains truly unknown symptoms
    """

    def __init__(
        self,
        llm=None,
        confidence_threshold: float = 0.8,
        history_weight: float = 0.15,
    ):
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

        vecs = [float(c.get("vector_score") or 0.0) for c in candidates]
        min_v, max_v = min(vecs), max(vecs)

        def norm(v: float) -> float:
            return 0.0 if max_v == min_v else (v - min_v) / (max_v - min_v)

        scored = []
        for c in candidates:
            j = float(c.get("jaccard") or 0.0)
            vs = norm(float(c.get("vector_score") or 0.0))

            hist_bonus = sum(
                history_index.get(s, 0.0)
                for s in c.get("matched_symptoms", []) or []
            )

            score = 0.6 * j + 0.4 * vs + hist_bonus
            score = float(np.clip(score, 0.0, 1.0))

            scored.append({
                "disease": c.get("disease"),
                "likelihood": score,
                "reason": "hybrid(jaccard+vector+history)",
                "matched_symptoms": c.get("matched_symptoms", []) or [],
                "missing_symptoms": c.get("missing_symptoms", []) or [],
            })

        scored.sort(key=lambda x: x["likelihood"], reverse=True)
        return scored

    # -----------------------------------------------------------
    # QUESTION SELECTION
    # -----------------------------------------------------------
    def _select_discriminators(
        self,
        candidates: List[Dict[str, Any]],
        resolved_symptoms: set,
        limit: int,
        history_index: Dict[str, float],
    ) -> List[str]:
        weights: Dict[str, float] = {}

        for c in candidates:
            base = float(c.get("jaccard") or 0.0)
            for s in c.get("missing_symptoms", []) or []:
                if s in resolved_symptoms:
                    continue

                w = base + 0.01
                if s in history_index:
                    w += history_index[s] * 2.0

                weights[s] = weights.get(s, 0.0) + w

        ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in ranked[:limit]]

    def _to_question(self, key: str) -> str:
        return f"Do you have {key.replace('_', ' ')}?"

    # -----------------------------------------------------------
    # MAIN RUN
    # -----------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        candidates = state.get("candidates", []) or []
        symptoms = state.get("symptoms", {}) or {}

        # -------------------------
        # Build history index
        # -------------------------
        history_index: Dict[str, float] = {}

        for src in (state.get("historical_symptoms"), state.get("relevant_history")):
            if isinstance(src, dict):
                for k, meta in src.items():
                    if isinstance(meta, dict):
                        history_index[k] = self.history_weight * (
                            2.0 if meta.get("chronic") else 1.0
                        )
            elif isinstance(src, list):
                for e in src:
                    if isinstance(e, dict) and e.get("symptom"):
                        history_index[e["symptom"]] = self.history_weight * (
                            2.0 if e.get("chronic") else 1.0
                        )

        state.setdefault("debug", []).append({
            "agent": "diagnoser",
            "num_candidates": len(candidates),
            "history_index_keys": list(history_index.keys()),
        })

        # -------------------------
        # Rank candidates
        # -------------------------
        ranked = self._hybrid_rank(candidates, history_index)
        state["ranked_candidates"] = ranked
        state["top_likelihood"] = ranked[0]["likelihood"] if ranked else 0.0
        state["uncertainty"] = 1.0 - state["top_likelihood"]

        # -------------------------
        # Confident → no questions
        # -------------------------
        if ranked and state["top_likelihood"] >= self.confidence_threshold:
            state["pending_questions"] = []
            return state

        # -------------------------
        # Need more info
        # -------------------------
        missing = state.get("missing_symptoms", []) or []

        # 🔑 YES and NO both count as resolved
        resolved = set(symptoms.keys())

        unknown = [s for s in missing if s not in resolved]

        if not unknown:
            state["pending_questions"] = []
            return state

        batch_size = min(3, len(unknown))

        discriminators = self._select_discriminators(
            candidates=candidates,
            resolved_symptoms=resolved,
            limit=batch_size,
            history_index=history_index,
        )

        state["pending_questions"] = [self._to_question(s) for s in discriminators]
        return state
