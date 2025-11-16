from typing import Dict, List, Any
import numpy as np

class DifferentialDiagnosisAgent:
    """
    Single-pass differential agent compatible with LangGraph:
    - Reads state['candidates'] and state['symptoms']
    - Writes state['ranked_candidates'], state['pending_questions'] (NL), state['uncertainty']
    - Uses hybrid ranking as fallback
    - Implements batching logic for follow-up questions based on missing symptoms
    """

    def __init__(self, llm=None, confidence_threshold: float = 0.8):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def _hybrid_rank(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        vecs = [c.get("vector_score") or 0.0 for c in candidates]
        min_v, max_v = min(vecs), max(vecs)
        def norm(v):
            return 0.0 if max_v - min_v == 0 else (v - min_v) / (max_v - min_v)
        scored = []
        for c in candidates:
            j = c.get("jaccard") or 0.0
            vs = norm(c.get("vector_score") or 0.0)
            score = 0.6 * j + 0.4 * vs
            scored.append({
                "disease": c.get("disease"),
                "likelihood": float(np.clip(score, 0.0, 1.0)),
                "reason": "hybrid(jaccard+vec)",
                "matched_symptoms": c.get("matched_symptoms", []),
                "missing_symptoms": c.get("missing_symptoms", [])
            })
        scored.sort(key=lambda x: x["likelihood"], reverse=True)
        return scored

    def _select_discriminators(self, candidates: List[Dict[str, Any]], known_symptoms: set, limit: int):
        """
        Score missing symptoms by frequency among candidates and by candidate weight.
        Return top 'limit' symptom keys (snake_case).
        """
        # weight each candidate by its jaccard (if present) or fallback equal weight
        weights = {}
        for c in candidates:
            w = c.get("jaccard") or 0.0
            for s in c.get("missing_symptoms", []):
                if s in known_symptoms:
                    continue
                weights[s] = weights.get(s, 0.0) + w + 0.01  # small base
        # sort by weight and return top
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_items[:limit]]

    def _to_natural_question(self, symptom_key: str) -> str:
        # transform snake_case into readable question
        return f"Do you have {symptom_key.replace('_',' ')}?"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        candidates = state.get("candidates", [])
        symptoms = state.get("symptoms", {})
        messages = state.get("messages", [])

        # produce ranked candidates (hybrid)
        ranked = self._hybrid_rank(candidates)

        top_likelihood = ranked[0]["likelihood"] if ranked else 0.0
        state["ranked_candidates"] = ranked
        state["top_likelihood"] = float(top_likelihood)
        state["uncertainty"] = 1.0 - float(top_likelihood)

        # If top_likelihood already high enough, we won't ask follow-ups
        if top_likelihood >= self.confidence_threshold:
            state["pending_questions"] = []
            state.setdefault("messages", []).append(f"[diagnoser] confident top {ranked[0]['disease']} ({top_likelihood:.2f})")
            return state

        # Build missing symptom candidates (unknown ones)
        missing = state.get("missing_symptoms", []) or []
        # remove symptoms already answered in session memory
        known_present = {k for k, v in (symptoms or {}).items() if str(v).strip().lower() not in {"no","none","absent","0","false",""}}
        unknown_missing = [s for s in missing if s not in known_present]

        # Batching logic
        n_missing = len(unknown_missing)
        if n_missing <= 3:
            batch_size = n_missing
        elif 4 <= n_missing <= 10:
            batch_size = 3
        else:
            # choose top weighted discriminators (limit 5)
            batch_size = 5

        # choose symptoms to ask based on weights if too many
        if n_missing > batch_size:
            discriminators = self._select_discriminators(candidates, known_present, batch_size)
        else:
            discriminators = unknown_missing[:batch_size]

        # Construct NL questions
        questions = [self._to_natural_question(s) for s in discriminators]

        state["pending_questions"] = questions
        state.setdefault("messages", []).append(f"[diagnoser] pending_questions: {questions}")
        return state