# rag/symptom_retriever.py
import os
from typing import Dict, List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

class SymptomRetriever:
    """
    FAISS-backed retriever with optional small weighting for historical symptoms.
    """
    def __init__(self, index_dir: str, hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=hf_model)
        self.vs = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)

    @staticmethod
    def _symptoms_from_dict(symptoms: Dict[str, str]) -> List[str]:
        present = []
        for k, v in symptoms.items():
            if v is None:
                continue
            sv = str(v).strip().lower()
            if sv and sv not in {"no", "none", "absent", "false", "0"}:
                present.append(k)
        return present

    def _make_query_text(self, positive_symptoms: List[str], historical_symptoms: Optional[List[str]] = None) -> str:
        if not positive_symptoms and not historical_symptoms:
            return "Symptoms: none"
        parts = []
        if positive_symptoms:
            parts.append("Symptoms: " + ", ".join(sorted(positive_symptoms)))
        if historical_symptoms:
            parts.append("History: " + ", ".join(sorted(historical_symptoms)))
        return " | ".join(parts)

    def retrieve(
        self,
        symptoms_dict: Dict[str, str],
        historical_symptoms: Optional[Dict[str, str]] = None,
        top_k: int = 8,
        rerank_by_jaccard: bool = True,
        history_weight: float = 0.3
    ) -> List[Dict]:
        """
        symptoms_dict: combined mapping used as primary query
        historical_symptoms: subset of historical items (keys->values) used for soft scoring
        history_weight: how much historical matches contribute in hybrid jaccard
        """
        positive = set(self._symptoms_from_dict(symptoms_dict))
        historical_pos = set(self._symptoms_from_dict(historical_symptoms or {}))

        query = self._make_query_text(list(positive), list(historical_pos) if historical_pos else None)
        docs = self.vs.similarity_search(query, k=top_k * 2)

        items = []
        for d in docs:
            meta = d.metadata or {}
            candidate = set(meta.get("positive_symptoms", []))
            jac_current = jaccard(positive, candidate)
            jac_history = jaccard(historical_pos, candidate) if historical_pos else 0.0
            # hybrid jaccard: weight current more than history
            hybrid_jac = (1.0 - history_weight) * jac_current + (history_weight) * jac_history

            items.append({
                "disease": meta.get("disease"),
                "vector_score": getattr(d, "score", None),
                "jaccard": hybrid_jac,
                "matched_symptoms": sorted(list(positive & candidate)),
                "missing_symptoms": sorted(list(candidate - positive)),
                "metadata": meta,
                "raw_text": d.page_content
            })

        if rerank_by_jaccard:
            items.sort(key=lambda x: (x["jaccard"], x["vector_score"] or 0.0), reverse=True)
        else:
            items = items[:top_k]

        return items[:top_k]
