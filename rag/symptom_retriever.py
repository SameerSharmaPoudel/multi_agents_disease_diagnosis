import os
from typing import Dict, List, Tuple
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
    Loads a FAISS vector store built from the symptom CSV and provides:
      - vector retrieval
      - optional one-hot set-based re-ranking (Jaccard)
    """
    def __init__(self, index_dir: str, hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=hf_model)
        self.vs = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)

    @staticmethod
    def _symptoms_from_dict(symptoms: Dict[str, str]) -> List[str]:
        """
        Given a {symptom: value} dict from the collector, return the list of 'present' symptoms.
        We treat any non-empty string as present; adapt if you store severity/booleans differently.
        """
        present = []
        for k, v in symptoms.items():
            if v is None:
                continue
            # mark as present if non-empty and not explicitly 'no'
            sv = str(v).strip().lower()
            if sv and sv not in {"no", "none", "absent", "false", "0"}:
                present.append(k)
        return present

    def _make_query_text(self, positive_symptoms: List[str]) -> str:
        if not positive_symptoms:
            return "Symptoms: none"
        return "Symptoms: " + ", ".join(sorted(positive_symptoms))

    def retrieve(
        self,
        symptoms_dict: Dict[str, str],
        top_k: int = 8,
        rerank_by_jaccard: bool = True
    ) -> List[Dict]:
        """
        Returns a list of dicts: {disease, score, matched_symptoms, missing_symptoms, metadata}
        """
        positive = set(self._symptoms_from_dict(symptoms_dict))
        query = self._make_query_text(list(positive))

        docs = self.vs.similarity_search(query, k=top_k * 2)  # overfetch, then rerank

        # Build scored items
        items = []
        for d in docs:
            meta = d.metadata or {}
            candidate = set(meta.get("positive_symptoms", []))
            jac = jaccard(positive, candidate)
            items.append({
                "disease": meta.get("disease"),
                "vector_score": getattr(d, "score", None),
                "jaccard": jac,
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

    @staticmethod
    def suggest_discriminative_questions(results: List[Dict], max_questions: int = 5) -> List[str]:
        """
        From the top candidates, collect symptoms that are present in some diseases but not all,
        and that the user hasn't confirmed yet. These become 'ask-next' questions.
        """
        # symptoms that appear among candidates (union)
        union = set()
        # symptoms common to all candidates (intersection)
        intersection = None
        for r in results:
            s = set(r.get("missing_symptoms", [])) | set(r.get("matched_symptoms", []))
            union |= s
            if intersection is None:
                intersection = s.copy()
            else:
                intersection &= s

        # Discriminators = in union but not in intersection
        discriminators = sorted(list(union - (intersection or set())))
        questions = [f"Do you have {sym.replace('_', ' ')}?" for sym in discriminators[:max_questions]]
        return questions