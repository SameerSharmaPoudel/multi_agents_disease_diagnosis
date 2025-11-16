from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from rag.symptom_retriever import SymptomRetriever  # your existing retriever

class SymptomAnalyzerAgent:
    """
    AnalyzerAgent:
    - expects state['symptoms'] (dict symptom -> value)
    - returns state['candidates'] (list of dicts) and state['missing_symptoms'] (list)
    - also appends a trace message in state['messages']
    """

    def __init__(self, index_dir: str = "./indices/symptoms_faiss", hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.retriever = SymptomRetriever(index_dir=index_dir, hf_model=hf_model)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = state.get("symptoms", {})
        messages = state.get("messages", [])
        if not isinstance(symptoms, dict) or not symptoms:
            msg = AIMessage(content="No symptoms found; please provide symptoms.")
            state["messages"] = [*messages, msg]
            state["candidates"] = []
            state["missing_symptoms"] = []
            return state

        # Retrieve candidate diseases (overfetch)
        results = self.retriever.retrieve(symptoms_dict=symptoms, top_k=8, rerank_by_jaccard=True)

        # convert results into candidates shape
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

        # Provide follow-up suggestions as raw symptom keys (not full NL)
        state["candidates"] = candidates
        state["missing_symptoms"] = sorted(list(missing_union))
        top_line = ", ".join([f"{c['disease']} (J={c['jaccard']:.2f})" for c in candidates[:3]]) if candidates else "no candidates yet"
        state.setdefault("messages", []).append(f"[analyzer] top candidates: {top_line}")
        return state