from typing import Dict, Optional
from langchain_core.messages import AIMessage
from rag.symptom_retriever import SymptomRetriever

class SymptomAnalyzerAgent:
    """
    LangGraph/LangChain-friendly node:
      input state:  expects `state["symptoms"]` from the Interviewer/Collector.
      output state: adds `candidate_diseases` and a brief message.
    """
    def __init__(self, index_dir: str = "./indices/symptoms_faiss"):
        self.retriever = SymptomRetriever(index_dir=index_dir)

    def run(self, state: Dict) -> Dict:
        # Pull symptoms dict populated by your Symptom Collector Agent
        symptoms = state.get("symptoms", {})
        if not isinstance(symptoms, dict) or not symptoms:
            msg = AIMessage(content="I couldn't find any symptoms to analyze yet. Please complete the interview first.")
            return {**state, "messages": [*state.get("messages", []), msg], "candidate_diseases": []}

        # Retrieve and rank
        results = self.retriever.retrieve(symptoms_dict=symptoms, top_k=5, rerank_by_jaccard=True)

        # Prepare compact payload for downstream agents
        candidates = []
        for r in results:
            candidates.append({
                "disease": r["disease"],
                "jaccard": r["jaccard"],
                "matched_symptoms": r["matched_symptoms"],
                "missing_symptoms": r["missing_symptoms"],
                "row_id": r["metadata"].get("row_id")
            })

        # Suggest next best questions for interviewer (optional loop-back)
        ask_next = self.retriever.suggest_discriminative_questions(results)

        # Add a short AI message (optional, for traceability)
        top_line = ", ".join([f"{c['disease']} (J={c['jaccard']:.2f})" for c in candidates[:3]]) or "no candidates yet"
        msg = AIMessage(content=f"I analyzed your symptoms and found these candidate diseases: {top_line}.\n"
                                f"To refine, consider asking: {ask_next if ask_next else 'no further questions needed.'}")

        # Return enriched state
        new_state = {
            **state,
            "messages": [*state.get("messages", []), msg],
            "candidate_diseases": candidates,
            "followup_questions": ask_next
        }
        return new_state