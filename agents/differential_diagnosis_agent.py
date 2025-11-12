from typing import List, Dict, Any
import json
import numpy as np
from langchain.chains import RetrievalQA
from utils.rag_utils import build_rag_chain


class DifferentialDiagnosisAgent:
    """
    Differential Diagnosis Agent:
    - Takes top-ranked disease candidates from the SymptomAnalyzerAgent
    - Performs deeper reasoning using RAG
    - Generates discriminative follow-up questions
    - Iteratively refines the diagnosis through a feedback loop
    """

    def __init__(self, rag_vectorstore, llm, max_candidates: int = 5, max_rounds: int = 3, confidence_threshold: float = 0.8):
        self.qa_chain = build_rag_chain(rag_vectorstore, llm)
        self.max_candidates = max_candidates
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold

    # -------------------------------------------------------------------------
    # 1️⃣ Core reasoning and ranking
    # -------------------------------------------------------------------------
    def analyze(self, symptoms: Dict[str, Any], candidates: List[Dict]) -> Dict:
        """
        Rank candidate diseases and explain reasoning using RAG.
        """
        observed = [k for k, v in symptoms.items() if str(v).strip().lower() not in {"no", "none", "", "absent", "false"}]

        # Prepare context for reasoning
        context = ""
        for c in candidates[:self.max_candidates]:
            context += (
                f"Disease: {c['disease']}\n"
                f"Matched symptoms: {', '.join(c['matched_symptoms'])}\n"
                f"Missing symptoms: {', '.join(c['missing_symptoms'])}\n\n"
            )

        # RAG reasoning query
        query = (
            f"Given the observed symptoms: {', '.join(observed)}.\n"
            f"Analyze and rank the likelihood of the following diseases:\n{context}\n"
            f"Provide a ranked JSON list like this:\n"
            f"{{'ranked_candidates': [{{'disease': str, 'likelihood': float, 'reason': str}}]}}\n"
        )

        response = self.qa_chain.run(query)

        # Try to parse structured JSON
        try:
            parsed = json.loads(response)
        except Exception:
            parsed = {"ranked_candidates": []}

        # Fallback to hybrid ranking if LLM fails
        if not parsed["ranked_candidates"]:
            parsed["ranked_candidates"] = self._hybrid_rank(candidates)

        return parsed

    # -------------------------------------------------------------------------
    # 2️⃣ Hybrid fallback ranking (numeric + similarity-based)
    # -------------------------------------------------------------------------
    def _hybrid_rank(self, candidates: List[Dict]) -> List[Dict]:
        ranked = []
        for c in candidates:
            score = 0.6 * (c.get("jaccard") or 0) + 0.4 * (c.get("vector_score") or 0)
            ranked.append({
                "disease": c["disease"],
                "likelihood": float(np.clip(score, 0.0, 1.0)),
                "reason": "Based on symptom overlap and embedding similarity."
            })
        ranked.sort(key=lambda x: x["likelihood"], reverse=True)
        return ranked

    # -------------------------------------------------------------------------
    # 3️⃣ Follow-up question generation (discriminators)
    # -------------------------------------------------------------------------
    @staticmethod
    def suggest_discriminative_questions(results: List[Dict], max_questions: int = 5) -> List[str]:
        """
        Identify symptoms that are present in some diseases but not all.
        These are the best next questions to ask the user.
        """
        union, intersection = set(), None
        for r in results:
            s = set(r.get("missing_symptoms", [])) | set(r.get("matched_symptoms", []))
            union |= s
            intersection = s.copy() if intersection is None else intersection & s

        discriminators = sorted(list(union - (intersection or set())))
        return [f"Do you have {sym.replace('_', ' ')}?" for sym in discriminators[:max_questions]]

    # -------------------------------------------------------------------------
    # 4️⃣ Feedback loop: human-in-the-loop refinement
    # -------------------------------------------------------------------------
    def iterative_diagnosis(self, initial_symptoms: Dict[str, Any], candidates: List[Dict]) -> Dict:
        """
        Run an iterative refinement loop where follow-up questions are asked
        and responses are integrated into updated reasoning.
        """
        symptoms = initial_symptoms.copy()

        for round_id in range(1, self.max_rounds + 1):
            print(f"\n--- Iteration {round_id} ---")

            # Analyze with current symptom set
            result = self.analyze(symptoms, candidates)
            ranked = result.get("ranked_candidates", [])

            if not ranked:
                print("⚠️ No valid results. Using fallback ranking.")
                ranked = self._hybrid_rank(candidates)

            # Display top candidates
            top = ranked[0]
            print(f"Top predicted disease: {top['disease']} (confidence: {top['likelihood']:.2f})")

            # Check stopping criteria
            if top["likelihood"] >= self.confidence_threshold:
                print("✅ Confidence threshold reached. Stopping iterative loop.")
                break

            # Otherwise, generate discriminative follow-up questions
            follow_up = self.suggest_discriminative_questions(candidates)
            if not follow_up:
                print("⚠️ No discriminative questions left. Stopping.")
                break

            print("\nFollow-up questions:")
            for q in follow_up:
                print(" -", q)

            # Human-in-the-loop feedback simulation (can be replaced with UI input)
            for q in follow_up:
                ans = input(f"{q} (yes/no): ").strip().lower()
                symptom_name = q.replace("Do you have ", "").replace("?", "").replace(" ", "_")
                symptoms[symptom_name] = "yes" if ans.startswith("y") else "no"

            # Reanalyze with updated symptoms
            print("🔁 Updating diagnosis based on feedback...\n")

        result["final_symptoms"] = symptoms
        result["final_candidates"] = ranked
        return result