from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from orchestrator import DiagnosisOrchestrator


def load_symptom_vectorstore(index_dir: str = "./indices/symptoms_faiss"):
    """Load FAISS index built from symptom dataset."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS symptom index from: {index_dir}")
    return vs


def main():
    # 1️⃣ Load your RAG vectorstore
    rag_vectorstore = load_symptom_vectorstore()

    # 2️⃣ Initialize the orchestrator
    orchestrator = DiagnosisOrchestrator(model_provider="groq", rag_vectorstore=rag_vectorstore)

    # 3️⃣ Collect user symptoms interactively (or use hardcoded ones for testing)
    print("\n🩺 Welcome to the AI Diagnostic Assistant!")
    print("Enter your symptoms one by one (type 'done' to finish):\n")

    user_symptoms = {}
    while True:
        symptom = input("Symptom: ").strip().lower()
        if symptom == "done":
            break
        user_symptoms[symptom] = "yes"

    if not user_symptoms:
        print("⚠️ No symptoms entered. Exiting.")
        return

    # 4️⃣ Run the orchestrator
    print("\n🚀 Running diagnostic workflow...\n")
    result = orchestrator.run(user_symptoms)

    # 5️⃣ Display the results
    print("\n=== Diagnosis Summary ===")
    diagnosis = result.get("diagnosis_result", [])
    followups = result.get("follow_up_questions", [])

    print(f"Top-ranked conditions: {diagnosis}")
    if followups:
        print("\nRecommended follow-up questions:")
        for q in followups:
            print(f" - {q}")


if __name__ == "__main__":
    main()