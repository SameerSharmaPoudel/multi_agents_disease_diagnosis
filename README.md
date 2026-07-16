# 🧠 Multi-Agents Disease Diagnostic System  
A LangGraph-powered, RAG-enhanced, multi-turn diagnosis engine with long-term memory.

This system uses multiple specialized agents (Symptom Collector, Analyzer, Differential Diagnoser, Explainer, Memory) orchestrated through **LangGraph** with **Human in the loop** for follow-up questions.  
It performs iterative reasoning until diagnostic uncertainty is reduced, while also saving patient history using **SQLite + FAISS** long-term memory.

---

# 🚀 System Overview

This project enables:

- Multi-turn diagnostic reasoning  
- Dynamic follow-up questions using discriminative symptom selection  
- RAG-enhanced ranking of candidate diseases  
- Hybrid fallback scoring (Jaccard + vector similarity)  
- Long-term patient memory with embedding search  
- LangGraph interrupts that pause and resume the pipeline  
- Session-based patient identity management

  # 🔄 Usage / Flow Summary
- User sends free-text symptoms to backend UI.
- Backend calls DiagnosisOrchestrator.start_session(user_text).
  - Graph runs:
      - memory_load (ensures patient_id, loads patient_history if any)
      - collector (extracts symptoms)
      - analyzer (RAG retrieval & missing symptom union)
      - diagnoser (computes pending_questions & uncertainty)
  - If diagnoser sets pending_questions, the graph pauses at ask_user interrupt.
    - state['pending_questions'] contains NL questions (batch per your batching logic).
- Frontend collects answers:
  - If batching, send back as dict mapping question->answer (or symptom_key->answer).
  - If one-by-one, send string answer for first question.
- Backend calls DiagnosisOrchestrator.resume_session_with_answer(state, user_response).
  - Graph resumes, update_symptoms maps answer(s) into state['symptoms'].
  - Graph loops back to analyzer and diagnoser.
- This repeats until diagnoser produces no pending_questions (confidence reached or no discriminators), then graph moves to explainer and memory_persist.


# Run

- Run Backend
  -- uv run uvicorn api.main:app --env-file .env.backend 

- Run Frontend
  -- uv run streamlit run streamlit_app/app.py
