"""
Full Memory Agent: session UUID + SQLite (structured) + FAISS (semantic)
- Uses SQLite for structured patient profiles + symptoms + visits
- Uses FAISS via langchain_community for semantic visit summaries storage
- If patient_id not present in state, will create one (session-based) and return it in state['patient_id']
- On finalization, writes visit summary and updates symptom timeline
"""

import sqlite3
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# For FAISS embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = Path("./memory_db.sqlite")
FAISS_INDEX_DIR = Path("./memory_faiss")

class MemoryAgent:
    def __init__(self, llm=None, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.llm = llm
        self.embed_model = embed_model
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_db()

        # Setup embeddings & vectorstore; index created lazily
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.faiss_index_dir = FAISS_INDEX_DIR
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # SQLite helpers
    # ---------------------------
    def _ensure_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # patient profiles
        cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            profile_json TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """)
        # symptom timeline
        cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            symptom TEXT,
            value TEXT,
            first_seen TEXT,
            last_updated TEXT
        )
        """)
        # visit history summaries
        cur.execute("""
        CREATE TABLE IF NOT EXISTS visit_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            visit_json TEXT,
            created_at TEXT
        )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self):
        return sqlite3.connect(DB_PATH)

    # ---------------------------
    # Patient helpers
    # ---------------------------
    def ensure_patient(self, state: Dict[str, Any]) -> str:
        pid = state.get("patient_id")
        if pid:
            return pid
        # create session-based id
        pid = str(uuid.uuid4())
        state["patient_id"] = pid
        # insert minimal profile row
        conn = self._get_conn()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()
        cur.execute("INSERT OR IGNORE INTO patients (patient_id, profile_json, created_at, updated_at) VALUES (?,?,?,?)",
                    (pid, json.dumps({}), now, now))
        conn.commit()
        conn.close()
        return pid

    def load_patient_history(self, patient_id: str) -> Dict[str, Any]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT profile_json FROM patients WHERE patient_id = ?", (patient_id,))
        row = cur.fetchone()
        profile = json.loads(row[0]) if row and row[0] else {}
        # load symptom timeline
        cur.execute("SELECT symptom, value, first_seen, last_updated FROM patient_symptoms WHERE patient_id = ?", (patient_id,))
        rows = cur.fetchall()
        known_symptoms = {}
        for sym, val, first, last in rows:
            known_symptoms[sym] = {"value": val, "first_seen": first, "last_updated": last}
        conn.close()
        return {"profile": profile, "known_symptoms": known_symptoms}

    def persist_visit(self, patient_id: str, state: Dict[str, Any]):
        """
        Save session summary to visit_history + update patient_symptoms table + append semantic summary to FAISS.
        """
        conn = self._get_conn()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()
        visit = {
            "timestamp": now,
            "symptoms": state.get("symptoms", {}),
            "ranked_candidates": state.get("ranked_candidates", []),
            "messages": state.get("messages", []),
            "final": state.get("diagnosis_result", [])
        }
        cur.execute("INSERT INTO visit_history (patient_id, visit_json, created_at) VALUES (?,?,?)",
                    (patient_id, json.dumps(visit), now))
        # update symptom timeline (upsert-like)
        for s, v in (state.get("symptoms") or {}).items():
            cur.execute("SELECT id FROM patient_symptoms WHERE patient_id=? AND symptom=?", (patient_id, s))
            r = cur.fetchone()
            if r:
                cur.execute("UPDATE patient_symptoms SET value=?, last_updated=? WHERE id=?", (str(v), now, r[0]))
            else:
                cur.execute("INSERT INTO patient_symptoms (patient_id, symptom, value, first_seen, last_updated) VALUES (?,?,?,?,?)",
                            (patient_id, s, str(v), now, now))
        conn.commit()
        conn.close()

        # Add semantic summary to FAISS vectorstore
        # create a short summary text
        summary = f"Visit {now} - symptoms: {json.dumps(visit['symptoms'])} - diagnosis: {json.dumps(visit['final'])}"
        # load/create local FAISS index
        try:
            vs = FAISS.load_local(str(self.faiss_index_dir), self.embeddings, allow_dangerous_deserialization=True)
        except Exception:
            # no index yet -> create from one doc
            from langchain_core.documents import Document
            doc = Document(page_content=summary, metadata={"patient_id": patient_id, "timestamp": now})
            vs = FAISS.from_documents([doc], self.embeddings)
            vs.save_local(str(self.faiss_index_dir))
            return
        # add doc
        from langchain_core.documents import Document
        doc = Document(page_content=summary, metadata={"patient_id": patient_id, "timestamp": now})
        vs.add_documents([doc])
        vs.save_local(str(self.faiss_index_dir))

    # ---------------------------
    # run() (called by Graph at memory node)
    # ---------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        If called mid-graph, this ensures patient_id exists and loads patient history into state.
        If called at the end, persist current visit.
        Expected call patterns:
          - At start: provide state with no patient_id -> memory.run injects patient_id + patient_history
          - At end: call memory.run with state containing final diagnosis -> memory persists visit
        """
        # Ensure patient id
        pid = self.ensure_patient(state)
        # Load and attach patient history into state
        history = self.load_patient_history(pid)
        state["patient_id"] = pid
        state["patient_history"] = history
        # If final result present, persist
        if state.get("diagnosis_result") or state.get("ranked_candidates"):
            try:
                self.persist_visit(pid, state)
                state.setdefault("messages", []).append("[memory] visit persisted")
            except Exception as e:
                state.setdefault("messages", []).append(f"[memory] persist failed: {e}")
        return state