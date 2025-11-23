# agents/memory_agent.py
import sqlite3
import uuid
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from utils.logging_config import get_logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

log = get_logger("MemoryAgent")

DB_PATH = Path("./memory_db.sqlite")
FAISS_INDEX_DIR = Path("./memory_faiss")

class MemoryAgent:
    def __init__(self, llm=None, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.llm = llm
        self.embed_model = embed_model
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_db()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.faiss_index_dir = str(FAISS_INDEX_DIR)

    def _ensure_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # add a 'chronic' column to patient_symptoms to indicate longitudinal conditions
        cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            profile_json TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            symptom TEXT,
            value TEXT,
            chronic INTEGER DEFAULT 0,
            first_seen TEXT,
            last_updated TEXT
        )
        """)
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

    def ensure_patient(self, state: Dict[str, Any]) -> str:
        pid = state.get("patient_id")
        if pid:
            log.info(f"Using provided patient_id: {pid}")
            return pid
        pid = str(uuid.uuid4())
        state["patient_id"] = pid
        conn = self._get_conn()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()
        cur.execute("INSERT OR IGNORE INTO patients (patient_id, profile_json, created_at, updated_at) VALUES (?,?,?,?)",
                    (pid, json.dumps({}), now, now))
        conn.commit()
        conn.close()
        log.info(f"Created new patient_id: {pid}")
        return pid

    def load_patient_history(self, patient_id: str) -> Dict[str, Any]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT profile_json FROM patients WHERE patient_id = ?", (patient_id,))
        row = cur.fetchone()
        profile = json.loads(row[0]) if row and row[0] else {}

        cur.execute("SELECT symptom, value, chronic, first_seen, last_updated FROM patient_symptoms WHERE patient_id = ?", (patient_id,))
        rows = cur.fetchall()
        known_symptoms = {}
        for sym, val, chronic, first, last in rows:
            known_symptoms[sym] = {
                "value": val,
                "chronic": bool(chronic),
                "first_seen": first,
                "last_updated": last
            }

        cur.execute("SELECT visit_json, created_at FROM visit_history WHERE patient_id = ? ORDER BY created_at DESC LIMIT 50", (patient_id,))
        visits = []
        for vjson, created in cur.fetchall():
            try:
                visits.append(json.loads(vjson))
            except Exception:
                visits.append({"raw": vjson, "created_at": created})

        conn.close()
        log.info("Loaded history for %s: %d symptoms, %d visits", patient_id, len(known_symptoms), len(visits))
        return {"profile": profile, "known_symptoms": known_symptoms, "visits": visits}

    def persist_visit(self, patient_id: str, state: Dict[str, Any]):
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
        try:
            cur.execute("INSERT INTO visit_history (patient_id, visit_json, created_at) VALUES (?,?,?)",
                        (patient_id, json.dumps(visit), now))
        except Exception as e:
            log.exception("Failed to insert visit_history: %s", e)

        # update symptom timeline (upsert-like)
        for s, v in (state.get("symptoms") or {}).items():
            # heuristic: if user says 'chronic' or the value contains 'chronic', mark chronic
            is_chronic = False
            try:
                if isinstance(v, str) and "chronic" in v.lower():
                    is_chronic = True
            except Exception:
                pass

            cur.execute("SELECT id FROM patient_symptoms WHERE patient_id=? AND symptom=?", (patient_id, s))
            r = cur.fetchone()
            if r:
                cur.execute("UPDATE patient_symptoms SET value=?, chronic=?, last_updated=? WHERE id=?", (str(v), int(is_chronic), now, r[0]))
            else:
                cur.execute("INSERT INTO patient_symptoms (patient_id, symptom, value, chronic, first_seen, last_updated) VALUES (?,?,?,?,?,?)",
                            (patient_id, s, str(v), int(is_chronic), now, now))
        conn.commit()
        conn.close()
        log.info("Persisted visit for %s", patient_id)

        # Append semantic summary to FAISS
        summary = f"Visit {now} - symptoms: {json.dumps(visit['symptoms'])} - diagnosis: {json.dumps(visit['final'])}"
        try:
            vs = FAISS.load_local(self.faiss_index_dir, self.embeddings, allow_dangerous_deserialization=True)
        except Exception:
            doc = Document(page_content=summary, metadata={"patient_id": patient_id, "timestamp": now})
            vs = FAISS.from_documents([doc], self.embeddings)
            vs.save_local(self.faiss_index_dir)
            log.info("Created FAISS index and saved first doc")
            return
        doc = Document(page_content=summary, metadata={"patient_id": patient_id, "timestamp": now})
        vs.add_documents([doc])
        vs.save_local(self.faiss_index_dir)
        log.info("Appended visit summary to FAISS")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pid = self.ensure_patient(state)
        history = self.load_patient_history(pid)
        # merge historical symptoms, but do NOT override user-provided current session symptoms
        current_symptoms = dict(state.get("symptoms") or {})
        hist_symptoms = {k: {"value": v.get("value"), "chronic": v.get("chronic"), "first_seen": v.get("first_seen"), "last_seen": v.get("last_updated")} for k, v in (history.get("known_symptoms") or {}).items()}
        # attach but don't override: collector will take current session inputs
        state["patient_history"] = history
        # ensure state['symptoms'] exists (may be empty)
        state["symptoms"] = current_symptoms
        state["patient_id"] = pid

        # if final results present then persist (end of flow)
        if state.get("diagnosis_result") or state.get("ranked_candidates"):
            try:
                self.persist_visit(pid, state)
                state.setdefault("messages", []).append({"agent": "memory", "content": "visit persisted"})
            except Exception as e:
                log.exception("Persist failed: %s", e)
                state.setdefault("messages", []).append({"agent": "memory", "content": f"persist failed: {e}"})
        return state
