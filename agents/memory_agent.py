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
    """
    MemoryAgent responsibilities (INTENTIONALLY LIMITED):

    ✔ Assign / reuse patient_id
    ✔ Load historical patient data
    ✔ Persist completed visits
    ✔ Maintain longitudinal symptom timeline
    ✔ Maintain FAISS visit summaries

    """

    def __init__(self, llm=None, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.llm = llm
        self.embed_model = embed_model

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        self._ensure_db()

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.faiss_index_dir = str(FAISS_INDEX_DIR)

    # ----------------------------------------------------
    # DB INITIALIZATION
    # ----------------------------------------------------
    def _ensure_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

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
            last_seen TEXT
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

    # ----------------------------------------------------
    # PATIENT ID MANAGEMENT
    # ----------------------------------------------------
    def ensure_patient(self, state: Dict[str, Any]) -> str:
        pid = state.get("patient_id")
        if pid:
            log.info("Using existing patient_id=%s", pid)
            state.setdefault("debug", []).append(
                {"agent": "memory", "msg": f"Using existing patient_id {pid}"}
            )
            return pid

        pid = str(uuid.uuid4())
        state["patient_id"] = pid

        conn = self._get_conn()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()

        cur.execute(
            "INSERT OR IGNORE INTO patients (patient_id, profile_json, created_at, updated_at) "
            "VALUES (?,?,?,?)",
            (pid, json.dumps({}), now, now),
        )

        conn.commit()
        conn.close()

        log.info("Created new patient_id=%s", pid)
        state.setdefault("debug", []).append(
            {"agent": "memory", "msg": f"Created new patient_id {pid}"}
        )

        return pid

    # ----------------------------------------------------
    # LOAD HISTORY
    # ----------------------------------------------------
    def load_patient_history(self, patient_id: str) -> Dict[str, Any]:
        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("SELECT profile_json FROM patients WHERE patient_id = ?", (patient_id,))
        row = cur.fetchone()
        profile = json.loads(row[0]) if row and row[0] else {}

        cur.execute(
            """
            SELECT symptom, value, chronic, first_seen, last_seen
            FROM patient_symptoms
            WHERE patient_id = ?
            """,
            (patient_id,),
        )

        rows = cur.fetchall()
        known_symptoms = {
            sym: {
                "value": val,
                "chronic": bool(chronic),
                "first_seen": first,
                "last_seen": last,
            }
            for sym, val, chronic, first, last in rows
        }

        cur.execute(
            """
            SELECT visit_json, created_at
            FROM visit_history
            WHERE patient_id = ?
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (patient_id,),
        )

        visits = []
        for vjson, created in cur.fetchall():
            try:
                visits.append(json.loads(vjson))
            except Exception:
                visits.append({"raw": vjson, "created_at": created})

        conn.close()

        log.info(
            "Loaded history for %s: %d symptoms, %d visits",
            patient_id,
            len(known_symptoms),
            len(visits),
        )

        return {
            "profile": profile,
            "known_symptoms": known_symptoms,
            "visits": visits,
        }

    # ----------------------------------------------------
    # PERSIST VISIT
    # ----------------------------------------------------
    def persist_visit(self, patient_id: str, state: Dict[str, Any]):
        """
        Persist ONLY completed visits.
        A visit is considered complete when diagnosis_result exists.
        """
        conn = self._get_conn()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()

        visit = {
            "timestamp": now,
            "symptoms": state.get("symptoms", {}),
            "ranked_candidates": state.get("ranked_candidates", []),
            "messages": state.get("messages", []),
            "final": state.get("diagnosis_result", []),
        }

        cur.execute(
            "INSERT INTO visit_history (patient_id, visit_json, created_at) VALUES (?,?,?)",
            (patient_id, json.dumps(visit), now),
        )

        # Upsert symptom timeline
        for symptom, value in (state.get("symptoms") or {}).items():
            is_chronic = isinstance(value, str) and "chronic" in value.lower()

            cur.execute(
                "SELECT id FROM patient_symptoms WHERE patient_id=? AND symptom=?",
                (patient_id, symptom),
            )
            row = cur.fetchone()

            if row:
                cur.execute(
                    """
                    UPDATE patient_symptoms
                    SET value=?, chronic=?, last_seen=?
                    WHERE id=?
                    """,
                    (str(value), int(is_chronic), now, row[0]),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO patient_symptoms
                    (patient_id, symptom, value, chronic, first_seen, last_seen)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (patient_id, symptom, str(value), int(is_chronic), now, now),
                )

        conn.commit()
        conn.close()

        log.info("Persisted completed visit for %s", patient_id)
        state.setdefault("debug", []).append(
            {"agent": "memory", "msg": "visit persisted"}
        )

        # Append visit summary to FAISS
        summary = (
            f"Visit {now} | "
            f"symptoms={json.dumps(visit['symptoms'])} | "
            f"diagnosis={json.dumps(visit['final'])}"
        )

        doc = Document(
            page_content=summary,
            metadata={"patient_id": patient_id, "timestamp": now},
        )

        try:
            vs = FAISS.load_local(
                self.faiss_index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            vs.add_documents([doc])
        except Exception:
            vs = FAISS.from_documents([doc], self.embeddings)

        vs.save_local(self.faiss_index_dir)

    # ----------------------------------------------------
    # MAIN ENTRYPOINT
    # ----------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pid = self.ensure_patient(state)
        history = self.load_patient_history(pid)

        state["patient_history"] = history
        state["patient_id"] = pid
        state["symptoms"] = dict(state.get("symptoms") or {})

        # Persist ONLY if final diagnosis exists
        if state.get("diagnosis_result"):
            self.persist_visit(pid, state)

        return state
