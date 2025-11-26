import sqlite3
import json
import pytest
from pathlib import Path

from agents import memory_agent as mem_mod
from agents.memory_agent import MemoryAgent
from langchain_community.vectorstores import FAISS


@pytest.fixture
def temp_memory_env(tmp_path, monkeypatch):
    """
    Redirect DB_PATH and FAISS_INDEX_DIR to temp directories.
    Returns a fresh MemoryAgent using those paths.
    """
    db_path = tmp_path / "test_memory_db.sqlite"
    faiss_dir = tmp_path / "faiss_idx"

    monkeypatch.setattr(mem_mod, "DB_PATH", db_path)
    monkeypatch.setattr(mem_mod, "FAISS_INDEX_DIR", faiss_dir)

    agent = MemoryAgent(llm=None)
    return agent, db_path, faiss_dir


# ------------------------------------------------------------
# 1. PATIENT CREATION & INITIAL HISTORY LOADING
# ------------------------------------------------------------
def test_memory_agent_creates_patient_and_history(temp_memory_env):
    agent, db_path, _ = temp_memory_env

    # First call initializes a session
    state = {}
    out = agent.run(state)

    assert "patient_id" in out
    pid = out["patient_id"]
    assert isinstance(pid, str)

    assert "patient_history" in out
    ph = out["patient_history"]
    assert isinstance(ph, dict)
    assert "known_symptoms" in ph
    assert "visits" in ph

    # DB should have 1 patient row
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patients")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 1


# ------------------------------------------------------------
# 2. RETURNING PATIENT DOES NOT CREATE NEW ID
# ------------------------------------------------------------
def test_existing_patient_id_respected(temp_memory_env):
    agent, db_path, _ = temp_memory_env

    # Create first session
    s1 = agent.run({})
    pid = s1["patient_id"]

    # Second call with same patient_id
    state = {"patient_id": pid}
    s2 = agent.run(state)

    assert s2["patient_id"] == pid

    # Ensure DB did not add new patient
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patients")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 1


# ------------------------------------------------------------
# 3. FIRST VISIT -> PERSIST + CHRONIC DETECTION
# ------------------------------------------------------------
def test_persist_visit_and_chronic_flag(temp_memory_env):
    agent, db_path, _ = temp_memory_env

    state = {
        "symptoms": {"fever": "chronic high"},
        "ranked_candidates": [{"disease": "flu", "likelihood": 0.9}],
        "diagnosis_result": [{"disease": "flu", "likelihood": 0.9}],
    }
    out = agent.run(state)
    pid = out["patient_id"]

    # Validate DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Visit row exists
    cur.execute("SELECT COUNT(*) FROM visit_history WHERE patient_id=?", (pid,))
    visit_count = cur.fetchone()[0]
    assert visit_count == 1

    # Chronic flag is 1
    cur.execute("SELECT symptom, chronic FROM patient_symptoms WHERE patient_id=?", (pid,))
    rows = cur.fetchall()
    conn.close()

    assert rows
    fever_row = [r for r in rows if r[0] == "fever"][0]
    assert fever_row[1] == 1  # chronic=True


# ------------------------------------------------------------
# 4. SYMPTOM UPDATE ACROSS VISITS
# ------------------------------------------------------------
def test_symptom_update_over_time(temp_memory_env):
    agent, db_path, _ = temp_memory_env

    # 1st visit
    s1 = {"symptoms": {"fever": "high"}}
    out1 = agent.run(s1)
    pid = out1["patient_id"]

    # overwrite / update symptoms in 2nd visit
    visit2 = {
        "patient_id": pid,
        "symptoms": {"fever": "mild", "cough": "yes"},
        "ranked_candidates": [{"disease": "cold", "likelihood": 0.7}],
        "diagnosis_result": [{"disease": "cold", "likelihood": 0.7}],
    }
    agent.run(visit2)

    # load history and check updated values
    history = agent.load_patient_history(pid)["known_symptoms"]

    assert history["fever"]["value"] == "mild"
    assert history["cough"]["value"] == "yes"
    assert "first_seen" in history["fever"]
    assert "last_seen" in history["fever"]


# ------------------------------------------------------------
# 5. VISIT JSON STRUCTURE IS VALID
# ------------------------------------------------------------
def test_visit_json_structure(temp_memory_env):
    agent, db_path, _ = temp_memory_env

    state = {
        "symptoms": {"headache": "severe"},
        "ranked_candidates": [{"disease": "migraine", "likelihood": 0.95}],
        "diagnosis_result": [{"disease": "migraine", "likelihood": 0.95}],
    }
    out = agent.run(state)
    pid = out["patient_id"]

    # read visit JSON
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT visit_json FROM visit_history WHERE patient_id=?", (pid,))
    row = cur.fetchone()
    conn.close()

    visit_json = json.loads(row[0])

    assert "symptoms" in visit_json
    assert "ranked_candidates" in visit_json
    assert "final" in visit_json
    assert visit_json["symptoms"]["headache"] == "severe"


# ------------------------------------------------------------
# 6. FAISS INDEX CREATION AND RETRIEVAL
# ------------------------------------------------------------
def test_faiss_index_updates(temp_memory_env):
    agent, _, faiss_dir = temp_memory_env

    state = {
        "symptoms": {"fever": "yes"},
        "ranked_candidates": [{"disease": "flu", "likelihood": 0.9}],
        "diagnosis_result": [{"disease": "flu", "likelihood": 0.9}],
    }
    out = agent.run(state)
    pid = out["patient_id"]

    # Index must exist now
    vs = FAISS.load_local(str(faiss_dir), agent.embeddings, allow_dangerous_deserialization=True)
    assert vs is not None

    # FAISS search for "fever"
    docs = vs.similarity_search("fever", k=1)
    assert len(docs) >= 1


# ------------------------------------------------------------
# 7. HISTORICAL SYMPTOMS RETURNED CORRECTLY
# ------------------------------------------------------------
def test_historical_symptoms_returned(temp_memory_env):
    agent, _, _ = temp_memory_env

    # Visit 1
    s1 = {"symptoms": {"fever": "mild"}}
    out1 = agent.run(s1)
    pid = out1["patient_id"]

    # Visit 2
    s2 = {
        "patient_id": pid,
        "symptoms": {"cough": "yes"},
        "ranked_candidates": [{"disease": "cold", "likelihood": 0.7}],
        "diagnosis_result": [{"disease": "cold", "likelihood": 0.7}],
    }
    agent.run(s2)

    # load history
    history = agent.load_patient_history(pid)
    known = history["known_symptoms"]

    assert "fever" in known
    assert "cough" in known
