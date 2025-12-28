import copy
import pytest

from workflow.run_graph import run_diagnosis_graph


@pytest.mark.integration
def test_end_to_end_with_final_diagnosis_and_memory_regression():
    """
    Verifies:
    1. Two-phase pause & resume works
    2. Diagnosis is deferred until sufficient information
    3. Final diagnosis is produced as a ranked list
    4. Memory persists ONLY after final diagnosis
    5. Memory does not regress or duplicate visits
    """

    # =========================================================
    # VISIT 1 — PHASE 1 (INCOMPLETE INPUT)
    # =========================================================
    initial_state = {
        "symptoms": {
            "itching": "yes",
            "skin_rash": "yes",
        },
        "messages": [],
        "debug": [],
    }

    state_phase1 = run_diagnosis_graph(copy.deepcopy(initial_state))

    # ---- Pause expected
    assert "pending_questions" in state_phase1
    assert state_phase1["pending_questions"]

    # ---- No diagnosis yet
    assert not state_phase1.get("diagnosis_result")

    patient_id = state_phase1.get("patient_id")
    assert patient_id is not None

    # =========================================================
    # VISIT 1 — PHASE 2 (PARTIAL FOLLOW-UP)
    # =========================================================
    resumed_state = copy.deepcopy(state_phase1)
    resumed_state["symptoms"].update(
        {
            "fever": "yes",
            "joint_pain": "no",
        }
    )

    state_phase2 = run_diagnosis_graph(resumed_state)

    # ---- Still deferred
    assert not state_phase2.get("diagnosis_result")
    assert "pending_questions" in state_phase2

    # =========================================================
    # VISIT 1 — PHASE 3 (FINAL FOLLOW-UP → DIAGNOSIS)
    # =========================================================
    final_state = copy.deepcopy(state_phase2)
    final_state["symptoms"].update(
        {
            "nodal_skin_eruptions": "yes",
            "spotting_urination": "no",
            "stomach_pain": "no",
        }
    )

    state_final = run_diagnosis_graph(final_state)

    # ---- Final diagnosis must exist
    diagnosis_list = state_final.get("diagnosis_result")

    assert diagnosis_list is not None
    assert isinstance(diagnosis_list, list)
    assert len(diagnosis_list) > 0

    top_diagnosis = diagnosis_list[0]

    assert isinstance(top_diagnosis, dict)
    assert "disease" in top_diagnosis
    assert "likelihood" in top_diagnosis

    # ---- Confidence sanity check (bounded)
    assert 0.0 <= top_diagnosis["likelihood"] <= 1.0

    # =========================================================
    # VISIT 2 — SESSION REVISIT (MEMORY LOAD)
    # =========================================================
    revisit_state = {
        "patient_id": patient_id,
        "symptoms": {
            "itching": "yes",
            "skin_rash": "yes",
        },
        "messages": [],
        "debug": [],
    }

    state_revisit = run_diagnosis_graph(revisit_state)

    # ---- Memory loaded
    assert "patient_history" in state_revisit
    history = state_revisit["patient_history"]

    assert "visits" in history
    assert len(history["visits"]) == 1  # exactly one completed visit

    stored_visit = history["visits"][0]

    assert "final" in stored_visit
    assert isinstance(stored_visit["final"], list)
    assert stored_visit["final"]

    stored_top = stored_visit["final"][0]
    assert stored_top["disease"] == top_diagnosis["disease"]

    # =========================================================
    # MEMORY REGRESSION CHECK
    # =========================================================
    # Re-running revisit must NOT create a new visit
    state_revisit_again = run_diagnosis_graph(revisit_state)

    history_again = state_revisit_again["patient_history"]
    assert len(history_again["visits"]) == 1
