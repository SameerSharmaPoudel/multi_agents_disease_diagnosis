# tests/test_differential_diagnosis_agent.py
import pytest
from agents.differential_diagnosis_agent import DifferentialDiagnosisAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_candidates():
    """Basic candidates for ranking tests."""
    return [
        {
            "disease": "flu",
            "jaccard": 0.6,
            "vector_score": 0.5,
            "matched_symptoms": ["fever", "cough"],
            "missing_symptoms": ["fatigue"]
        },
        {
            "disease": "cold",
            "jaccard": 0.4,
            "vector_score": 0.3,
            "matched_symptoms": ["cough"],
            "missing_symptoms": ["fever"]
        },
        {
            "disease": "covid",
            "jaccard": 0.5,
            "vector_score": 0.7,
            "matched_symptoms": ["fever"],
            "missing_symptoms": ["shortness_of_breath"]
        }
    ]


@pytest.fixture
def history_entries():
    """Baseline historical symptoms."""
    return [
        {"symptom": "fever", "chronic": False},
        {"symptom": "cough", "chronic": True},   # chronic should get a double weight
    ]


@pytest.fixture
def agent():
    """Agent with low threshold to avoid auto-confidence in some tests."""
    return DifferentialDiagnosisAgent(confidence_threshold=0.8, history_weight=0.2)


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------

def test_ranking_without_history(agent, simple_candidates):
    """Ensure ranking works without history."""
    history_index = {}
    ranked = agent._hybrid_rank(simple_candidates, history_index)

    assert len(ranked) == 3
    assert ranked[0]["likelihood"] >= ranked[1]["likelihood"]


def test_ranking_with_history_boost(agent, simple_candidates, history_entries):
    """History should boost diseases with overlapping matched symptoms."""
    history_index = {
        "fever": 0.2,
        "cough": 0.4   # chronic → doubled weight → 0.2*2=0.4
    }

    ranked = agent._hybrid_rank(simple_candidates, history_index)

    # "flu" has matched ["fever", "cough"] → gets the largest boost
    assert ranked[0]["disease"] == "flu"


def test_chronic_history_has_stronger_effect(agent, simple_candidates):
    """Chronic history produces a larger boost than non-chronic."""
    non_chronic_history = {"fever": agent.history_weight}
    chronic_history = {"fever": agent.history_weight * 2.0}

    ranked_nc = agent._hybrid_rank(simple_candidates, non_chronic_history)
    ranked_c = agent._hybrid_rank(simple_candidates, chronic_history)

    flu_nc = [r for r in ranked_nc if r["disease"] == "flu"][0]["likelihood"]
    flu_c = [r for r in ranked_c if r["disease"] == "flu"][0]["likelihood"]

    assert flu_c > flu_nc


# ---------------------------------------------------------------------------
# Confidence branch
# ---------------------------------------------------------------------------

def test_confident_diagnosis_skips_questions(agent, simple_candidates):
    state = {
        "candidates": simple_candidates,
        "symptoms": {"fever": "yes"},
        "relevant_history": [],
        "missing_symptoms": [],
    }

    # Force top candidate to have extremely high probability
    for c in simple_candidates:
        if c["disease"] == "flu":
            c["jaccard"] = 1.0
            c["vector_score"] = 1.0

    out = agent.run(state)

    assert out["pending_questions"] == []
    assert "confident top" in out["messages"][-1]


# ---------------------------------------------------------------------------
# Missing symptom and question generation
# ---------------------------------------------------------------------------

def test_followup_questions_generated(agent, simple_candidates):
    state = {
        "candidates": simple_candidates,
        "symptoms": {"fever": "yes"},
        "missing_symptoms": ["fatigue", "nausea"],
        "relevant_history": [],
    }

    out = agent.run(state)

    # Should ask 2 questions
    assert len(out["pending_questions"]) == 2
    assert all(q.startswith("Do you have") for q in out["pending_questions"])


def test_question_batching_large_missing(agent):
    candidates = [{
        "disease": "flu",
        "jaccard": 0.5,
        "vector_score": 0.4,
        "matched_symptoms": ["fever"],
        "missing_symptoms": [f"s{i}" for i in range(20)]
    }]

    state = {
        "candidates": candidates,
        "symptoms": {"fever": "yes"},
        "missing_symptoms": [f"s{i}" for i in range(20)],
        "relevant_history": [],
    }

    out = agent.run(state)

    # For >10 missing → batch size = 5
    assert len(out["pending_questions"]) == 5


# ---------------------------------------------------------------------------
# History-aware discriminator selection
# ---------------------------------------------------------------------------

def test_history_boosts_discriminator_choice(agent):
    candidates = [
        {
            "disease": "flu",
            "jaccard": 0.6,
            "vector_score": 0.3,
            "matched_symptoms": [],
            "missing_symptoms": ["fatigue", "cough"]
        }
    ]

    known = set()
    history_index = {"cough": 0.5}   # cough should dominate

    discriminators = agent._select_discriminators(
        candidates=candidates,
        known_symptoms=known,
        limit=1,
        history_index=history_index
    )

    assert discriminators == ["cough"]  # History-chronic missing symptom chosen


# ---------------------------------------------------------------------------
# End-to-end behavior
# ---------------------------------------------------------------------------

def test_full_run_pipeline(agent, simple_candidates, history_entries):
    state = {
        "symptoms": {"fever": "yes"},
        "candidates": simple_candidates,
        "missing_symptoms": ["fatigue", "shortness_of_breath"],
        "relevant_history": history_entries,
    }

    out = agent.run(state)

    assert "ranked_candidates" in out
    assert "top_likelihood" in out
    assert "uncertainty" in out
    assert "pending_questions" in out
    assert isinstance(out["pending_questions"], list)
