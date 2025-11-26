# tests/test_symptom_analyzer_agent.py
import pytest
from datetime import datetime, timedelta

from agents.symptom_analyzer_agent import SymptomAnalyzerAgent


# ---------------------------------------------------------------------
# Mock SymptomRetriever so tests do NOT require FAISS or embeddings
# ---------------------------------------------------------------------
class MockRetriever:
    """
    Mock retriever that simply returns a canned result and captures the input.
    This allows asserting how the analyzer merged history + current symptoms.
    """
    def __init__(self, return_items):
        self.return_items = return_items
        self.last_query = None

    def retrieve(self, symptoms_dict, historical_symptoms=None,
                 top_k=8, rerank_by_jaccard=True, history_weight=0.3):

        # capture query for inspection
        self.last_query = {
            "symptoms_dict": symptoms_dict,
            "historical_symptoms": historical_symptoms,
            "top_k": top_k,
            "rerank_by_jaccard": rerank_by_jaccard,
            "history_weight": history_weight,
        }

        return self.return_items


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def mock_results():
    """Simulated retriever outputs."""
    return [
        {
            "disease": "flu",
            "vector_score": 0.82,
            "jaccard": 0.5,
            "matched_symptoms": ["fever", "cough"],
            "missing_symptoms": ["fatigue"],
            "metadata": {"row_id": 1},
        }
    ]


@pytest.fixture
def analyzer(mock_results):
    mock = MockRetriever(return_items=mock_results)
    agent = SymptomAnalyzerAgent(retriever=mock)
    return agent


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_no_symptoms_return_message(analyzer):
    state = {"symptoms": {}}
    out = analyzer.run(state)

    assert out["candidates"] == []
    assert out["missing_symptoms"] == []
    assert "provide symptoms" in out["messages"][0].content


def test_relevant_history_chronic(analyzer):
    # chronic historical symptom must be used
    hist = {
        "asthma": {
            "value": "intermittent",
            "chronic": True,
            "first_seen": "2022-01-01T12:00",
            "last_seen": "2024-02-01T12:00",
        }
    }
    current = {"cough": "mild"}

    state = {"symptoms": current, "historical_symptoms": hist, "messages": []}
    out = analyzer.run(state)

    # Check that chronic history included
    query = analyzer.retriever.last_query["symptoms_dict"]
    assert "asthma" in query

    # Ensure message reflects used history
    assert "asthma" in out["messages"][0]["content"]


def test_relevant_history_same_key(analyzer):
    hist = {
        "fever": {
            "value": "low",
            "chronic": False,
            "first_seen": "2024-01-01T12:00",
            "last_seen": "2024-01-01T12:00",
        }
    }
    current = {"fever": "high"}  # same key

    state = {"symptoms": current, "historical_symptoms": hist, "messages": []}
    out = analyzer.run(state)

    # Should include fever because key matches, BUT must not override current
    query = analyzer.retriever.last_query["symptoms_dict"]
    assert query["fever"] == "high"  # current has priority

    # history should still be reported as "used"
    assert "fever" in out["messages"][0]["content"]


def test_relevant_history_recent(analyzer):
    recent_date = (datetime.utcnow() - timedelta(days=5)).isoformat()
    hist = {
        "fatigue": {
            "value": "mild",
            "chronic": False,
            "first_seen": recent_date,
            "last_seen": recent_date
        }
    }
    current = {"fever": "high"}

    state = {"symptoms": current, "historical_symptoms": hist, "messages": []}
    out = analyzer.run(state)

    # Should include fatigue due to recency
    query = analyzer.retriever.last_query["symptoms_dict"]
    assert "fatigue" in query


def test_irrelevant_history_ignored(analyzer):
    # old historical symptom - not chronic, not same key, not recent
    old_date = (datetime.utcnow() - timedelta(days=400)).isoformat()
    hist = {
        "headache": {
            "value": "mild",
            "chronic": False,
            "first_seen": old_date,
            "last_seen": old_date,
        }
    }
    current = {"fever": "high"}

    state = {"symptoms": current, "historical_symptoms": hist, "messages": []}
    out = analyzer.run(state)

    query = analyzer.retriever.last_query["symptoms_dict"]

    # headache should be excluded
    assert "headache" not in query


def test_candidates_and_missing_symptoms_built(analyzer, mock_results):
    state = {
        "symptoms": {"fever": "high"},
        "historical_symptoms": {},
        "messages": []
    }
    out = analyzer.run(state)

    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["disease"] == "flu"
    assert out["missing_symptoms"] == ["fatigue"]
    assert "top_candidates" in out["messages"][-1]["content"]


def test_combined_query_prioritizes_current(analyzer):
    hist = {
        "cough": {
            "value": "old_value",
            "chronic": False,
            "first_seen": "2024-01-01T12:00",
            "last_seen": "2024-01-01T12:00",
        }
    }
    current = {"cough": "new_value"}  # should override history

    state = {"symptoms": current, "historical_symptoms": hist, "messages": []}
    analyzer.run(state)

    q = analyzer.retriever.last_query["symptoms_dict"]
    assert q["cough"] == "new_value"


def test_retriever_called_with_correct_extra_params(analyzer):
    state = {"symptoms": {"fever": "high"}, "historical_symptoms": {}, "messages": []}
    analyzer.run(state)

    last = analyzer.retriever.last_query
    assert last["top_k"] == 8
    assert last["rerank_by_jaccard"] is True
    assert last["history_weight"] == 0.3
