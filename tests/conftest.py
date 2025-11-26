# tests/conftest.py
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

# Ensure project root on sys.path (so `import agents...` works when running from project root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeLLM:
    """
    Simple fake LLM used across tests.
    - Returns predefined responses (as if they were AI messages with `.content`)
    - Records prompts for assertion.
    """
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.responses:
            text = self.responses.pop(0)
        else:
            text = "fake-llm-response"
        # mimic AIMessage-like object (has .content)
        return SimpleNamespace(content=text)


@pytest.fixture
def fake_llm():
    return FakeLLM()
