from typing import Dict
from uuid import UUID
import threading


class InMemoryStateStore:
    """
    In-memory session store.

    - Thread-safe
    - Stateless API compatible
    - Redis-replaceable later
    """

    def __init__(self):
        self._store: Dict[UUID, dict] = {}
        self._lock = threading.Lock()

    # -------------------------
    # Public API
    # -------------------------

    def create(self, session_id: UUID, state: dict) -> None:
        """
        Create a new session state.
        """
        with self._lock:
            self._store[session_id] = state

    def get(self, session_id: UUID) -> dict:
        """
        Retrieve an existing session state.
        """
        with self._lock:
            if session_id not in self._store:
                raise KeyError(f"Session not found: {session_id}")
            return self._store[session_id]

    def update(self, session_id: UUID, state: dict) -> None:
        """
        Overwrite session state.
        """
        with self._lock:
            if session_id not in self._store:
                raise KeyError(f"Session not found: {session_id}")
            self._store[session_id] = state

    def delete(self, session_id: UUID) -> None:
        """
        Delete a session state.
        """
        with self._lock:
            self._store.pop(session_id, None)
