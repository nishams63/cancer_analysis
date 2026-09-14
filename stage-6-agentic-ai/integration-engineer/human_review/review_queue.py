"""Tracks active human reviews across runs."""
from typing import Dict, Optional, List
import threading
from orchestration.session_manager import AADARunSession, get_session_manager


class ReviewQueue:
    """Manages runs pending analyst intervention."""

    def __init__(self, session_manager=None):
        self.session_manager = session_manager or get_session_manager()
        self._lock = threading.Lock()

    def get_pending_reviews(self) -> List[AADARunSession]:
        with self._lock:
            all_sessions = self.session_manager.list_sessions()
            return [s for s in all_sessions if s.human_review.required and s.human_review.status == "PENDING"]

    def get_review(self, run_id: str) -> Optional[AADARunSession]:
        session = self.session_manager.get_session(run_id)
        if session and session.human_review.required:
            return session
        return None


_GLOBAL_REVIEW_QUEUE = ReviewQueue()

def get_review_queue() -> ReviewQueue:
    return _GLOBAL_REVIEW_QUEUE
