"""
Feedback-Adaptive Memory Store.
Logs user corrections and provides dynamic in-context exemplars for self-improvement.
"""

from typing import List, Dict, Any, Optional
import sqlite3
import time
from pathlib import Path
from pydantic import BaseModel, Field
from src.config import FEEDBACK_DIR


class UserFeedbackRecord(BaseModel):
    feedback_id: str
    unit_id: str
    original_aspect: Optional[str] = None
    corrected_aspect: Optional[str] = None
    original_sentiment: Optional[int] = None
    corrected_sentiment: Optional[int] = None
    upvoted: bool = True
    comment: str = ""
    timestamp: float = Field(default_factory=time.time)


class FeedbackAdaptiveStore:
    """
    Persists user corrections and provides few-shot exemplars for dynamic adaptation.
    """

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or (FEEDBACK_DIR / "feedback.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    unit_id TEXT,
                    original_aspect TEXT,
                    corrected_aspect TEXT,
                    original_sentiment INTEGER,
                    corrected_sentiment INTEGER,
                    upvoted INTEGER,
                    comment TEXT,
                    timestamp REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exemplar_memory (
                    exemplar_id TEXT PRIMARY KEY,
                    aspect TEXT,
                    sentiment INTEGER,
                    text_span TEXT,
                    added_at REAL
                )
            """)
            conn.commit()

    def record_feedback(self, feedback: UserFeedbackRecord, text_span: Optional[str] = None):
        """Save user correction and optionally add to exemplar cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_feedback (
                    feedback_id, unit_id, original_aspect, corrected_aspect,
                    original_sentiment, corrected_sentiment, upvoted, comment, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id, feedback.unit_id,
                feedback.original_aspect, feedback.corrected_aspect,
                feedback.original_sentiment, feedback.corrected_sentiment,
                1 if feedback.upvoted else 0, feedback.comment, feedback.timestamp
            ))

            # If user corrected an aspect or sentiment, cache as in-context exemplar
            if text_span and (feedback.corrected_aspect or feedback.corrected_sentiment is not None):
                target_asp = feedback.corrected_aspect or feedback.original_aspect or "General"
                target_sent = feedback.corrected_sentiment if feedback.corrected_sentiment is not None else (feedback.original_sentiment or 0)
                cursor.execute("""
                    INSERT OR REPLACE INTO exemplar_memory (exemplar_id, aspect, sentiment, text_span, added_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"EX_{feedback.unit_id}", target_asp, target_sent, text_span, time.time()))

            conn.commit()

    def get_learned_exemplars(self, aspect: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve dynamic few-shot exemplars learned from user feedback."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if aspect:
                cursor.execute("""
                    SELECT * FROM exemplar_memory WHERE aspect = ? ORDER BY added_at DESC LIMIT ?
                """, (aspect, limit))
            else:
                cursor.execute("""
                    SELECT * FROM exemplar_memory ORDER BY added_at DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_upvoted_unit_ids(self) -> List[str]:
        """Returns list of unit IDs positively rated by users."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT unit_id FROM user_feedback WHERE upvoted = 1")
            return [r[0] for r in cursor.fetchall()]
