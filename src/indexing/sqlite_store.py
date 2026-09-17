"""
SQLite Metadata & Global Distribution Store.
Computes mathematically exact, unbiased corpus-level aspect-sentiment distributions.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import SQLITE_DB_PATH
from src.ingestion.loader import ReviewItem
from src.ingestion.absa_pipeline import OpinionUnit


class SQLiteStore:
    """
    Manages structured storage for raw reviews and extracted opinion units.
    Enables uncorrupted global count calculations P(S|A).
    """

    def __init__(self, db_path: Path | str = SQLITE_DB_PATH):
        self.db_path = str(db_path)
        self._is_memory = (self.db_path == ":memory:")
        self._mem_conn = None
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self):
        """Close memory connection if open."""
        if self._mem_conn:
            self._mem_conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Reviews table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    entity_id TEXT,
                    text TEXT,
                    rating REAL,
                    date TEXT,
                    metadata_json TEXT
                )
            """)

            # Opinion units table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opinion_units (
                    unit_id TEXT PRIMARY KEY,
                    review_id TEXT,
                    entity_id TEXT,
                    canonical_aspect TEXT,
                    raw_aspect TEXT,
                    sentiment INTEGER,
                    sentiment_label TEXT,
                    confidence REAL,
                    text_span TEXT,
                    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
                )
            """)

            # Create indices for sub-millisecond distribution queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_units_entity_aspect ON opinion_units (entity_id, canonical_aspect)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_units_sentiment ON opinion_units (sentiment)")
            conn.commit()

    def insert_reviews_and_units(self, reviews: List[ReviewItem], units: List[OpinionUnit]):
        """Batch insert reviews and their extracted opinion units."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert reviews
            cursor.executemany("""
                INSERT OR REPLACE INTO reviews (review_id, entity_id, text, rating, date, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (r.review_id, r.entity_id, r.text, r.rating, r.date, json.dumps(r.metadata))
                for r in reviews
            ])

            # Insert opinion units
            cursor.executemany("""
                INSERT OR REPLACE INTO opinion_units (
                    unit_id, review_id, entity_id, canonical_aspect, raw_aspect, sentiment, sentiment_label, confidence, text_span
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (u.unit_id, u.review_id, u.entity_id, u.canonical_aspect, u.raw_aspect, u.sentiment, u.sentiment_label, u.confidence, u.text_span)
                for u in units
            ])
            conn.commit()

    def get_global_aspect_distribution(
        self,
        entity_id: Optional[str] = None,
        aspects: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates exact global P(S|A) distributions directly across all indexed units.
        Returns:
            Dict mapping aspect -> {
                'total': int,
                'positive': int, 'positive_pct': float,
                'neutral': int, 'neutral_pct': float,
                'negative': int, 'negative_pct': float
            }
        """
        query = """
            SELECT canonical_aspect, sentiment, COUNT(*) as cnt
            FROM opinion_units
            WHERE 1=1
        """
        params = []

        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)

        if aspects:
            placeholders = ",".join("?" for _ in aspects)
            query += f" AND canonical_aspect IN ({placeholders})"
            params.extend(aspects)

        query += " GROUP BY canonical_aspect, sentiment"

        distribution: Dict[str, Dict[str, Any]] = {}

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                asp = row["canonical_aspect"]
                sent = row["sentiment"]
                count = row["cnt"]

                if asp not in distribution:
                    distribution[asp] = {
                        "total": 0,
                        "positive": 0, "positive_pct": 0.0,
                        "neutral": 0,  "neutral_pct": 0.0,
                        "negative": 0, "negative_pct": 0.0
                    }

                distribution[asp]["total"] += count
                if sent == 1:
                    distribution[asp]["positive"] += count
                elif sent == -1:
                    distribution[asp]["negative"] += count
                else:
                    distribution[asp]["neutral"] += count

            # Calculate percentages
            for asp, stats in distribution.items():
                total = stats["total"]
                if total > 0:
                    stats["positive_pct"] = round((stats["positive"] / total) * 100, 1)
                    stats["neutral_pct"] = round((stats["neutral"] / total) * 100, 1)
                    stats["negative_pct"] = round((stats["negative"] / total) * 100, 1)

        return distribution

    def detect_conflicts(
        self,
        entity_id: Optional[str] = None,
        aspects: Optional[List[str]] = None,
        bifurcation_threshold: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Detects aspects where user opinion is sharply divided.
        An aspect has a conflict if BOTH positive and negative percentages exceed the bifurcation threshold.
        """
        dist = self.get_global_aspect_distribution(entity_id=entity_id, aspects=aspects)
        conflicts = []

        for asp, stats in dist.items():
            pos_pct = stats["positive_pct"]
            neg_pct = stats["negative_pct"]

            if pos_pct >= bifurcation_threshold and neg_pct >= bifurcation_threshold:
                conflicts.append({
                    "aspect": asp,
                    "positive_pct": pos_pct,
                    "negative_pct": neg_pct,
                    "neutral_pct": stats["neutral_pct"],
                    "total_mentions": stats["total"],
                    "description": f"Divided opinion: {pos_pct}% positive vs {neg_pct}% negative"
                })

        return conflicts

    def get_review(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a full raw review by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_opinion_units_by_ids(self, unit_ids: List[str]) -> List[OpinionUnit]:
        """Fetch specific opinion units by their IDs."""
        if not unit_ids:
            return []

        placeholders = ",".join("?" for _ in unit_ids)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM opinion_units WHERE unit_id IN ({placeholders})", unit_ids)
            rows = cursor.fetchall()
            return [
                OpinionUnit(
                    unit_id=r["unit_id"],
                    review_id=r["review_id"],
                    entity_id=r["entity_id"],
                    canonical_aspect=r["canonical_aspect"],
                    raw_aspect=r["raw_aspect"],
                    sentiment=r["sentiment"],
                    sentiment_label=r["sentiment_label"],
                    confidence=r["confidence"],
                    text_span=r["text_span"]
                )
                for r in rows
            ]
