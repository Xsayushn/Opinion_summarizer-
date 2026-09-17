"""
Test SQLite Exact Global Distribution & Conflict Detection.
"""

from src.indexing.sqlite_store import SQLiteStore
from src.ingestion.loader import ReviewItem
from src.ingestion.absa_pipeline import OpinionUnit


def test_sqlite_global_distribution():
    store = SQLiteStore(db_path=":memory:")

    reviews = [
        ReviewItem(review_id="R1", entity_id="hotel_A", text="Clean room"),
        ReviewItem(review_id="R2", entity_id="hotel_A", text="Great staff but bad service")
    ]

    units = [
        OpinionUnit(
            unit_id="U1", review_id="R1", entity_id="hotel_A",
            canonical_aspect="Cleanliness", raw_aspect="Clean",
            sentiment=1, sentiment_label="positive", confidence=0.9, text_span="Clean room"
        ),
        OpinionUnit(
            unit_id="U2", review_id="R2", entity_id="hotel_A",
            canonical_aspect="Service", raw_aspect="staff",
            sentiment=1, sentiment_label="positive", confidence=0.8, text_span="Great staff"
        ),
        OpinionUnit(
            unit_id="U3", review_id="R2", entity_id="hotel_A",
            canonical_aspect="Service", raw_aspect="service",
            sentiment=-1, sentiment_label="negative", confidence=0.85, text_span="bad service"
        )
    ]

    store.insert_reviews_and_units(reviews, units)

    # Check Cleanliness: 100% positive
    dist = store.get_global_aspect_distribution(entity_id="hotel_A", aspects=["Cleanliness", "Service"])
    assert "Cleanliness" in dist
    assert dist["Cleanliness"]["positive_pct"] == 100.0
    assert dist["Cleanliness"]["negative_pct"] == 0.0

    # Check Service: 50% positive, 50% negative -> Sharp Conflict
    assert "Service" in dist
    assert dist["Service"]["positive_pct"] == 50.0
    assert dist["Service"]["negative_pct"] == 50.0

    conflicts = store.detect_conflicts(entity_id="hotel_A", aspects=["Service"], bifurcation_threshold=20.0)
    assert len(conflicts) == 1
    assert conflicts[0]["aspect"] == "Service"
