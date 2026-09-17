"""
Test MinHash LSH Deduplication.
"""

from src.ingestion.loader import ReviewLoader, ReviewItem


def test_minhash_deduplication():
    loader = ReviewLoader(jaccard_threshold=0.80)
    reviews = [
        ReviewItem(
            review_id="R1",
            text="The hotel room was very clean, bed was comfortable, and staff was polite."
        ),
        ReviewItem(
            review_id="R2",
            text="The hotel room was very clean, bed was comfortable, and staff was polite!"  # Near duplicate
        ),
        ReviewItem(
            review_id="R3",
            text="Terrible location, far away from everything and awful noise outside."
        )
    ]

    unique, stats = loader.deduplicate(reviews)
    assert len(unique) == 2
    assert stats["duplicates_removed"] == 1
    assert unique[0].review_id == "R1"
    assert unique[1].review_id == "R3"
