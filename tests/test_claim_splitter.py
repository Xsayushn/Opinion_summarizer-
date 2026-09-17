"""
Test Atomic Claim Decomposer.
"""

from src.verification.claim_splitter import ClaimSplitter


def test_atomic_claim_splitting():
    splitter = ClaimSplitter()
    compound_text = "While reception staff was extremely courteous [E1], room service arrived two hours late [E2]."

    claims = splitter.split_into_atomic_claims(compound_text)
    assert len(claims) >= 2

    # Verify citation tags are properly parsed and assigned
    all_tags = [t for c in claims for t in c.citation_tags]
    assert "[E1]" in all_tags
    assert "[E2]" in all_tags
