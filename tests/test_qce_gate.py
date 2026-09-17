"""
Test Quantifier Calibration Error (QCE) Gate.
"""

from src.verification.qce_gate import QCEGate
from src.verification.claim_splitter import AtomicClaim


def test_qce_gate_calibrated():
    gate = QCEGate()
    claim = AtomicClaim(
        claim_id="C1",
        original_sentence="Most users praised the cleanliness.",
        atomic_text="Most users praised the cleanliness.",
        citation_tags=["[E1]"]
    )
    # Global stats show 85% positive -> "most" is valid
    stats = {"Cleanliness": {"positive_pct": 85.0, "neutral_pct": 5.0, "negative_pct": 10.0}}

    result = gate.check_claim_quantifier(claim, aspect="Cleanliness", global_stats=stats, sentiment_polarity="positive")
    assert result.is_calibrated is True
    assert result.detected_quantifier == "most"


def test_qce_gate_uncalibrated():
    gate = QCEGate()
    claim = AtomicClaim(
        claim_id="C2",
        original_sentence="Most users complained about the noise.",
        atomic_text="Most users complained about the noise.",
        citation_tags=["[E2]"]
    )
    # Global stats show negative is only 15% -> "most" is wildly uncalibrated!
    stats = {"Rooms": {"positive_pct": 75.0, "neutral_pct": 10.0, "negative_pct": 15.0}}

    result = gate.check_claim_quantifier(claim, aspect="Rooms", global_stats=stats, sentiment_polarity="negative")
    assert result.is_calibrated is False
    assert result.feedback_message is not None
    assert "only 15%" in result.feedback_message
