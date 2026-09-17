"""
Quantifier Calibration Error (QCE) Gate.
Verifies that verbal quantifiers (e.g., 'most', 'all', 'a minority')
mathematically match the empirical global corpus statistics.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from pydantic import BaseModel, Field
from src.config import QUANTIFIER_BOUNDARIES
from .claim_splitter import AtomicClaim


class QuantifierCheckResult(BaseModel):
    claim_id: str
    text: str
    detected_quantifier: Optional[str] = None
    expected_range: Optional[Tuple[float, float]] = None
    empirical_percentage: Optional[float] = None
    calibration_error: float = 0.0
    is_calibrated: bool = True
    feedback_message: Optional[str] = None


class QCEGate:
    """
    Evaluates and enforces quantifier consistency against empirical review distributions.
    """

    def __init__(self, boundaries: Optional[Dict[str, Any]] = None):
        self.boundaries = boundaries or QUANTIFIER_BOUNDARIES

    def check_claim_quantifier(
        self,
        claim: AtomicClaim,
        aspect: str,
        global_stats: Dict[str, Dict[str, Any]],
        sentiment_polarity: str = "positive"
    ) -> QuantifierCheckResult:
        """
        Validates whether a claim's quantifier accurately represents the aspect's empirical ratio.
        """
        text_lower = claim.atomic_text.lower()
        detected_category = None
        matched_term = None

        # Detect verbal quantifier
        for cat, conf in self.boundaries.items():
            for term in conf["terms"]:
                if re.search(rf'\b{re.escape(term)}\b', text_lower):
                    detected_category = cat
                    matched_term = term
                    break
            if detected_category:
                break

        if not detected_category:
            return QuantifierCheckResult(
                claim_id=claim.claim_id,
                text=claim.atomic_text,
                is_calibrated=True
            )

        expected_min = self.boundaries[detected_category]["min"]
        expected_max = self.boundaries[detected_category]["max"]
        midpoint = (expected_min + expected_max) / 2.0

        # Retrieve empirical proportion
        asp_stat = global_stats.get(aspect, {})
        pct_key = f"{sentiment_polarity}_pct"
        empirical_pct = asp_stat.get(pct_key, 50.0) / 100.0

        # Calibration Error
        error = abs(midpoint - empirical_pct)
        is_valid = expected_min <= empirical_pct <= expected_max

        msg = None
        if not is_valid:
            msg = (
                f"Quantifier '{matched_term}' implies {int(expected_min*100)}-{int(expected_max*100)}%, "
                f"but empirical {aspect} {sentiment_polarity} is only {int(empirical_pct*100)}%."
            )

        return QuantifierCheckResult(
            claim_id=claim.claim_id,
            text=claim.atomic_text,
            detected_quantifier=matched_term,
            expected_range=(expected_min, expected_max),
            empirical_percentage=empirical_pct,
            calibration_error=round(error, 3),
            is_calibrated=is_valid,
            feedback_message=msg
        )

    def compute_summary_qce(
        self,
        checks: List[QuantifierCheckResult]
    ) -> float:
        """Computes mean QCE across all checked claims."""
        errors = [c.calibration_error for c in checks if c.detected_quantifier is not None]
        if not errors:
            return 0.0
        return round(sum(errors) / len(errors), 4)
