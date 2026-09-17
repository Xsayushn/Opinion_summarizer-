"""
Verification module for Atomic Claim Decomposition, NLI Faithfulness, and Quantifier Calibration.
"""

from .claim_splitter import ClaimSplitter, AtomicClaim
from .nli_verifier import NLIVerifier, VerificationResult
from .qce_gate import QCEGate, QuantifierCheckResult

__all__ = ["ClaimSplitter", "AtomicClaim", "NLIVerifier", "VerificationResult", "QCEGate", "QuantifierCheckResult"]
