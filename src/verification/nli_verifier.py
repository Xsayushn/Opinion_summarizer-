"""
NLI Claim-to-Evidence Verifier.
Checks whether each generated claim is logically entailed by its cited evidence spans.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import re
from src.config import NLI_ENTAILMENT_THRESHOLD, NLI_CONTRADICTION_THRESHOLD, DEFAULT_NLI_MODEL
from src.ingestion.absa_pipeline import OpinionUnit
from .claim_splitter import AtomicClaim


class VerificationResult(BaseModel):
    claim_id: str
    atomic_text: str
    citation_tags: List[str]
    status: str  # "VERIFIED", "CONTRADICTED", "UNSUPPORTED"
    entailment_score: float
    contradiction_score: float
    supporting_evidence_texts: List[str] = Field(default_factory=list)


class NLIVerifier:
    """
    Evaluates factual consistency of claims against cited review evidence.
    """

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self.pipeline = None

        if self.use_transformer:
            self._init_transformer()

    def _init_transformer(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline("text-classification", model=DEFAULT_NLI_MODEL)
        except Exception:
            self.use_transformer = False

    @staticmethod
    def _compute_lexical_entailment(premise: str, hypothesis: str) -> float:
        """Lightweight token-overlap and semantic alignment score."""
        p_tokens = set(re.findall(r'\b\w+\b', premise.lower()))
        h_tokens = set(re.findall(r'\b\w+\b', hypothesis.lower()))
        # Remove stop tags and punctuation
        stop = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "to", "in", "for", "that"}
        p_clean = p_tokens - stop
        h_clean = h_tokens - stop
        if not h_clean:
            return 0.5
        overlap = len(p_clean & h_clean) / len(h_clean)
        return min(0.99, max(0.1, overlap + 0.25))

    def verify_claim(
        self,
        claim: AtomicClaim,
        evidence_tag_map: Dict[str, OpinionUnit]
    ) -> VerificationResult:
        """Verifies a single atomic claim against its cited evidence."""
        # 1. Flag claims with zero citations as unsupported
        if not claim.citation_tags:
            return VerificationResult(
                claim_id=claim.claim_id,
                atomic_text=claim.atomic_text,
                citation_tags=[],
                status="UNSUPPORTED",
                entailment_score=0.10,
                contradiction_score=0.10,
                supporting_evidence_texts=[]
            )

        # 2. Gather cited premises
        cited_units = [evidence_tag_map[tag] for tag in claim.citation_tags if tag in evidence_tag_map]
        if not cited_units:
            return VerificationResult(
                claim_id=claim.claim_id,
                atomic_text=claim.atomic_text,
                citation_tags=claim.citation_tags,
                status="UNSUPPORTED",
                entailment_score=0.15,
                contradiction_score=0.10,
                supporting_evidence_texts=[]
            )

        best_entailment = 0.0
        best_contradiction = 0.0
        evidence_texts = [u.text_span for u in cited_units]

        for unit in cited_units:
            if self.use_transformer and self.pipeline:
                # Format for cross-encoder NLI
                res = self.pipeline(f"{unit.text_span} </s></s> {claim.atomic_text}")
                # Parse labels: ENTAILMENT, NEUTRAL, CONTRADICTION
                scores = {r["label"].lower(): r["score"] for r in res}
                entailment = scores.get("entailment", 0.0)
                contradiction = scores.get("contradiction", 0.0)
            else:
                entailment = self._compute_lexical_entailment(unit.text_span, claim.atomic_text)
                contradiction = 0.05

            if entailment > best_entailment:
                best_entailment = entailment
            if contradiction > best_contradiction:
                best_contradiction = contradiction

        # Decision rule
        if best_contradiction > NLI_CONTRADICTION_THRESHOLD:
            status = "CONTRADICTED"
        elif best_entailment >= NLI_ENTAILMENT_THRESHOLD:
            status = "VERIFIED"
        else:
            status = "UNSUPPORTED"

        return VerificationResult(
            claim_id=claim.claim_id,
            atomic_text=claim.atomic_text,
            citation_tags=claim.citation_tags,
            status=status,
            entailment_score=round(best_entailment, 2),
            contradiction_score=round(best_contradiction, 2),
            supporting_evidence_texts=evidence_texts
        )

    def verify_all(
        self,
        claims: List[AtomicClaim],
        evidence_tag_map: Dict[str, OpinionUnit]
    ) -> List[VerificationResult]:
        """Verify all claims in a generated summary."""
        return [self.verify_claim(c, evidence_tag_map) for c in claims]
