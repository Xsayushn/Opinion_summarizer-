"""
Atomic Claim Decomposer.
Splits compound summary sentences into atomic propositions and binds each to its specific citation tag.
"""

from typing import List, Dict, Set, Optional
import re
from pydantic import BaseModel, Field


class AtomicClaim(BaseModel):
    claim_id: str
    original_sentence: str
    atomic_text: str
    citation_tags: List[str] = Field(default_factory=list)
    aspect: Optional[str] = None


class ClaimSplitter:
    """
    Decomposes generated text into atomic verifiable claims.
    """

    @staticmethod
    def extract_citation_tags(text: str) -> List[str]:
        """Extracts tags like [E1], [E2, E3] from text."""
        raw_matches = re.findall(r'\[(E\d+(?:,\s*E\d+)*)\]', text)
        tags = []
        for m in raw_matches:
            for sub in m.split(','):
                tags.append(f"[{sub.strip()}]")
        return list(dict.fromkeys(tags))  # deduplicate preserving order

    def split_into_atomic_claims(self, summary_text: str) -> List[AtomicClaim]:
        """
        Splits summary paragraphs into individual sentences and clauses with assigned citations.
        """
        lines = summary_text.strip().splitlines()
        atomic_claims: List[AtomicClaim] = []
        claim_counter = 1

        for line in lines:
            line_clean = line.strip()
            # Ignore markdown headers
            if not line_clean or line_clean.startswith("#"):
                continue

            # Split sentences
            sentences = re.split(r'(?<=[.!?])\s+', line_clean)
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 10:
                    continue

                tags = self.extract_citation_tags(sent_clean)

                # Check if sentence has multiple distinct citation anchors
                clauses = []
                if len(tags) > 1:
                    raw_splits = re.split(
                        r'\b(?:while|although|however|whereas|but)\b|(?<=\])\s*,\s*',
                        sent_clean,
                        flags=re.IGNORECASE
                    )
                    clauses = [c.strip().strip(",").strip() for c in raw_splits if len(c.strip().strip(",").strip()) > 8]

                if len(clauses) > 1:
                    for clause in clauses:
                        cl_tags = self.extract_citation_tags(clause)
                        if not cl_tags:
                            cl_tags = tags
                        atomic_claims.append(AtomicClaim(
                            claim_id=f"C_{claim_counter}",
                            original_sentence=sent_clean,
                            atomic_text=clause,
                            citation_tags=cl_tags
                        ))
                        claim_counter += 1
                else:
                    atomic_claims.append(AtomicClaim(
                        claim_id=f"C_{claim_counter}",
                        original_sentence=sent_clean,
                        atomic_text=sent_clean,
                        citation_tags=tags
                    ))
                    claim_counter += 1

        return atomic_claims
