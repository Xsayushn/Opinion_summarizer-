"""
BM25 Keyword & Aspect Inverted Indexer.
Provides sparse retrieval to complement dense embeddings.
"""

from typing import List, Tuple, Optional, Dict
import re
from rank_bm25 import BM25Okapi
from src.ingestion.absa_pipeline import OpinionUnit


class BM25Indexer:
    """
    In-memory BM25 index over opinion units for keyword and aspect matching.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.unit_ids: List[str] = []
        self.aspects: List[str] = []
        self.corpus_tokens: List[List[str]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def build_index(self, units: List[OpinionUnit]):
        """Builds BM25 index over opinion units."""
        self.unit_ids = []
        self.aspects = []
        self.corpus_tokens = []

        for unit in units:
            # Combine aspect and text span to give weight to aspect matches
            doc_text = f"{unit.canonical_aspect} {unit.text_span}"
            tokens = self._tokenize(doc_text)
            self.corpus_tokens.append(tokens)
            self.unit_ids.append(unit.unit_id)
            self.aspects.append(unit.canonical_aspect)

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        target_aspect: Optional[str] = None,
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Search for matching opinion units using BM25.
        Returns:
            List of (unit_id, bm25_score)
        """
        if not self.bm25 or not self.unit_ids:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        results = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            if target_aspect and self.aspects[i].lower() != target_aspect.lower():
                continue
            results.append((self.unit_ids[i], float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
