"""
Hybrid Retriever combining Dense (ChromaDB) and Sparse (BM25) search via Reciprocal Rank Fusion.
"""

from typing import List, Tuple, Optional, Dict
from src.indexing.chroma_store import ChromaVectorStore
from src.indexing.bm25_index import BM25Indexer


class HybridRetriever:
    """
    Performs Reciprocal Rank Fusion (RRF) between dense semantic embeddings
    and sparse keyword matching over opinion units.
    """

    def __init__(self, chroma_store: ChromaVectorStore, bm25_indexer: BM25Indexer, k_rrf: int = 60):
        self.chroma = chroma_store
        self.bm25 = bm25_indexer
        self.k_rrf = k_rrf

    def retrieve_candidates(
        self,
        query: str,
        target_aspect: Optional[str] = None,
        entity_id: Optional[str] = None,
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Retrieves top candidate unit IDs scored by Reciprocal Rank Fusion.
        Returns:
            List of (unit_id, rrf_score) sorted descending by relevance.
        """
        # 1. Dense Semantic Search
        dense_results = self.chroma.search(
            query=query,
            target_aspect=target_aspect,
            entity_id=entity_id,
            top_k=top_k
        )

        # 2. Sparse BM25 Search
        bm25_results = self.bm25.search(
            query=query,
            target_aspect=target_aspect,
            top_k=top_k
        )

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}

        for rank, (uid, _) in enumerate(dense_results):
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (self.k_rrf + (rank + 1)))

        for rank, (uid, _) in enumerate(bm25_results):
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (self.k_rrf + (rank + 1)))

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]
