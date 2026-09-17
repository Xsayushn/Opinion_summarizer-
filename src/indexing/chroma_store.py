"""
ChromaDB Dense Vector Store for Opinion Units.
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from src.config import CHROMA_DIR, DEFAULT_EMBEDDING_MODEL
from src.ingestion.absa_pipeline import OpinionUnit

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


class ChromaVectorStore:
    """
    Manages dense semantic embeddings and similarity queries for opinion units.
    """

    def __init__(self, collection_name: str = "opinion_units", persist_dir: Path | str = CHROMA_DIR):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.collection = None

        if HAS_CHROMA:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Use sentence-transformers embedding function
            self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=DEFAULT_EMBEDDING_MODEL
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.emb_fn,
                metadata={"hnsw:space": "cosine"}
            )

    def add_units(self, units: List[OpinionUnit]):
        """Indexes opinion units into ChromaDB."""
        if not HAS_CHROMA or not self.collection or not units:
            return

        documents = []
        metadatas = []
        ids = []

        for u in units:
            documents.append(u.text_span)
            ids.append(u.unit_id)
            metadatas.append({
                "unit_id": u.unit_id,
                "review_id": u.review_id,
                "entity_id": u.entity_id,
                "canonical_aspect": u.canonical_aspect,
                "sentiment": u.sentiment,
                "sentiment_label": u.sentiment_label,
                "confidence": float(u.confidence)
            })

        # Batch insert into ChromaDB
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

    def search(
        self,
        query: str,
        target_aspect: Optional[str] = None,
        entity_id: Optional[str] = None,
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Dense semantic search using cosine similarity.
        Returns:
            List of (unit_id, similarity_score)
        """
        if not HAS_CHROMA or not self.collection:
            return []

        where_filter = {}
        if target_aspect and entity_id:
            where_filter = {"$and": [{"canonical_aspect": target_aspect}, {"entity_id": entity_id}]}
        elif target_aspect:
            where_filter = {"canonical_aspect": target_aspect}
        elif entity_id:
            where_filter = {"entity_id": entity_id}

        kwargs = {"query_texts": [query], "n_results": top_k}
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        out = []
        if results and "ids" in results and results["ids"]:
            unit_ids = results["ids"][0]
            # Chroma returns distances; with cosine distance, similarity = 1 - distance
            distances = results["distances"][0] if "distances" in results else [0.0] * len(unit_ids)
            for uid, dist in zip(unit_ids, distances):
                similarity = max(0.0, 1.0 - dist)
                out.append((uid, float(similarity)))

        return out
