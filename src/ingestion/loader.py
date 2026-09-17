"""
Review loader and deduplication engine using MinHash LSH.
"""

from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import json
import re
from pathlib import Path
from src.config import MINHASH_PERMUTATIONS, DEDUP_JACCARD_THRESHOLD

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


class ReviewItem(BaseModel):
    review_id: str
    text: str
    entity_id: str = "default_entity"
    rating: Optional[float] = None
    date: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReviewLoader:
    """
    Loads review corpora and removes identical or near-duplicate reviews using MinHash LSH.
    """

    def __init__(
        self,
        jaccard_threshold: float = DEDUP_JACCARD_THRESHOLD,
        num_perm: int = MINHASH_PERMUTATIONS
    ):
        self.jaccard_threshold = jaccard_threshold
        self.num_perm = num_perm

    @staticmethod
    def _shingle_text(text: str, k: int = 3) -> set:
        """Create word k-shingles for Jaccard similarity estimation."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < k:
            return set(words)
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

    def _compute_minhash(self, text: str) -> Optional[Any]:
        if not HAS_DATASKETCH:
            return None
        minhash = MinHash(num_perm=self.num_perm)
        shingles = self._shingle_text(text)
        if not shingles:
            return None
        for s in shingles:
            minhash.update(s.encode('utf-8'))
        return minhash

    def deduplicate(self, reviews: List[ReviewItem]) -> Tuple[List[ReviewItem], Dict[str, int]]:
        """
        Deduplicate reviews using MinHash LSH.
        Returns:
            (unique_reviews, stats_dictionary)
        """
        if not reviews:
            return [], {"total": 0, "unique": 0, "duplicates_removed": 0}

        if not HAS_DATASKETCH:
            # Fallback to exact-string deduplication
            seen = set()
            unique_reviews = []
            for r in reviews:
                norm = r.text.strip().lower()
                if norm not in seen:
                    seen.add(norm)
                    unique_reviews.append(r)
            return unique_reviews, {
                "total": len(reviews),
                "unique": len(unique_reviews),
                "duplicates_removed": len(reviews) - len(unique_reviews),
                "method": "exact_fallback"
            }

        lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=self.num_perm)
        unique_reviews = []
        duplicates_count = 0

        for review in reviews:
            mh = self._compute_minhash(review.text)
            if mh is None:
                continue

            # Query LSH to see if a near-duplicate has already been indexed
            matches = lsh.query(mh)
            if matches:
                duplicates_count += 1
                continue

            # Insert new unique review into LSH
            lsh.insert(review.review_id, mh)
            unique_reviews.append(review)

        stats = {
            "total": len(reviews),
            "unique": len(unique_reviews),
            "duplicates_removed": duplicates_count,
            "method": "minhash_lsh"
        }
        return unique_reviews, stats

    def load_from_json(self, file_path: str | Path) -> List[ReviewItem]:
        """Load reviews from a JSON array or JSON-lines file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Review file not found: {path}")

        reviews = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                raw_list = json.loads(content)
                for item in raw_list:
                    reviews.append(ReviewItem(**item))
            else:
                # JSONL format
                for line in content.splitlines():
                    if line.strip():
                        reviews.append(ReviewItem(**json.loads(line)))
        return reviews
