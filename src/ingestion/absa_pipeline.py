"""
Aspect-Based Sentiment Analysis (ABSA) Pipeline.
Performs Sentence Splitting, Aspect Term Extraction (ATE),
Aspect Canonicalization, and Sentiment Classification.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import re
import uuid
from .canonicalizer import AspectCanonicalizer
from .loader import ReviewItem


class OpinionUnit(BaseModel):
    unit_id: str
    review_id: str
    entity_id: str = "default_entity"
    canonical_aspect: str
    raw_aspect: str
    sentiment: int  # +1 (Positive), 0 (Neutral), -1 (Negative)
    sentiment_label: str  # "positive", "neutral", "negative"
    confidence: float = 1.0
    text_span: str
    rating: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ABSAPipeline:
    """
    Decomposes reviews into fine-grained opinion units.
    """

    # Common sentiment polar words with negation awareness
    POSITIVE_WORDS = {
        "great", "excellent", "good", "amazing", "wonderful", "love", "loved",
        "best", "perfect", "clean", "spotless", "helpful", "friendly", "fast",
        "spacious", "comfortable", "smooth", "crisp", "stunning", "fantastic",
        "awesome", "worth", "pleasant", "impressed", "superb", "durable"
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "horrible", "awful", "poor", "dirty", "unclean",
        "rude", "slow", "unhelpful", "noisy", "loud", "cramped", "expensive",
        "overpriced", "lag", "laggy", "heating", "warm", "broke", "broken",
        "worst", "disappointed", "disappointing", "drain", "useless", "smell"
    }

    NEGATION_WORDS = {"not", "never", "no", "hardly", "barely", "scarcely", "isn't", "wasn't", "doesn't", "don't", "didn't"}

    def __init__(self, canonicalizer: Optional[AspectCanonicalizer] = None, use_deberta: bool = False):
        self.canonicalizer = canonicalizer or AspectCanonicalizer()
        self.use_deberta = use_deberta
        self.deberta_model = None
        self.deberta_tokenizer = None

        if self.use_deberta:
            self._load_deberta()

    def _load_deberta(self):
        """Lazy loader for DeBERTa ABSA model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from src.config import DEFAULT_ABSA_MODEL
            self.deberta_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ABSA_MODEL)
            self.deberta_model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_ABSA_MODEL)
        except Exception as e:
            # Fall back gracefully to rule-based sentiment if weights are unavailable
            self.use_deberta = False

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split a review into discrete sentence and clause propositions."""
        # Split on period, exclamation, question mark, or strong semicolons/newlines
        raw_sentences = re.split(r'(?<=[.!?;\n])\s+', text)
        cleaned = []
        for s in raw_sentences:
            s_clean = s.strip()
            if len(s_clean) > 8:  # Filter out trivial fragments
                cleaned.append(s_clean)
        return cleaned if cleaned else [text.strip()]

    def _extract_aspect_mentions(self, sentence: str) -> List[str]:
        """
        Identify which canonical aspects are discussed in this sentence.
        """
        sentence_lower = sentence.lower()
        matched_aspects = set()

        for syn, canonical in self.canonicalizer.synonym_map.items():
            pattern = rf'\b{re.escape(syn)}\b'
            if re.search(pattern, sentence_lower):
                matched_aspects.add(canonical)

        return list(matched_aspects)

    def _classify_sentiment_rule_based(self, sentence: str, aspect: str) -> Tuple[int, str, float]:
        """
        Robust polarity classifier with windowed negation handling.
        Returns:
            (sentiment_int, sentiment_label, confidence)
        """
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        pos_score = 0
        neg_score = 0

        for i, token in enumerate(tokens):
            # Check negation in previous 3 tokens
            is_negated = any(tokens[j] in self.NEGATION_WORDS for j in range(max(0, i - 3), i))

            if token in self.POSITIVE_WORDS:
                if is_negated:
                    neg_score += 1.5
                else:
                    pos_score += 1.0
            elif token in self.NEGATIVE_WORDS:
                if is_negated:
                    pos_score += 1.2
                else:
                    neg_score += 1.5

        if pos_score > neg_score:
            conf = min(0.95, 0.65 + 0.1 * (pos_score - neg_score))
            return 1, "positive", conf
        elif neg_score > pos_score:
            conf = min(0.95, 0.65 + 0.1 * (neg_score - pos_score))
            return -1, "negative", conf
        else:
            return 0, "neutral", 0.50

    def extract_opinion_units(self, review: ReviewItem) -> List[OpinionUnit]:
        """
        Decomposes a single review into opinion units.
        """
        units = []
        sentences = self.split_sentences(review.text)

        unit_counter = 0
        for sent in sentences:
            aspects = self._extract_aspect_mentions(sent)
            if not aspects:
                # If no specific aspect is matched, treat as 'General'
                aspects = ["General"]

            for asp in aspects:
                sentiment, label, conf = self._classify_sentiment_rule_based(sent, asp)

                unit_id = f"U_{review.review_id}_{unit_counter}"
                unit_counter += 1

                unit = OpinionUnit(
                    unit_id=unit_id,
                    review_id=review.review_id,
                    entity_id=review.entity_id,
                    canonical_aspect=asp,
                    raw_aspect=asp,
                    sentiment=sentiment,
                    sentiment_label=label,
                    confidence=conf,
                    text_span=sent,
                    rating=review.rating,
                    metadata=review.metadata
                )
                units.append(unit)

        return units

    def process_reviews(self, reviews: List[ReviewItem]) -> List[OpinionUnit]:
        """Batch process a list of reviews into opinion units."""
        all_units = []
        for r in reviews:
            all_units.extend(self.extract_opinion_units(r))
        return all_units
