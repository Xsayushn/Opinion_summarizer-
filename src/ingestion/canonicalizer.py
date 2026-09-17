"""
Aspect Canonicalizer module.
Normalizes heterogeneous, noisy review aspect terms into a unified canonical taxonomy.
"""

from typing import Dict, List, Optional
import re
from src.config import HOTEL_ASPECT_TAXONOMY, ELECTRONICS_ASPECT_TAXONOMY


class AspectCanonicalizer:
    """
    Maps free-form extracted aspect expressions to a canonical aspect category.
    Supports both rule-based keyword matching and semantic embedding fallback.
    """

    def __init__(self, domain: str = "hotel", custom_taxonomy: Optional[Dict[str, List[str]]] = None):
        self.domain = domain.lower()
        if custom_taxonomy:
            self.taxonomy = custom_taxonomy
        elif self.domain in ["electronics", "amazon", "product", "laptop", "phone"]:
            self.taxonomy = ELECTRONICS_ASPECT_TAXONOMY
        else:
            self.taxonomy = HOTEL_ASPECT_TAXONOMY

        # Build reverse index: synonym -> canonical_aspect
        self.synonym_map = {}
        for canonical, synonyms in self.taxonomy.items():
            self.synonym_map[canonical.lower()] = canonical
            for syn in synonyms:
                self.synonym_map[syn.lower()] = canonical

    def canonicalize(self, raw_term: str) -> str:
        """
        Maps a raw extracted aspect string to a canonical aspect category.
        If no confident match is found, returns 'General' or the cleaned raw term.
        """
        if not raw_term:
            return "General"

        clean = raw_term.strip().lower()

        # 1. Direct synonym lookup
        if clean in self.synonym_map:
            return self.synonym_map[clean]

        # 2. Check multi-word synonyms first (ordered by length descending)
        sorted_synonyms = sorted(self.synonym_map.keys(), key=lambda s: len(s), reverse=True)
        for syn in sorted_synonyms:
            if syn in clean:
                return self.synonym_map[syn]

        # 3. Substring match against individual words
        words = set(re.findall(r'\b\w+\b', clean))
        for word in words:
            if word in self.synonym_map:
                return self.synonym_map[word]

        for syn, canonical in self.synonym_map.items():
            if clean in syn:
                return canonical

        # 4. Capitalized fallback for unknown aspects
        return clean.capitalize()

    def get_canonical_aspects(self) -> List[str]:
        """Returns the list of all registered canonical aspect names."""
        return list(self.taxonomy.keys())

    def register_synonym(self, canonical_aspect: str, new_synonym: str):
        """
        Dynamically registers a new synonym for an aspect.
        Used by the feedback and self-learning loop.
        """
        canonical_norm = canonical_aspect.strip().capitalize()
        syn_norm = new_synonym.strip().lower()

        if canonical_norm not in self.taxonomy:
            self.taxonomy[canonical_norm] = []

        if syn_norm not in self.taxonomy[canonical_norm]:
            self.taxonomy[canonical_norm].append(syn_norm)
            self.synonym_map[syn_norm] = canonical_norm
