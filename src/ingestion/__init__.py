"""
Ingestion module for loading, deduplicating, and extracting opinion units.
"""

from .loader import ReviewLoader, ReviewItem
from .canonicalizer import AspectCanonicalizer
from .absa_pipeline import ABSAPipeline, OpinionUnit

__all__ = ["ReviewLoader", "ReviewItem", "AspectCanonicalizer", "ABSAPipeline", "OpinionUnit"]
