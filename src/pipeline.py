"""
Master Opinion Summarizer Pipeline.
Orchestrates Ingestion, Global Count Querying, Hybrid RRF, Proportional Sampling,
Decision Tracing, Dialectical Generation, Atomic NLI Verification, and Feedback Adaptation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from src.ingestion.loader import ReviewLoader, ReviewItem
from src.ingestion.canonicalizer import AspectCanonicalizer
from src.ingestion.absa_pipeline import ABSAPipeline, OpinionUnit
from src.indexing.sqlite_store import SQLiteStore
from src.indexing.bm25_index import BM25Indexer
from src.indexing.chroma_store import ChromaVectorStore
from src.retrieval.hybrid_rrf import HybridRetriever
from src.retrieval.proportional import ProportionalSampler
from src.explanation.decision_trace import DecisionTrace
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient
from src.verification.claim_splitter import ClaimSplitter
from src.verification.nli_verifier import NLIVerifier
from src.verification.qce_gate import QCEGate
from src.feedback.adaptive_store import FeedbackAdaptiveStore
from src.config import DEFAULT_CONTEXT_EXEMPLAR_K, DEFAULT_CANDIDATE_POOL_K


class OpinionSummarizerPipeline:
    """
    End-to-end framework implementing Explainable, Conflict-Aware,
    and Feedback-Adaptive Opinion Summarization.
    """

    def __init__(self, domain: str = "hotel", use_gpu_models: bool = False):
        self.domain = domain
        self.canonicalizer = AspectCanonicalizer(domain=domain)
        self.absa = ABSAPipeline(canonicalizer=self.canonicalizer, use_deberta=use_gpu_models)
        self.loader = ReviewLoader()

        # Storage & Indexing
        self.sqlite = SQLiteStore()
        self.bm25 = BM25Indexer()
        self.chroma = ChromaVectorStore()

        # Retrieval
        self.hybrid = HybridRetriever(self.chroma, self.bm25)
        self.proportional = ProportionalSampler(self.sqlite)

        # Generation & Verification
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient()
        self.claim_splitter = ClaimSplitter()
        self.nli_verifier = NLIVerifier(use_transformer=use_gpu_models)
        self.qce_gate = QCEGate()

        # Feedback
        self.feedback_store = FeedbackAdaptiveStore()

    def index_reviews(self, reviews: List[ReviewItem], deduplicate: bool = True) -> Dict[str, Any]:
        """
        Loads, deduplicates, extracts opinion units, and indexes into SQLite, BM25, and ChromaDB.
        """
        dedup_stats = {}
        if deduplicate:
            reviews, dedup_stats = self.loader.deduplicate(reviews)

        # Extract opinion units
        units = self.absa.process_reviews(reviews)

        # Insert into storage
        self.sqlite.insert_reviews_and_units(reviews, units)
        self.bm25.build_index(units)
        self.chroma.add_units(units)

        return {
            "total_reviews_indexed": len(reviews),
            "total_opinion_units_extracted": len(units),
            "deduplication_stats": dedup_stats
        }

    def summarize(
        self,
        query: str,
        entity_id: Optional[str] = None,
        target_aspects: Optional[List[str]] = None,
        context_budget: int = DEFAULT_CONTEXT_EXEMPLAR_K
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline with real-time decision tracing.
        """
        trace = DecisionTrace()

        # Step 1: Query & Aspect Intent Parsing
        resolved_aspects = []
        if target_aspects:
            resolved_aspects = [self.canonicalizer.canonicalize(a) for a in target_aspects]
        else:
            # Detect aspects directly from query text
            for syn, canonical in self.canonicalizer.synonym_map.items():
                if syn in query.lower():
                    resolved_aspects.append(canonical)
            resolved_aspects = list(dict.fromkeys(resolved_aspects))

        if not resolved_aspects:
            resolved_aspects = self.canonicalizer.get_canonical_aspects()[:3]

        trace.log_step(
            name="Query Intent Parsing",
            description=f"Identified {len(resolved_aspects)} target canonical aspects",
            details={"query": query, "aspects": resolved_aspects}
        )

        # Step 2: Global Corpus Distribution (Mathematically unbiased from SQLite)
        global_stats = self.sqlite.get_global_aspect_distribution(
            entity_id=entity_id,
            aspects=resolved_aspects
        )
        trace.log_step(
            name="Global Distribution Query",
            description="Queried uncorrupted corpus-level sentiment proportions from SQLite",
            details={"global_distribution": global_stats}
        )

        # Step 3: Conflict Discovery
        conflicts = self.sqlite.detect_conflicts(
            entity_id=entity_id,
            aspects=resolved_aspects
        )
        conflict_status = "warning" if conflicts else "completed"
        trace.log_step(
            name="Conflict Detection",
            description=f"Flagged {len(conflicts)} aspects with divided polarities",
            details={"conflicts": [c["aspect"] for c in conflicts]},
            status=conflict_status
        )

        # Step 4: Hybrid Retrieval & Proportional Exemplar Sampling
        candidates = self.hybrid.retrieve_candidates(
            query=query,
            entity_id=entity_id,
            top_k=DEFAULT_CANDIDATE_POOL_K
        )
        candidate_ids = [c[0] for c in candidates]

        exemplars, sampling_audit = self.proportional.sample_exemplars(
            candidate_ids=candidate_ids,
            target_aspects=resolved_aspects,
            global_stats=global_stats,
            context_budget=context_budget
        )
        trace.log_step(
            name="Proportional Exemplar Sampling",
            description=f"Stratified {len(exemplars)} evidence exemplars preserving true ratios & minority guarantee",
            details={"exemplar_count": len(exemplars), "audit": sampling_audit}
        )

        # Step 5: Dialectical Generation with Citations
        prompt, evidence_tag_map = self.prompt_builder.build_summary_prompt(
            query=query,
            target_aspects=resolved_aspects,
            global_stats=global_stats,
            conflicts=conflicts,
            exemplars=exemplars
        )
        summary_text = self.llm.generate(prompt)
        trace.log_step(
            name="Constrained Generation",
            description="Generated dialectical summary with inline citation anchors [E#]",
            details={"provider": self.llm.provider}
        )

        # Step 6: Atomic Claim Decomposition & NLI Verification
        claims = self.claim_splitter.split_into_atomic_claims(summary_text)

        # Map tag -> unit
        tag_to_unit = {}
        for idx, u in enumerate(exemplars, 1):
            tag_to_unit[f"[E{idx}]"] = u

        verification_results = self.nli_verifier.verify_all(claims, tag_to_unit)
        verified_count = sum(1 for v in verification_results if v.status == "VERIFIED")

        trace.log_step(
            name="Atomic NLI Verification",
            description=f"Cross-examined {len(claims)} atomic propositions against cited evidence",
            details={"total_claims": len(claims), "verified": verified_count}
        )

        # Step 7: Quantifier Calibration Gate
        qce_checks = []
        for claim in claims:
            # Check against first resolved aspect
            target_asp = resolved_aspects[0] if resolved_aspects else "General"
            res = self.qce_gate.check_claim_quantifier(claim, target_asp, global_stats)
            qce_checks.append(res)

        mean_qce = self.qce_gate.compute_summary_qce(qce_checks)
        trace.log_step(
            name="Quantifier Calibration Gate",
            description=f"Evaluated quantifier accuracy against true counts (Mean QCE: {mean_qce})",
            details={"mean_qce": mean_qce}
        )

        return {
            "query": query,
            "summary_text": summary_text,
            "target_aspects": resolved_aspects,
            "global_distribution": global_stats,
            "conflicts": conflicts,
            "decision_trace": trace.to_dict(),
            "decision_trace_summary": trace.format_summary_text(),
            "claims_verification": [v.model_dump() for v in verification_results],
            "quantifier_checks": [q.model_dump() for q in qce_checks],
            "mean_qce": mean_qce,
            "exemplars": [u.model_dump() for u in exemplars]
        }
