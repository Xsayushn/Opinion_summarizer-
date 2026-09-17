"""
End-to-End Pipeline Integration Test.
"""

from pathlib import Path
from src.pipeline import OpinionSummarizerPipeline
from src.ingestion.loader import ReviewLoader


def test_end_to_end_pipeline():
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample_reviews.json"
    assert sample_file.exists()

    loader = ReviewLoader()
    raw_reviews = loader.load_from_json(sample_file)

    # Initialize pipeline with in-memory or test SQLite database
    pipeline = OpinionSummarizerPipeline(domain="hotel", use_gpu_models=False)

    # 1. Index reviews
    index_res = pipeline.index_reviews(raw_reviews, deduplicate=True)
    assert index_res["total_reviews_indexed"] > 0
    assert index_res["total_opinion_units_extracted"] > 0

    # 2. Run summarization
    result = pipeline.summarize(
        query="How is the cleanliness, service, and location?",
        entity_id="grand_hotel",
        context_budget=30
    )

    # Assertions on pipeline outputs
    assert "summary_text" in result
    assert len(result["summary_text"]) > 20
    assert "Cleanliness" in result["global_distribution"]
    assert "Service" in result["global_distribution"]

    # Check that the sharp conflict on Service is detected
    conflict_aspects = [c["aspect"] for c in result["conflicts"]]
    assert "Service" in conflict_aspects

    # Check Decision Trace has all recorded milestones
    trace = result["decision_trace"]
    assert trace["total_steps"] >= 6
    step_names = [s["name"] for s in trace["steps"]]
    assert "Query Intent Parsing" in step_names
    assert "Global Distribution Query" in step_names
    assert "Conflict Detection" in step_names
    assert "Proportional Exemplar Sampling" in step_names
    assert "Atomic NLI Verification" in step_names

    # Check evidence exemplars were returned
    assert len(result["exemplars"]) > 0
