"""
FastAPI REST API Service for Opinion Summarizer Framework.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.pipeline import OpinionSummarizerPipeline
from src.ingestion.loader import ReviewItem

app = FastAPI(
    title="Explainable Opinion Summarizer API",
    description="Conflict-Aware, Distributionally Grounded, and Feedback-Adaptive RAG for Opinion Analysis",
    version="1.0.0"
)

# Initialize pipeline instance
pipeline = OpinionSummarizerPipeline()


class IndexRequest(BaseModel):
    reviews: List[ReviewItem]
    deduplicate: bool = True


class QueryRequest(BaseModel):
    query: str
    entity_id: Optional[str] = None
    target_aspects: Optional[List[str]] = None
    context_budget: int = 40


@app.get("/")
def root():
    return {"message": "Opinion Summarizer API is running.", "status": "healthy"}


@app.post("/api/index")
def index_corpus(payload: IndexRequest):
    try:
        res = pipeline.index_reviews(payload.reviews, deduplicate=payload.deduplicate)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
def summarize(payload: QueryRequest):
    try:
        res = pipeline.summarize(
            query=payload.query,
            entity_id=payload.entity_id,
            target_aspects=payload.target_aspects,
            context_budget=payload.context_budget
        )
        return {"status": "success", "data": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/distribution")
def get_distribution(entity_id: Optional[str] = None):
    try:
        dist = pipeline.sqlite.get_global_aspect_distribution(entity_id=entity_id)
        conflicts = pipeline.sqlite.detect_conflicts(entity_id=entity_id)
        return {"status": "success", "distribution": dist, "conflicts": conflicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
