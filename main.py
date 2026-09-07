"""
main.py — PatentRank Production FastAPI Service

Exposes high-performance REST APIs for:
1. /search (GET & POST): Two-stage hybrid semantic retrieval + cross-encoder re-ranking.
2. /segment (POST): Structural patent segmentation (Rule-based, ML Classifier, LLM).
3. /ingest (POST): Asynchronous document embedding and indexing pipeline (Celery + Redis).
4. /tasks/{task_id} (GET): Background task status tracking.
5. /health (GET): Service health and backend cluster diagnostics.
6. /metrics (GET): Empirical IR evaluation benchmark comparisons.
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from search_service import get_patentrank_service, PatentRankService

# Configure Structured Logging (Google Cloud Logging compatible)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("patentrank.api")


# ---------------------------------------------------------------------------
# Lifespan Handler
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models, indices, and database connections on startup."""
    logger.info("Initializing PatentRank service engine on startup...")
    t0 = time.time()
    service = get_patentrank_service()
    logger.info(f"PatentRank service engine ready in {time.time() - t0:.2f}s.")
    yield
    logger.info("Shutting down PatentRank service.")


# ---------------------------------------------------------------------------
# FastAPI Application Declaration
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PatentRank API",
    description="Production-grade Hybrid Semantic Search & Ranking Engine for Patent Intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Cloud Logging Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def cloud_logging_middleware(request: Request, call_next):
    """Emit GCP Cloud Logging structured JSON format for request telemetry."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    log_entry = {
        "severity": "INFO" if response.status_code < 400 else "WARNING" if response.status_code < 500 else "ERROR",
        "httpRequest": {
            "requestMethod": request.method,
            "requestUrl": str(request.url),
            "status": response.status_code,
            "latency": f"{duration:.4f}s",
            "remoteIp": request.client.host if request.client else "unknown"
        },
        "message": f"{request.method} {request.url.path} responded {response.status_code} in {duration*1000.0:.2f}ms"
    }
    # Log structured message
    logger.info(json.dumps(log_entry))
    return response


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------

class SearchResultItem(BaseModel):
    rank: int = Field(..., description="Final rank of the candidate")
    doc_id: str = Field(..., description="Unique patent document identifier")
    score: float = Field(..., description="Relevance score (Cross-Encoder probability or Stage 1 score)")
    stage1_rank: Optional[int] = Field(None, description="Candidate rank from Stage 1 retrieval")
    stage1_score: Optional[float] = Field(None, description="Raw Stage 1 score (BM25/Cosine/RRF)")
    raw_logit: Optional[float] = Field(None, description="Raw unnormalized Cross-Encoder logit")
    title: str = Field(..., description="Patent title")
    abstract: Optional[str] = Field(None, description="Patent abstract snippet")
    category: Optional[str] = Field(None, description="CPC classification category")


class SearchResponse(BaseModel):
    query: str
    mode: str
    rerank_applied: bool
    total_candidates: int
    results_count: int
    latency: Dict[str, float]
    results: List[SearchResultItem]
    explanation: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, description="Search query string")
    mode: Literal["hybrid", "bm25", "dense"] = Field("hybrid", description="Retrieval mode")
    rerank: bool = Field(True, description="Whether to apply Stage 2 Cross-Encoder re-ranking")
    top_k: int = Field(10, ge=1, le=100, description="Number of final results to return")
    top_k_candidates: int = Field(50, ge=1, le=200, description="Number of Stage 1 candidates to retrieve")
    explain: bool = Field(False, description="Include diagnostic debugging metadata")


class SegmentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Patent specification text to segment")
    doc_id: Optional[str] = Field(None, description="Document ID identifier")
    engine: Literal["rule", "ml", "llm"] = Field("rule", description="Segmentation parser engine")


class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Unique patent document ID (e.g. US-10123456-B2)")
    title: str = Field(..., description="Patent title")
    abstract: Optional[str] = Field("", description="Patent abstract")
    summary: Optional[str] = Field("", description="Patent summary")
    text: Optional[str] = Field(None, description="Full patent specification or claims text")
    async_mode: bool = Field(True, description="Queue asynchronously via Celery (True) or process synchronously (False)")


class IngestResponse(BaseModel):
    task_id: Optional[str] = None
    status: str
    doc_id: str
    message: str
    result: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    """Service metadata and navigation links."""
    return {
        "service": "PatentRank API",
        "description": "Production Hybrid Semantic Search & Ranking Engine for Patent Intelligence",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search?q={query}",
            "segment": "/segment",
            "ingest": "/ingest",
            "tasks": "/tasks/{task_id}",
            "metrics": "/metrics",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", tags=["System"])
def health():
    """Health check reporting status of all subservices and model checkpoints."""
    service = get_patentrank_service()
    return service.health_status()


@app.get("/search", response_model=SearchResponse, tags=["Search"])
def search_get(
    q: str = Query(..., min_length=1, description="Search query string"),
    mode: Literal["hybrid", "bm25", "dense"] = Query("hybrid", description="Retrieval mode"),
    rerank: bool = Query(True, description="Apply Cross-Encoder re-ranking"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    top_k_candidates: int = Query(50, ge=1, le=200, description="Stage 1 candidates count"),
    explain: bool = Query(False, description="Include diagnostic explanations")
):
    """
    Search patents using hybrid semantic retrieval and fine-tuned cross-encoder re-ranking.
    """
    service = get_patentrank_service()
    return service.search(
        query=q,
        mode=mode,
        rerank=rerank,
        top_k=top_k,
        top_k_candidates=top_k_candidates,
        explain=explain
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search_post(request: SearchRequest):
    """
    POST search endpoint supporting complex search request bodies.
    """
    service = get_patentrank_service()
    return service.search(
        query=request.q,
        mode=request.mode,
        rerank=request.rerank,
        top_k=request.top_k,
        top_k_candidates=request.top_k_candidates,
        explain=request.explain
    )


@app.post("/segment", tags=["Segmentation"])
def segment_document(request: SegmentRequest):
    """
    Decompose patent document text into functional sections, paragraphs, and claims tree.
    Supports Rule-Based (Regex), ML Boundary Classifier (Random Forest), and LLM (Gemini).
    """
    service = get_patentrank_service()
    try:
        return service.segment_document(
            text=request.text,
            doc_id=request.doc_id,
            engine=request.engine
        )
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_patent(request: IngestRequest):
    """
    Ingest a patent document for segmentation, embedding generation, and indexing.
    Supports asynchronous Celery worker queuing with graceful synchronous fallback.
    """
    doc_payload = {
        "doc_id": request.doc_id,
        "title": request.title,
        "abstract": request.abstract,
        "summary": request.summary,
        "text": request.text or f"{request.title}\n{request.abstract}\n{request.summary}"
    }

    if request.async_mode:
        try:
            from tasks import ingest_patent_task
            async_result = ingest_patent_task.delay(doc_payload)
            return IngestResponse(
                task_id=async_result.id,
                status="QUEUED",
                doc_id=request.doc_id,
                message=f"Document {request.doc_id} submitted to Celery background worker."
            )
        except Exception as e:
            logger.warning(f"Celery queueing failed ({e}). Falling back to synchronous processing.")

    # Synchronous processing fallback
    try:
        from tasks import ingest_patent_task
        sync_result = ingest_patent_task(doc_payload)
        return IngestResponse(
            task_id="sync-execution",
            status="SUCCESS",
            doc_id=request.doc_id,
            message=f"Document {request.doc_id} processed synchronously.",
            result=sync_result
        )
    except Exception as e:
        logger.error(f"Synchronous ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.get("/tasks/{task_id}", tags=["Ingestion"])
def get_task_status(task_id: str):
    """
    Check the status and result of a Celery background ingestion task.
    """
    if task_id == "sync-execution":
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "message": "Task was executed synchronously."
        }

    try:
        from celery_app import celery_app
        result = celery_app.AsyncResult(task_id)
        response = {
            "task_id": task_id,
            "status": result.status
        }
        if result.ready():
            if result.successful():
                response["result"] = result.result
            else:
                response["error"] = str(result.result)
        return response
    except Exception as e:
        logger.warning(f"Error querying Celery task {task_id}: {e}")
        return {
            "task_id": task_id,
            "status": "UNKNOWN",
            "message": f"Unable to reach Celery result backend: {str(e)}"
        }


@app.get("/metrics", tags=["Evaluation"])
def get_benchmark_metrics():
    """
    Retrieve empirical IR benchmark results across all 4 retrieval paradigms:
    BM25 vs Dense (Gemini) vs Hybrid RRF vs Hybrid + Cross-Encoder Re-Ranking.
    """
    service = get_patentrank_service()
    return service.get_ir_metrics()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
