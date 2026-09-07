"""
test_api.py — Comprehensive Test Suite for PatentRank FastAPI Service

Tests all REST endpoints:
- GET / (Root metadata)
- GET /health (Health & readiness check)
- GET /search & POST /search (BM25, Dense, Hybrid RRF, Cross-Encoder Re-Ranking)
- POST /segment (Rule-based and ML boundary classifier)
- POST /ingest (Synchronous and asynchronous ingestion)
- GET /tasks/{task_id} (Task status tracking)
- GET /metrics (Evaluation benchmark data)
"""

import os
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SAMPLE_PATENT_TEXT = """
[0001] TECHNICAL FIELD
The present invention relates to quantum key distribution in optical fiber communication networks.

[0002] BACKGROUND OF THE INVENTION
Prior art quantum communication systems suffer from signal degradation and high bit-error rates over extended fiber distances.

[0003] SUMMARY OF THE INVENTION
An apparatus and method for adaptive phase error correction in quantum cryptosystems is disclosed herein. The system includes an interferometer and an active phase modulator.

CLAIMS
What is claimed is:
1. A quantum communication system comprising:
an optical transmitter configured to emit single photon pulses;
an optical fiber transmission channel coupled to said transmitter; and
an active phase modulator for compensating phase drift in real time.

2. The quantum communication system of claim 1, further comprising a balanced homodyne detector.
"""


def test_root_endpoint():
    """Verify root discovery endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "PatentRank API"
    assert "endpoints" in data
    assert data["endpoints"]["search"] == "/search?q={query}"


def test_health_endpoint():
    """Verify health and dependency diagnostic endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "services" in data
    assert "elasticsearch" in data["services"]
    assert "pgvector" in data["services"]
    assert "redis" in data["services"]
    assert "models" in data
    assert data["models"]["cross_encoder"]["loaded"] is True


def test_search_bm25_mode():
    """Verify BM25 keyword retrieval mode."""
    response = client.get("/search?q=optical+fiber&mode=bm25&rerank=false&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "bm25"
    assert data["rerank_applied"] is False
    assert len(data["results"]) <= 5
    if len(data["results"]) > 0:
        first = data["results"][0]
        assert "doc_id" in first
        assert "score" in first
        assert "rank" in first
        assert first["rank"] == 1


def test_search_dense_mode():
    """Verify dense semantic retrieval mode."""
    response = client.get("/search?q=neural+network+controller&mode=dense&rerank=false&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "dense"
    assert "results" in data


def test_search_hybrid_mode():
    """Verify Stage 1 Hybrid RRF mode."""
    response = client.get("/search?q=magnetic+resonance+imaging&mode=hybrid&rerank=false&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert data["rerank_applied"] is False
    assert len(data["results"]) <= 5


def test_search_hybrid_cross_encoder_rerank():
    """Verify two-stage search with fine-tuned Cross-Encoder re-ranking."""
    response = client.get("/search?q=semiconductor+substrate+etching&mode=hybrid&rerank=true&top_k=3&top_k_candidates=15")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert data["rerank_applied"] is True
    assert len(data["results"]) <= 3
    if len(data["results"]) > 0:
        first = data["results"][0]
        assert "raw_logit" in first
        assert "stage1_rank" in first
        assert 0.0 <= first["score"] <= 1.0


def test_search_post_endpoint():
    """Verify POST /search with JSON payload."""
    payload = {
        "q": "distributed blockchain consensus mechanism",
        "mode": "hybrid",
        "rerank": True,
        "top_k": 4,
        "top_k_candidates": 10,
        "explain": True
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["q"]
    assert "latency" in data
    assert "explanation" in data


def test_segment_rule_based():
    """Verify rule-based patent text segmentation."""
    payload = {
        "text": SAMPLE_PATENT_TEXT,
        "doc_id": "US-TEST-2026",
        "engine": "rule"
    }
    response = client.post("/segment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "US-TEST-2026"
    assert data["engine_used"] == "regex_rule_based"
    assert len(data["sections"]) > 0
    assert len(data["paragraphs"]) > 0
    assert len(data["claims"]) >= 2
    # Verify claim parsing
    first_claim = data["claims"][0]
    assert first_claim["metadata"]["claim_type"] == "INDEPENDENT"


def test_segment_ml_classifier():
    """Verify ML boundary classifier segmentation."""
    payload = {
        "text": SAMPLE_PATENT_TEXT,
        "doc_id": "US-TEST-ML-01",
        "engine": "ml"
    }
    response = client.post("/segment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "US-TEST-ML-01"
    assert "sections" in data


def test_ingest_synchronous():
    """Verify synchronous document ingestion fallback."""
    payload = {
        "doc_id": "US-SYNCTEST-001",
        "title": "Quantum Error Mitigating Photonic Router",
        "abstract": "A photonic router utilizing quantum states for high-speed packet switching.",
        "summary": "The router includes optical waveguide arrays and phase modulators.",
        "text": SAMPLE_PATENT_TEXT,
        "async_mode": False
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["doc_id"] == "US-SYNCTEST-001"
    assert data["result"]["embedding_dim"] == 768
    assert data["result"]["segmentation"]["claims_count"] >= 2


def test_sync_task_status():
    """Verify querying task status for synchronous execution."""
    response = client.get("/tasks/sync-execution")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"


def test_metrics_endpoint():
    """Verify empirical benchmark metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "num_queries" in data
    assert "comparison" in data
    assert "hybrid_cross_encoder" in data["comparison"]
    ce_metrics = data["comparison"]["hybrid_cross_encoder"]
    assert ce_metrics["hit@1"] >= 0.95
    assert ce_metrics["mrr"] >= 0.95
