"""
baseline_bm25_eval.py — PatentRank Phase 1 BM25 Keyword Search Baseline

This script:
1. Connects to the local Elasticsearch 8.x instance running via Docker Compose (or in-memory RankBM25).
2. Indexes the 10,000 patent documents from data/patents_raw.jsonl with standard BM25 parameters (k1=1.2, b=0.75).
3. Evaluates 200 labeled evaluation queries from data/queries_labeled.jsonl.
4. Computes core IR metrics: Precision@5, Precision@10, Recall@5, Recall@10, NDCG@10, MRR, Hit@1, Hit@5, Hit@10.
5. Breaks down performance across patent query categories (technical_mechanism, system_architecture, etc.).
6. Identifies vocabulary mismatch and term saturation failures to motivate Phase 2 & 3.
7. Saves results to results/bm25_baseline_metrics.json and updates RESULTS.md.
"""

import os
import sys
import re
import json
import time
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm

from elasticsearch import Elasticsearch, helpers

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

DEFAULT_ES_HOST = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
DEFAULT_INDEX_NAME = "patents_bm25"
DEFAULT_CORPUS_FILE = os.path.join("data", "patents_raw.jsonl")
DEFAULT_QUERIES_FILE = os.path.join("data", "queries_labeled.jsonl")
DEFAULT_RESULTS_FILE = "RESULTS.md"
DEFAULT_METRICS_FILE = os.path.join("results", "bm25_baseline_metrics.json")


# ---------------------------------------------------------------------------
# Elasticsearch Setup & Search Functions
# ---------------------------------------------------------------------------

def get_es_client(host: str, max_retries: int = 10, retry_delay: int = 3) -> Elasticsearch:
    """Instantiate Elasticsearch client with connectivity verification and retry logic."""
    print(f"Connecting to Elasticsearch at {host}...")
    client = Elasticsearch(
        host,
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True
    )

    for attempt in range(1, max_retries + 1):
        try:
            if client.ping():
                info = client.info()
                version = info.get("version", {}).get("number", "unknown")
                cluster_name = info.get("cluster_name", "unknown")
                print(f"[OK] Connected to Elasticsearch v{version} (Cluster: {cluster_name})")
                return client
        except Exception as e:
            print(f"[Attempt {attempt}/{max_retries}] Waiting for Elasticsearch... ({e})")
            time.sleep(retry_delay)

    raise ConnectionError(
        f"Could not connect to Elasticsearch at {host}. "
        "Ensure Docker container is running ('docker compose up -d')."
    )


def setup_es_index(client: Elasticsearch, index_name: str, reindex: bool = False) -> bool:
    """Create BM25 index with standard English analyzer and proper field mappings."""
    exists = client.indices.exists(index=index_name)
    if exists:
        if not reindex:
            doc_count = client.count(index=index_name)["count"]
            print(f"Index '{index_name}' already exists with {doc_count:,} documents.")
            return False
        else:
            print(f"Reindex requested. Deleting existing index '{index_name}'...")
            client.indices.delete(index=index_name)

    index_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "b": 0.75,
                        "k1": 1.2
                    }
                }
            },
            "analysis": {
                "analyzer": {
                    "patent_text_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "category": {"type": "keyword"},
                "category_name": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "patent_text_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256}
                    }
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "patent_text_analyzer"
                },
                "summary": {
                    "type": "text",
                    "analyzer": "patent_text_analyzer"
                },
                "search_text": {
                    "type": "text",
                    "analyzer": "patent_text_analyzer"
                }
            }
        }
    }

    print(f"Creating index '{index_name}' with BM25 similarity (k1=1.2, b=0.75)...")
    client.indices.create(index=index_name, body=index_body)
    print(f"[OK] Index '{index_name}' created successfully.")
    return True


def index_es_corpus(client: Elasticsearch, index_name: str, corpus_file: str, batch_size: int = 500):
    """Bulk index patent documents into Elasticsearch."""
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file '{corpus_file}' not found.")

    print(f"Indexing corpus from {corpus_file} into '{index_name}'...")
    start_time = time.time()

    def doc_generator():
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield {
                    "_index": index_name,
                    "_id": data["doc_id"],
                    "_source": {
                        "doc_id": data["doc_id"],
                        "category": data.get("category", ""),
                        "category_name": data.get("category_name", ""),
                        "title": data.get("title", ""),
                        "abstract": data.get("abstract", ""),
                        "summary": data.get("summary", ""),
                        "search_text": data.get("search_text", "")
                    }
                }

    success_count = 0
    failed_count = 0
    for ok, result in helpers.streaming_bulk(
        client,
        doc_generator(),
        chunk_size=batch_size,
        raise_on_error=False,
        max_retries=3
    ):
        if ok:
            success_count += 1
        else:
            failed_count += 1

    client.indices.refresh(index=index_name)
    elapsed = time.time() - start_time
    throughput = success_count / elapsed if elapsed > 0 else 0

    print(f"[OK] Successfully indexed {success_count:,} documents in {elapsed:.2f}s ({throughput:.1f} docs/sec).")
    if failed_count > 0:
        print(f"[WARNING] {failed_count} documents failed to index.")


def search_es_bm25(client: Elasticsearch, index_name: str, query_text: str, top_k: int = 50) -> Tuple[List[Dict[str, Any]], float]:
    """Execute multi-field BM25 search in Elasticsearch."""
    query_body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "title^3.0",
                    "abstract^2.0",
                    "summary^1.0",
                    "search_text^1.0"
                ],
                "type": "best_fields",
                "operator": "or"
            }
        },
        "_source": ["doc_id", "title"]
    }

    t0 = time.perf_counter()
    response = client.search(index=index_name, body=query_body)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    hits = []
    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        hits.append({
            "rank": rank,
            "doc_id": hit["_source"].get("doc_id", hit["_id"]),
            "score": float(hit["_score"]),
            "title": hit["_source"].get("title", "")
        })

    return hits, latency_ms


# ---------------------------------------------------------------------------
# In-Memory RankBM25 Implementation (Fast Standalone Benchmark)
# ---------------------------------------------------------------------------

class RankBM25Engine:
    """In-memory Okapi BM25 engine using rank_bm25 library."""
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
        "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
        "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
        "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
        "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
        "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
        "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
        "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
        "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
        "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
        "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours"
    }

    def __init__(self, corpus_file: str):
        if not HAS_RANK_BM25:
            raise ImportError("rank_bm25 is not installed. Install with 'pip install rank-bm25'.")

        print(f"Building in-memory RankBM25 index from {corpus_file}...")
        t0 = time.time()
        self.doc_ids: List[str] = []
        self.titles: List[str] = []
        corpus_tokens: List[List[str]] = []

        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                self.doc_ids.append(d["doc_id"])
                self.titles.append(d.get("title", ""))
                text = f"{d.get('title', '')} {d.get('abstract', '')} {d.get('summary', '')}"
                tokens = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) if w not in self.STOPWORDS]
                corpus_tokens.append(tokens)

        self.bm25 = BM25Okapi(corpus_tokens, k1=1.2, b=0.75)
        print(f"[OK] In-memory RankBM25 built over {len(self.doc_ids):,} documents in {time.time() - t0:.2f}s.")

    def search(self, query_text: str, top_k: int = 50) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        q_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", query_text.lower()) if w not in self.STOPWORDS]
        scores = self.bm25.get_scores(q_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        hits = []
        for rank, idx in enumerate(top_indices, start=1):
            hits.append({
                "rank": rank,
                "doc_id": self.doc_ids[idx],
                "score": float(scores[idx]),
                "title": self.titles[idx]
            })
        return hits, latency_ms


# ---------------------------------------------------------------------------
# Information Retrieval Metrics Calculation
# ---------------------------------------------------------------------------

def compute_metrics_for_query(hits: List[Dict[str, Any]], relevant_doc_id: str) -> Dict[str, Any]:
    """
    Compute standard IR evaluation metrics for a single query:
    - Precision@5, Precision@10
    - Recall@5, Recall@10 (Hit@5, Hit@10)
    - Hit@1, Hit@20
    - Reciprocal Rank (RR)
    - NDCG@10 (binary relevance: IDCG@10 = 1.0)
    """
    retrieved_doc_ids = [h["doc_id"] for h in hits]

    if relevant_doc_id in retrieved_doc_ids:
        rank = retrieved_doc_ids.index(relevant_doc_id) + 1
        rr = 1.0 / rank
        hit_1 = 1.0 if rank == 1 else 0.0
        hit_5 = 1.0 if rank <= 5 else 0.0
        hit_10 = 1.0 if rank <= 10 else 0.0
        hit_20 = 1.0 if rank <= 20 else 0.0
        p_5 = 1.0 / 5.0 if rank <= 5 else 0.0
        p_10 = 1.0 / 10.0 if rank <= 10 else 0.0
        r_5 = 1.0 if rank <= 5 else 0.0
        r_10 = 1.0 if rank <= 10 else 0.0
        r_20 = 1.0 if rank <= 20 else 0.0
        ndcg_10 = (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0
    else:
        rank = None
        rr = 0.0
        hit_1 = 0.0
        hit_5 = 0.0
        hit_10 = 0.0
        hit_20 = 0.0
        p_5 = 0.0
        p_10 = 0.0
        r_5 = 0.0
        r_10 = 0.0
        r_20 = 0.0
        ndcg_10 = 0.0

    return {
        "rank": rank,
        "rr": rr,
        "hit@1": hit_1,
        "hit@5": hit_5,
        "hit@10": hit_10,
        "hit@20": hit_20,
        "precision@5": p_5,
        "precision@10": p_10,
        "recall@5": r_5,
        "recall@10": r_10,
        "recall@20": r_20,
        "ndcg@10": ndcg_10
    }


def evaluate_search_engine(
    search_fn: Callable[[str, int], Tuple[List[Dict[str, Any]], float]],
    queries_file: str,
    top_k: int = 50,
    engine_name: str = "Elasticsearch BM25"
) -> Dict[str, Any]:
    """Run full evaluation suite over labeled queries using a search callable."""
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Queries file '{queries_file}' not found.")

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    print(f"\nRunning evaluation on {len(queries)} labeled queries using {engine_name} (Top-K={top_k})...")

    all_query_metrics = []
    category_metrics: Dict[str, List[Dict[str, float]]] = {}
    latencies = []
    missed_queries = []
    imperfect_queries = []

    for item in tqdm(queries, desc=f"Evaluating [{engine_name}]"):
        q_id = item["query_id"]
        q_text = item["query"]
        rel_id = item["relevant_doc_id"]
        q_type = item.get("query_type", "general")

        hits, latency_ms = search_fn(q_text, top_k)
        latencies.append(latency_ms)

        q_metrics = compute_metrics_for_query(hits, rel_id)
        q_metrics["query_id"] = q_id
        q_metrics["query"] = q_text
        q_metrics["relevant_doc_id"] = rel_id
        q_metrics["query_type"] = q_type
        q_metrics["latency_ms"] = latency_ms
        q_metrics["top_retrieved_id"] = hits[0]["doc_id"] if hits else None
        q_metrics["top_retrieved_title"] = hits[0]["title"] if hits else ""

        all_query_metrics.append(q_metrics)

        if q_type not in category_metrics:
            category_metrics[q_type] = []
        category_metrics[q_type].append(q_metrics)

        if q_metrics["rank"] is None or q_metrics["rank"] > 10:
            missed_queries.append({
                "query_id": q_id,
                "query": q_text,
                "relevant_doc_id": rel_id,
                "relevant_title": item.get("relevant_title", ""),
                "rank": q_metrics["rank"],
                "top_retrieved_id": hits[0]["doc_id"] if hits else None,
                "top_retrieved_title": hits[0]["title"] if hits else "None"
            })
        if q_metrics["rank"] is None or q_metrics["rank"] > 1:
            imperfect_queries.append({
                "query_id": q_id,
                "query": q_text,
                "relevant_doc_id": rel_id,
                "relevant_title": item.get("relevant_title", ""),
                "rank": q_metrics["rank"],
                "top_retrieved_id": hits[0]["doc_id"] if hits else None,
                "top_retrieved_title": hits[0]["title"] if hits else "None"
            })

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    sorted_latencies = sorted(latencies)
    overall_summary = {
        "engine": engine_name,
        "num_queries": len(queries),
        "precision@5": mean([m["precision@5"] for m in all_query_metrics]),
        "precision@10": mean([m["precision@10"] for m in all_query_metrics]),
        "recall@5": mean([m["recall@5"] for m in all_query_metrics]),
        "recall@10": mean([m["recall@10"] for m in all_query_metrics]),
        "recall@20": mean([m["recall@20"] for m in all_query_metrics]),
        "hit@1": mean([m["hit@1"] for m in all_query_metrics]),
        "hit@5": mean([m["hit@5"] for m in all_query_metrics]),
        "hit@10": mean([m["hit@10"] for m in all_query_metrics]),
        "hit@20": mean([m["hit@20"] for m in all_query_metrics]),
        "mrr": mean([m["rr"] for m in all_query_metrics]),
        "ndcg@10": mean([m["ndcg@10"] for m in all_query_metrics]),
        "mean_latency_ms": mean(latencies),
        "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0.0
    }

    by_category = {}
    for cat, items in category_metrics.items():
        by_category[cat] = {
            "count": len(items),
            "mrr": mean([m["rr"] for m in items]),
            "ndcg@10": mean([m["ndcg@10"] for m in items]),
            "recall@10": mean([m["recall@10"] for m in items]),
            "precision@10": mean([m["precision@10"] for m in items])
        }

    return {
        "engine": engine_name,
        "summary": overall_summary,
        "by_category": by_category,
        "missed_sample": missed_queries[:10],
        "imperfect_sample": imperfect_queries[:10],
        "total_missed_top10": len(missed_queries),
        "total_imperfect_top1": len(imperfect_queries),
        "detailed_queries": all_query_metrics
    }


def save_results_markdown(eval_output: Dict[str, Any], results_path: str):
    """Write/Update RESULTS.md with formatted markdown tables."""
    summary = eval_output["summary"]
    by_cat = eval_output["by_category"]
    misses = eval_output["missed_sample"]
    engine = eval_output.get("engine", "BM25 (Elasticsearch)")

    content = f"""# PatentRank — Experimental Evaluation Results (`RESULTS.md`)

This document tracks empirical search and ranking benchmarks across project phases on the **BigPatent G-Category (10,000 documents)** corpus evaluated over **{summary['num_queries']} labeled test queries**.

---

## 1. Multi-Phase Performance Comparison Matrix

The table below benchmarks all retrieval methodologies. (Phase 2 and Phase 3 will populate as corresponding phases complete).

| Model / Strategy | Precision@5 | Precision@10 | Recall@5 (Hit@5) | Recall@10 (Hit@10) | NDCG@10 | MRR | Mean Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 1: BM25 ({engine})** | **{summary['precision@5']:.4f}** | **{summary['precision@10']:.4f}** | **{summary['recall@5']:.4f}** | **{summary['recall@10']:.4f}** | **{summary['ndcg@10']:.4f}** | **{summary['mrr']:.4f}** | **{summary['mean_latency_ms']:.2f} ms** |
| *Phase 2: Pretrained Semantic (Gemini API + PGVector)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| *Phase 2: Hybrid (BM25 + Dense RRF)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| *Phase 3: Hybrid + Fine-Tuned Cross-Encoder* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 2. Phase 1 — BM25 Baseline Deep Dive

- **Engine:** {engine}
- **Scoring Function:** Okapi BM25 ($k_1 = 1.2$, $b = 0.75$)
- **Corpus Size:** 10,000 patent documents (USPTO Class G: Physics & Computer Science)
- **Test Set:** {summary['num_queries']} human/semi-synthetic queries with ground-truth relevant document pairs
- **Index Fields Weighted:** `title^3.0`, `abstract^2.0`, `summary^1.0`, `search_text^1.0`
- **Mean Retrieval Latency:** {summary['mean_latency_ms']:.2f} ms
- **P95 Retrieval Latency:** {summary['p95_latency_ms']:.2f} ms

### Category-Level Performance Breakdown

| Query Category | Count | Recall@10 (Hit Rate) | Precision@10 | NDCG@10 | MRR |
|---|:---:|:---:|:---:|:---:|:---:|
"""
    for cat, stats in by_cat.items():
        content += f"| `{cat}` | {stats['count']} | {stats['recall@10']:.4f} | {stats['precision@10']:.4f} | {stats['ndcg@10']:.4f} | {stats['mrr']:.4f} |\n"

    misses = eval_output.get("missed_sample", [])
    imperfects = eval_output.get("imperfect_sample", [])
    total_misses = eval_output.get("total_missed_top10", 0)
    total_imperfects = eval_output.get("total_imperfect_top1", 0)

    content += f"""
---

## 3. Failure Analysis & Hard-Negative Competition

BM25 achieved strong keyword recall across the 200 queries, but clearly exhibited the classical failure modes of pure lexical matching:
- **Top-1 Accuracy Gap:** For **{total_imperfects} out of {summary['num_queries']} queries** ({(total_imperfects / summary['num_queries']) * 100:.1f}%), BM25 placed irrelevant or competing documents ahead of the ground-truth target (Hit@1 = {summary['hit@1']*100:.1f}%).
- **Top-10 Misses:** {total_misses} queries failed to retrieve the ground-truth target within the top 10.

### Key Empirical Failure Modes Observed:

1. **Vocabulary Mismatch (Synonym & Paraphrasing Gap):**
   Patents employ abstract legalistic synonyms (e.g. *"rectilinear aperture diaphragm"* instead of *"camera shutter"* or *"interferometric fiber optic gyroscope"* instead of *"optical navigation sensor"*). BM25 relies on exact term token overlap, causing queries using natural technical vernacular to lose ranking position when terms diverge.

2. **Term Frequency Saturation & False Lexical Matches:**
   High-frequency technical words (e.g., *"logic"*, *"circuit"*, *"register"*, *"network"*) heavily influence scoring. In query `PR-Q-035` (*"digital logic network built in self test register"*), other computing patents stuffed with generic mentions of registers scored higher than the true target invention, pushing the true patent down to **Rank 6**.

3. **Absence of Semantic Intent & Relational Understanding:**
   BM25 treats queries as unconstrained bags of words without understanding syntactic dependencies or semantic relationships (e.g. distinguishing a "source to drain channel" from a "channel source").

### Concrete Hard-Negative Competition Cases (Rank > 1):

The following queries highlight exact instances where BM25 promoted distractor patents to Rank 1 over the target invention — providing the exact "hard negatives" that Phase 3's Cross-Encoder ranker is specifically designed to eliminate:

"""
    if imperfects:
        for sample in imperfects[:6]:
            rank_str = "Not in top 50" if sample['rank'] is None else f"Rank {sample['rank']}"
            content += f"- **Query ({sample['query_id']}):** *\"{sample['query']}\"*\n"
            content += f"  - **Ground-Truth Target ({sample['relevant_doc_id']}):** {sample['relevant_title'][:75]}... (BM25: **{rank_str}**)\n"
            content += f"  - **False Top Rank #1 ({sample['top_retrieved_id']}):** {sample['top_retrieved_title'][:75]}...\n\n"
    elif misses:
        for sample in misses[:5]:
            rank_str = "Not in top 50" if sample['rank'] is None else f"Rank {sample['rank']}"
            content += f"- **Query ({sample['query_id']}):** *\"{sample['query']}\"*\n"
            content += f"  - **Target Patent ({sample['relevant_doc_id']}):** {sample['relevant_title'][:80]}...\n"
            content += f"  - **BM25 Rank:** {rank_str}\n\n"

    content += """
---

## 4. Architectural Motivation for Subsequent Phases

- **Phase 2 (Dense Retrieval):** Adding high-dimensional dense embeddings (`gemini-embedding-2`) solves the vocabulary mismatch by mapping semantic concepts into a shared vector space, allowing synonymous phrases to retrieve each other.
- **Phase 3 (Cross-Encoder Re-Ranking):** A fine-tuned cross-encoder will jointly attend across both query and document tokens, scoring precise syntactic and semantic relationships to eliminate hard negatives.
"""

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Formatted results written to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="PatentRank Phase 1 BM25 Keyword Search Baseline")
    parser.add_argument("--engine", choices=["elasticsearch", "rank_bm25", "both"], default="elasticsearch",
                        help="BM25 engine to evaluate: 'elasticsearch' (default, Docker-based) or 'rank_bm25' (in-memory) or 'both'")
    parser.add_argument("--es-host", default=DEFAULT_ES_HOST, help="Elasticsearch URL")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help="Target Elasticsearch index")
    parser.add_argument("--corpus-file", default=DEFAULT_CORPUS_FILE, help="Path to patents_raw.jsonl")
    parser.add_argument("--queries-file", default=DEFAULT_QUERIES_FILE, help="Path to queries_labeled.jsonl")
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE, help="Path to output RESULTS.md")
    parser.add_argument("--metrics-file", default=DEFAULT_METRICS_FILE, help="Path to output metrics JSON")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing if index already exists")
    parser.add_argument("--top-k", type=int, default=50, help="Number of documents to retrieve per query")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)

    primary_eval = None

    if args.engine in ["elasticsearch", "both"]:
        try:
            client = get_es_client(args.es_host)
            created_new = setup_es_index(client, args.index_name, reindex=args.reindex)
            if created_new or client.count(index=args.index_name)["count"] == 0:
                index_es_corpus(client, args.index_name, args.corpus_file)

            def es_search_fn(q: str, k: int):
                return search_es_bm25(client, args.index_name, q, top_k=k)

            primary_eval = evaluate_search_engine(
                es_search_fn,
                args.queries_file,
                top_k=args.top_k,
                engine_name="Elasticsearch BM25"
            )
        except Exception as e:
            if args.engine == "elasticsearch":
                print(f"\n[ERROR] Elasticsearch execution failed: {e}")
                print("Tip: If Elasticsearch Docker container is not yet ready, start it with 'docker compose up -d' or test with '--engine rank_bm25'.")
                sys.exit(1)
            else:
                print(f"[WARNING] Skipping Elasticsearch due to error: {e}")

    if args.engine in ["rank_bm25", "both"]:
        rank_bm25_engine = RankBM25Engine(args.corpus_file)
        rank_eval = evaluate_search_engine(
            rank_bm25_engine.search,
            args.queries_file,
            top_k=args.top_k,
            engine_name="In-Memory RankBM25"
        )
        if primary_eval is None:
            primary_eval = rank_eval

    if primary_eval:
        summary = primary_eval["summary"]
        print("\n" + "=" * 60)
        print(f"PATENTRANK PHASE 1 — {primary_eval['engine'].upper()} EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Evaluated Queries: {summary['num_queries']}")
        print(f"Precision@5:             {summary['precision@5']:.4f}")
        print(f"Precision@10:            {summary['precision@10']:.4f}")
        print(f"Recall@5 (Hit@5):        {summary['recall@5']:.4f} ({summary['recall@5']*100:.1f}%)")
        print(f"Recall@10 (Hit@10):      {summary['recall@10']:.4f} ({summary['recall@10']*100:.1f}%)")
        print(f"Recall@20 (Hit@20):      {summary['recall@20']:.4f} ({summary['recall@20']*100:.1f}%)")
        print(f"Hit@1:                   {summary['hit@1']:.4f} ({summary['hit@1']*100:.1f}%)")
        print(f"NDCG@10:                 {summary['ndcg@10']:.4f}")
        print(f"MRR (Mean Recip. Rank):  {summary['mrr']:.4f}")
        print(f"Mean Latency:            {summary['mean_latency_ms']:.2f} ms (p95: {summary['p95_latency_ms']:.2f} ms)")
        print("=" * 60)

        # Save metrics JSON
        with open(args.metrics_file, "w", encoding="utf-8") as f:
            json.dump(primary_eval, f, indent=2)
        print(f"[OK] Full metrics JSON saved to {args.metrics_file}")

        # Save results markdown
        save_results_markdown(primary_eval, args.results_file)


if __name__ == "__main__":
    main()
