"""
scripts/mine_hard_negatives.py — PatentRank Phase 3: Hard-Negative Mining Pipeline

This script:
1. Loads the 10,000 patent corpus (data/patents_raw.jsonl) and 200 labeled queries (data/queries_labeled.jsonl).
2. Indexes the corpus with RankBM25 to retrieve lexical candidates.
3. Computes dense semantic similarity using cached Gemini embeddings (data/embeddings_cache.jsonl and data/query_embeddings_cache.jsonl).
4. Mines hard negatives for each query:
   - BM25 hard negatives: Top-scoring documents retrieved by BM25 that are NOT the ground-truth target.
   - Dense hard negatives: Top-scoring documents retrieved by dense embeddings that are NOT the ground-truth target.
   - Orthogonal negatives: The labeled irrelevant document from queries_labeled.jsonl.
5. Produces stratified Train (80% / 160 queries) and Validation (20% / 40 queries) datasets:
   - data/train_triples.jsonl and data/val_triples.jsonl (for margin ranking loss)
   - data/train_pairs.jsonl and data/val_pairs.jsonl (for BCE classification loss)
   - data/hard_negatives_summary.json (mining statistics and distribution)
"""

import os
import re
import json
import random
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

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


def tokenize(text: str) -> List[str]:
    """Clean and tokenize text, removing punctuation and standard stopwords."""
    return [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) if w not in STOPWORDS]


def format_doc_text(doc: Dict[str, Any]) -> str:
    """Format document into title + abstract representation for cross-encoder scoring."""
    title = doc.get("title", "").strip()
    abstract = doc.get("abstract", "").strip()
    if abstract:
        return f"{title} — {abstract}"
    return title


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives for Cross-Encoder training")
    parser.add_argument("--corpus-file", default="data/patents_raw.jsonl", help="Path to raw patents corpus")
    parser.add_argument("--queries-file", default="data/queries_labeled.jsonl", help="Path to labeled queries")
    parser.add_argument("--doc-cache-file", default="data/embeddings_cache.jsonl", help="Path to document embeddings")
    parser.add_argument("--query-cache-file", default="data/query_embeddings_cache.jsonl", help="Path to query embeddings")
    parser.add_argument("--out-dir", default="data", help="Output directory for training datasets")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/Val split ratio")
    parser.add_argument("--bm25-negatives", type=int, default=3, help="Number of BM25 hard negatives per query")
    parser.add_argument("--dense-negatives", type=int, default=2, help="Number of Dense hard negatives per query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("PATENTRANK PHASE 3: HARD-NEGATIVE MINING PIPELINE")
    print("=" * 70)

    # 1. Load Patents Corpus
    print(f"Loading patent corpus from {args.corpus_file}...")
    corpus_dict: Dict[str, Dict[str, Any]] = {}
    doc_ids: List[str] = []
    bm25_tokens: List[List[str]] = []

    with open(args.corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            corpus_dict[doc_id] = doc
            doc_ids.append(doc_id)
            full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('summary', '')}"
            bm25_tokens.append(tokenize(full_text))

    print(f"[OK] Loaded {len(corpus_dict):,} patent documents.")

    # 2. Build RankBM25 Index
    print("Building RankBM25 index over full corpus...")
    bm25 = BM25Okapi(bm25_tokens, k1=1.2, b=0.75)
    print("[OK] RankBM25 index constructed.")

    # 3. Load Cached Embeddings (if available)
    doc_embeddings: Dict[str, np.ndarray] = {}
    if os.path.exists(args.doc_cache_file):
        print(f"Loading document embeddings from {args.doc_cache_file}...")
        with open(args.doc_cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    doc_embeddings[item["doc_id"]] = np.array(item["embedding"], dtype=np.float32)
        print(f"[OK] Loaded {len(doc_embeddings):,} document embeddings.")

    query_embeddings: Dict[str, np.ndarray] = {}
    if os.path.exists(args.query_cache_file):
        print(f"Loading query embeddings from {args.query_cache_file}...")
        with open(args.query_cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    query_embeddings[item["query"]] = np.array(item["embedding"], dtype=np.float32)
        print(f"[OK] Loaded {len(query_embeddings):,} query embeddings.")

    # 4. Load Labeled Queries
    print(f"Loading labeled queries from {args.queries_file}...")
    with open(args.queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    print(f"[OK] Loaded {len(queries)} labeled queries.")

    # 5. Stratified Train / Validation Split by query_type
    queries_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for q in queries:
        q_type = q.get("query_type", "general")
        queries_by_type.setdefault(q_type, []).append(q)

    train_queries: List[Dict[str, Any]] = []
    val_queries: List[Dict[str, Any]] = []

    for q_type, q_list in queries_by_type.items():
        random.shuffle(q_list)
        n_train = int(len(q_list) * args.train_ratio)
        train_queries.extend(q_list[:n_train])
        val_queries.extend(q_list[n_train:])

    print(f"[OK] Stratified split: {len(train_queries)} train queries, {len(val_queries)} validation queries.")

    # 6. Mine Hard Negatives Function
    def process_query_set(query_list: List[Dict[str, Any]], split_name: str):
        pairs_data: List[Dict[str, Any]] = []
        triples_data: List[Dict[str, Any]] = []
        stats = {
            "num_queries": len(query_list),
            "num_positives": 0,
            "num_bm25_hard_negatives": 0,
            "num_dense_hard_negatives": 0,
            "num_orthogonal_negatives": 0,
            "total_pairs": 0,
            "total_triples": 0
        }

        for q in query_list:
            q_id = q["query_id"]
            q_text = q["query"]
            rel_id = q["relevant_doc_id"]
            ortho_id = q.get("irrelevant_doc_id")
            q_type = q.get("query_type", "general")

            rel_doc = corpus_dict.get(rel_id)
            if not rel_doc:
                continue
            pos_text = format_doc_text(rel_doc)

            # Positive Pair
            pairs_data.append({
                "query_id": q_id,
                "query": q_text,
                "doc_id": rel_id,
                "doc_text": pos_text,
                "label": 1.0,
                "type": "positive",
                "query_type": q_type
            })
            stats["num_positives"] += 1

            # A. Mine BM25 Hard Negatives
            q_tokens = tokenize(q_text)
            scores = bm25.get_scores(q_tokens)
            top_bm25_indices = np.argsort(scores)[::-1][:50]

            bm25_hard_neg_ids = []
            for idx in top_bm25_indices:
                cand_id = doc_ids[idx]
                if cand_id != rel_id and cand_id != ortho_id:
                    bm25_hard_neg_ids.append(cand_id)
                if len(bm25_hard_neg_ids) >= args.bm25_negatives:
                    break

            for rank_bm25, neg_id in enumerate(bm25_hard_neg_ids, start=1):
                neg_doc = corpus_dict.get(neg_id)
                if not neg_doc:
                    continue
                neg_text = format_doc_text(neg_doc)
                pairs_data.append({
                    "query_id": q_id,
                    "query": q_text,
                    "doc_id": neg_id,
                    "doc_text": neg_text,
                    "label": 0.0,
                    "type": f"hard_negative_bm25_rank{rank_bm25}",
                    "query_type": q_type
                })
                triples_data.append({
                    "query_id": q_id,
                    "query": q_text,
                    "positive_id": rel_id,
                    "positive_text": pos_text,
                    "negative_id": neg_id,
                    "negative_text": neg_text,
                    "negative_type": "hard_negative_bm25",
                    "margin": 1.0,
                    "query_type": q_type
                })
                stats["num_bm25_hard_negatives"] += 1

            # B. Mine Dense Hard Negatives (if embeddings available)
            q_emb = query_embeddings.get(q_text)
            if q_emb is not None and doc_embeddings:
                dense_sims = []
                for d_id, d_emb in doc_embeddings.items():
                    if d_id != rel_id and d_id != ortho_id and d_id not in bm25_hard_neg_ids:
                        sim = cosine_similarity(q_emb, d_emb)
                        dense_sims.append((d_id, sim))
                dense_sims.sort(key=lambda x: x[1], reverse=True)

                dense_hard_neg_ids = [d[0] for d in dense_sims[:args.dense_negatives]]
                for rank_dense, neg_id in enumerate(dense_hard_neg_ids, start=1):
                    neg_doc = corpus_dict.get(neg_id)
                    if not neg_doc:
                        continue
                    neg_text = format_doc_text(neg_doc)
                    pairs_data.append({
                        "query_id": q_id,
                        "query": q_text,
                        "doc_id": neg_id,
                        "doc_text": neg_text,
                        "label": 0.0,
                        "type": f"hard_negative_dense_rank{rank_dense}",
                        "query_type": q_type
                    })
                    triples_data.append({
                        "query_id": q_id,
                        "query": q_text,
                        "positive_id": rel_id,
                        "positive_text": pos_text,
                        "negative_id": neg_id,
                        "negative_text": neg_text,
                        "negative_type": "hard_negative_dense",
                        "margin": 1.0,
                        "query_type": q_type
                    })
                    stats["num_dense_hard_negatives"] += 1

            # C. Orthogonal / Irrelevant Negative
            if ortho_id and ortho_id in corpus_dict and ortho_id != rel_id:
                ortho_doc = corpus_dict[ortho_id]
                ortho_text = format_doc_text(ortho_doc)
                pairs_data.append({
                    "query_id": q_id,
                    "query": q_text,
                    "doc_id": ortho_id,
                    "doc_text": ortho_text,
                    "label": 0.0,
                    "type": "orthogonal_negative",
                    "query_type": q_type
                })
                triples_data.append({
                    "query_id": q_id,
                    "query": q_text,
                    "positive_id": rel_id,
                    "positive_text": pos_text,
                    "negative_id": ortho_id,
                    "negative_text": ortho_text,
                    "negative_type": "orthogonal_negative",
                    "margin": 1.0,
                    "query_type": q_type
                })
                stats["num_orthogonal_negatives"] += 1

        stats["total_pairs"] = len(pairs_data)
        stats["total_triples"] = len(triples_data)
        return pairs_data, triples_data, stats

    print("\nMining hard negatives for Training set...")
    train_pairs, train_triples, train_stats = process_query_set(train_queries, "train")

    print("Mining hard negatives for Validation set...")
    val_pairs, val_triples, val_stats = process_query_set(val_queries, "val")

    # 7. Write Datasets to Disk
    os.makedirs(args.out_dir, exist_ok=True)
    train_pairs_path = os.path.join(args.out_dir, "train_pairs.jsonl")
    val_pairs_path = os.path.join(args.out_dir, "val_pairs.jsonl")
    train_triples_path = os.path.join(args.out_dir, "train_triples.jsonl")
    val_triples_path = os.path.join(args.out_dir, "val_triples.jsonl")
    summary_path = os.path.join(args.out_dir, "hard_negatives_summary.json")

    with open(train_pairs_path, "w", encoding="utf-8") as f:
        for item in train_pairs:
            f.write(json.dumps(item) + "\n")

    with open(val_pairs_path, "w", encoding="utf-8") as f:
        for item in val_pairs:
            f.write(json.dumps(item) + "\n")

    with open(train_triples_path, "w", encoding="utf-8") as f:
        for item in train_triples:
            f.write(json.dumps(item) + "\n")

    with open(val_triples_path, "w", encoding="utf-8") as f:
        for item in val_triples:
            f.write(json.dumps(item) + "\n")

    summary_info = {
        "train": train_stats,
        "validation": val_stats,
        "config": {
            "bm25_negatives": args.bm25_negatives,
            "dense_negatives": args.dense_negatives,
            "train_ratio": args.train_ratio,
            "seed": args.seed
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_info, f, indent=2)

    print("\n" + "=" * 70)
    print("HARD-NEGATIVE MINING COMPLETE")
    print("=" * 70)
    print(f"Train Queries:       {train_stats['num_queries']}")
    print(f"Train Total Pairs:   {train_stats['total_pairs']} (160 pos + {train_stats['total_pairs']-160} negs)")
    print(f"Train Total Triples: {train_stats['total_triples']}")
    print(f"Val Queries:         {val_stats['num_queries']}")
    print(f"Val Total Pairs:     {val_stats['total_pairs']} (40 pos + {val_stats['total_pairs']-40} negs)")
    print(f"Val Total Triples:   {val_stats['total_triples']}")
    print(f"\nFiles generated:")
    print(f"  - {train_pairs_path}")
    print(f"  - {val_pairs_path}")
    print(f"  - {train_triples_path}")
    print(f"  - {val_triples_path}")
    print(f"  - {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
