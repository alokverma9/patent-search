# PatentRank Dataset & Ground Truth Documentation

This directory contains the patent corpus and labeled evaluation query set powering **PatentRank**, a hybrid semantic search and ranking engine.

---

## 1. Patent Corpus (`patents_raw.jsonl`)

### 1.1 Source & Domain
- **Source:** [BigPatent Benchmark](https://huggingface.co/datasets/NortheasternUniversity/big_patent) (academic benchmark dataset compiled from official USPTO utility patent publications).
- **Technology Focus:** **Cooperative Patent Classification (CPC) Category G: Physics, Computing & Information Technology**.
  - Covers computing architectures, databases, network routing, cryptography, signal/image processing, optical systems, and human-computer interfaces.
- **Corpus Size:** **10,000 documents** (~53 MB uncompressed).
- **Average Document Length:** **423.1 words** per search document.

### 1.2 Schema (`patents_raw.jsonl`)
Each line is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `doc_id` | `string` | Unique identifier formatted as `US-G-XXXXXX` (e.g., `US-G-000001`). |
| `category` | `string` | Primary CPC category symbol (`G`). |
| `category_name` | `string` | Full category description: `"Physics, Computing & Information Technology"`. |
| `title` | `string` | Core inventive title or introductory invention statement. |
| `abstract` | `string` | Full official patent abstract outlining the invention. |
| `summary` | `string` | Extracted "Summary of the Invention" section from patent specification. |
| `search_text` | `string` | Unified representation combining `Title`, `Abstract`, and `Summary`, structured for BM25 and vector embedding indexing. |

---

## 2. Labeled Query Set (`queries_labeled.jsonl`)

### 2.1 Methodology & Rationale
Evaluating hybrid search and ranking requires a trustworthy ground-truth test collection. Rather than relying purely on keyword-heavy titles or unverified synthetic queries, we built **200 evaluation triples** `(query, relevant_doc, irrelevant_doc)`:

1. **Stratified Sampling:** 200 documents were uniformly sampled across the 10,000-patent corpus to ensure broad coverage across diverse subfields (distributed systems, medical imaging, optical waveguides, GPS routing, cryptography, sensor calibration, etc.).
2. **Realistic Query Formulation:** Query formulation mirrors how a patent examiner or patent attorney searches for prior art. Queries are concise (5–12 words), concept-rich, and avoid exact verbatim keyword copying, creating realistic vocabulary mismatch challenges for baseline keyword search.
3. **Relevance Triples:**
   - **Relevant Document (`relevant_doc_id`):** The ground-truth patent that specifically solves the query's technical problem.
   - **Irrelevant Document (`irrelevant_doc_id`):** A challenging negative from the same broad CPC category (Category G) operating in an orthogonal technical domain (e.g., GPS technician dispatch vs. 3D particle measurement in fluid).
4. **Human Verification:** All queries and triple pairs were reviewed for domain coherence, grammatical plausibility, and clear semantic separation between positive and negative documents.

### 2.2 Schema (`queries_labeled.jsonl`)
Each line is a JSON object with the following structure:

```json
{
  "query_id": "PR-Q-001",
  "query": "gps routing service technician customer location coordinates",
  "relevant_doc_id": "US-G-000001",
  "irrelevant_doc_id": "US-G-005001",
  "query_type": "technical_mechanism",
  "relevant_title": "Methods and systems are provided for obtaining information related to a customer service location...",
  "irrelevant_title": "A method measures the three dimensional position of particles in a fluid...",
  "verified": true
}
```

---

## 3. How to Reproduce

All corpus ingestion and query generation scripts are version-controlled in `scripts/`:

```bash
# 1. Pull and format the 10,000 patent documents from BigPatent
python scripts/pull_corpus.py

# 2. Generate the 200 labeled evaluation triples
python scripts/build_labeled_queries.py

# 3. Verify environment, CPU mode, and API connectivity
python scripts/verify_setup.py
```
