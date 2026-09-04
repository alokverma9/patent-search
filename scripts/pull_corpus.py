"""
pull_corpus.py — Download and structure patent corpus for PatentRank
Pulls 10,000 documents from BigPatent (CPC Category G: Physics & Information Technology).
Saves output to data/patents_raw.jsonl.
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm

PARQUET_URL = "https://huggingface.co/datasets/NortheasternUniversity/big_patent/resolve/main/g/train-00000-of-00018.parquet"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "patents_raw.jsonl")
NUM_DOCS = 10000

def clean_text(text: str) -> str:
    """Normalize whitespace and formatting."""
    if not text:
        return ""
    # Remove excessive whitespace/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_sections(abstract: str, description: str):
    """Extract clean title, summary, and description snippet from patent text."""
    abstract = clean_text(abstract)
    description = clean_text(description)
    
    # Title derivation from abstract
    first_sentence = abstract.split('.')[0].strip()
    if 15 <= len(first_sentence) <= 200:
        title = first_sentence
    else:
        title = abstract[:150].rsplit(' ', 1)[0] + '...'
    
    # Remove leading numbers/bulleting from title if present
    title = re.sub(r'^(?:\[\d+\]|\d+\.|\([a-z]\))\s*', '', title).strip()

    # Extract Summary from description
    summary = ""
    m = re.search(
        r'(?:SUMMARY OF THE INVENTION|BRIEF SUMMARY OF THE INVENTION|OBJECTS AND SUMMARY OF THE INVENTION|SUMMARY)\s*[\n\r]+',
        description,
        re.IGNORECASE
    )
    if m:
        start = m.end()
        next_m = re.search(
            r'(?:BRIEF DESCRIPTION OF THE (?:DRAWINGS|FIGURES)|DETAILED DESCRIPTION)\s*[\n\r]+',
            description[start:],
            re.IGNORECASE
        )
        if next_m:
            summary = description[start : start + next_m.start()].strip()
        else:
            summary = description[start : start + 2500].strip()

    if not summary or len(summary) < 50:
        # Fallback to initial paragraphs of description
        paragraphs = [p for p in description.split('\n\n') if len(p.strip()) > 40]
        summary = "\n\n".join(paragraphs[:2]) if paragraphs else description[:1500]

    # Limit summary to first ~2000 characters to keep representations dense & relevant
    summary = clean_text(summary[:2000])

    # Search text for BM25 and dense retrieval
    search_text = f"Title: {title}\n\nAbstract: {abstract}\n\nSummary: {summary}"

    return title, abstract, summary, search_text

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading parquet from {PARQUET_URL} ...")
    df = pd.read_parquet(PARQUET_URL, columns=['abstract', 'description'])
    print(f"Total available rows in partition: {len(df)}")
    
    # Filter out empty or excessively short documents
    valid_df = df[df['abstract'].str.len() > 100].head(NUM_DOCS)
    print(f"Selected {len(valid_df)} documents for PatentRank corpus.")

    print(f"Processing and saving to {OUTPUT_FILE} ...")
    count = 0
    total_words = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Formatting patents"):
            doc_id = f"US-G-{count+1:06d}"
            title, abstract, summary, search_text = extract_sections(row['abstract'], row['description'])
            
            doc_record = {
                "doc_id": doc_id,
                "category": "G",
                "category_name": "Physics, Computing & Information Technology",
                "title": title,
                "abstract": abstract,
                "summary": summary,
                "search_text": search_text
            }
            f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
            count += 1
            total_words += len(search_text.split())

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    avg_words = total_words / count if count > 0 else 0
    print(f"\n[OK] Successfully wrote {count} patent documents to {OUTPUT_FILE}")
    print(f"     File size: {file_size_mb:.2f} MB")
    print(f"     Average search_text words: {avg_words:.1f} words/doc")

if __name__ == "__main__":
    main()
