"""
segment_documents.py — Patent Document Segmentation Engine for PatentRank

Covers Phase 4 of BUILD_PLAN.md:
1. Multi-Granularity Patent Segmentation:
   - Functional Section Level: Title, Abstract, Cross-References, Background,
     Summary, Drawings Description, Detailed Description, Claims.
   - Paragraph Level: Bracketed USPTO paragraph markers ([0001], [0002]) and
     adaptive semantic chunking.
   - Claim Level: Independent vs. Dependent claim classification, dependency graph
     tracing, preamble & transitional phrase extraction, limitation decomposition.
2. Dual-Engine Architecture:
   - RegexPatentSegmenter: Ultra-fast (<1 ms/doc), deterministic, rule-based parser.
   - MLBoundarySegmenter: Scikit-learn boundary classifier robust against OCR noise,
     unstructured filings, and missing section headers.
   - GeminiSemanticSegmenter: LLM-assisted semantic structuring via Gemini API.
3. Downstream Retrieval Evaluation:
   - Compares whole-document BM25 retrieval against passage/segment-level retrieval
     with MaxP (Maximum Passage Score) aggregation.
"""

import os
import re
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi

# Load environment configuration
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_ML_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "boundary_classifier.joblib")


# ============================================================================
# 1. DOMAIN DATA CLASSES & ENUMS
# ============================================================================

class SectionType(str, Enum):
    TITLE = "TITLE"
    ABSTRACT = "ABSTRACT"
    CROSS_REFERENCE = "CROSS_REFERENCE"
    FIELD_OF_INVENTION = "FIELD_OF_INVENTION"
    BACKGROUND = "BACKGROUND"
    SUMMARY = "SUMMARY"
    DRAWINGS_DESCRIPTION = "DRAWINGS_DESCRIPTION"
    DETAILED_DESCRIPTION = "DETAILED_DESCRIPTION"
    CLAIMS = "CLAIMS"
    OTHER = "OTHER"


class ClaimType(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    DEPENDENT = "DEPENDENT"


@dataclass
class TextSegment:
    segment_id: str
    segment_type: str
    heading: str
    start_char: int
    end_char: int
    text: str
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentedPatent:
    doc_id: str
    title: str
    abstract: str
    sections: List[TextSegment] = field(default_factory=list)
    paragraphs: List[TextSegment] = field(default_factory=list)
    claims: List[TextSegment] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "abstract": self.abstract,
            "sections": [s.to_dict() for s in self.sections],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "claims": [c.to_dict() for c in self.claims],
            "stats": self.stats
        }

    def get_searchable_passages(self) -> List[Dict[str, Any]]:
        """
        Extract granular searchable passages for passage-level search:
        - Abstract (1 passage)
        - Summary paragraphs or overall summary
        - Key detailed description paragraphs
        - Individual claims
        """
        passages = []
        # Abstract passage
        if self.abstract.strip():
            passages.append({
                "doc_id": self.doc_id,
                "passage_id": f"{self.doc_id}-SEC-ABSTRACT",
                "segment_type": "ABSTRACT",
                "text": f"Title: {self.title}\nAbstract: {self.abstract}",
                "metadata": {"section": "ABSTRACT"}
            })

        # Paragraph passages (from Summary or Detailed Description)
        for p in self.paragraphs:
            passages.append({
                "doc_id": self.doc_id,
                "passage_id": f"{self.doc_id}-{p.segment_id}",
                "segment_type": p.segment_type,
                "text": f"Title: {self.title}\n[{p.heading or p.segment_type}]: {p.text}",
                "metadata": p.metadata
            })

        # Claim passages
        for c in self.claims:
            c_type = c.metadata.get("claim_type", "CLAIM")
            c_num = c.metadata.get("claim_id", "")
            passages.append({
                "doc_id": self.doc_id,
                "passage_id": f"{self.doc_id}-{c.segment_id}",
                "segment_type": "CLAIM",
                "text": f"Title: {self.title}\nClaim {c_num} ({c_type}): {c.text}",
                "metadata": c.metadata
            })

        return passages


# ============================================================================
# 2. RULE-BASED REGEX SEGMENTER
# ============================================================================

class RegexPatentSegmenter:
    """
    Deterministic rule-based patent segmenter leveraging canonical USPTO/EPO
    structural formats and claim syntax patterns.
    """

    # Major section header regex definitions
    SECTION_SPECS = [
        (SectionType.ABSTRACT, r'(?:^|\n)\s*ABSTRACT\s*(?:\n|$)'),
        (SectionType.CROSS_REFERENCE, r'(?:^|\n)\s*(?:CROSS[- ]REFERENCE TO RELATED APPLICATIONS?|RELATED APPLICATIONS?)\s*(?:\n|$)'),
        (SectionType.FIELD_OF_INVENTION, r'(?:^|\n)\s*(?:FIELD OF (?:THE )?INVENTION|TECHNICAL FIELD)\s*(?:\n|$)'),
        (SectionType.BACKGROUND, r'(?:^|\n)\s*(?:BACKGROUND OF (?:THE )?INVENTION|BACKGROUND|DESCRIPTION OF (?:THE )?RELATED ART|PRIOR ART)\s*(?:\n|$)'),
        (SectionType.SUMMARY, r'(?:^|\n)\s*(?:SUMMARY OF (?:THE )?INVENTION|BRIEF SUMMARY(?: OF (?:THE )?INVENTION)?|OBJECTS? OF (?:THE )?INVENTION|SUMMARY\b)\s*(?:\n|$)'),
        (SectionType.DRAWINGS_DESCRIPTION, r'(?:^|\n)\s*(?:BRIEF DESCRIPTION OF (?:THE )?(?:DRAWINGS|FIGURES)|DESCRIPTION OF (?:THE )?DRAWINGS)\s*(?:\n|$)'),
        (SectionType.DETAILED_DESCRIPTION, r'(?:^|\n)\s*(?:DETAILED DESCRIPTION(?: OF (?:THE )?(?:PREFERRED )?EMBODIMENTS?)?|BEST MODE FOR CARRYING OUT THE INVENTION)\s*(?:\n|$)'),
        (SectionType.CLAIMS, r'(?:^|\n)\s*(?:WHAT IS CLAIMED IS|WE CLAIM|I CLAIM|CLAIMS\b)\s*[:\n\r]+')
    ]

    TRANSITIONAL_PHRASES = [
        r'\bcomprising\b',
        r'\bcomprises\b',
        r'\bconsisting of\b',
        r'\bconsists of\b',
        r'\bconsisting essentially of\b',
        r'\bcharacterized in that\b',
        r'\bcharacterized by\b',
        r'\bincluding\b',
        r'\bcomposed of\b'
    ]

    def segment_document(self, doc_id: str, title: str, full_text: str, abstract: str = "") -> SegmentedPatent:
        """
        End-to-end segmentation of a patent document into sections, paragraphs, and claims.
        """
        start_time = time.perf_counter()
        normalized_text = full_text.replace('\r\n', '\n').replace('\r', '\n')

        # 1. Extract functional sections
        sections = self._extract_sections(normalized_text)

        # 2. Extract paragraphs across sections
        paragraphs = self._extract_paragraphs(normalized_text, sections)

        # 3. Extract and parse claims
        claims = self._extract_claims(normalized_text, sections)

        indep_count = sum(1 for c in claims if c.metadata.get("claim_type") == ClaimType.INDEPENDENT.value)
        dep_count = sum(1 for c in claims if c.metadata.get("claim_type") == ClaimType.DEPENDENT.value)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        stats = {
            "total_chars": len(normalized_text),
            "total_words": len(normalized_text.split()),
            "section_count": len(sections),
            "paragraph_count": len(paragraphs),
            "claim_count": len(claims),
            "independent_claims": indep_count,
            "dependent_claims": dep_count,
            "parse_time_ms": round(elapsed_ms, 3)
        }

        return SegmentedPatent(
            doc_id=doc_id,
            title=title.strip(),
            abstract=abstract.strip(),
            sections=sections,
            paragraphs=paragraphs,
            claims=claims,
            stats=stats
        )

    def _extract_sections(self, text: str) -> List[TextSegment]:
        matches = []
        for sec_type, pat in self.SECTION_SPECS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                matches.append((m.start(), m.end(), sec_type.value, m.group(0).strip()))

        if not matches:
            # Fallback if no canonical headers are found: treat as single unified section
            return [
                TextSegment(
                    segment_id="SEC-001",
                    segment_type=SectionType.OTHER.value,
                    heading="BODY",
                    start_char=0,
                    end_char=len(text),
                    text=text.strip(),
                    word_count=len(text.split()),
                    metadata={"fallback": True}
                )
            ]

        # Sort matches chronologically by text offset
        matches.sort(key=lambda x: x[0])
        sections = []

        # Preamble text before the first detected header (e.g. Title or Abstract)
        if matches[0][0] > 100:
            lead_text = text[:matches[0][0]].strip()
            if len(lead_text) > 30:
                sections.append(
                    TextSegment(
                        segment_id="SEC-000",
                        segment_type=SectionType.OTHER.value,
                        heading="PREAMBLE",
                        start_char=0,
                        end_char=matches[0][0],
                        text=lead_text,
                        word_count=len(lead_text.split()),
                        metadata={}
                    )
                )

        for i, (start, end, sec_type, header_text) in enumerate(matches):
            next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            content = text[end:next_start].strip()
            sections.append(
                TextSegment(
                    segment_id=f"SEC-{i+1:03d}",
                    segment_type=sec_type,
                    heading=header_text,
                    start_char=start,
                    end_char=next_start,
                    text=content,
                    word_count=len(content.split()),
                    metadata={"header_text": header_text}
                )
            )

        return sections

    def _extract_paragraphs(self, text: str, sections: List[TextSegment]) -> List[TextSegment]:
        """
        Extracts numbered paragraphs ([0001], [0002]) or semantic chunk paragraphs.
        """
        paragraphs = []
        bracket_pattern = r'(?:^|\n)\s*(\[\d{4}\]|\(\d{4}\))\s*'
        matches = list(re.finditer(bracket_pattern, text))

        if len(matches) >= 2:
            # Bracketed paragraph numbering detected
            for i, m in enumerate(matches):
                pid = m.group(1).strip()
                p_start = m.end()
                p_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

                p_text = text[p_start:p_end].strip()
                if not p_text:
                    continue

                # Assign section affinity
                sec_affinity = "DETAILED_DESCRIPTION"
                for s in sections:
                    if s.start_char <= m.start() <= s.end_char:
                        sec_affinity = s.segment_type
                        break

                paragraphs.append(
                    TextSegment(
                        segment_id=f"PAR-{pid.strip('[]()')}",
                        segment_type="PARAGRAPH",
                        heading=pid,
                        start_char=m.start(),
                        end_char=p_end,
                        text=p_text,
                        word_count=len(p_text.split()),
                        metadata={"paragraph_id": pid, "section_affinity": sec_affinity}
                    )
                )
        else:
            # Fallback to double newline semantic blocks for sections outside claims
            p_counter = 1
            for sec in sections:
                if sec.segment_type == SectionType.CLAIMS.value:
                    continue
                blocks = [b.strip() for b in re.split(r'\n\s*\n', sec.text) if len(b.strip()) > 60]
                for b in blocks:
                    paragraphs.append(
                        TextSegment(
                            segment_id=f"PAR-{p_counter:04d}",
                            segment_type="PARAGRAPH",
                            heading=f"Paragraph {p_counter}",
                            start_char=0,
                            end_char=len(b),
                            text=b,
                            word_count=len(b.split()),
                            metadata={"section_affinity": sec.segment_type, "fallback_split": True}
                        )
                    )
                    p_counter += 1

        return paragraphs

    def _extract_claims(self, text: str, sections: List[TextSegment]) -> List[TextSegment]:
        """
        Isolates and parses individual claims into structured representations.
        """
        claims_text = ""
        # 1. Locate claims section from extracted sections
        for s in sections:
            if s.segment_type == SectionType.CLAIMS.value:
                claims_text = s.text
                break

        # Fallback: search for claims header directly if sections didn't isolate it
        if not claims_text:
            m = re.search(r'(?:^|\n)\s*(?:WHAT IS CLAIMED IS|WE CLAIM|I CLAIM|CLAIMS\b)\s*[:\n\r]+', text, re.IGNORECASE)
            if m:
                claims_text = text[m.end():]

        if not claims_text:
            return []

        # Split on claim numbers: e.g. "1. ", "2. ", "Claim 1: ", etc.
        claim_splits = re.split(r'(?:^|\n)\s*(?:Claim\s+)?(\d+)[\.\:]\s+', claims_text)
        claims = []

        for i in range(1, len(claim_splits), 2):
            try:
                cid = int(claim_splits[i])
            except ValueError:
                continue

            ctext = claim_splits[i + 1].strip()
            # Truncate if it captures subsequent non-claim signatures
            ctext = re.split(r'\n\s*(?:[A-Z\s]{5,}:|Dated:|Inventor Signature)', ctext)[0].strip()
            if not ctext:
                continue

            # Dependency analysis
            parent_matches = re.findall(r'(?:claim|claims)\s+(\d+)', ctext, re.IGNORECASE)
            parents = sorted(list(set(int(p) for p in parent_matches if int(p) != cid)))
            claim_type = ClaimType.DEPENDENT.value if parents else ClaimType.INDEPENDENT.value

            # Transitional phrase detection
            preamble = ""
            trans_phrase = ""
            limitations = []

            for trans_pat in self.TRANSITIONAL_PHRASES:
                tm = re.search(trans_pat, ctext, re.IGNORECASE)
                if tm:
                    preamble = ctext[:tm.start()].strip()
                    trans_phrase = tm.group(0).lower()
                    body = ctext[tm.end():].strip().lstrip(':').strip()
                    # Limitation decomposition on semicolons or clause letters (a), (b)
                    raw_lims = re.split(r';|\n\s*[-–•]|\n\s*\([a-z\d]+\)', body)
                    limitations = [lim.strip() for lim in raw_lims if len(lim.strip()) > 10]
                    break

            if not preamble:
                preamble = ctext.split(',')[0].strip()
                limitations = [ctext]

            claims.append(
                TextSegment(
                    segment_id=f"CLM-{cid:03d}",
                    segment_type=SectionType.CLAIMS.value,
                    heading=f"Claim {cid}",
                    start_char=0,
                    end_char=len(ctext),
                    text=ctext,
                    word_count=len(ctext.split()),
                    metadata={
                        "claim_id": cid,
                        "claim_type": claim_type,
                        "parent_claims": parents,
                        "preamble": preamble,
                        "transitional_phrase": trans_phrase,
                        "limitations_count": len(limitations),
                        "limitations": limitations
                    }
                )
            )

        return claims


# ============================================================================
# 3. MACHINE LEARNING BOUNDARY CLASSIFIER
# ============================================================================

class MLBoundarySegmenter:
    """
    Supervised boundary classification model for detecting section, paragraph,
    and claim boundaries in unstructured or OCR-degraded patent text.
    """

    FEATURE_NAMES = [
        "char_length", "word_count", "uppercase_ratio", "titlecase_ratio",
        "digit_ratio", "punct_ratio", "starts_with_digit", "starts_with_bracket",
        "has_claim_keyword", "has_section_keyword", "has_transitional_keyword",
        "ends_with_colon", "ends_with_period", "prev_line_empty", "is_short_line"
    ]

    def __init__(self, model_path: str = DEFAULT_ML_MODEL_PATH):
        self.model_path = model_path
        self.classifier: Optional[RandomForestClassifier] = None

    def extract_features(self, line: str, prev_line: str = "", next_line: str = "") -> List[float]:
        """Extract a 15-dimensional numerical feature representation for a candidate line."""
        stripped = line.strip()
        if not stripped:
            return [0.0] * len(self.FEATURE_NAMES)

        chars = len(stripped)
        words = stripped.split()
        n_words = len(words)

        caps_count = sum(1 for c in stripped if c.isupper())
        digit_count = sum(1 for c in stripped if c.isdigit())
        punct_count = sum(1 for c in stripped if not c.isalnum() and not c.isspace())

        caps_ratio = caps_count / chars if chars > 0 else 0.0
        digit_ratio = digit_count / chars if chars > 0 else 0.0
        punct_ratio = punct_count / chars if chars > 0 else 0.0

        title_words = sum(1 for w in words if w.istitle())
        title_ratio = title_words / n_words if n_words > 0 else 0.0

        starts_with_digit = 1.0 if re.match(r'^\s*\d+[\.\:\)]', stripped) else 0.0
        starts_with_bracket = 1.0 if re.match(r'^\s*\[\d+\]|^\s*\(\d+\)', stripped) else 0.0

        has_claim_kw = 1.0 if re.search(r'\bclaims?\b', stripped, re.IGNORECASE) else 0.0
        has_section_kw = 1.0 if re.search(
            r'\b(invention|background|summary|drawings?|figures?|embodiments?|field|prior art|cross[- ]reference|detailed description)\b',
            stripped, re.IGNORECASE
        ) else 0.0
        has_trans_kw = 1.0 if re.search(
            r'\b(comprising|consisting|characterized)\b', stripped, re.IGNORECASE
        ) else 0.0

        ends_with_colon = 1.0 if stripped.endswith(':') else 0.0
        ends_with_period = 1.0 if stripped.endswith('.') else 0.0

        prev_blank = 1.0 if not prev_line.strip() else 0.0
        is_short = 1.0 if chars < 60 else 0.0

        return [
            float(chars), float(n_words), caps_ratio, title_ratio, digit_ratio, punct_ratio,
            starts_with_digit, starts_with_bracket, has_claim_kw, has_section_kw,
            has_trans_kw, ends_with_colon, ends_with_period, prev_blank, is_short
        ]

    def build_training_dataset(self, patents: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract labeled line transitions from patent documents:
        Positive (1): Section header, paragraph start [0001], claim boundary.
        Negative (0): Internal sentence / body text line.
        """
        X = []
        y = []

        for p in patents:
            text = p.get("full_description", "") or p.get("summary", "")
            lines = [ln for ln in text.split('\n') if ln.strip()]

            for i, line in enumerate(lines):
                prev_l = lines[i - 1] if i > 0 else ""
                next_l = lines[i + 1] if i + 1 < len(lines) else ""
                feats = self.extract_features(line, prev_l, next_l)

                stripped = line.strip()
                is_boundary = 0
                if re.match(r'^(?:\[\d{4}\]|\(\d{4}\))', stripped):
                    is_boundary = 1
                elif re.match(r'^(?:Claim\s+)?\d+[\.\:]\s+', stripped):
                    is_boundary = 1
                elif re.match(r'^(?:[A-Z\s]{4,}(?:INVENTION|APPLICATIONS|DRAWINGS|EMBODIMENTS|CLAIMS|BACKGROUND|SUMMARY))\b', stripped):
                    is_boundary = 1
                elif stripped.isupper() and len(stripped.split()) <= 7:
                    is_boundary = 1

                X.append(feats)
                y.append(is_boundary)

        return np.array(X), np.array(y)

    def train_and_evaluate(self, patents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train a boundary classifier on extracted line features and report evaluation metrics."""
        print("\n[MLBoundarySegmenter] Extracting feature matrix from training documents...")
        X, y = self.build_training_dataset(patents)

        n_pos = int(np.sum(y))
        n_neg = len(y) - n_pos
        print(f"  Dataset: {len(y)} total lines | {n_pos} boundaries (positive) | {n_neg} non-boundaries (negative)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = round(time.perf_counter() - t0, 3)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)
        roc = roc_auc_score(y_test, y_prob)

        feature_importances = {
            name: round(float(imp), 4)
            for name, imp in zip(self.FEATURE_NAMES, clf.feature_importances_)
        }
        sorted_fi = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

        metrics = {
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "positive_test_count": int(np.sum(y_test)),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "training_time_sec": train_time,
            "feature_importances": sorted_fi
        }

        self.classifier = clf
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(clf, self.model_path)
        print(f"  [OK] Saved ML boundary model to {self.model_path}")

        return metrics

    def predict_boundaries(self, text: str) -> List[Dict[str, Any]]:
        """Predict boundary status for every non-empty line in a document."""
        if self.classifier is None:
            if os.path.exists(self.model_path):
                self.classifier = joblib.load(self.model_path)
            else:
                raise ValueError("Model not trained or loaded. Run train_and_evaluate() first.")

        lines = [ln for ln in text.split('\n') if ln.strip()]
        results = []

        for i, line in enumerate(lines):
            prev_l = lines[i - 1] if i > 0 else ""
            next_l = lines[i + 1] if i + 1 < len(lines) else ""
            feats = self.extract_features(line, prev_l, next_l)
            prob = float(self.classifier.predict_proba([feats])[0][1])
            is_bound = bool(prob >= 0.5)

            results.append({
                "line_idx": i,
                "text": line[:80],
                "is_boundary": is_bound,
                "confidence": round(prob, 4)
            })

        return results


# ============================================================================
# 4. LLM SEMANTIC STRUCTURING (GEMINI API)
# ============================================================================

class GeminiSemanticSegmenter:
    """
    LLM-assisted semantic structuring for dense patent disclosures.
    Uses Google AI Studio's Gemini 3.5 Flash Lite to extract structured inventive units.
    """

    def __init__(self, cache_file: str = os.path.join(DEFAULT_DATA_DIR, "llm_segmentation_cache.jsonl")):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.client = None

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and "your_" not in api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"[GeminiSemanticSegmenter] Note: Could not initialize Gemini client: {e}")

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        cache = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        cache[item["doc_id"]] = item["structured_data"]
        return cache

    def _save_cache(self, doc_id: str, data: Dict[str, Any]):
        self.cache[doc_id] = data
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": doc_id, "structured_data": data}, ensure_ascii=False) + "\n")

    def structure_disclosure(self, doc_id: str, title: str, text: str) -> Dict[str, Any]:
        """
        Decomposes patent text into structured semantic units:
        - Problem Statement
        - Inventive Core
        - Technical Elements
        - Independent Claims Summary
        """
        if doc_id in self.cache:
            return self.cache[doc_id]

        if not self.client:
            return {
                "doc_id": doc_id,
                "problem_statement": "Gemini API client not initialized.",
                "inventive_core": "Rule-based fallback representation.",
                "technical_elements": [],
                "claim_summary": "Unstructured text."
            }

        prompt = f"""You are an expert USPTO patent analyst. Analyze the following patent disclosure and return ONLY valid JSON with no markdown backticks:
{{
  "problem_statement": "<technical problem addressed by the patent>",
  "inventive_core": "<core novelty or solution provided>",
  "technical_elements": ["<element 1>", "<element 2>", "<element 3>"],
  "claim_summary": "<concise functional summary of independent claims>"
}}

Patent Title: {title}
Patent Text:
{text[:4000]}
"""
        try:
            resp = self.client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )
            raw_text = resp.text.strip()
            raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
            raw_text = re.sub(r'\s*```$', '', raw_text)
            data = json.loads(raw_text)
            data["doc_id"] = doc_id
            self._save_cache(doc_id, data)
            return data
        except Exception as e:
            fallback = {
                "doc_id": doc_id,
                "error": str(e),
                "problem_statement": f"Extraction error: {e}",
                "inventive_core": title,
                "technical_elements": [],
                "claim_summary": "Fallback representation"
            }
            return fallback


# ============================================================================
# 5. DOWNSTREAM RETRIEVAL EVALUATION (PASSAGE VS. DOCUMENT BM25)
# ============================================================================

class SegmentRetrievalEvaluator:
    """
    Evaluates the tangible retrieval benefit of text segmentation:
    Whole-document monolithic BM25 vs. Segment-level BM25 with MaxP aggregation.
    """

    def __init__(self, patents: List[Dict[str, Any]], queries: List[Dict[str, Any]]):
        self.patents = patents
        self.queries = queries
        self.segmenter = RegexPatentSegmenter()

    def run_comparison(self, num_queries: int = 100) -> Dict[str, Any]:
        eval_queries = self.queries[:num_queries]
        print(f"\n[SegmentRetrievalEvaluator] Running comparison across {len(eval_queries)} queries...")

        # 1. Segment all patents and build passage corpus
        doc_corpus = []
        doc_ids = []
        passage_corpus = []
        passage_parent_ids = []
        passage_types = []

        print("  Segmenting patent corpus into functional passages...")
        for p in self.patents:
            did = p.get("doc_id", p.get("sample_id", "DOC"))
            title = p.get("title", "")
            raw_text = p.get("search_text", "") or f"Title: {title}\nAbstract: {p.get('abstract','')}\nSummary: {p.get('summary','')}"

            # Monolithic document
            doc_corpus.append(raw_text.lower().split())
            doc_ids.append(did)

            # Segmented document
            segmented = self.segmenter.segment_document(did, title, raw_text, p.get("abstract", ""))
            passages = segmented.get_searchable_passages()

            for pass_rec in passages:
                passage_corpus.append(pass_rec["text"].lower().split())
                passage_parent_ids.append(did)
                passage_types.append(pass_rec["segment_type"])

        print(f"  Indexed: {len(doc_corpus)} monolithic documents vs. {len(passage_corpus)} granular passages.")

        # Fit BM25 indices
        bm25_doc = BM25Okapi(doc_corpus)
        bm25_passage = BM25Okapi(passage_corpus)

        # Evaluate monolithic vs passage MaxP
        doc_metrics = self._eval_bm25_model(bm25_doc, doc_ids, eval_queries, is_passage=False)
        pass_metrics = self._eval_bm25_model(bm25_passage, passage_parent_ids, eval_queries, is_passage=True)

        return {
            "evaluated_queries": len(eval_queries),
            "monolithic_document_count": len(doc_corpus),
            "granular_passage_count": len(passage_corpus),
            "monolithic_bm25": doc_metrics,
            "passage_segmented_bm25_maxp": pass_metrics,
            "improvement_analysis": {
                "mrr_delta": round(pass_metrics["mrr"] - doc_metrics["mrr"], 4),
                "top1_hit_delta": round(pass_metrics["hit_at_1"] - doc_metrics["hit_at_1"], 4),
                "ndcg_delta": round(pass_metrics["ndcg_at_10"] - doc_metrics["ndcg_at_10"], 4)
            }
        }

    def _eval_bm25_model(self, model: BM25Okapi, id_index: List[str], queries: List[Dict[str, Any]], is_passage: bool) -> Dict[str, Any]:
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        latencies = []

        for q in queries:
            target_id = q.get("relevant_doc_id", "")
            q_text = q.get("query", "").lower().split()

            t0 = time.perf_counter()
            scores = model.get_scores(q_text)
            latency = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency)

            if is_passage:
                # MaxP aggregation: group passage scores by parent document
                parent_scores: Dict[str, float] = {}
                for idx, sc in enumerate(scores):
                    pid = id_index[idx]
                    if pid not in parent_scores or sc > parent_scores[pid]:
                        parent_scores[pid] = sc
                sorted_docs = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)
                ranked_ids = [d[0] for d in sorted_docs]
            else:
                top_indices = np.argsort(scores)[::-1]
                ranked_ids = [id_index[idx] for idx in top_indices]

            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 5:
                    hits_at_5 += 1
                if rank <= 10:
                    hits_at_10 += 1
                mrr_sum += 1.0 / rank
                ndcg_sum += 1.0 / np.log2(rank + 1)

        n = len(queries)
        return {
            "hit_at_1": round(hits_at_1 / n, 4),
            "recall_at_5": round(hits_at_5 / n, 4),
            "recall_at_10": round(hits_at_10 / n, 4),
            "mrr": round(mrr_sum / n, 4),
            "ndcg_at_10": round(ndcg_sum / n, 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2)
        }


# ============================================================================
# 6. CLI & ORCHESTRATION PIPELINES
# ============================================================================

def run_demo(use_llm: bool = False):
    """Demonstrates all three segmentation engines on a comprehensive patent specification."""
    print("=" * 70)
    print("PatentRank — Phase 4 Text Segmentation Demo")
    print("=" * 70)

    demo_text = """
A SYSTEM AND METHOD FOR ADAPTIVE MAGNETIC FIELD COMPENSATION

ABSTRACT
A system for compensating electromagnetic interfering fields is provided that includes two triaxial magnetic field sensors for outputting real sensor signals; six compensation coils, which are arranged as a cage around an object to be protected, and may individually be actuated; a control unit having six inputs, and six outputs, and a digital processor receiving the sensor signals on the input side, and processing the signals to control signals for the compensation coils.

CROSS-REFERENCE TO RELATED APPLICATIONS
[0001] This application is a continuation of application Ser. No. 10/445,861 filed May 27, 2003, which is a continuation of application Ser. No. 10/032,853 filed Oct. 25, 2001 and now U.S. Pat. No. 6,772,064.

BACKGROUND OF THE INVENTION
[0002] 1. Field of the Invention
[0003] The present invention relates to magnetic field compensation systems, and more particularly relates to compensating dynamic magnetic interference around sensitive electron microscopes.
[0004] 2. Description of the Related Art
[0005] Conventional systems suffer from phase delays and spatial inhomogeneity when compensating dynamic environmental noise. Existing analog feedback loops cannot dynamically adjust matrix transformations in real time, leading to degradation of image resolution.

SUMMARY OF THE INVENTION
[0006] It is an object of the present invention to overcome the aforementioned deficiencies.
[0007] In detail, a system for compensating electromagnetic interfering fields is provided, which has two real triaxial magnetic field sensors, three pairs of compensation coils, and one control unit in order to protect an object against influences of an interfering field.

BRIEF DESCRIPTION OF THE DRAWINGS
[0008] FIG. 1 is a schematic diagram depicting the sensor cage configuration.
[0009] FIG. 2 is a block diagram of the digital signal controller matrix multiplication.

DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS
[0010] Referring now to FIG. 1, two triaxial magnetic field sensors are positioned orthogonally.
[0011] The six output signals are converted to virtual sensor signals by a first matrix multiplication. The virtual signals are mapped to control currents for six compensation coils.

WHAT IS CLAIMED IS:
1. An electromagnetic compensation system comprising:
   two triaxial magnetic field sensors configured to output real sensor signals;
   six compensation coils arranged as a cage around an object to be protected; and
   a digital processor configured to convert the real sensor signals to virtual sensor signals via matrix multiplication and drive the six compensation coils.
2. The system of claim 1, further comprising:
   a field programmable gate array coupled to the digital processor to accelerate matrix calculations.
3. The system of claim 1, wherein each of the two triaxial magnetic field sensors outputs three orthogonal field components.
4. A method for compensating magnetic interfering fields, comprising:
   measuring interfering magnetic field components using multiple triaxial sensors;
   multiplying the measured components by a transformation matrix to synthesize virtual sensor coordinates; and
   generating actuation currents in six surrounding cage coils based on the synthesized coordinates.
5. The method according to claim 4, wherein the transformation matrix is updated dynamically based on real-time feedback signals.
"""

    segmenter = RegexPatentSegmenter()
    segmented = segmenter.segment_document("DEMO-US-001", "Adaptive Magnetic Field Compensation", demo_text)

    print(f"\n[1] Rule-Based Parser Statistics:")
    for k, v in segmented.stats.items():
        print(f"    {k}: {v}")

    print(f"\n[2] Extracted Functional Sections ({len(segmented.sections)}):")
    for s in segmented.sections:
        print(f"    - [{s.segment_type:20}] {s.heading:35} | {s.word_count} words")

    print(f"\n[3] Extracted Paragraphs ({len(segmented.paragraphs)}):")
    for p in segmented.paragraphs[:4]:
        print(f"    - {p.heading}: {p.text[:90]}...")

    print(f"\n[4] Parsed Claim Hierarchy ({len(segmented.claims)}):")
    for c in segmented.claims:
        cid = c.metadata.get("claim_id")
        ctype = c.metadata.get("claim_type")
        parents = c.metadata.get("parent_claims", [])
        trans = c.metadata.get("transitional_phrase", "")
        lim_cnt = c.metadata.get("limitations_count", 0)
        parent_str = f"-> Depends on Claim {parents}" if parents else "-> Independent Root Claim"
        print(f"    - Claim {cid} [{ctype}] {parent_str}")
        print(f"        Preamble: {c.metadata.get('preamble')[:70]}")
        print(f"        Transition: '{trans}' | Limitations: {lim_cnt}")

    # ML Classifier Demo
    print(f"\n[5] ML Boundary Classifier Demonstration:")
    ml_segmenter = MLBoundarySegmenter()
    if os.path.exists(DEFAULT_ML_MODEL_PATH):
        predictions = ml_segmenter.predict_boundaries(demo_text)
        boundary_lines = [p for p in predictions if p["is_boundary"]]
        print(f"    Detected {len(boundary_lines)} boundary lines with ML classifier:")
        for bl in boundary_lines[:5]:
            print(f"      Line {bl['line_idx']:2d} (conf {bl['confidence']:.2f}): {bl['text']}")
    else:
        print("    [Info] ML model not yet trained. Run --mode train_ml first.")

    # LLM Structuring Demo
    if use_llm:
        print(f"\n[6] Gemini LLM Semantic Structuring:")
        gemini_seg = GeminiSemanticSegmenter()
        structured = gemini_seg.structure_disclosure("DEMO-US-001", segmented.title, demo_text)
        print("    Problem Statement:", structured.get("problem_statement"))
        print("    Inventive Core:   ", structured.get("inventive_core"))
        print("    Key Elements:     ", structured.get("technical_elements"))
        print("    Claim Summary:    ", structured.get("claim_summary"))

    print("\n" + "=" * 70)


def run_benchmark():
    """Runs end-to-end benchmark across datasets, ML boundary model, and downstream retrieval."""
    print("=" * 70)
    print("PatentRank — Running Phase 4 Comprehensive Benchmark")
    print("=" * 70)

    # 1. Load full patent sample and raw patents
    full_sample_path = os.path.join(DEFAULT_DATA_DIR, "patents_full_sample.jsonl")
    raw_patents_path = os.path.join(DEFAULT_DATA_DIR, "patents_raw.jsonl")
    queries_path = os.path.join(DEFAULT_DATA_DIR, "queries_labeled.jsonl")

    full_patents = []
    if os.path.exists(full_sample_path):
        with open(full_sample_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    full_patents.append(json.loads(line))
        print(f"Loaded {len(full_patents)} full patent specifications from {full_sample_path}.")

    raw_patents = []
    with open(raw_patents_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            raw_patents.append(json.loads(line))
    print(f"Loaded {len(raw_patents)} local corpus documents from {raw_patents_path}.")

    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} labeled evaluation queries.")

    # 2. Benchmark Rule-Based Parser
    print("\n--- [Step 1] Benchmarking Rule-Based Parser ---")
    segmenter = RegexPatentSegmenter()
    parse_times = []
    section_counts = []
    para_counts = []
    claim_counts = []

    segmented_records = []
    eval_docs = full_patents if full_patents else raw_patents[:50]

    for p in eval_docs:
        did = p.get("sample_id", p.get("doc_id", "DOC"))
        title = p.get("title", "")
        text = p.get("full_description", "") or p.get("search_text", "")
        abstract = p.get("abstract", "")

        t0 = time.perf_counter()
        seg_doc = segmenter.segment_document(did, title, text, abstract)
        elapsed = (time.perf_counter() - t0) * 1000.0

        parse_times.append(elapsed)
        section_counts.append(len(seg_doc.sections))
        para_counts.append(len(seg_doc.paragraphs))
        claim_counts.append(len(seg_doc.claims))
        segmented_records.append(seg_doc.to_dict())

    # Save segmented samples
    segmented_out_path = os.path.join(DEFAULT_DATA_DIR, "patents_segmented_sample.jsonl")
    with open(segmented_out_path, "w", encoding="utf-8") as f:
        for rec in segmented_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(segmented_records)} segmented patent specifications to {segmented_out_path}.")

    regex_metrics = {
        "evaluated_documents": len(eval_docs),
        "mean_parse_time_ms": round(float(np.mean(parse_times)), 3),
        "p95_parse_time_ms": round(float(np.percentile(parse_times, 95)), 3),
        "mean_sections_per_doc": round(float(np.mean(section_counts)), 1),
        "mean_paragraphs_per_doc": round(float(np.mean(para_counts)), 1),
        "mean_claims_per_doc": round(float(np.mean(claim_counts)), 1)
    }
    print("Regex Parser Metrics:", json.dumps(regex_metrics, indent=2))

    # 3. Train & Evaluate ML Boundary Classifier
    print("\n--- [Step 2] Training & Evaluating ML Boundary Classifier ---")
    ml_segmenter = MLBoundarySegmenter()
    ml_train_corpus = full_patents if full_patents else raw_patents[:100]
    ml_metrics = ml_segmenter.train_and_evaluate(ml_train_corpus)
    print("ML Boundary Classifier Metrics:", json.dumps(ml_metrics, indent=2))

    # 4. Evaluate Downstream Retrieval (Monolithic vs. Segmented MaxP)
    print("\n--- [Step 3] Downstream Retrieval Benchmark (Monolithic vs Segmented) ---")
    target_doc_ids = set()
    for q in queries:
        target_doc_ids.add(q["relevant_doc_id"])
        target_doc_ids.add(q["irrelevant_doc_id"])

    eval_patents = []
    with open(raw_patents_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["doc_id"] in target_doc_ids:
                eval_patents.append(doc)
    print(f"Loaded {len(eval_patents)} target & distractor patents for downstream retrieval benchmark.")

    retrieval_eval = SegmentRetrievalEvaluator(eval_patents, queries)
    retrieval_metrics = retrieval_eval.run_comparison(num_queries=len(queries))
    print("Retrieval Comparison Results:", json.dumps(retrieval_metrics, indent=2))

    # 5. Export combined metrics
    all_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule_based_segmenter": regex_metrics,
        "ml_boundary_classifier": ml_metrics,
        "retrieval_evaluation": retrieval_metrics
    }

    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(DEFAULT_RESULTS_DIR, "segmentation_metrics.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[OK] Successfully persisted benchmark results to {results_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Patent Document Segmentation Engine (PatentRank Phase 4)")
    parser.add_argument("--mode", choices=["demo", "benchmark", "train_ml", "segment_file"], default="demo",
                        help="Execution mode (default: demo)")
    parser.add_argument("--input", type=str, default="", help="Path to input file for segment_file mode")
    parser.add_argument("--output", type=str, default="", help="Path to output file")
    parser.add_argument("--use_llm", action="store_true", help="Enable Gemini LLM semantic structuring")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(use_llm=args.use_llm)
    elif args.mode == "benchmark":
        run_benchmark()
    elif args.mode == "train_ml":
        full_sample_path = os.path.join(DEFAULT_DATA_DIR, "patents_full_sample.jsonl")
        patents = []
        with open(full_sample_path, "r", encoding="utf-8") as f:
            for line in f:
                patents.append(json.loads(line))
        ml_segmenter = MLBoundarySegmenter()
        metrics = ml_segmenter.train_and_evaluate(patents)
        print("Training complete. Metrics:", json.dumps(metrics, indent=2))
    elif args.mode == "segment_file":
        if not args.input or not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found.")
            sys.exit(1)
        segmenter = RegexPatentSegmenter()
        out_path = args.output or "segmented_output.jsonl"
        with open(args.input, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                doc = json.loads(line)
                res = segmenter.segment_document(
                    doc.get("doc_id", "DOC"),
                    doc.get("title", ""),
                    doc.get("full_description", "") or doc.get("search_text", ""),
                    doc.get("abstract", "")
                )
                fout.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
        print(f"Successfully segmented {args.input} -> {out_path}")


if __name__ == "__main__":
    main()
