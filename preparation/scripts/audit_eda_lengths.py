"""Aggregate-only EDA of the pinned ViOCD cohort; never writes review text or row IDs."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "preparation"
OUT = ROOT / "results" / "eda" / "viocd_length_token_audit.json"
SPLITS = ("test", "val", "train")
EXPECTED = {
    "numpy": "2.1.3",
    "pandas": "2.3.3",
    "underthesea": "9.5.0",
    "py_vncorenlp": "0.1.4",
    "pyjnius": "1.9.0",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stats(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    arr = np.asarray(values, dtype=np.float64)
    q = np.percentile(arr, [0, 25, 50, 75, 90, 95, 100], method="linear")
    return {
        "n": int(arr.size),
        "sum": int(arr.sum()),
        "mean": round(float(arr.mean()), 4),
        "min": int(q[0]),
        "p25": round(float(q[1]), 2),
        "median_p50": round(float(q[2]), 2),
        "p75": round(float(q[3]), 2),
        "p90": round(float(q[4]), 2),
        "p95": round(float(q[5]), 2),
        "max": int(q[6]),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate audit: {OUT}")
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    sys.path[:0] = [str(ROOT / "src"), str(PREP / "assets" / "python_tokenizer"), str(PREP / "assets" / "python_rdr")]

    from nlp_btl.data import _parse_split
    from nlp_btl.preprocessing import (
        RDRSegmenter,
        UndertheseaSegmenter,
    )

    for package, expected_version in EXPECTED.items():
        actual = importlib.metadata.version(package)
        if actual != expected_version:
            raise RuntimeError(f"Locked package mismatch: {package}={actual}")

    protocol_bytes = (PREP / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest_bytes = (PREP / "data_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    lock = json.loads((PREP / "preparation_lock.json").read_text(encoding="utf-8"))
    manifest_sha = sha256(manifest_bytes)
    if sha256(protocol_bytes) != lock["hashes"]["protocol.json"]:
        raise RuntimeError("Protocol does not match the preparation lock")
    if manifest_sha != protocol["dataset"]["manifest_sha256"]:
        raise RuntimeError("Manifest does not match the protocol")
    if manifest_sha != lock["hashes"]["data_manifest.json"]:
        raise RuntimeError("Manifest does not match the preparation lock")
    if manifest["revision"] != protocol["dataset"]["revision"]:
        raise RuntimeError("Dataset revision mismatch")

    tokenizer_audit_bytes = (PREP / "tokenizer_input_audit.json").read_bytes()
    tokenizer_audit = json.loads(tokenizer_audit_bytes)
    if sha256(tokenizer_audit_bytes) != lock["hashes"]["tokenizer_input_audit.json"]:
        raise RuntimeError("Tokenizer audit does not match the preparation lock")
    if tokenizer_audit["data_manifest_sha256"] != manifest_sha:
        raise RuntimeError("Tokenizer audit is bound to a different data manifest")
    if tokenizer_audit["all_checks_passed"] is not True:
        raise RuntimeError("Pinned tokenizer audit is not passing")

    seen_text: set[str] = set()
    records_by_split = {}
    source_hashes = {}
    for split in SPLITS:
        source_url = manifest["source_base_url"] + split + ".csv"
        with urlopen(source_url, timeout=60) as response:
            payload = response.read()
        source_hash = sha256(payload)
        if source_hash != manifest["splits"][split]["source_sha256"]:
            raise RuntimeError(f"Pinned source hash mismatch for {split}")
        records, _ = _parse_split(
            split,
            payload,
            manifest["splits"][split],
            manifest["revision"],
            seen_text,
        )
        if len(records) != manifest["splits"][split]["retained_rows"]:
            raise RuntimeError(f"Retained row count mismatch for {split}")
        records_by_split[split] = records
        source_hashes[split] = source_hash

    underthesea = UndertheseaSegmenter()
    rdr = RDRSegmenter(PREP)
    split_results = {}
    combined = {
        key: []
        for key in (
            "normalized_unicode_codepoints",
            "normalized_whitespace_units",
            "rdr_sentence_segments",
            "underthesea_word_tokens",
            "rdr_word_tokens",
        )
    }

    for split in SPLITS:
        records = records_by_split[split]
        texts = [record.text for record in records]
        normalized_chars = [len(text) for text in texts]
        whitespace_units = [len(text.split()) for text in texts]
        p0_segments = [underthesea(text) for text in texts]
        p1_segments = []
        sentence_counts = []
        for text in texts:
            sentences = rdr._rdr.word_segment(text)
            if not isinstance(sentences, list) or not sentences or not all(
                isinstance(sentence, str) for sentence in sentences
            ):
                raise RuntimeError(f"Invalid RDR sentence output in {split}")
            p1_segments.append(" ".join(sentences))
            sentence_counts.append(len(sentences))

        p0_word_counts = [len(value.split()) for value in p0_segments]
        p1_word_counts = [len(value.split()) for value in p1_segments]
        if any(not value for value in p0_segments + p1_segments):
            raise RuntimeError(f"Empty segmenter output in {split}")

        counts = {
            "normalized_unicode_codepoints": normalized_chars,
            "normalized_whitespace_units": whitespace_units,
            "rdr_sentence_segments": sentence_counts,
            "underthesea_word_tokens": p0_word_counts,
            "rdr_word_tokens": p1_word_counts,
        }
        for key, values in counts.items():
            combined[key].extend(values)

        label_counts = Counter(str(record.label) for record in records)
        domain_counts = Counter(record.domain for record in records)
        split_results[split] = {
            "reviews": len(records),
            "labels": dict(sorted(label_counts.items())),
            "domains": dict(sorted(domain_counts.items())),
            "lengths": {key: stats(values) for key, values in counts.items()},
            "phobert_truncation_counts_from_locked_tokenizer_audit": {
                "P0": tokenizer_audit["configurations"]["P0"]["truncation_counts"][split],
                "P1": tokenizer_audit["configurations"]["P1"]["truncation_counts"][split],
            },
        }

    total_rows = sum(len(records) for records in records_by_split.values())
    if total_rows != manifest["retained_total"]:
        raise RuntimeError("Total cohort count differs from the pinned manifest")
    split_results["overall"] = {
        "reviews": total_rows,
        "lengths": {key: stats(values) for key, values in combined.items()},
        "phobert_truncation_counts_from_locked_tokenizer_audit": {
            "P0": tokenizer_audit["configurations"]["P0"]["truncation_counts"],
            "P1": tokenizer_audit["configurations"]["P1"]["truncation_counts"],
        },
    }

    result = {
        "schema_version": 1,
        "status": "PASS",
        "audit_kind": "aggregate-only descriptive EDA; no model fit or selection",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_repository": manifest["repository"],
        "dataset_revision": manifest["revision"],
        "manifest_sha256": manifest_sha,
        "protocol_sha256": sha256(protocol_bytes),
        "tokenizer_audit_sha256": sha256(tokenizer_audit_bytes),
        "source_csv_sha256": source_hashes,
        "retention_priority": list(SPLITS),
        "unit": "one retained review; rows match the locked canonical cohort",
        "definitions": {
            "normalized_unicode_codepoints": "Python Unicode code-point length after NFC, lowercase, and whitespace collapse",
            "normalized_whitespace_units": "whitespace-separated units after the same locked normalization",
            "rdr_sentence_segments": "number of sentence strings returned by the pinned VnCoreNLP word_segment API; not manual sentence annotation",
            "underthesea_word_tokens": "space-separated tokens in pinned Underthesea 9.5.0 output",
            "rdr_word_tokens": "space-separated tokens in pinned VnCoreNLP/RDR output",
            "phobert_truncation_counts": "Counts are copied from the existing hash-locked tokenizer audit; this EDA does not run the model tokenizer",
            "percentiles": "NumPy percentile with linear interpolation; p50 is the median",
            "privacy": "No review text, source row IDs, token IDs, or per-review measurements are persisted; only split-level aggregate distributions are written",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "packages": {name: importlib.metadata.version(name) for name in EXPECTED},
            "device": "CPU only; no model weights loaded; no training/inference",
        },
        "splits": split_results,
        "limits": [
            "The cohort has prior historical test exposure; descriptive lengths do not constitute independent validation.",
            "RDR sentence spans are the segmenter's output, not human-annotated sentence boundaries.",
            "The EDA is descriptive and was not used to select a model, alter the protocol, or change the cohort.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(OUT),
        "reviews": total_rows,
        "source_hashes_match": True,
        "raw_text_saved": False,
        "token_ids_saved": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
