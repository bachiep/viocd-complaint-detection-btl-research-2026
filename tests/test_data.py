"""Offline contract tests for the pinned-data loader using synthetic review text."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unicodedata
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_btl.data import DataIntegrityError, load_dataset


REVISION = "synthetic-revision"
BASE_URL = "https://example.invalid/pinned/"
FIELDS = ["review", "label", "domain"]
ROWS = {
    "test": [
        {"review": "Café review", "label": "0", "domain": "app"},
        {"review": "  cafe\u0301   review  ", "label": "0", "domain": "app"},
        {"review": "shared", "label": "1", "domain": "mobile"},
        {"review": "   ", "label": "0", "domain": "app"},
        {"review": "conflict", "label": "0", "domain": "app"},
        {"review": "CONFLICT", "label": "1", "domain": "app"},
        {"review": None, "label": "1", "domain": "app"},
        {"review": "missing label", "label": None, "domain": "app"},
    ],
    "val": [
        {"review": "SHARED", "label": "1", "domain": "mobile"},
        {"review": "val only", "label": "1", "domain": "fashion"},
        {"review": " val   only ", "label": "1", "domain": "fashion"},
    ],
    "train": [
        {"review": "VAL ONLY", "label": "1", "domain": "fashion"},
        {"review": "train only", "label": "0", "domain": "cosmetic"},
        {"review": " train   only ", "label": "0", "domain": "cosmetic"},
    ],
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def csv_bytes(rows: list[dict[str, str]], fields: list[str] = FIELDS) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def expected_manifest(
    payloads: dict[str, bytes],
    rows_by_split: dict[str, list[dict[str, str]]],
) -> dict:
    """Small independent reference for the synthetic fixture's expected cohort."""
    seen: set[str] = set()
    splits: dict[str, dict] = {}
    for split in ("test", "val", "train"):
        rows = rows_by_split[split]
        candidates = []
        missing = 0
        blank = 0
        for source_row, row in enumerate(rows):
            review, label = row.get("review"), row.get("label")
            if review in (None, "") or label in (None, ""):
                missing += 1
                continue
            text = normalize(review)
            if not text:
                blank += 1
                continue
            candidates.append((source_row, text, int(label), row.get("domain", "")))
        labels_by_text: dict[str, set[int]] = {}
        for _, text, label, _ in candidates:
            labels_by_text.setdefault(text, set()).add(label)
        conflicts = {text for text, labels in labels_by_text.items() if len(labels) > 1}
        consistent = [row for row in candidates if row[1] not in conflicts]
        duplicate_texts: set[str] = set()
        deduplicated = []
        for row in consistent:
            if row[1] in duplicate_texts:
                continue
            duplicate_texts.add(row[1])
            deduplicated.append(row)
        overlap_rows = [row for row in deduplicated if row[1] in seen]
        kept = [row for row in deduplicated if row[1] not in seen]
        seen.update(row[1] for row in kept)
        membership = "".join(
            f"{split}\t{source_row}\t{label}\t{domain}\t{sha256(text.encode('utf-8'))}\n"
            for source_row, text, label, domain in kept
        ).encode("utf-8")
        prediction_binding = "".join(
            f"{split}\t{source_row}\t{label}\t{domain}\n"
            for source_row, _, label, domain in kept
        ).encode("utf-8")
        labels: dict[str, int] = {}
        domains: dict[str, int] = {}
        for _, _, label, _ in kept:
            labels[str(label)] = labels.get(str(label), 0) + 1
        for _, _, _, domain in kept:
            domains[domain] = domains.get(domain, 0) + 1
        payload = payloads[split]
        splits[split] = {
            "source_file": f"{split}.csv",
            "source_bytes": len(payload),
            "source_sha256": sha256(payload),
            "raw_rows": len(rows),
            "missing_rows": missing,
            "blank_rows": blank,
            "conflict_rows": sum(1 for _, text, _, _ in candidates if text in conflicts),
            "duplicate_rows": len(consistent) - len(deduplicated),
            "cross_split_overlap_removed": len(overlap_rows),
            "retained_rows": len(kept),
            "labels": dict(sorted(labels.items(), key=lambda item: int(item[0]))),
            "domains": dict(sorted(domains.items())),
            "ordered_membership_digest_sha256": sha256(membership),
            "ordered_prediction_binding_digest_sha256": sha256(prediction_binding),
        }
    return {
        "schema_version": 1,
        "manifest_id": "synthetic-fixture",
        "repository": "synthetic/test-only",
        "revision": REVISION,
        "source_base_url": BASE_URL,
        "source_description_rows": sum(len(rows_by_split[s]) for s in rows_by_split),
        "raw_rows_observed": sum(len(rows_by_split[s]) for s in rows_by_split),
        "retained_total": sum(value["retained_rows"] for value in splits.values()),
        "retention_priority": ["test", "val", "train"],
        "normalization": "NFC -> lowercase -> whitespace split/join",
        "membership_digest_serialization": "synthetic only",
        "prediction_binding_digest_serialization": "synthetic only",
        "splits": splits,
    }


class DataLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.preparation = Path(self.temp.name)
        self.rows = {name: [dict(row) for row in values] for name, values in ROWS.items()}
        self.payloads = {name: csv_bytes(values) for name, values in self.rows.items()}
        self.manifest = expected_manifest(self.payloads, self.rows)
        self.write_contract()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_contract(self) -> None:
        manifest_bytes = (json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        (self.preparation / "data_manifest.json").write_bytes(manifest_bytes)
        protocol = {
            "protocol_id": "synthetic-test-protocol",
            "environment": {"explicit_package_targets": {"pandas": "2.3.3"}},
            "dataset": {
                "revision": REVISION,
                "source_base_url": BASE_URL,
                "manifest_file": "data_manifest.json",
                "manifest_sha256": sha256(manifest_bytes),
                "retention_priority": ["test", "val", "train"],
                "expected_counts": {
                    name: self.manifest["splits"][name]["retained_rows"]
                    for name in ("train", "val", "test")
                },
                "domains": ["app", "cosmetic", "fashion", "mobile"],
            },
        }
        protocol_bytes = json.dumps(protocol).encode("utf-8")
        (self.preparation / "protocol.json").write_bytes(protocol_bytes)
        lock = {
            "hashes": {
                "protocol.json": sha256(protocol_bytes),
                "data_manifest.json": sha256(manifest_bytes),
            }
        }
        (self.preparation / "preparation_lock.json").write_text(json.dumps(lock), encoding="utf-8")

    def fetcher(self, url: str) -> bytes:
        name = url.removeprefix(BASE_URL).removesuffix(".csv")
        return self.payloads[name]

    def test_reconstructs_development_cohort_and_stable_source_keys(self) -> None:
        bundle = load_dataset(self.preparation / "protocol.json", self.fetcher)
        self.assertEqual(set(bundle.splits), {"train", "val"})
        self.assertEqual([record.key for record in bundle.splits["val"]], [(REVISION, "val", 1)])
        self.assertEqual(
            [record.key for record in bundle.splits["train"]],
            [(REVISION, "train", 1)],
        )
        self.assertEqual(bundle.audit["test"]["retained_rows"], 2)
        self.assertEqual(bundle.audit["test"]["missing_rows"], 2)
        self.assertEqual(bundle.audit["test"]["conflict_rows"], 2)
        self.assertEqual(bundle.audit["test"]["duplicate_rows"], 1)
        self.assertEqual(bundle.audit["val"]["cross_split_overlap_removed"], 1)
        self.assertEqual(bundle.audit["train"]["cross_split_overlap_removed"], 1)

    def test_aggregate_report_contains_no_review_text_or_row_ids(self) -> None:
        report = json.dumps(
            load_dataset(self.preparation / "protocol.json", self.fetcher).audit_report(),
            ensure_ascii=False,
        )
        self.assertNotIn("café review", report)
        self.assertNotIn("source_row", report)
        self.assertIn('"raw_review_text_saved": false', report)
        self.assertIn('"test_review_rows_returned": false', report)

    def test_tampered_membership_digest_fails_closed(self) -> None:
        self.manifest["splits"]["test"]["ordered_membership_digest_sha256"] = "0" * 64
        self.write_contract()
        with self.assertRaisesRegex(DataIntegrityError, "test.canonical_cohort"):
            load_dataset(self.preparation / "protocol.json", self.fetcher)

    def test_tampered_prediction_binding_digest_fails_closed(self) -> None:
        self.manifest["splits"]["test"]["ordered_prediction_binding_digest_sha256"] = "0" * 64
        self.write_contract()
        with self.assertRaisesRegex(DataIntegrityError, "test.canonical_cohort"):
            load_dataset(self.preparation / "protocol.json", self.fetcher)

    def test_modified_protocol_fails_before_source_fetch(self) -> None:
        protocol_path = self.preparation / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["dataset"]["manifest_sha256"] = "0" * 64
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        called = False

        def should_not_fetch(_: str) -> bytes:
            nonlocal called
            called = True
            raise AssertionError("source fetch must not run")

        with self.assertRaisesRegex(DataIntegrityError, "protocol_sha256"):
            load_dataset(protocol_path, should_not_fetch)
        self.assertFalse(called)

    def test_source_byte_change_fails_before_csv_parsing(self) -> None:
        original = self.payloads["test"]
        self.payloads["test"] = original.replace(b"Caf", b"Xaf", 1)
        with self.assertRaisesRegex(DataIntegrityError, "test.source_sha256"):
            load_dataset(self.preparation / "protocol.json", self.fetcher)

    def test_nonbinary_label_is_rejected_even_when_source_hash_matches_manifest(self) -> None:
        self.rows["test"][0]["label"] = "2"
        self.payloads = {name: csv_bytes(values) for name, values in self.rows.items()}
        self.manifest = expected_manifest(self.payloads, self.rows)
        self.write_contract()
        with self.assertRaisesRegex(DataIntegrityError, "Nonbinary source label"):
            load_dataset(self.preparation / "protocol.json", self.fetcher)

    def test_missing_required_column_is_rejected(self) -> None:
        self.payloads["test"] = csv_bytes(self.rows["test"], fields=["review", "label"])
        self.manifest["splits"]["test"]["source_bytes"] = len(self.payloads["test"])
        self.manifest["splits"]["test"]["source_sha256"] = sha256(self.payloads["test"])
        self.write_contract()
        with self.assertRaisesRegex(DataIntegrityError, "lacks required columns"):
            load_dataset(self.preparation / "protocol.json", self.fetcher)

    def test_missing_domain_fails_closed(self) -> None:
        self.rows["test"][0]["domain"] = ""
        self.payloads = {name: csv_bytes(values) for name, values in self.rows.items()}
        self.manifest = expected_manifest(self.payloads, self.rows)
        self.write_contract()
        with self.assertRaisesRegex(DataIntegrityError, "Missing source domain"):
            load_dataset(self.preparation / "protocol.json", self.fetcher)
    def test_unknown_domain_is_rejected(self) -> None:
        self.rows["test"][0]["domain"] = "unknown"
        self.payloads = {name: csv_bytes(values) for name, values in self.rows.items()}
        self.manifest = expected_manifest(self.payloads, self.rows)
        self.write_contract()
        with self.assertRaisesRegex(DataIntegrityError, "Unexpected source domain"):
            load_dataset(self.preparation / "protocol.json", self.fetcher)


if __name__ == "__main__":
    unittest.main()
