"""Load and verify the pinned ViOCD cohort without persisting review text."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping
import unicodedata
from urllib.request import urlopen

import pandas as pd


class DataIntegrityError(ValueError):
    """Raised when a source or cohort differs from the locked data contract."""


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """One retained review with a stable, source-derived key."""

    revision: str
    split: str
    source_row: int
    text: str
    label: int
    domain: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.revision, self.split, self.source_row)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Verified in-memory development cohort and aggregate audit metadata."""

    protocol_id: str
    revision: str
    manifest_sha256: str
    protocol_sha256: str
    loader_sha256: str
    pandas_version: str
    splits: Mapping[str, tuple[ReviewRecord, ...]]
    audit: Mapping[str, Mapping[str, object]]

    def audit_report(self) -> dict[str, object]:
        """Return aggregate evidence without review text or row-level keys."""
        return {
            "schema_version": 1,
            "status": "PASS",
            "protocol_id": self.protocol_id,
            "dataset_revision": self.revision,
            "manifest_sha256": self.manifest_sha256,
            "protocol_sha256": self.protocol_sha256,
            "loader_sha256": self.loader_sha256,
            "pandas_version": self.pandas_version,
            "splits": {name: dict(values) for name, values in self.audit.items()},
            "normalized_text_splits_disjoint": True,
            "development_exposed_splits": list(self.splits),
            "test_review_rows_returned": False,
            "raw_review_text_saved": False,
            "row_level_ids_saved": False,
            "training_performed": False,
            "limits": [
                "The pinned cohort is historical and has prior test exposure.",
                "Exact normalized-text disjointness does not establish user/product independence.",
            ],
        }


@dataclass(frozen=True, slots=True)
class TerminalTestBundle:
    """Test records exposed only after validation selection is verified."""

    protocol_id: str
    revision: str
    manifest_sha256: str
    protocol_sha256: str
    selection_sha256: str
    source_sha256: str
    loader_sha256: str
    records: tuple[ReviewRecord, ...]
    audit: Mapping[str, object]
    selection: Mapping[str, object]

    def audit_report(self) -> dict[str, object]:
        """Return terminal-test provenance without review text or row keys."""
        return {
            "schema_version": 1,
            "status": "PASS",
            "protocol_id": self.protocol_id,
            "dataset_revision": self.revision,
            "manifest_sha256": self.manifest_sha256,
            "protocol_sha256": self.protocol_sha256,
            "selection_sha256": self.selection_sha256,
            "test_source_sha256": self.source_sha256,
            "loader_sha256": self.loader_sha256,
            "test_rows": len(self.records),
            "raw_review_text_saved": False,
            "row_level_ids_saved": False,
            "limits": [
                "The pinned cohort is historical and has prior test exposure.",
                "This terminal bundle does not establish user or product independence.",
            ],
        }


def normalize_review(value: str) -> str:
    """Apply the exact normalization used by the locked cohort manifest."""
    if not isinstance(value, str):
        raise DataIntegrityError("Review values must be strings")
    return " ".join(unicodedata.normalize("NFC", value).lower().split())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ordered_prediction_binding_digest(
    rows: Iterable[tuple[str, int, int, str]],
) -> str:
    payload = "".join(
        f"{split}\t{source_row}\t{label}\t{domain}\n"
        for split, source_row, label, domain in sorted(rows, key=lambda row: row[1])
    ).encode("utf-8")
    return _sha256(payload)


def _fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=40) as response:
        return response.read()


def _require_equal(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise DataIntegrityError(f"Pinned data contract mismatch at {field}")


def _parse_split(
    split: str,
    payload: bytes,
    expected: Mapping[str, object],
    revision: str,
    seen_text: set[str],
) -> tuple[tuple[ReviewRecord, ...], dict[str, object]]:
    """Validate, normalize, deduplicate and fingerprint one source CSV."""
    actual_sha = _sha256(payload)
    _require_equal(actual_sha, expected.get("source_sha256"), f"{split}.source_sha256")
    _require_equal(len(payload), expected.get("source_bytes"), f"{split}.source_bytes")

    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise DataIntegrityError(f"Cannot parse pinned source CSV for {split}") from exc
    if not {"review", "label", "domain"}.issubset(raw.columns):
        raise DataIntegrityError(f"Pinned source CSV for {split} lacks required columns")

    frame = raw.dropna(subset=["review", "label"]).copy()
    missing = len(raw) - len(frame)
    if not frame["review"].map(lambda value: isinstance(value, str)).all():
        raise DataIntegrityError(f"Non-string review value in {split}")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise DataIntegrityError(f"Nonbinary source label in {split}")
    frame["label"] = labels.astype("int64")
    if frame["domain"].isna().any():
        raise DataIntegrityError(f"Missing source domain in {split}")
    if not frame["domain"].map(lambda value: isinstance(value, str)).all():
        raise DataIntegrityError(f"Non-string source domain in {split}")
    allowed_domains = {"app", "cosmetic", "fashion", "mobile"}
    if not set(frame["domain"]).issubset(allowed_domains):
        raise DataIntegrityError(f"Unexpected source domain in {split}")

    frame["text"] = frame["review"].map(normalize_review)
    blank = int(frame["text"].eq("").sum())
    frame = frame.loc[frame["text"].ne("")]
    conflict = frame.groupby("text")["label"].transform("nunique").gt(1)
    consistent = frame.loc[~conflict]
    deduplicated = consistent.drop_duplicates("text")
    overlap = deduplicated["text"].isin(seen_text)
    kept = deduplicated.loc[~overlap]
    seen_text.update(kept["text"])

    records = tuple(
        ReviewRecord(
            revision=revision,
            split=split,
            source_row=int(index),
            text=str(row.text),
            label=int(row.label),
            domain=str(row.domain),
        )
        for index, row in kept.iterrows()
    )
    membership = "".join(
        f"{split}\t{record.source_row}\t{record.label}\t{record.domain}\t"
        f"{_sha256(record.text.encode('utf-8'))}\n"
        for record in sorted(records, key=lambda item: item.source_row)
    ).encode("utf-8")
    prediction_binding_digest = _ordered_prediction_binding_digest(
        (record.split, record.source_row, record.label, record.domain)
        for record in records
    )
    label_counts = Counter(record.label for record in records)
    domain_counts = Counter(record.domain for record in records)
    audit: dict[str, object] = {
        "source_file": f"{split}.csv",
        "source_bytes": len(payload),
        "source_sha256": actual_sha,
        "raw_rows": len(raw),
        "missing_rows": missing,
        "blank_rows": blank,
        "conflict_rows": int(conflict.sum()),
        "duplicate_rows": int(len(consistent) - len(deduplicated)),
        "cross_split_overlap_removed": int(overlap.sum()),
        "retained_rows": len(records),
        "labels": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "domains": {key: int(value) for key, value in sorted(domain_counts.items())},
        "ordered_membership_digest_sha256": _sha256(membership),
        "ordered_prediction_binding_digest_sha256": prediction_binding_digest,
    }
    _require_equal(audit, dict(expected), f"{split}.canonical_cohort")
    return records, audit


def load_dataset(
    protocol_path: str | Path,
    fetch_bytes: Callable[[str], bytes] = _fetch_bytes,
) -> DatasetBundle:
    """Verify the pinned cohort and return only train/validation records.

    Test is fetched and audited for cohort integrity but never returned. A
    separate M1-05 terminal evaluator must expose test only after validating
    the saved model-selection artifact.
    """
    protocol_file = Path(protocol_path).resolve(strict=True)
    try:
        protocol_bytes = protocol_file.read_bytes()
        protocol = json.loads(protocol_bytes)
        dataset = protocol["dataset"]
        manifest_name = dataset["manifest_file"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataIntegrityError("Cannot read dataset settings from protocol") from exc
    if not isinstance(protocol, dict) or not isinstance(dataset, dict):
        raise DataIntegrityError("Protocol has an invalid schema")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise DataIntegrityError("Manifest path must be a filename within preparation")

    protocol_sha = _sha256(protocol_bytes)
    lock_path = protocol_file.parent / "preparation_lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError("Preparation lock is unavailable or invalid") from exc
    if not isinstance(lock, dict) or not isinstance(lock.get("hashes"), dict):
        raise DataIntegrityError("Preparation lock has an invalid schema")
    lock_hashes = lock["hashes"]
    _require_equal(lock_hashes.get("protocol.json"), protocol_sha, "protocol_sha256")

    manifest_path = protocol_file.parent / manifest_name
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise DataIntegrityError("Pinned data manifest is unavailable") from exc
    manifest_sha = _sha256(manifest_bytes)
    _require_equal(manifest_sha, dataset.get("manifest_sha256"), "manifest_sha256")
    _require_equal(lock_hashes.get(manifest_name), manifest_sha, "locked_manifest_sha256")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise DataIntegrityError("Pinned data manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise DataIntegrityError("Pinned data manifest has an invalid schema")

    revision = dataset.get("revision")
    base_url = dataset.get("source_base_url")
    priority = dataset.get("retention_priority")
    if not isinstance(revision, str) or not isinstance(base_url, str):
        raise DataIntegrityError("Protocol is missing pinned dataset identity")
    _require_equal(manifest.get("revision"), revision, "dataset.revision")
    _require_equal(manifest.get("source_base_url"), base_url, "dataset.source_base_url")
    _require_equal(manifest.get("retention_priority"), priority, "retention_priority")
    _require_equal(
        manifest.get("normalization"),
        "NFC -> lowercase -> whitespace split/join",
        "normalization",
    )
    domains = dataset.get("domains")
    if not isinstance(domains, list):
        raise DataIntegrityError("Protocol domain list is invalid")
    _require_equal(set(domains), {"app", "cosmetic", "fashion", "mobile"}, "dataset.domains")
    if not isinstance(priority, list) or set(priority) != {"train", "val", "test"}:
        raise DataIntegrityError("Unexpected split retention order")

    expected_counts = dataset.get("expected_counts")
    manifest_splits = manifest.get("splits")
    if not isinstance(expected_counts, dict) or not isinstance(manifest_splits, dict):
        raise DataIntegrityError("Protocol or manifest split counts are missing")
    for split, count in expected_counts.items():
        expected_split = manifest_splits.get(split)
        if not isinstance(expected_split, dict):
            raise DataIntegrityError(f"Missing canonical split {split}")
        _require_equal(expected_split.get("retained_rows"), count, f"{split}.retained_rows")
    _require_equal(set(expected_counts), {"train", "val", "test"}, "expected_counts")
    _require_equal(
        manifest.get("retained_total"), sum(expected_counts.values()), "retained_total"
    )

    environment = protocol.get("environment", {})
    if not isinstance(environment, dict):
        raise DataIntegrityError("Protocol environment settings are invalid")
    package_targets = environment.get("explicit_package_targets", {})
    if not isinstance(package_targets, dict):
        raise DataIntegrityError("Protocol package targets are invalid")
    target_pandas = package_targets.get("pandas")
    if target_pandas is not None:
        _require_equal(pd.__version__, target_pandas, "pandas_version")

    all_splits: dict[str, tuple[ReviewRecord, ...]] = {}
    audits: dict[str, Mapping[str, object]] = {}
    seen_text: set[str] = set()
    for split in priority:
        expected = manifest_splits[split]
        expected_file = f"{split}.csv"
        _require_equal(expected.get("source_file"), expected_file, f"{split}.source_file")
        try:
            payload = fetch_bytes(base_url + expected_file)
        except Exception as exc:
            raise DataIntegrityError(f"Could not retrieve pinned source {expected_file}") from exc
        if not isinstance(payload, bytes):
            raise DataIntegrityError(f"Source reader did not return bytes for {split}")
        records, audit = _parse_split(split, payload, expected, revision, seen_text)
        all_splits[split] = records
        audits[split] = MappingProxyType(audit)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_text = {record.text for record in all_splits[left]}
        if left_text.intersection(record.text for record in all_splits[right]):
            raise DataIntegrityError(f"Normalized text overlap remains across {left}/{right}")

    return DatasetBundle(
        protocol_id=str(protocol.get("protocol_id", "")),
        revision=revision,
        manifest_sha256=manifest_sha,
        protocol_sha256=protocol_sha,
        loader_sha256=_sha256(Path(__file__).resolve().read_bytes()),
        pandas_version=pd.__version__,
        splits=MappingProxyType({"train": all_splits["train"], "val": all_splits["val"]}),
        audit=MappingProxyType(audits),
    )


def load_terminal_test(
    protocol_path: str | Path,
    selection_path: str | Path,
    fetch_bytes: Callable[[str], bytes] = _fetch_bytes,
) -> TerminalTestBundle:
    """Verify saved selection before retrieving only the pinned test split."""
    from .selection import _read_locked_protocol, verify_selection_artifact

    selection, selection_sha256 = verify_selection_artifact(
        selection_path, protocol_path
    )
    protocol, protocol_sha256, manifest_sha256, revision = _read_locked_protocol(
        protocol_path
    )
    protocol_file = Path(protocol_path).resolve(strict=True)
    dataset = protocol.get("dataset")
    if not isinstance(dataset, dict):
        raise DataIntegrityError("Locked protocol lacks dataset settings")
    manifest_name = dataset.get("manifest_file")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise DataIntegrityError("Manifest path must be a filename within preparation")
    try:
        manifest = json.loads((protocol_file.parent / manifest_name).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError("Pinned data manifest is unavailable or invalid") from exc
    if not isinstance(manifest, dict):
        raise DataIntegrityError("Pinned data manifest has an invalid schema")

    expected_counts = dataset.get("expected_counts")
    manifest_splits = manifest.get("splits")
    priority = dataset.get("retention_priority")
    if (
        not isinstance(expected_counts, dict)
        or not isinstance(manifest_splits, dict)
        or not isinstance(priority, list)
        or set(expected_counts) != {"train", "val", "test"}
        or set(manifest_splits) != {"train", "val", "test"}
        or set(priority) != {"train", "val", "test"}
        or priority[0] != "test"
    ):
        raise DataIntegrityError("Locked split contract is incomplete for terminal evaluation")
    for split, count in expected_counts.items():
        expected_split = manifest_splits[split]
        if not isinstance(expected_split, dict) or expected_split.get("retained_rows") != count:
            raise DataIntegrityError(f"Locked manifest count differs for {split}")
    validation_split = manifest_splits.get("val")
    if (
        not isinstance(validation_split, dict)
        or selection.get("validation_membership_sha256")
        != validation_split.get("ordered_membership_digest_sha256")
    ):
        raise DataIntegrityError("Selection validation population differs from the locked cohort")
    if manifest.get("retained_total") != sum(expected_counts.values()):
        raise DataIntegrityError("Locked manifest total differs from the protocol")
    _require_equal(manifest.get("revision"), revision, "dataset.revision")
    _require_equal(manifest.get("source_base_url"), dataset.get("source_base_url"), "dataset.source_base_url")
    _require_equal(manifest.get("retention_priority"), priority, "retention_priority")

    environment = protocol.get("environment")
    package_targets = environment.get("explicit_package_targets") if isinstance(environment, dict) else None
    if isinstance(package_targets, dict) and package_targets.get("pandas") is not None:
        _require_equal(pd.__version__, package_targets["pandas"], "pandas_version")

    expected_test = manifest_splits["test"]
    if not isinstance(expected_test, dict) or expected_test.get("source_file") != "test.csv":
        raise DataIntegrityError("Pinned test split metadata is invalid")
    base_url = dataset.get("source_base_url")
    if not isinstance(base_url, str):
        raise DataIntegrityError("Pinned dataset source URL is invalid")
    try:
        payload = fetch_bytes(base_url + "test.csv")
    except Exception as exc:
        raise DataIntegrityError("Could not retrieve pinned terminal test source") from exc
    if not isinstance(payload, bytes):
        raise DataIntegrityError("Terminal test source reader did not return bytes")
    records, audit = _parse_split("test", payload, expected_test, revision, set())
    if (
        selection.get("protocol_sha256") != protocol_sha256
        or selection.get("manifest_sha256") != manifest_sha256
        or selection.get("dataset_revision") != revision
    ):
        raise DataIntegrityError("Verified selection binding differs from terminal test source")
    return TerminalTestBundle(
        protocol_id=str(protocol.get("protocol_id", "")),
        revision=revision,
        manifest_sha256=manifest_sha256,
        protocol_sha256=protocol_sha256,
        selection_sha256=selection_sha256,
        source_sha256=str(audit["source_sha256"]),
        loader_sha256=_sha256(Path(__file__).resolve().read_bytes()),
        records=records,
        audit=MappingProxyType(audit),
        selection=MappingProxyType(selection),
    )
