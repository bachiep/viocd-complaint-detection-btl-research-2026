"""Shared text processing and aggregate-only pre-fit representation audits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import sys
import tarfile
import time
from typing import Callable, Mapping, Sequence

from .data import (
    DataIntegrityError,
    DatasetBundle,
    ReviewRecord,
    _fetch_bytes,
    _parse_split,
    load_dataset,
    normalize_review,
)


Segmenter = Callable[[str], str]
SPLIT_PRIORITY = ("test", "val", "train")


def _verify_rdr_import_origin(module, python_assets: Path) -> bool:
    package_root = (python_assets / "py_vncorenlp").resolve(strict=True)
    expected_package = (package_root / "__init__.py").resolve(strict=True)
    expected_implementation = (package_root / "vncorenlp.py").resolve(strict=True)
    try:
        actual_package = Path(module.__file__).resolve(strict=True)
        actual_implementation = Path(module.vncorenlp.__file__).resolve(strict=True)
    except (AttributeError, OSError) as exc:
        raise DataIntegrityError("Cannot verify imported RDR wrapper origin") from exc
    if actual_package != expected_package or actual_implementation != expected_implementation:
        raise DataIntegrityError("Imported RDR wrapper does not match verified pinned source")
    return True


@dataclass(frozen=True, slots=True)
class ProcessedReview:
    """A source-keyed, segmented review ready for a model tokenizer."""

    key: tuple[str, str, int]
    segmented_text: str
    label: int
    domain: str


class UndertheseaSegmenter:
    """Pinned Underthesea word segmentation used by S0/P0."""

    def __init__(self) -> None:
        from importlib.metadata import version

        if version("underthesea") != "9.5.0":
            raise DataIntegrityError("Underthesea version differs from locked protocol")
        from underthesea import word_tokenize

        self._word_tokenize = word_tokenize

    def __call__(self, normalized_text: str) -> str:
        return self._word_tokenize(normalized_text, format="text")


class RDRSegmenter:
    """Pinned VnCoreNLP RDR segmenter used by P1."""

    def __init__(self, preparation_dir: str | Path) -> None:
        from importlib.metadata import version

        preparation = Path(preparation_dir).resolve(strict=True)
        python_assets = (preparation / "assets" / "python_rdr").resolve(strict=True)
        model_assets = (preparation / "assets" / "vncorenlp").resolve(strict=True)
        if str(python_assets) not in sys.path:
            sys.path.insert(0, str(python_assets))
        if version("py_vncorenlp") != "0.1.4" or version("pyjnius") != "1.9.0":
            raise DataIntegrityError("RDR Python dependencies differ from locked protocol")
        import py_vncorenlp

        self._import_origin_verified = _verify_rdr_import_origin(
            py_vncorenlp, python_assets
        )
        original_cwd = os.getcwd()
        try:
            self._rdr = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=str(model_assets)
            )
        except Exception as exc:
            raise DataIntegrityError("Cannot initialize pinned RDR segmenter") from exc
        finally:
            os.chdir(original_cwd)

    @property
    def import_origin_verified(self) -> bool:
        return self._import_origin_verified

    def __call__(self, normalized_text: str) -> str:
        try:
            sentences = self._rdr.word_segment(normalized_text)
        except Exception as exc:
            raise DataIntegrityError("RDR segmentation failed") from exc
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise DataIntegrityError("RDR returned an invalid segmentation result")
        return " ".join(sentences)


def preprocess_review(review: str, segmenter: Segmenter) -> str:
    """Apply the single normalization-plus-segmentation path used by all phases."""
    normalized = normalize_review(review)
    if not normalized:
        raise DataIntegrityError("Review is empty after normalization")
    segmented = segmenter(normalized)
    if not isinstance(segmented, str) or not segmented.strip():
        raise DataIntegrityError("Segmenter returned an empty or non-text result")
    return segmented


def preprocess_records(
    records: Sequence[ReviewRecord], segmenter: Segmenter
) -> tuple[ProcessedReview, ...]:
    """Process source records while preserving stable keys and gold labels."""
    return tuple(
        ProcessedReview(
            key=record.key,
            segmented_text=preprocess_review(record.text, segmenter),
            label=record.label,
            domain=record.domain,
        )
        for record in records
    )


def tokenize_batch(
    tokenizer,
    records: Sequence[ProcessedReview],
    max_length: int = 128,
):
    """Tokenize a batch with locked truncation and dynamic padding semantics."""
    if not records:
        raise DataIntegrityError("Cannot tokenize an empty batch")
    if max_length != 128:
        raise DataIntegrityError("max_length differs from locked PhoBERT protocol")
    return tokenizer(
        [record.segmented_text for record in records],
        truncation=True,
        max_length=max_length,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_locked_json(
    preparation: Path, lock: Mapping[str, object], name: str
) -> tuple[dict, bytes]:
    path = preparation / name
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DataIntegrityError(f"Locked preparation file is missing: {name}") from exc
    hashes = lock.get("hashes")
    if not isinstance(hashes, dict) or hashes.get(name) != _sha256(payload):
        raise DataIntegrityError(f"Locked preparation file changed: {name}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"Locked preparation JSON is invalid: {name}") from exc
    if not isinstance(value, dict):
        raise DataIntegrityError(f"Locked preparation JSON has invalid schema: {name}")
    return value, payload


def _write_new_audit_receipt(
    preparation: Path,
    output_path: Path,
    report: Mapping[str, object],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preparation_root = preparation.resolve(strict=True)
    resolved_output = output_path.parent.resolve(strict=True) / output_path.name
    try:
        resolved_output.relative_to(preparation_root)
    except ValueError:
        pass
    else:
        raise DataIntegrityError("Audit output cannot be written inside locked preparation")
    try:
        with resolved_output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise DataIntegrityError(
            "Audit output already exists; preserve prior receipt and choose a new path"
        ) from exc
    return resolved_output


def _verify_rdr_assets(preparation: Path, manifest: Mapping[str, object]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise DataIntegrityError("RDR asset manifest is invalid")
    root = preparation.resolve(strict=True)
    hashes: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise DataIntegrityError("RDR asset entry is invalid")
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str):
            raise DataIntegrityError("RDR asset path is invalid")
        path = (root / relative_path).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise DataIntegrityError("RDR asset path escapes preparation directory") from exc
        payload = path.read_bytes()
        if (
            len(payload) != artifact.get("bytes")
            or _sha256(payload) != artifact.get("sha256")
        ):
            raise DataIntegrityError(f"Pinned RDR asset hash mismatch: {relative_path}")
        hashes[relative_path] = _sha256(payload)
    return hashes


def _verify_rdr_wrapper(
    preparation: Path, protocol: Mapping[str, object]
) -> dict[str, object]:
    rdr = protocol["preprocessing"]["rdr"]
    expected_sha = rdr.get("wrapper_sdist_sha256")
    if not isinstance(expected_sha, str):
        raise DataIntegrityError("RDR wrapper source hash is missing from protocol")
    wrapper_root = (preparation / "assets" / "python_rdr").resolve(strict=True)
    archive_path = wrapper_root / "py_vncorenlp-0.1.4.tar.gz"
    try:
        payload = archive_path.read_bytes()
    except OSError as exc:
        raise DataIntegrityError("Pinned RDR wrapper source distribution is missing") from exc
    if _sha256(payload) != expected_sha:
        raise DataIntegrityError("RDR wrapper source distribution hash mismatch")
    source_paths = ("py_vncorenlp/__init__.py", "py_vncorenlp/vncorenlp.py")
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            package_root = f"py_vncorenlp-{rdr['wrapper_version']}"
            verified_hashes = {}
            for relative_path in source_paths:
                member = archive.extractfile(f"{package_root}/{relative_path}")
                if member is None:
                    raise DataIntegrityError("RDR wrapper source archive lacks a required file")
                expected_source = member.read()
                installed_path = (wrapper_root / relative_path).resolve(strict=True)
                installed_path.relative_to(wrapper_root)
                installed_source = installed_path.read_bytes()
                if installed_source != expected_source:
                    raise DataIntegrityError(
                        f"Installed RDR wrapper source differs: {relative_path}"
                    )
                verified_hashes[relative_path] = _sha256(installed_source)
    except (OSError, tarfile.TarError, KeyError, ValueError) as exc:
        if isinstance(exc, DataIntegrityError):
            raise
        raise DataIntegrityError("Cannot verify pinned RDR wrapper source files") from exc
    return {
        "sdist_sha256": expected_sha,
        "installed_source_sha256": verified_hashes,
        "installed_sources_match_sdist": True,
    }


def _validate_preprocessing_contract(
    protocol: Mapping[str, object], tokenizer_audit: Mapping[str, object]
) -> None:
    preprocessing = protocol.get("preprocessing")
    model = protocol.get("model")
    environment = protocol.get("environment")
    audit_packages = tokenizer_audit.get("packages")
    if not all(isinstance(value, dict) for value in (preprocessing, model, environment, audit_packages)):
        raise DataIntegrityError("Locked preprocessing protocol has an invalid schema")
    targets = environment.get("explicit_package_targets")
    normalization = [
        "Unicode NFC",
        "lowercase",
        "collapse whitespace with str.split then single-space join",
    ]
    if not isinstance(targets, dict) or preprocessing.get("normalization") != normalization:
        raise DataIntegrityError("Preprocessing normalization differs from locked protocol")
    if preprocessing.get("stopword_removal") is not False or preprocessing.get(
        "stemming_or_lemmatization"
    ) is not False:
        raise DataIntegrityError("Unexpected preprocessing transform in locked protocol")
    underthesea = preprocessing.get("underthesea")
    rdr = preprocessing.get("rdr")
    if not isinstance(underthesea, dict) or underthesea.get("version") != "9.5.0":
        raise DataIntegrityError("Underthesea version differs from implemented pipeline")
    if underthesea.get("call") != "word_tokenize(normalized_review, format='text')":
        raise DataIntegrityError("Underthesea call differs from implemented pipeline")
    if not isinstance(rdr, dict) or (
        rdr.get("wrapper_version") != "0.1.4"
        or rdr.get("pyjnius_version") != "1.9.0"
        or rdr.get("annotators") != ["wseg"]
    ):
        raise DataIntegrityError("RDR wrapper configuration differs from implementation")
    if targets.get("underthesea") != underthesea["version"]:
        raise DataIntegrityError("Underthesea runtime target differs from preprocessing protocol")
    if targets.get("py_vncorenlp") != rdr["wrapper_version"] or targets.get(
        "pyjnius"
    ) != rdr["pyjnius_version"]:
        raise DataIntegrityError("RDR runtime targets differ from preprocessing protocol")
    for package in ("transformers", "tokenizers", "underthesea", "py_vncorenlp", "pyjnius"):
        if targets.get(package) != audit_packages.get(package):
            raise DataIntegrityError(f"Preparation audit package target differs: {package}")
    if model.get("id") != tokenizer_audit.get("model") or model.get(
        "revision"
    ) != tokenizer_audit.get("model_revision"):
        raise DataIntegrityError("PhoBERT tokenizer audit is bound to another model revision")
    if model.get("tokenizer_use_fast") is not False or tokenizer_audit.get("use_fast") is not False:
        raise DataIntegrityError("PhoBERT tokenizer fast-mode differs from locked protocol")


def _segmentation_report(
    records_by_split: Mapping[str, Sequence[ReviewRecord]],
    outputs_by_split: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    seen: set[str] = set()
    result = []
    for split in SPLIT_PRIORITY:
        records = records_by_split[split]
        outputs = tuple(outputs_by_split[split])
        if len(records) != len(outputs):
            raise DataIntegrityError(f"Segmented row count differs for {split}")
        digest_payload = "".join(
            f"{split}\t{record.source_row}\t{_sha256(text.encode('utf-8'))}\n"
            for record, text in sorted(
                zip(records, outputs), key=lambda item: item[0].source_row
            )
        ).encode("utf-8")
        row = {
            "split": split,
            "rows": len(outputs),
            "empty_output_rows": sum(not text.strip() for text in outputs),
            "within_split_duplicate_rows": len(outputs) - len(set(outputs)),
            "cross_split_overlap_rows": sum(text in seen for text in outputs),
            "ordered_segmented_membership_digest_sha256": _sha256(digest_payload),
        }
        result.append(row)
        seen.update(outputs)
    return result


def _tokenizer_report(
    tokenizer,
    records_by_split: Mapping[str, Sequence[ReviewRecord]],
    segmented_by_split: Mapping[str, Sequence[str]],
    max_length: int,
) -> dict[str, object]:
    representations: dict[str, dict[str, object]] = {}
    for split in SPLIT_PRIORITY:
        strings = list(segmented_by_split[split])
        full = tokenizer(
            strings,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        truncated = tokenizer(
            strings,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        full_ids = [tuple(int(token_id) for token_id in row) for row in full]
        truncated_ids = [tuple(int(token_id) for token_id in row) for row in truncated]
        if (
            len(full_ids) != len(records_by_split[split])
            or len(truncated_ids) != len(full_ids)
        ):
            raise DataIntegrityError(f"Tokenized row count differs for {split}")
        representations[split] = {
            "full": full_ids,
            "truncated": truncated_ids,
            "full_lengths": [len(row) for row in full_ids],
        }

    within: dict[str, dict[str, int]] = {}
    within_labels: dict[str, dict[str, dict[str, int]]] = {}
    cross: dict[str, dict[str, dict[str, int]]] = {}
    for kind in ("full", "truncated"):
        within[kind] = {}
        within_labels[kind] = {}
        for split in SPLIT_PRIORITY:
            rows = representations[split][kind]
            labels = [record.label for record in records_by_split[split]]
            within[kind][split] = len(rows) - len(set(rows))
            grouped: dict[tuple[int, ...], list[int]] = {}
            for tokens, label in zip(rows, labels):
                grouped.setdefault(tokens, []).append(label)
            duplicate_groups = [values for values in grouped.values() if len(values) > 1]
            conflicting = [values for values in duplicate_groups if len(set(values)) > 1]
            within_labels[kind][split] = {
                "duplicate_groups": len(duplicate_groups),
                "rows_in_duplicate_groups": sum(map(len, duplicate_groups)),
                "conflicting_label_groups": len(conflicting),
                "rows_in_conflicting_label_groups": sum(map(len, conflicting)),
            }
        cross[kind] = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            left_counts = Counter(representations[left][kind])
            right_counts = Counter(representations[right][kind])
            shared = set(left_counts).intersection(right_counts)
            cross[kind][f"{left}__{right}"] = {
                "shared_unique_input_sequences": len(shared),
                "left_rows": sum(left_counts[key] for key in shared),
                "right_rows": sum(right_counts[key] for key in shared),
            }

    return {
        "rows": {split: len(records_by_split[split]) for split in SPLIT_PRIORITY},
        "truncation_counts": {
            split: sum(length > max_length for length in representations[split]["full_lengths"])
            for split in SPLIT_PRIORITY
        },
        "max_full_input_tokens": {
            split: max(representations[split]["full_lengths"]) for split in SPLIT_PRIORITY
        },
        "within_split_duplicate_rows": within,
        "within_split_duplicate_group_labels": within_labels,
        "cross_split_collisions": cross,
        "full_bpe_duplicate_includes_unk_behavior": True,
    }


def _audit_tokenizer_batch_interface(tokenizer, max_length: int) -> dict[str, object]:
    """Exercise real tensor, dynamic-padding and truncation behavior synthetically."""
    try:
        import torch
    except Exception as exc:
        raise DataIntegrityError("PyTorch is unavailable for tokenizer batch smoke") from exc

    short_batch = (
        ProcessedReview(("synthetic", "audit", 0), "tốt", 0, "app"),
        ProcessedReview(("synthetic", "audit", 1), "sản_phẩm rất tệ", 1, "cosmetic"),
    )
    padded = tokenize_batch(tokenizer, short_batch, max_length=max_length)
    input_ids = padded.get("input_ids")
    attention_mask = padded.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise DataIntegrityError("Tokenizer batch did not return PyTorch tensors")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise DataIntegrityError("Tokenizer batch returned invalid tensor shapes")
    width = int(input_ids.shape[1])
    if input_ids.shape[0] != len(short_batch) or width % 8 or width > max_length:
        raise DataIntegrityError("Tokenizer dynamic padding differs from locked batch protocol")
    mask_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
    if not any(length < width for length in mask_lengths):
        raise DataIntegrityError("Synthetic tokenizer batch did not exercise padding")
    pad_id = tokenizer.pad_token_id
    if pad_id is None or not bool(input_ids[attention_mask == 0].eq(pad_id).all()):
        raise DataIntegrityError("Tokenizer padding does not use the pinned pad token")

    long_text = "tốt " * (max_length + 32)
    full_ids = tokenizer(
        long_text,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    full_length = len(full_ids)
    if full_length <= max_length:
        raise DataIntegrityError("Synthetic long input did not exceed locked max_length")
    truncated = tokenize_batch(
        tokenizer,
        (ProcessedReview(("synthetic", "audit", 2), long_text, 1, "app"),),
        max_length=max_length,
    )
    truncated_ids = truncated["input_ids"]
    truncated_mask = truncated["attention_mask"]
    if truncated_ids.shape != (1, max_length) or int(truncated_mask.sum()) != max_length:
        raise DataIntegrityError("Tokenizer did not truncate the synthetic long input correctly")
    return {
        "status": "PASS",
        "backend": f"torch {torch.__version__}",
        "tensor_type": type(input_ids).__name__,
        "short_batch_rows": len(short_batch),
        "dynamic_padded_width": width,
        "dynamic_padding_applied": True,
        "attention_mask_lengths": mask_lengths,
        "pad_to_multiple_of": 8,
        "pad_token_id": int(pad_id),
        "synthetic_long_input_tokens_before_truncation": full_length,
        "synthetic_long_input_tokens_after_truncation": int(truncated_mask.sum()),
        "max_length": max_length,
        "synthetic_input_text_saved": False,
        "token_id_sequences_saved": False,
    }


def _load_pinned_tokenizer(
    preparation: Path,
    audit: Mapping[str, object],
    protocol: Mapping[str, object],
):
    from importlib.metadata import version

    target = preparation / "assets" / "python_tokenizer"
    if target.is_dir():
        resolved_target = target.resolve(strict=True)
        if str(resolved_target) not in sys.path:
            sys.path.insert(0, str(resolved_target))
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise DataIntegrityError("Pinned PhoBERT tokenizer packages are unavailable") from exc
    expected_packages = audit["packages"]
    for package in ("transformers", "tokenizers", "huggingface_hub"):
        expected_version = expected_packages.get(package)
        if not isinstance(expected_version, str) or version(package) != expected_version:
            raise DataIntegrityError(f"Tokenizer package version mismatch: {package}")
    assets = audit.get("model_assets")
    if not isinstance(assets, list):
        raise DataIntegrityError("Tokenizer asset manifest is invalid")
    for asset in assets:
        if not isinstance(asset, dict):
            raise DataIntegrityError("Tokenizer asset entry is invalid")
        relative_path = asset.get("relative_path")
        if not isinstance(relative_path, str) or not isinstance(asset.get("sha256"), str):
            raise DataIntegrityError("Tokenizer asset manifest entry is invalid")
        asset_path = (preparation / relative_path).resolve(strict=True)
        try:
            asset_path.relative_to(preparation.resolve(strict=True))
        except ValueError as exc:
            raise DataIntegrityError("Tokenizer asset path escapes preparation directory") from exc
        if _sha256(asset_path.read_bytes()) != asset["sha256"]:
            raise DataIntegrityError(
                f"Pinned tokenizer asset hash mismatch: {relative_path}"
            )
    model = protocol["model"]
    if audit.get("model_revision") != model["revision"]:
        raise DataIntegrityError("Tokenizer/model revision mismatch")
    tokenizer_path = (preparation / "assets" / "phobert_tokenizer").resolve(strict=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), use_fast=False, local_files_only=True
        )
    except Exception as exc:
        raise DataIntegrityError("Cannot load pinned PhoBERT tokenizer locally") from exc
    if type(tokenizer).__name__ != audit.get("tokenizer_class"):
        raise DataIntegrityError("PhoBERT tokenizer class differs from locked audit")
    return tokenizer


def audit_full_preprocessing(
    protocol_path: str | Path,
    output_path: str | Path,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict[str, object]:
    """Audit segmentation and token IDs across all splits; return aggregates only.

    Test records are consumed internally for the pre-fit collision gate. The
    public development loader still returns only train and validation records.
    No review text, row key, token-ID sequence or model weights are written.
    """
    report, _ = _audit_full_preprocessing_with_segmenter(
        protocol_path, output_path, fetch_bytes
    )
    return report


def _audit_full_preprocessing_with_segmenter(
    protocol_path: str | Path,
    output_path: str | Path,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> tuple[dict[str, object], RDRSegmenter]:
    """Run the aggregate audit and return its in-memory RDR runtime to the pilot."""
    from importlib.metadata import version
    protocol_file = Path(protocol_path).resolve(strict=True)
    output = Path(output_path).resolve()
    preparation = protocol_file.parent
    try:
        output.relative_to(preparation.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise DataIntegrityError("Audit output cannot be written inside locked preparation")
    if output.exists():
        raise DataIntegrityError(
            "Audit output already exists; preserve prior receipt and choose a new path"
        )
    fetch = fetch_bytes or _fetch_bytes
    raw_cache: dict[str, bytes] = {}

    def cached_fetch(url: str) -> bytes:
        if url not in raw_cache:
            raw_cache[url] = fetch(url)
        return raw_cache[url]

    started = time.perf_counter()
    bundle: DatasetBundle = load_dataset(protocol_file, cached_fetch)
    try:
        protocol = json.loads(protocol_file.read_bytes())
        lock = json.loads((preparation / "preparation_lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError("Cannot read locked preprocessing inputs") from exc
    manifest, _ = _read_locked_json(preparation, lock, "data_manifest.json")
    segmentation_expected, segmentation_bytes = _read_locked_json(
        preparation, lock, "segmentation_audit.json"
    )
    tokenizer_expected, tokenizer_bytes = _read_locked_json(
        preparation, lock, "tokenizer_input_audit.json"
    )
    rdr_assets, _ = _read_locked_json(preparation, lock, "rdr_assets_verification.json")
    if _sha256(protocol_file.read_bytes()) != bundle.protocol_sha256:
        raise DataIntegrityError("Protocol changed while preprocessing audit was running")
    if _sha256((preparation / "data_manifest.json").read_bytes()) != bundle.manifest_sha256:
        raise DataIntegrityError("Data manifest changed while preprocessing audit was running")
    if tokenizer_expected.get("data_manifest_sha256") != bundle.manifest_sha256:
        raise DataIntegrityError("Tokenizer audit is bound to another data manifest")
    if tokenizer_expected.get("dataset_revision") != bundle.revision:
        raise DataIntegrityError("Tokenizer audit is bound to another dataset revision")
    _validate_preprocessing_contract(protocol, tokenizer_expected)

    data_config = protocol["dataset"]
    test_url = data_config["source_base_url"] + "test.csv"
    test_payload = raw_cache.get(test_url)
    if test_payload is None:
        raise DataIntegrityError("Pinned test bytes were not retained in the in-memory audit cache")
    test_records, _ = _parse_split(
        "test",
        test_payload,
        manifest["splits"]["test"],
        bundle.revision,
        set(),
    )
    records_by_split: dict[str, Sequence[ReviewRecord]] = {
        "test": test_records,
        "val": bundle.splits["val"],
        "train": bundle.splits["train"],
    }
    preprocessing = protocol["preprocessing"]
    rdr_revision = preprocessing["rdr"]["revision"]
    if rdr_assets.get("revision") != rdr_revision:
        raise DataIntegrityError("RDR asset revision differs from locked protocol")
    if segmentation_expected.get("data_manifest_sha256") != bundle.manifest_sha256:
        raise DataIntegrityError("Segmentation audit is bound to another data manifest")
    if segmentation_expected.get("rdr_assets_revision") != rdr_revision:
        raise DataIntegrityError("Segmentation audit is bound to another RDR revision")
    if not segmentation_expected.get("all_checks_passed"):
        raise DataIntegrityError("Locked segmentation audit is not passing")
    if not tokenizer_expected.get("all_checks_passed") or tokenizer_expected.get(
        "cross_split_collision_found"
    ):
        raise DataIntegrityError("Locked tokenizer audit does not pass the collision gate")
    asset_hashes = _verify_rdr_assets(preparation, rdr_assets)
    wrapper_verification = _verify_rdr_wrapper(preparation, protocol)
    tokenizer = _load_pinned_tokenizer(preparation, tokenizer_expected, protocol)
    tokenizer_batch_smoke = _audit_tokenizer_batch_interface(
        tokenizer, int(protocol["training"]["max_length"])
    )
    original_cwd = os.getcwd()
    segmenters: dict[str, Segmenter] = {"underthesea": UndertheseaSegmenter()}
    rdr_segmenter = RDRSegmenter(preparation)
    segmenters["vncorenlp_rdr"] = rdr_segmenter
    if os.getcwd() != original_cwd:
        raise DataIntegrityError("RDR initialization changed the working directory")
    segmented_results = {}
    segmented_by_arm: dict[str, dict[str, tuple[str, ...]]] = {}
    segment_expected_results = segmentation_expected.get("results", {})
    for name, segmenter in segmenters.items():
        outputs = {
            split: tuple(preprocess_review(record.text, segmenter) for record in records)
            for split, records in records_by_split.items()
        }
        rows = _segmentation_report(records_by_split, outputs)
        expected_rows = segment_expected_results.get(name, {}).get("splits")
        if rows != expected_rows:
            raise DataIntegrityError(f"Segmented inputs differ from locked audit: {name}")
        if any(
            row[key]
            for row in rows
            for key in (
                "empty_output_rows",
                "within_split_duplicate_rows",
                "cross_split_overlap_rows",
            )
        ):
            raise DataIntegrityError(f"Exact segmentation collision gate failed: {name}")
        segmented_results[name] = {"splits": rows, "matches_locked_audit": True}
        segmented_by_arm[name] = outputs

    tokenizer_results = {}
    max_length = int(protocol["training"]["max_length"])
    if max_length != int(tokenizer_expected["max_length"]):
        raise DataIntegrityError("Tokenizer max_length differs from locked protocol")
    arm_ids = {"P0": "underthesea", "P1": "vncorenlp_rdr"}
    for config, arm in arm_ids.items():
        result = _tokenizer_report(
            tokenizer, records_by_split, segmented_by_arm[arm], max_length
        )
        expected = tokenizer_expected["configurations"][config]
        comparable = {key: value for key, value in expected.items() if key != "segmentation"}
        if result != comparable:
            raise DataIntegrityError(f"PhoBERT input audit differs from locked result: {config}")
        tokenizer_results[config] = {
            "segmentation": expected["segmentation"],
            **result,
            "matches_locked_audit": True,
        }

    has_cross_collision = any(
        collision["shared_unique_input_sequences"] > 0
        for result in tokenizer_results.values()
        for representation in result["cross_split_collisions"].values()
        for collision in representation.values()
    )
    if has_cross_collision:
        raise DataIntegrityError("Exact cross-split tokenizer collision; halt before fitting")

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": bundle.protocol_id,
        "protocol_sha256": bundle.protocol_sha256,
        "data_manifest_sha256": bundle.manifest_sha256,
        "dataset_revision": bundle.revision,
        "segmentation_audit_sha256": _sha256(segmentation_bytes),
        "tokenizer_input_audit_sha256": _sha256(tokenizer_bytes),
        "preprocessing_code_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "runtime_packages": {
            "underthesea": version("underthesea"),
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
            "huggingface_hub": version("huggingface_hub"),
            "py_vncorenlp": version("py_vncorenlp"),
            "pyjnius": version("pyjnius"),
        },
        "rdr_asset_sha256": asset_hashes,
        "rdr_wrapper": wrapper_verification,
        "rdr_import_origin_verified": rdr_segmenter.import_origin_verified,
        "tokenizer_class": type(tokenizer).__name__,
        "max_length": max_length,
        "tokenizer_batch_interface": tokenizer_batch_smoke,
        "runtime": {
            "python_version": platform.python_version(),
            "java_version": _java_runtime_version(),
        },
        "segmentation": segmented_results,
        "tokenizer_inputs": tokenizer_results,
        "cross_split_collision_found": False,
        "development_loader_exposes": ["train", "val"],
        "rdr_initialization_preserved_working_directory": True,
        "test_used_only_for_pre_fit_integrity_audit": True,
        "raw_review_text_saved": False,
        "row_level_ids_saved": False,
        "token_id_sequences_saved": False,
        "model_weights_loaded": False,
        "training_performed": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limits": [
            "Exact token sequences do not establish paraphrase/product/user independence.",
            "Historical corpus remains post-hoc and is not independent confirmation.",
            "The local CPU audit is not Kaggle runtime or GPU verification.",
        ],
    }
    _write_new_audit_receipt(preparation, output, report)
    return report, rdr_segmenter


def _java_runtime_version() -> str:
    import re
    import shutil
    import subprocess

    java = shutil.which("java")
    if java is None:
        raise DataIntegrityError("Java runtime is not available on PATH")
    try:
        completed = subprocess.run(
            [java, "-version"], capture_output=True, text=True, check=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DataIntegrityError("Could not query the Java runtime version") from exc
    match = re.search(r'version "([^"]+)"', completed.stderr + completed.stdout)
    if match is None:
        raise DataIntegrityError("Java runtime returned an unrecognized version string")
    return match.group(1)
