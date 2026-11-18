"""Synthetic, non-research artifacts for selection and terminal-boundary tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_btl import baseline, phobert, preprocessing
from nlp_btl.data import ReviewRecord, _sha256
from nlp_btl.selection import create_selection_artifact


REVISION = "synthetic-locked-revision"
VALIDATION = (
    ReviewRecord(REVISION, "val", 0, "validation neutral phrase", 0, "app"),
    ReviewRecord(REVISION, "val", 1, "validation complaint phrase", 1, "app"),
)
TEST_CSV = b"review,label,domain\nSatisfactory,0,app\nBroken,1,app\n"
TEST_LABELS = (0, 1)


def _write_json(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _test_split_manifest() -> dict[str, object]:
    texts = ("satisfactory", "broken")
    membership = "".join(
        f"test\t{index}\t{label}\tapp\t{_sha256(text.encode('utf-8'))}\n"
        for index, (text, label) in enumerate(zip(texts, TEST_LABELS, strict=True))
    ).encode("utf-8")
    prediction_binding = "".join(
        f"test\t{index}\t{label}\tapp\n"
        for index, label in enumerate(TEST_LABELS)
    ).encode("utf-8")
    return {
        "source_file": "test.csv",
        "source_bytes": len(TEST_CSV),
        "source_sha256": _sha256(TEST_CSV),
        "raw_rows": 2,
        "missing_rows": 0,
        "blank_rows": 0,
        "conflict_rows": 0,
        "duplicate_rows": 0,
        "cross_split_overlap_removed": 0,
        "retained_rows": 2,
        "labels": {"0": 1, "1": 1},
        "domains": {"app": 2},
        "ordered_membership_digest_sha256": _sha256(membership),
        "ordered_prediction_binding_digest_sha256": _sha256(prediction_binding),
    }


def _make_preparation(root: Path) -> Path:
    preparation_dir = root / "preparation"
    preparation_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "preparation" / "protocol.json").read_text(encoding="utf-8"))
    test_split = _test_split_manifest()
    val_membership = "".join(
        f"val\t{record.source_row}\t{record.label}\t{record.domain}\t"
        f"{_sha256(record.text.encode('utf-8'))}\n"
        for record in sorted(VALIDATION, key=lambda item: item.source_row)
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "revision": REVISION,
        "source_base_url": "https://fixture.invalid/viocd/",
        "retention_priority": ["test", "val", "train"],
        "normalization": "NFC -> lowercase -> whitespace split/join",
        "retained_total": 6,
        "splits": {
            "train": {"retained_rows": 2},
            "val": {
                "retained_rows": 2,
                "ordered_membership_digest_sha256": _sha256(val_membership),
            },
            "test": test_split,
        },
    }
    manifest_bytes = _write_json(preparation_dir / "data_manifest.json", manifest)
    protocol["dataset"]["revision"] = REVISION
    protocol["dataset"]["source_base_url"] = manifest["source_base_url"]
    protocol["dataset"]["manifest_sha256"] = _sha256(manifest_bytes)
    protocol["dataset"]["expected_counts"] = {"train": 2, "val": 2, "test": 2}
    protocol_bytes = _write_json(preparation_dir / "protocol.json", protocol)
    _write_json(
        preparation_dir / "preparation_lock.json",
        {
            "schema_version": 1,
            "hashes": {
                "protocol.json": _sha256(protocol_bytes),
                "data_manifest.json": _sha256(manifest_bytes),
            },
        },
    )
    return preparation_dir / "protocol.json"


def _dump_prediction(path: Path, configuration: str, seed: int, predictions: tuple[int, int], epoch: int | None) -> bytes:
    payload: dict[str, object] = {
        "schema_version": 1,
        "configuration": configuration,
        "seed": seed,
        "split": "val",
        "predictions": [
            {
                "key": list(record.key),
                "y_true": record.label,
                "y_pred": prediction,
            }
            for record, prediction in zip(VALIDATION, predictions, strict=True)
        ],
    }
    if epoch is not None:
        payload["epoch"] = epoch
    return _write_json(path, payload)


def build_selection_fixture(
    root: Path, scenario: str = "p1_wins"
) -> tuple[Path, Path, Path, tuple[ReviewRecord, ...]]:
    """Create a self-consistent seven-run fixture; weights are opaque test bytes."""
    protocol_path = _make_preparation(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_bytes = protocol_path.read_bytes()
    manifest_sha = protocol["dataset"]["manifest_sha256"]
    protocol_sha = _sha256(protocol_bytes)
    revision = protocol["dataset"]["revision"]
    experiment_dir = root / "results"
    runs_root = experiment_dir / "runs"
    runs_root.mkdir(parents=True)
    run_dirs: dict[str, Path] = {}

    s0_dir = runs_root / "S0_seed42"
    s0_dir.mkdir()
    s0_labels = (0, 1) if scenario == "s0_tie" else (0, 0)
    s0_predictions = _dump_prediction(
        s0_dir / "validation_predictions.json", "S0", 42, s0_labels, None
    )
    pipeline_bytes = b"synthetic TF-IDF/LinearSVC pipeline placeholder"
    (s0_dir / "pipeline.joblib").write_bytes(pipeline_bytes)
    _write_json(
        s0_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "status": "FIT_COMPLETE",
            "configuration": "S0",
            "seed": 42,
            "run_id": "S0_seed42_attempt1",
            "attempt_id": "S0_seed42_attempt1",
            "dataset_revision": revision,
            "protocol_sha256": protocol_sha,
            "manifest_sha256": manifest_sha,
            "raw_review_text_saved": False,
            "fit_performed": True,
            "fit_data_split": "train",
            "fit_scope": "train only",
            "test_split_available_to_runner": False,
            "prediction_keys_validated": True,
            "row_level_predictions_saved": True,
            "train_rows": 2,
            "validation_rows": len(VALIDATION),
            "validation_prediction_rows": len(VALIDATION),
            "scikit_learn_version": protocol["environment"]["explicit_package_targets"]["scikit-learn"],
            "baseline_code_sha256": _sha256(Path(baseline.__file__).read_bytes()),
            "preprocessing_code_sha256": _sha256(Path(preprocessing.__file__).read_bytes()),
            "validation_macro_f1": 1.0 if scenario == "s0_tie" else 1 / 3,
            "pipeline_file": "pipeline.joblib",
            "pipeline_sha256": _sha256(pipeline_bytes),
            "validation_predictions_file": "validation_predictions.json",
            "validation_predictions_sha256": _sha256(s0_predictions),
        },
    )
    run_dirs["S0_seed42"] = s0_dir

    for run_key in protocol["training"]["run_order"]:
        configuration, seed_text = run_key.split("_seed", 1)
        seed = int(seed_text)
        run_dir = runs_root / run_key
        checkpoint_dir = run_dir / "best_checkpoint"
        checkpoint_dir.mkdir(parents=True)
        run_id = f"fixture-{run_key}"
        attempt_id = f"attempt-{run_key}"
        family_tie = scenario in {"p0_p1_tie", "s0_tie"}
        has_perfect_checkpoint = configuration == "P1" or family_tie
        epoch_predictions = (
            ((0, 0), (0, 1), (0, 1))
            if has_perfect_checkpoint
            else ((0, 0), (0, 0), (0, 0))
        )
        scores = (
            (1 / 3, 1.0, 1.0)
            if has_perfect_checkpoint
            else (1 / 3, 1 / 3, 1 / 3)
        )
        history_rows: list[dict[str, object]] = []
        best_score: float | None = None
        best_epoch = 0
        for epoch, (predictions, score) in enumerate(zip(epoch_predictions, scores, strict=True), 1):
            became_best = best_score is None or score > best_score
            if became_best:
                best_score = score
                best_epoch = epoch
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": 0.5,
                    "validation_macro_f1": score,
                    "optimizer_steps_succeeded": 1,
                    "amp_overflow_steps": 0,
                    "learning_rate_after_epoch": 0.00001,
                    "checkpoint_saved": became_best,
                    "checkpoint_became_best": became_best,
                }
            )
            _dump_prediction(
                run_dir / f"epoch_{epoch:02d}_validation_predictions.json",
                configuration,
                seed,
                predictions,
                epoch,
            )

        source_hashes = {
            "trainer_sha256": _sha256(Path(phobert.__file__).read_bytes()),
            "preprocessing_sha256": _sha256(Path(preprocessing.__file__).read_bytes()),
        }
        binding = {
            "configuration": configuration,
            "seed": seed,
            "dataset_revision": revision,
            "manifest_sha256": manifest_sha,
            "protocol_sha256": protocol_sha,
            "preparation_lock_sha256": "fixture-preparation-lock-sha256",
            "pre_fit_audit_sha256": "fixture-pre-fit-audit-sha256",
            "readiness_receipt_sha256": "fixture-readiness-sha256",
            "model_id": protocol["model"]["id"],
            "model_revision": protocol["model"]["revision"],
            "model_config_sha256": protocol["model"]["config_sha256"],
            "frozen_base_state_sha256": "fixture-frozen-base-sha256",
            "segmenter_id": "fixture-segmenter",
            "tokenizer_revision": protocol["model"]["revision"],
            "tokenizer_audit_sha256": "fixture-tokenizer-audit-sha256",
            "max_length": protocol["training"]["max_length"],
            "trainable_scope": "encoder_layers_10_11_and_classifier",
            **source_hashes,
        }
        weights = f"fixture weights for {run_key}".encode("ascii")
        weights_path = checkpoint_dir / "trainable_state.pt"
        weights_path.write_bytes(weights)
        checkpoint = {
            **binding,
            "selected_epoch": best_epoch,
            "validation_macro_f1": best_score,
            "weights_file": weights_path.name,
            "weights_sha256": _sha256(weights),
        }
        checkpoint_bytes = _write_json(checkpoint_dir / "checkpoint.json", checkpoint)
        _write_json(
            run_dir / "training_history.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "configuration": configuration,
                "seed": seed,
                "epochs": history_rows,
            },
        )
        _write_json(
            run_dir / "run_metadata.json",
            {
                "schema_version": 1,
                "status": "FIT_COMPLETE",
                "configuration": configuration,
                "seed": seed,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "dataset_revision": revision,
                "protocol_sha256": protocol_sha,
                "manifest_sha256": manifest_sha,
                "raw_review_text_saved": False,
                "train_rows": 2,
                "validation_rows": len(VALIDATION),
                "row_level_validation_predictions_saved": True,
                "validation_prediction_rows": len(VALIDATION),
                "selected_epoch": best_epoch,
                "validation_macro_f1": best_score,
                "checkpoint_metadata_sha256": _sha256(checkpoint_bytes),
                "checkpoint_weights_sha256": _sha256(weights),
                **binding,
            },
        )
        run_dirs[run_key] = run_dir

    selection_path = experiment_dir / "selection.json"
    create_selection_artifact(protocol_path, VALIDATION, run_dirs, selection_path)
    return protocol_path, selection_path, experiment_dir, VALIDATION
