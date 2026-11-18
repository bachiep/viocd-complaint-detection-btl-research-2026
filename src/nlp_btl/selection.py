"""Validation-only model selection over the locked seven-run experiment matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from . import baseline, data, phobert, preprocessing
from .data import DataIntegrityError, ReviewRecord, _sha256


class SelectionError(DataIntegrityError):
    """Raised when development artifacts cannot support the locked selection."""


def _read_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"Required selection input is unavailable or invalid: {name}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"Required selection input has an invalid schema: {name}")
    return value, payload


def _finite_score(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SelectionError(f"Stored validation score is invalid: {name}")
    return float(value)


def _verify_metric_runtime(protocol: Mapping[str, object]) -> str:
    from importlib.metadata import PackageNotFoundError, version

    environment = protocol.get("environment")
    targets = environment.get("explicit_package_targets") if isinstance(environment, dict) else None
    expected = targets.get("scikit-learn") if isinstance(targets, dict) else None
    if not isinstance(expected, str) or not expected:
        raise SelectionError("Protocol does not pin the metric package version")
    try:
        observed = version("scikit-learn")
    except PackageNotFoundError as exc:
        raise SelectionError("Pinned metric package is not installed") from exc
    if observed != expected:
        raise SelectionError("Installed scikit-learn version differs from the locked protocol")
    return observed


def _read_locked_protocol(protocol_path: str | Path) -> tuple[dict[str, object], str, str, str]:
    try:
        path = Path(protocol_path).resolve(strict=True)
        protocol_bytes = path.read_bytes()
        protocol = json.loads(protocol_bytes)
        if not isinstance(protocol, dict):
            raise SelectionError("Locked protocol has an invalid schema")
        protocol_sha256 = _sha256(protocol_bytes)
        dataset = protocol.get("dataset")
        if not isinstance(dataset, dict):
            raise SelectionError("Locked protocol lacks dataset identity")
        manifest_name = dataset.get("manifest_file")
        if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
            raise SelectionError("Locked manifest path is invalid")
        manifest_bytes = (path.parent / manifest_name).read_bytes()
        manifest_sha256 = _sha256(manifest_bytes)
        if manifest_sha256 != dataset.get("manifest_sha256"):
            raise SelectionError("Dataset manifest differs from the locked protocol")
        lock = json.loads((path.parent / "preparation_lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(
            "Locked protocol, manifest or preparation lock is unavailable or invalid"
        ) from exc
    hashes = lock.get("hashes") if isinstance(lock, dict) else None
    if (
        not isinstance(hashes, dict)
        or hashes.get("protocol.json") != protocol_sha256
        or hashes.get(manifest_name) != manifest_sha256
    ):
        raise SelectionError("Protocol or dataset manifest differs from the preparation lock")
    revision = dataset.get("revision")
    if not isinstance(revision, str) or not revision:
        raise SelectionError("Dataset revision is invalid")
    return protocol, protocol_sha256, manifest_sha256, revision


def expected_run_keys(protocol: Mapping[str, object]) -> tuple[str, ...]:
    training = protocol.get("training")
    run_order = training.get("run_order") if isinstance(training, dict) else None
    if not isinstance(run_order, list) or run_order != [
        "P0_seed42",
        "P1_seed42",
        "P1_seed43",
        "P0_seed43",
        "P0_seed44",
        "P1_seed44",
    ]:
        raise SelectionError("Protocol run order differs from the locked six-fit PhoBERT matrix")
    return ("S0_seed42", *run_order)


def _locked_validation_contract(
    protocol_path: str | Path, protocol: Mapping[str, object]
) -> tuple[int, str]:
    dataset = protocol.get("dataset")
    if not isinstance(dataset, dict):
        raise SelectionError("Locked protocol lacks dataset counts")
    counts = dataset.get("expected_counts")
    manifest_name = dataset.get("manifest_file")
    if not isinstance(counts, dict) or type(counts.get("val")) is not int:
        raise SelectionError("Locked validation row count is invalid")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise SelectionError("Locked manifest path is invalid")
    try:
        manifest = json.loads(
            (Path(protocol_path).resolve(strict=True).parent / manifest_name).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError("Locked validation manifest is unavailable or invalid") from exc
    splits = manifest.get("splits") if isinstance(manifest, dict) else None
    validation = splits.get("val") if isinstance(splits, dict) else None
    if (
        not isinstance(validation, dict)
        or validation.get("retained_rows") != counts["val"]
        or not isinstance(validation.get("ordered_membership_digest_sha256"), str)
        or len(validation["ordered_membership_digest_sha256"]) != 64
    ):
        raise SelectionError("Locked validation manifest contract is incomplete")
    return counts["val"], validation["ordered_membership_digest_sha256"]


def _validate_validation_population(
    records: Sequence[ReviewRecord],
    revision: str,
    expected_count: int,
    expected_membership_sha256: str,
) -> tuple[dict[tuple[str, str, int], int], str, str]:
    if not records:
        raise SelectionError("Validation population is empty")
    expected: dict[tuple[str, str, int], int] = {}
    ordered_keys: list[list[object]] = []
    for record in records:
        if (
            not isinstance(record, ReviewRecord)
            or record.split != "val"
            or record.revision != revision
            or type(record.label) is not int
            or record.label not in (0, 1)
            or record.key in expected
        ):
            raise SelectionError("Selection accepts only unique, locked validation records")
        expected[record.key] = record.label
        ordered_keys.append(list(record.key))
    if len(expected) != expected_count:
        raise SelectionError("Validation records do not match the locked cohort size")
    membership = "".join(
        f"val\t{record.source_row}\t{record.label}\t{record.domain}\t"
        f"{_sha256(record.text.encode('utf-8'))}\n"
        for record in sorted(records, key=lambda item: item.source_row)
    ).encode("utf-8")
    membership_sha256 = _sha256(membership)
    if membership_sha256 != expected_membership_sha256:
        raise SelectionError("Validation records differ from the locked cohort membership")
    population_sha256 = _sha256(
        json.dumps(ordered_keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return expected, population_sha256, membership_sha256


def _prediction_score(
    path: Path,
    expected: Mapping[tuple[str, str, int], int],
    configuration: str,
    seed: int,
    epoch: int | None,
) -> tuple[float, str]:
    from sklearn.metrics import f1_score

    payload, raw = _read_json(path, path.name)
    required = {"schema_version", "configuration", "seed", "split", "predictions"}
    if epoch is not None:
        required.add("epoch")
    if set(payload) != required or payload.get("schema_version") != 1:
        raise SelectionError("Validation prediction artifact has an unexpected schema")
    if (
        payload.get("configuration") != configuration
        or payload.get("seed") != seed
        or payload.get("split") != "val"
        or (epoch is not None and payload.get("epoch") != epoch)
    ):
        raise SelectionError("Validation prediction artifact has the wrong run identity")
    rows = payload.get("predictions")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise SelectionError("Validation predictions do not cover the complete population")
    observed_keys: list[tuple[str, str, int]] = []
    gold: list[int] = []
    predicted: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"key", "y_true", "y_pred"}:
            raise SelectionError("Validation prediction row has an invalid schema")
        key = row.get("key")
        if (
            not isinstance(key, list)
            or len(key) != 3
            or not isinstance(key[0], str)
            or key[1] != "val"
            or type(key[2]) is not int
            or type(row.get("y_true")) is not int
            or type(row.get("y_pred")) is not int
            or row["y_pred"] not in (0, 1)
        ):
            raise SelectionError("Validation prediction row contains an invalid key or label")
        stable_key = (key[0], key[1], key[2])
        if stable_key not in expected or row["y_true"] != expected[stable_key]:
            raise SelectionError("Validation key or gold label differs from the locked population")
        observed_keys.append(stable_key)
        gold.append(row["y_true"])
        predicted.append(row["y_pred"])
    if len(set(observed_keys)) != len(expected):
        raise SelectionError("Validation predictions contain duplicate source keys")
    score = float(f1_score(gold, predicted, labels=[0, 1], average="macro", zero_division=0))
    if not math.isfinite(score):
        raise SelectionError("Validation Macro-F1 is not finite")
    return score, _sha256(raw)


def _source_hashes() -> dict[str, str]:
    return {
        "selection.py": _sha256(Path(__file__).resolve().read_bytes()),
        "baseline.py": _sha256(Path(baseline.__file__).resolve().read_bytes()),
        "phobert.py": _sha256(Path(phobert.__file__).resolve().read_bytes()),
        "preprocessing.py": _sha256(Path(preprocessing.__file__).resolve().read_bytes()),
        "data.py": _sha256(Path(data.__file__).resolve().read_bytes()),
    }


def _safe_artifact_path(
    run_dir: Path, relative: str, *, must_be_file: bool = True
) -> Path:
    candidate_rel = Path(relative)
    if candidate_rel.is_absolute() or ".." in candidate_rel.parts:
        raise SelectionError("Run artifact path escapes its run directory")
    candidate = (run_dir / candidate_rel).resolve(strict=True)
    try:
        candidate.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise SelectionError("Run artifact resolves outside its run directory") from exc
    if must_be_file and not candidate.is_file():
        raise SelectionError("Run artifact is not a file")
    if not must_be_file and not candidate.is_dir():
        raise SelectionError("Run directory is not a directory")
    return candidate


def _validate_run_binding(
    metadata: Mapping[str, object],
    configuration: str,
    seed: int,
    protocol_sha256: str,
    manifest_sha256: str,
    revision: str,
) -> None:
    if any(
        metadata.get(key) != value
        for key, value in {
            "status": "FIT_COMPLETE",
            "configuration": configuration,
            "seed": seed,
            "protocol_sha256": protocol_sha256,
            "manifest_sha256": manifest_sha256,
            "dataset_revision": revision,
            "raw_review_text_saved": False,
        }.items()
    ):
        raise SelectionError(f"Development run binding is invalid: {configuration}/{seed}")
    if not isinstance(metadata.get("run_id"), str) or not metadata["run_id"]:
        raise SelectionError("Development run identifier is missing")


def _load_run(
    run_key: str,
    run_dir: Path,
    expected: Mapping[tuple[str, str, int], int],
    protocol_sha256: str,
    manifest_sha256: str,
    revision: str,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    if run_key == "S0_seed42":
        configuration, seed = "S0", 42
    else:
        configuration, seed_text = run_key.split("_seed", 1)
        seed = int(seed_text)
    metadata_path = _safe_artifact_path(run_dir, "run_metadata.json")
    metadata, metadata_bytes = _read_json(metadata_path, "run_metadata.json")
    if _contains_test_result(metadata):
        raise SelectionError("Development run metadata contains terminal-test output")
    _validate_run_binding(
        metadata, configuration, seed, protocol_sha256, manifest_sha256, revision
    )
    artifact_hashes = {"run_metadata.json": _sha256(metadata_bytes)}

    if configuration == "S0":
        baseline_config = protocol.get("baseline")
        dataset_config = protocol.get("dataset")
        expected_counts = (
            dataset_config.get("expected_counts")
            if isinstance(dataset_config, dict)
            else None
        )
        expected_train_rows = (
            expected_counts.get("train") if isinstance(expected_counts, dict) else None
        )
        environment = protocol.get("environment")
        package_targets = (
            environment.get("explicit_package_targets")
            if isinstance(environment, dict)
            else None
        )
        current_source_hashes = _source_hashes()
        if (
            not isinstance(baseline_config, dict)
            or not isinstance(package_targets, dict)
            or metadata.get("fit_performed") is not True
            or metadata.get("fit_data_split") != "train"
            or metadata.get("fit_scope") != "train only"
            or metadata.get("test_split_available_to_runner") is not False
            or metadata.get("prediction_keys_validated") is not True
            or metadata.get("train_rows") != expected_train_rows
            or metadata.get("validation_rows") != len(expected)
            or metadata.get("validation_prediction_rows") != len(expected)
            or metadata.get("row_level_predictions_saved") is not True
            or metadata.get("scikit_learn_version") != package_targets.get("scikit-learn")
            or metadata.get("baseline_code_sha256") != current_source_hashes["baseline.py"]
            or metadata.get("preprocessing_code_sha256") != current_source_hashes["preprocessing.py"]
        ):
            raise SelectionError("S0 run metadata does not satisfy its locked train-only contract")
        prediction_path = _safe_artifact_path(run_dir, "validation_predictions.json")
        score, prediction_hash = _prediction_score(
            prediction_path, expected, configuration, seed, None
        )
        model_name = metadata.get("pipeline_file")
        if model_name != "pipeline.joblib":
            raise SelectionError("S0 pipeline artifact name is invalid")
        model_path = _safe_artifact_path(run_dir, model_name)
        model_hash = _sha256(model_path.read_bytes())
        recorded_score = _finite_score(metadata.get("validation_macro_f1"), "S0 metadata")
        if (
            prediction_hash != metadata.get("validation_predictions_sha256")
            or model_hash != metadata.get("pipeline_sha256")
            or not math.isclose(
                score, recorded_score, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise SelectionError("S0 model or validation artifact differs from run metadata")
        artifact_hashes.update(
            {"validation_predictions.json": prediction_hash, "pipeline.joblib": model_hash}
        )
        return {
            "run_key": run_key,
            "configuration": configuration,
            "seed": seed,
            "best_epoch": None,
            "validation_macro_f1": score,
            "run_id": metadata.get("run_id"),
            "run_metadata_sha256": artifact_hashes["run_metadata.json"],
            "artifact_sha256": artifact_hashes,
        }

    if configuration not in {"P0", "P1"}:
        raise SelectionError("Run key is outside the locked model family")
    current_source_hashes = _source_hashes()
    dataset_config = protocol.get("dataset")
    expected_counts = (
        dataset_config.get("expected_counts")
        if isinstance(dataset_config, dict)
        else None
    )
    expected_train_rows = (
        expected_counts.get("train") if isinstance(expected_counts, dict) else None
    )
    if (
        metadata.get("trainer_sha256") != current_source_hashes["phobert.py"]
        or metadata.get("preprocessing_sha256") != current_source_hashes["preprocessing.py"]
        or metadata.get("train_rows") != expected_train_rows
        or metadata.get("validation_rows") != len(expected)
        or metadata.get("row_level_validation_predictions_saved") is not True
        or metadata.get("validation_prediction_rows") != len(expected)
    ):
        raise SelectionError("PhoBERT run was produced by different code or lacks keyed validation rows")
    history_path = _safe_artifact_path(run_dir, "training_history.json")
    history, history_bytes = _read_json(history_path, "training_history.json")
    if _contains_test_result(history):
        raise SelectionError("Development history contains terminal-test output")
    epochs = history.get("epochs")
    if (
        history.get("schema_version") != 1
        or history.get("configuration") != configuration
        or history.get("seed") != seed
        or history.get("run_id") != metadata.get("run_id")
        or history.get("attempt_id") != metadata.get("attempt_id")
        or not isinstance(epochs, list)
        or [row.get("epoch") if isinstance(row, dict) else None for row in epochs] != [1, 2, 3]
    ):
        raise SelectionError("PhoBERT training history is incomplete or bound to another run")
    artifact_hashes["training_history.json"] = _sha256(history_bytes)

    epoch_scores: list[tuple[int, float]] = []
    for epoch in range(1, 4):
        name = f"epoch_{epoch:02d}_validation_predictions.json"
        pred_path = _safe_artifact_path(run_dir, name)
        score, digest = _prediction_score(pred_path, expected, configuration, seed, epoch)
        recorded = _finite_score(
            epochs[epoch - 1].get("validation_macro_f1"), f"{run_key}/epoch-{epoch}"
        )
        if not math.isclose(score, recorded, rel_tol=0.0, abs_tol=1e-12):
            raise SelectionError("Stored epoch score differs from keyed validation predictions")
        expected_best = not epoch_scores or score > max(previous for _, previous in epoch_scores)
        if (
            epochs[epoch - 1].get("checkpoint_became_best") is not expected_best
            or epochs[epoch - 1].get("checkpoint_saved") is not expected_best
        ):
            raise SelectionError("Stored best-checkpoint history differs from strict validation selection")
        epoch_scores.append((epoch, score))
        artifact_hashes[name] = digest

    best_score = max(score for _, score in epoch_scores)
    best_epoch = next(epoch for epoch, score in epoch_scores if score == best_score)
    checkpoint_json = _safe_artifact_path(run_dir, "best_checkpoint/checkpoint.json")
    checkpoint_weights = _safe_artifact_path(run_dir, "best_checkpoint/trainable_state.pt")
    checkpoint, checkpoint_bytes = _read_json(checkpoint_json, "checkpoint.json")
    if _contains_test_result(checkpoint):
        raise SelectionError("Development checkpoint contains terminal-test output")
    checkpoint_hash = _sha256(checkpoint_bytes)
    weights_hash = _sha256(checkpoint_weights.read_bytes())
    metadata_score = _finite_score(
        metadata.get("validation_macro_f1"), f"{run_key} metadata"
    )
    checkpoint_score = _finite_score(
        checkpoint.get("validation_macro_f1"), f"{run_key} checkpoint"
    )
    if (
        metadata.get("selected_epoch") != best_epoch
        or checkpoint.get("selected_epoch") != best_epoch
        or metadata.get("checkpoint_metadata_sha256") != checkpoint_hash
        or metadata.get("checkpoint_weights_sha256") != weights_hash
        or checkpoint.get("weights_sha256") != weights_hash
        or not math.isclose(
            best_score, metadata_score, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            best_score, checkpoint_score, rel_tol=0.0, abs_tol=1e-12
        )
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("manifest_sha256") != manifest_sha256
        or checkpoint.get("configuration") != configuration
        or checkpoint.get("seed") != seed
        or checkpoint.get("weights_file") != "trainable_state.pt"
        or any(
            checkpoint.get(key) != metadata.get(key)
            for key in phobert.CHECKPOINT_BINDING_FIELDS
        )
    ):
        raise SelectionError("Best checkpoint differs from recomputed validation selection")
    artifact_hashes.update(
        {
            "best_checkpoint/checkpoint.json": checkpoint_hash,
            "best_checkpoint/trainable_state.pt": weights_hash,
        }
    )
    return {
        "run_key": run_key,
        "configuration": configuration,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "run_id": metadata.get("run_id"),
        "run_metadata_sha256": artifact_hashes["run_metadata.json"],
        "artifact_sha256": artifact_hashes,
    }


def _validate_run_directories(
    run_directories: Mapping[str, str | Path], expected_keys: Sequence[str]
) -> dict[str, Path]:
    if set(run_directories) != set(expected_keys) or len(run_directories) != len(expected_keys):
        raise SelectionError("Selection requires exactly the seven declared development runs")
    resolved: dict[str, Path] = {}
    for run_key in expected_keys:
        path = Path(run_directories[run_key]).resolve(strict=True)
        if not path.is_dir():
            raise SelectionError(f"Development run directory is invalid: {run_key}")
        resolved[run_key] = path
    if len(set(resolved.values())) != len(resolved):
        raise SelectionError("A development run directory cannot represent multiple matrix cells")
    return resolved


def create_selection_artifact(
    protocol_path: str | Path,
    validation_records: Sequence[ReviewRecord],
    run_directories: Mapping[str, str | Path],
    selection_path: str | Path,
) -> tuple[dict[str, object], str]:
    """Recompute validation results and create a digest-bound selection exactly once."""
    protocol, protocol_sha256, manifest_sha256, revision = _read_locked_protocol(protocol_path)
    metric_version = _verify_metric_runtime(protocol)
    validation_count, validation_membership_sha256 = _locked_validation_contract(
        protocol_path, protocol
    )
    expected, population_sha256, checked_membership_sha256 = _validate_validation_population(
        validation_records,
        revision,
        validation_count,
        validation_membership_sha256,
    )
    run_keys = expected_run_keys(protocol)
    resolved_dirs = _validate_run_directories(run_directories, run_keys)
    target = Path(selection_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(target.suffix + ".sha256").exists():
        raise SelectionError("Selection artifact is create-only and already exists")
    root = target.parent.resolve(strict=True)
    run_rows: list[dict[str, object]] = []
    for run_key in run_keys:
        run_dir = resolved_dirs[run_key]
        try:
            relative_dir = run_dir.relative_to(root).as_posix()
        except ValueError as exc:
            raise SelectionError("Development run directory must be inside the selection root") from exc
        row = _load_run(
            run_key,
            run_dir,
            expected,
            protocol_sha256,
            manifest_sha256,
            revision,
            protocol,
        )
        row["run_directory"] = relative_dir
        run_rows.append(row)

    by_key = {row["run_key"]: row for row in run_rows}
    family_scores = {
        "S0": float(by_key["S0_seed42"]["validation_macro_f1"]),
        "P0": sum(float(by_key[f"P0_seed{seed}"]["validation_macro_f1"]) for seed in (42, 43, 44)) / 3,
        "P1": sum(float(by_key[f"P1_seed{seed}"]["validation_macro_f1"]) for seed in (42, 43, 44)) / 3,
    }
    selection_rules = protocol.get("selection")
    if not isinstance(selection_rules, dict):
        raise SelectionError("Protocol selection rules are missing")
    tie_tolerance = selection_rules.get("tie_tolerance")
    tie_order = selection_rules.get("tie_break_order")
    if (
        not isinstance(tie_tolerance, (int, float))
        or isinstance(tie_tolerance, bool)
        or tie_tolerance < 0
        or not math.isfinite(tie_tolerance)
        or tie_order != ["S0", "P0", "P1"]
    ):
        raise SelectionError("Protocol family tie rule is invalid")
    maximum = max(family_scores.values())
    tied = [family for family in tie_order if maximum - family_scores[family] <= tie_tolerance]
    selected_configuration = tied[0]
    demo_run_key = f"{selected_configuration}_seed42"
    demo_score = float(by_key[demo_run_key]["validation_macro_f1"])

    source_hashes = _source_hashes()
    selection: dict[str, object] = {
        "schema_version": 1,
        "status": "SELECTION_SAVED",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "dataset_revision": revision,
        "scikit_learn_version": metric_version,
        "validation_population_rows": len(validation_records),
        "validation_population_sha256": population_sha256,
        "validation_membership_sha256": checked_membership_sha256,
        "run_order": list(run_keys),
        "runs": run_rows,
        "family_validation_macro_f1": family_scores,
        "selected_configuration": selected_configuration,
        "selected_demo_run": demo_run_key,
        "selected_demo_validation_macro_f1": demo_score,
        "selection_rule": _selection_rule(protocol),
        "source_sha256": source_hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    encoded = (
        json.dumps(selection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    digest = _sha256(encoded)
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    sidecar = target.with_suffix(target.suffix + ".sha256")
    with sidecar.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return selection, digest


def verify_selection_artifact(
    selection_path: str | Path, protocol_path: str | Path
) -> tuple[dict[str, object], str]:
    """Verify immutable selection bytes, protocol binding, exact matrix and run hashes."""
    protocol, protocol_sha256, manifest_sha256, revision = _read_locked_protocol(protocol_path)
    metric_version = _verify_metric_runtime(protocol)
    validation_count, validation_membership_sha256 = _locked_validation_contract(
        protocol_path, protocol
    )
    path = Path(selection_path).resolve(strict=True)
    payload = path.read_bytes()
    digest = _sha256(payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        recorded_digest = sidecar.read_text(encoding="ascii").strip()
        selection = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError("Selection hash or JSON sidecar is missing/invalid") from exc
    if not isinstance(selection, dict) or recorded_digest != digest:
        raise SelectionError("Selection artifact digest does not match its sidecar")
    expected_keys = expected_run_keys(protocol)
    required_fields = {
        "schema_version",
        "status",
        "protocol_id",
        "protocol_sha256",
        "manifest_sha256",
        "dataset_revision",
        "scikit_learn_version",
        "validation_population_rows",
        "validation_population_sha256",
        "validation_membership_sha256",
        "run_order",
        "runs",
        "family_validation_macro_f1",
        "selected_configuration",
        "selected_demo_run",
        "selected_demo_validation_macro_f1",
        "selection_rule",
        "source_sha256",
        "created_at_utc",
    }
    if set(selection) != required_fields:
        raise SelectionError("Selection artifact has an unexpected schema")
    if any(
        selection.get(key) != value
        for key, value in {
            "schema_version": 1,
            "status": "SELECTION_SAVED",
            "protocol_id": protocol.get("protocol_id"),
            "protocol_sha256": protocol_sha256,
            "manifest_sha256": manifest_sha256,
            "dataset_revision": revision,
            "scikit_learn_version": metric_version,
            "run_order": list(expected_keys),
            "selection_rule": _selection_rule(protocol),
            "source_sha256": _source_hashes(),
        }.items()
    ):
        raise SelectionError("Selection artifact is stale or bound to different inputs")
    runs = selection.get("runs")
    if not isinstance(runs, list) or [row.get("run_key") if isinstance(row, dict) else None for row in runs] != list(expected_keys):
        raise SelectionError("Selection artifact does not contain the exact seven-run matrix")
    if _contains_test_result(selection):
        raise SelectionError("Selection artifact contains terminal-test information")
    root = path.parent.resolve(strict=True)
    by_key = {row["run_key"]: row for row in runs if isinstance(row, dict)}
    s0_row = by_key["S0_seed42"]
    s0_directory = s0_row.get("run_directory")
    if not isinstance(s0_directory, str):
        raise SelectionError("S0 selection run directory is invalid")
    s0_dir = _safe_artifact_path(root, s0_directory, must_be_file=False)
    expected = _validation_gold_from_s0(
        _safe_artifact_path(s0_dir, "validation_predictions.json"), revision
    )
    ordered_keys = [list(key) for key in expected]
    population_sha256 = _sha256(
        json.dumps(ordered_keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if (
        selection.get("validation_population_rows") != len(expected)
        or selection.get("validation_population_rows") != validation_count
        or selection.get("validation_population_sha256") != population_sha256
        or selection.get("validation_membership_sha256")
        != validation_membership_sha256
    ):
        raise SelectionError("Selection validation population differs from saved S0 keys")
    for row in runs:
        if not isinstance(row, dict) or set(row) != {
            "run_key",
            "configuration",
            "seed",
            "best_epoch",
            "validation_macro_f1",
            "run_id",
            "run_metadata_sha256",
            "artifact_sha256",
            "run_directory",
        }:
            raise SelectionError("Selection run row has an unexpected schema")
        run_key = row["run_key"]
        run_directory = row["run_directory"]
        if not isinstance(run_directory, str):
            raise SelectionError("Selection run directory is invalid")
        run_dir = _safe_artifact_path(root, run_directory, must_be_file=False)
        recomputed = _load_run(
            run_key,
            run_dir,
            expected,
            protocol_sha256,
            manifest_sha256,
            revision,
            protocol,
        )
        if any(row.get(key) != value for key, value in recomputed.items()):
            raise SelectionError("Selection summary differs from recomputed development artifacts")
        if (
            type(row.get("seed")) is not int
            or type(row.get("validation_macro_f1")) not in (int, float)
            or isinstance(row.get("validation_macro_f1"), bool)
            or (
                row["run_key"] == "S0_seed42"
                and row.get("best_epoch") is not None
            )
            or (
                row["run_key"] != "S0_seed42"
                and type(row.get("best_epoch")) is not int
            )
            or not isinstance(row.get("run_id"), str)
            or not isinstance(row.get("artifact_sha256"), dict)
        ):
            raise SelectionError("Selection summary has invalid run types")
    _recompute_family_decision(selection, protocol)
    return selection, digest


def _validation_gold_from_s0(
    path: Path, revision: str
) -> dict[tuple[str, str, int], int]:
    payload, _ = _read_json(path, "S0 validation predictions")
    if (
        set(payload) != {"schema_version", "configuration", "seed", "split", "predictions"}
        or payload.get("schema_version") != 1
        or payload.get("configuration") != "S0"
        or payload.get("seed") != 42
        or payload.get("split") != "val"
    ):
        raise SelectionError("S0 predictions cannot define the validation population")
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise SelectionError("S0 validation predictions are empty")
    expected: dict[tuple[str, str, int], int] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"key", "y_true", "y_pred"}:
            raise SelectionError("S0 validation prediction row has an invalid schema")
        key = row.get("key")
        if (
            not isinstance(key, list)
            or len(key) != 3
            or key[0] != revision
            or key[1] != "val"
            or type(key[2]) is not int
            or type(row.get("y_true")) is not int
            or row["y_true"] not in (0, 1)
            or type(row.get("y_pred")) is not int
            or row["y_pred"] not in (0, 1)
        ):
            raise SelectionError("S0 predictions contain an invalid validation key or label")
        stable_key = (key[0], key[1], key[2])
        if stable_key in expected:
            raise SelectionError("S0 validation source keys are duplicated")
        expected[stable_key] = row["y_true"]
    return expected


def _selection_rule(protocol: Mapping[str, object]) -> dict[str, object]:
    rules = protocol.get("selection")
    if not isinstance(rules, dict):
        raise SelectionError("Protocol selection rules are missing")
    fields = (
        "primary_metric",
        "labels",
        "zero_division",
        "epoch_rule",
        "configuration_rule",
        "tie_tolerance",
        "tie_break_order",
        "no_refit_on_validation",
        "no_test_driven_retry_or_extension",
    )
    return {field: rules.get(field) for field in fields}


def _contains_test_result(value: object, parent_key: str = "") -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key).lower()
            if name not in {
                "test_split_available_to_runner",
                "no_test_driven_retry_or_extension",
            } and "test_" in name:
                return True
            if _contains_test_result(child, name):
                return True
    elif isinstance(value, list):
        return any(_contains_test_result(child, parent_key) for child in value)
    return False


def _recompute_family_decision(selection: Mapping[str, object], protocol: Mapping[str, object]) -> None:
    runs = selection.get("runs")
    if not isinstance(runs, list):
        raise SelectionError("Selection run list is invalid")
    by_key = {row.get("run_key"): row for row in runs if isinstance(row, dict)}
    scores = {
        "S0": _finite_score(
            by_key["S0_seed42"].get("validation_macro_f1"), "S0 selection row"
        ),
        "P0": sum(
            _finite_score(
                by_key[f"P0_seed{seed}"].get("validation_macro_f1"), f"P0/{seed}"
            )
            for seed in (42, 43, 44)
        )
        / 3,
        "P1": sum(
            _finite_score(
                by_key[f"P1_seed{seed}"].get("validation_macro_f1"), f"P1/{seed}"
            )
            for seed in (42, 43, 44)
        )
        / 3,
    }
    rules = protocol.get("selection")
    tolerance = rules.get("tie_tolerance") if isinstance(rules, dict) else None
    order = rules.get("tie_break_order") if isinstance(rules, dict) else None
    if (
        type(tolerance) not in (int, float)
        or not math.isfinite(tolerance)
        or tolerance < 0
        or order != ["S0", "P0", "P1"]
    ):
        raise SelectionError("Protocol tie rule is invalid")
    maximum = max(scores.values())
    selected = next(family for family in order if maximum - scores[family] <= tolerance)
    demo_key = f"{selected}_seed42"
    if (
        selection.get("family_validation_macro_f1") != scores
        or selection.get("selected_configuration") != selected
        or selection.get("selected_demo_run") != demo_key
        or selection.get("selected_demo_validation_macro_f1") != by_key[demo_key].get("validation_macro_f1")
    ):
        raise SelectionError("Selection decision differs from the locked validation rule")
