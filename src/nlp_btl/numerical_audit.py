"""Independent numerical audit of persisted terminal predictions."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Mapping, Sequence

from .data import DataIntegrityError, _ordered_prediction_binding_digest, _sha256
from .selection import (
    SelectionError,
    _read_locked_protocol,
    expected_run_keys,
    verify_selection_artifact,
)


class NumericalAuditError(DataIntegrityError):
    """Persisted terminal artifacts fail independent numerical verification."""


def _private_results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_bound_json(path: Path) -> tuple[dict[str, object], str]:
    try:
        payload = path.read_bytes()
        recorded_hash = path.with_suffix(path.suffix + ".sha256").read_text(
            encoding="ascii"
        ).strip()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise NumericalAuditError(f"Missing or invalid audit input: {path.name}") from exc
    digest = _sha256(payload)
    if recorded_hash != digest or not isinstance(value, dict):
        raise NumericalAuditError(f"Artifact hash/schema check failed: {path.name}")
    return value, digest


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_binary_metrics(
    gold: Sequence[int], predicted: Sequence[int]
) -> dict[str, object]:
    """Compute binary metrics from integer counts without sklearn/evaluator code."""
    if len(gold) != len(predicted) or not gold:
        raise NumericalAuditError("Metric inputs must have equal, nonzero row counts")
    if any(type(value) is not int or value not in (0, 1) for value in (*gold, *predicted)):
        raise NumericalAuditError("Metric inputs contain nonbinary labels")
    tn = sum(actual == 0 and guess == 0 for actual, guess in zip(gold, predicted, strict=True))
    fp = sum(actual == 0 and guess == 1 for actual, guess in zip(gold, predicted, strict=True))
    fn = sum(actual == 1 and guess == 0 for actual, guess in zip(gold, predicted, strict=True))
    tp = sum(actual == 1 and guess == 1 for actual, guess in zip(gold, predicted, strict=True))
    counts = {0: (tn, fn, fp), 1: (tp, fp, fn)}
    per_class: dict[str, dict[str, object]] = {}
    for label in (0, 1):
        true_positive, false_positive, false_negative = counts[label]
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        )
        support = tn + fp if label == 0 else fn + tp
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    accuracy = (tn + tp) / len(gold)
    return {
        "n": len(gold),
        "accuracy": accuracy,
        "macro_precision": sum(float(per_class[str(label)]["precision"]) for label in (0, 1)) / 2,
        "macro_recall": sum(float(per_class[str(label)]["recall"]) for label in (0, 1)) / 2,
        "macro_f1": sum(float(per_class[str(label)]["f1"]) for label in (0, 1)) / 2,
        "per_class": per_class,
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
    }


def _assert_metric_equal(
    calculated: Mapping[str, object], saved: object, context: str
) -> None:
    if not isinstance(saved, dict) or set(saved) != set(calculated):
        raise NumericalAuditError(f"Saved metric schema differs from independent result: {context}")
    for key, value in calculated.items():
        observed = saved.get(key)
        if isinstance(value, dict):
            _assert_nested_metric_equal(value, observed, f"{context}.{key}")
        elif isinstance(value, list):
            if observed != value:
                raise NumericalAuditError(f"Saved confusion matrix differs: {context}")
        elif isinstance(value, float):
            if type(observed) not in (int, float) or isinstance(observed, bool) or not math.isclose(
                value, float(observed), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise NumericalAuditError(f"Saved metric differs from independent result: {context}.{key}")
        elif observed != value or type(observed) is not type(value):
            raise NumericalAuditError(f"Saved metric differs from independent result: {context}.{key}")


def _assert_nested_metric_equal(
    calculated: Mapping[str, object], saved: object, context: str
) -> None:
    if not isinstance(saved, dict) or set(saved) != set(calculated):
        raise NumericalAuditError(f"Saved metric schema differs: {context}")
    for key, value in calculated.items():
        observed = saved.get(key)
        if isinstance(value, dict):
            _assert_nested_metric_equal(value, observed, f"{context}.{key}")
        elif isinstance(value, float):
            if type(observed) not in (int, float) or isinstance(observed, bool) or not math.isclose(
                value, float(observed), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise NumericalAuditError(f"Saved metric differs: {context}.{key}")
        elif observed != value or type(observed) is not type(value):
            raise NumericalAuditError(f"Saved metric differs: {context}.{key}")


def _validate_prediction_rows(
    rows: object,
    run_order: Sequence[str],
    revision: str,
    expected_n: int,
    expected_labels: Mapping[str, object],
    expected_domains: Mapping[str, object],
    expected_binding_digest: str,
) -> tuple[dict[str, list[dict[str, object]]], str]:
    if not isinstance(rows, list) or len(rows) != len(run_order) * expected_n:
        raise NumericalAuditError("Saved predictions do not contain exact seven-run population size")
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in run_order}
    seen: dict[str, set[tuple[str, str, int]]] = {key: set() for key in run_order}
    reference_keys: list[tuple[str, str, int]] | None = None
    reference_values: dict[tuple[str, str, int], tuple[int, str]] = {}
    cursor = 0
    for run_key in run_order:
        current_keys: list[tuple[str, str, int]] = []
        for row in rows[cursor : cursor + expected_n]:
            cursor += 1
            if not isinstance(row, dict) or set(row) != {
                "run_key", "key", "domain", "y_true", "y_pred"
            }:
                raise NumericalAuditError("Prediction row has an invalid/private-data schema")
            key_value = row.get("key")
            if (
                row.get("run_key") != run_key
                or not isinstance(key_value, list)
                or len(key_value) != 3
                or key_value[0] != revision
                or key_value[1] != "test"
                or type(key_value[2]) is not int
                or type(row.get("y_true")) is not int
                or row["y_true"] not in (0, 1)
                or type(row.get("y_pred")) is not int
                or row["y_pred"] not in (0, 1)
                or not isinstance(row.get("domain"), str)
            ):
                raise NumericalAuditError("Prediction run/key/label/domain does not match contract")
            stable_key = (key_value[0], key_value[1], key_value[2])
            if stable_key in seen[run_key]:
                raise NumericalAuditError("Duplicate test prediction key")
            seen[run_key].add(stable_key)
            current_keys.append(stable_key)
            value = (row["y_true"], row["domain"])
            if run_key == run_order[0]:
                reference_values[stable_key] = value
            elif reference_values.get(stable_key) != value:
                raise NumericalAuditError("Runs do not share identical test keys, gold labels, and domains")
            grouped[run_key].append(row)
        if current_keys != sorted(current_keys, key=lambda key: key[2]):
            raise NumericalAuditError("Prediction rows are not in stable source-row order")
        if reference_keys is None:
            reference_keys = current_keys
        elif current_keys != reference_keys:
            raise NumericalAuditError("Runs do not contain the exact same ordered test keys")

    labels = {str(label): 0 for label in (0, 1)}
    domains: dict[str, int] = {}
    for label, domain in reference_values.values():
        labels[str(label)] += 1
        domains[domain] = domains.get(domain, 0) + 1
    if labels != dict(expected_labels) or domains != dict(expected_domains):
        raise NumericalAuditError("Saved gold/domain counts differ from the locked test manifest")
    actual_binding_digest = _ordered_prediction_binding_digest(
        (
            key[1],
            key[2],
            reference_values[key][0],
            reference_values[key][1],
        )
        for key in reference_keys or []
    )
    if actual_binding_digest != expected_binding_digest:
        raise NumericalAuditError("Saved key/label/domain rows differ from the locked cohort binding")
    ordered_digest = _sha256(
        json.dumps([list(key) for key in reference_keys or []], separators=(",", ":")).encode("utf-8")
    )
    return grouped, ordered_digest


def paired_bootstrap_differences(
    gold: Sequence[int],
    predictions_by_run: Mapping[str, Sequence[int]],
    resamples: int,
    random_seed: int,
    *,
    batch_size: int = 128,
) -> list[float]:
    """Paired, label-stratified review bootstrap for the P1−P0 mean seed contrast."""
    import numpy as np

    expected = {f"{arm}_seed{seed}" for arm in ("P0", "P1") for seed in (42, 43, 44)}
    if set(predictions_by_run) != expected or any(
        len(predictions_by_run[key]) != len(gold) for key in expected
    ):
        raise NumericalAuditError("Bootstrap inputs do not contain the paired P0/P1 seed matrix")
    if resamples <= 0 or batch_size <= 0:
        raise NumericalAuditError("Bootstrap resamples and batch size must be positive")
    y = np.asarray(gold, dtype=np.int8)
    if y.ndim != 1 or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise NumericalAuditError("Bootstrap gold labels must be a nonempty binary vector")
    pred = {
        key: np.asarray(predictions_by_run[key], dtype=np.int8)
        for key in expected
    }
    if any(array.ndim != 1 or not np.isin(array, [0, 1]).all() for array in pred.values()):
        raise NumericalAuditError("Bootstrap predictions must be binary vectors")
    strata = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(indices.size == 0 for indices in strata):
        raise NumericalAuditError("Paired stratified bootstrap requires both gold labels")
    rng = np.random.default_rng(random_seed)
    differences = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch_size):
        size = min(batch_size, resamples - start)
        sampled = np.concatenate(
            [rng.choice(indices, size=(size, indices.size), replace=True) for indices in strata],
            axis=1,
        )
        sampled_gold = np.concatenate(
            [np.full((size, len(indices)), label, dtype=np.int8) for label, indices in enumerate(strata)],
            axis=1,
        )
        seed_differences = np.zeros(size, dtype=np.float64)
        for seed in (42, 43, 44):
            p0_f1 = _batch_macro_f1(sampled_gold, pred[f"P0_seed{seed}"][sampled])
            p1_f1 = _batch_macro_f1(sampled_gold, pred[f"P1_seed{seed}"][sampled])
            seed_differences += (p1_f1 - p0_f1) / 3
        differences[start : start + size] = seed_differences
    return [float(value) for value in differences]


def _batch_macro_f1(gold, predicted):
    import numpy as np

    class_scores = []
    for label in (0, 1):
        tp = np.sum((gold == label) & (predicted == label), axis=1)
        fp = np.sum((gold != label) & (predicted == label), axis=1)
        fn = np.sum((gold == label) & (predicted != label), axis=1)
        denominator = 2 * tp + fp + fn
        class_scores.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0))
    return (class_scores[0] + class_scores[1]) / 2


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    fields = [
        "row_type", "configuration", "run_key", "seed", "domain", "n",
        "accuracy", "accuracy_sd", "macro_precision", "macro_precision_sd",
        "macro_recall", "macro_recall_sd", "macro_f1", "macro_f1_sd",
        "f1_label_0", "f1_label_1", "tn", "fp", "fn", "tp",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_new(path: Path, payload: bytes) -> str:
    digest = _sha256(payload)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    with path.with_suffix(path.suffix + ".sha256").open(
        "x", encoding="ascii", newline="\n"
    ) as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def audit_terminal_artifacts(
    protocol_path: str | Path,
    selection_path: str | Path,
    terminal_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Verify production terminal artifacts, recompute metrics, and create analysis."""
    protocol_file = Path(protocol_path).resolve(strict=True)
    selection_file = Path(selection_path).resolve(strict=True)
    terminal = Path(terminal_dir).resolve(strict=True)
    private_root = _private_results_root().resolve(strict=False)
    expected_terminal = (selection_file.parent / "terminal").resolve(strict=True)
    if terminal != expected_terminal or not _is_within(terminal, private_root):
        raise NumericalAuditError("Terminal evidence must be inside the selection's private results folder")
    output = Path(output_dir).resolve(strict=False) if output_dir else terminal / "analysis"
    if output != terminal / "analysis" or not _is_within(output, private_root):
        raise NumericalAuditError("Numerical audit output must use the private analysis folder")
    if output.exists():
        raise NumericalAuditError("Numerical audit output is create-only")

    try:
        selection, selection_sha256 = verify_selection_artifact(selection_file, protocol_file)
        protocol, protocol_sha256, manifest_sha256, revision = _read_locked_protocol(protocol_file)
    except SelectionError as exc:
        raise NumericalAuditError("Selection verification failed during numerical audit") from exc
    dataset = protocol.get("dataset")
    environment = protocol.get("environment")
    package_targets = environment.get("explicit_package_targets") if isinstance(environment, dict) else None
    if not isinstance(dataset, dict) or not isinstance(package_targets, dict):
        raise NumericalAuditError("Locked protocol lacks dataset or runtime package settings")
    try:
        numpy_version = __import__("importlib.metadata", fromlist=["version"]).version("numpy")
        import numpy as np
    except Exception as exc:
        raise NumericalAuditError("Pinned NumPy audit runtime is unavailable") from exc
    if numpy_version != package_targets.get("numpy"):
        raise NumericalAuditError("NumPy version differs from the locked protocol")

    manifest_path = protocol_file.parent / str(dataset.get("manifest_file"))
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise NumericalAuditError("Locked dataset manifest cannot be read") from exc
    if _sha256(manifest_bytes) != manifest_sha256 or not isinstance(manifest, dict):
        raise NumericalAuditError("Locked dataset manifest hash/schema is invalid")
    test_manifest = manifest.get("splits", {}).get("test")
    expected_counts = dataset.get("expected_counts")
    if not isinstance(test_manifest, dict) or not isinstance(expected_counts, dict):
        raise NumericalAuditError("Locked test cohort contract is incomplete")
    expected_n = expected_counts.get("test")
    if type(expected_n) is not int or expected_n <= 0:
        raise NumericalAuditError("Locked test cohort row count is invalid")

    claim_id = _sha256(
        f"viocd-terminal-cohort-v1\n{revision}\n{manifest_sha256}\n".encode("utf-8")
    )
    claim_root = private_root / ".terminal_evaluation_claims"
    claim_path = claim_root / f"{claim_id}.claim.json"
    outcome_path = claim_root / f"{claim_id}.outcome.json"
    claim, claim_sha256 = _read_bound_json(claim_path)
    outcome, _outcome_sha256 = _read_bound_json(outcome_path)
    summary, summary_sha256 = _read_bound_json(terminal / "terminal_evaluation.json")
    prediction_payload, predictions_sha256 = _read_bound_json(terminal / "test_predictions.json")
    expected_output_rel = terminal.relative_to(private_root).as_posix()
    expected_selection_rel = selection_file.parent.relative_to(private_root).as_posix()
    if claim.keys() != {
        "schema_version", "status", "cohort_id", "dataset_revision", "manifest_sha256",
        "protocol_sha256", "selection_sha256", "run_order", "selection_directory",
        "output_directory", "predictor_backend", "terminal_evaluator_sha256",
        "test_prediction_binding_digest_sha256", "created_at_utc",
    }:
        raise NumericalAuditError("Global terminal claim has an unexpected schema")
    if outcome.keys() != {
        "schema_version", "status", "stage", "test_fetch_started", "test_fetch_returned_bytes",
        "predictor_backend", "created_at_utc", "cohort_id", "claim_sha256",
        "selection_sha256", "summary_sha256", "predictions_sha256", "output_directory",
    }:
        raise NumericalAuditError("Terminal outcome has an unexpected completion schema")
    expected_summary_fields = {
        "schema_version", "status", "predictor_backend", "protocol_id", "protocol_sha256",
        "manifest_sha256", "dataset_revision", "scikit_learn_version", "selection_sha256",
        "selected_configuration", "selected_demo_run", "test_source_sha256", "test_rows",
        "runs", "primary_contrast_descriptive", "terminal_loader_sha256",
        "terminal_preprocessing_sha256", "terminal_evaluator_sha256", "created_at_utc",
        "raw_review_text_saved",
    }
    if summary.keys() != expected_summary_fields:
        raise NumericalAuditError("Terminal summary has an unexpected schema")
    if (
        claim.get("status") != "EVALUATING"
        or claim.get("cohort_id") != claim_id
        or claim.get("dataset_revision") != revision
        or claim.get("manifest_sha256") != manifest_sha256
        or claim.get("protocol_sha256") != protocol_sha256
        or claim.get("selection_sha256") != selection_sha256
        or claim.get("test_prediction_binding_digest_sha256")
        != test_manifest.get("ordered_prediction_binding_digest_sha256")
        or claim.get("run_order") != selection.get("run_order")
        or claim.get("selection_directory") != expected_selection_rel
        or claim.get("output_directory") != expected_output_rel
        or claim.get("predictor_backend") != "LOCKED_ARTIFACTS"
    ):
        raise NumericalAuditError("Global terminal claim does not bind this locked production run")
    if (
        outcome.get("status") != "COMPLETE"
        or outcome.get("stage") != "artifact_commit"
        or outcome.get("test_fetch_started") is not True
        or outcome.get("test_fetch_returned_bytes") is not True
        or outcome.get("predictor_backend") != "LOCKED_ARTIFACTS"
        or outcome.get("cohort_id") != claim_id
        or outcome.get("claim_sha256") != claim_sha256
        or outcome.get("selection_sha256") != selection_sha256
        or outcome.get("summary_sha256") != summary_sha256
        or outcome.get("predictions_sha256") != predictions_sha256
        or outcome.get("output_directory") != expected_output_rel
    ):
        raise NumericalAuditError("Terminal outcome receipt does not bind complete production artifacts")

    run_order = expected_run_keys(protocol)
    if (
        summary.get("status") != "TERMINAL_EVALUATION_COMPLETE"
        or summary.get("predictor_backend") != "LOCKED_ARTIFACTS"
        or prediction_payload.get("status") != "TERMINAL_PREDICTIONS_SAVED"
        or summary.get("protocol_sha256") != protocol_sha256
        or summary.get("manifest_sha256") != manifest_sha256
        or summary.get("dataset_revision") != revision
        or summary.get("selection_sha256") != selection_sha256
        or summary.get("selected_configuration") != selection.get("selected_configuration")
        or summary.get("selected_demo_run") != selection.get("selected_demo_run")
        or summary.get("protocol_id") != protocol.get("protocol_id")
        or prediction_payload.get("protocol_sha256") != protocol_sha256
        or prediction_payload.get("selection_sha256") != selection_sha256
        or summary.get("test_source_sha256") != test_manifest.get("source_sha256")
        or prediction_payload.get("test_source_sha256") != test_manifest.get("source_sha256")
        or summary.get("terminal_evaluator_sha256")
        != claim.get("terminal_evaluator_sha256")
        or summary.get("test_rows") != expected_n
        or summary.get("scikit_learn_version") != selection.get("scikit_learn_version")
        or prediction_payload.get("raw_review_text_saved") is not False
    ):
        raise NumericalAuditError("Terminal artifacts do not match the locked protocol/claim")
    if prediction_payload.keys() != {
        "schema_version", "status", "protocol_sha256", "selection_sha256",
        "test_source_sha256", "predictions", "raw_review_text_saved",
    }:
        raise NumericalAuditError("Terminal prediction artifact has an unexpected schema")
    if summary.get("schema_version") != 1 or summary.get("raw_review_text_saved") is not False:
        raise NumericalAuditError("Terminal summary schema/privacy assertion is invalid")

    expected_label_counts = test_manifest.get("labels")
    expected_domain_counts = test_manifest.get("domains")
    expected_binding_digest = test_manifest.get("ordered_prediction_binding_digest_sha256")
    if (
        not isinstance(expected_label_counts, dict)
        or not isinstance(expected_domain_counts, dict)
        or not isinstance(expected_binding_digest, str)
        or len(expected_binding_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_binding_digest)
    ):
        raise NumericalAuditError("Locked test label/domain counts or binding digest are invalid")
    grouped, ordered_key_sha256 = _validate_prediction_rows(
        prediction_payload.get("predictions"),
        run_order,
        revision,
        expected_n,
        expected_label_counts,
        expected_domain_counts,
        expected_binding_digest,
    )

    selection_runs = selection.get("runs")
    summary_runs = summary.get("runs")
    if not isinstance(selection_runs, list) or not isinstance(summary_runs, list):
        raise NumericalAuditError("Selection or terminal summary lacks the declared run matrix")
    selection_by_key = {row["run_key"]: row for row in selection_runs if isinstance(row, dict)}
    if len(selection_by_key) != len(run_order) or len(summary_runs) != len(run_order):
        raise NumericalAuditError("Selection/summary run count differs from the locked matrix")
    if [row.get("run_key") if isinstance(row, dict) else None for row in summary_runs] != list(run_order):
        raise NumericalAuditError("Terminal summary run order differs from the locked matrix")

    run_results: dict[str, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []
    score_by_run: dict[str, float] = {}
    for summary_row in summary_runs:
        if not isinstance(summary_row, dict) or summary_row.keys() != {
            "run_key", "configuration", "seed", "best_epoch", "validation_macro_f1",
            "test_metrics", "test_metrics_by_domain",
        }:
            raise NumericalAuditError("Terminal summary run row has an invalid schema")
        run_key = summary_row.get("run_key")
        selected = selection_by_key.get(run_key)
        if not isinstance(run_key, str) or selected is None:
            raise NumericalAuditError("Terminal summary run is outside the verified selection")
        for field in ("configuration", "seed", "best_epoch", "validation_macro_f1"):
            if summary_row.get(field) != selected.get(field):
                raise NumericalAuditError(f"Run metadata differs from selection: {run_key}.{field}")
        rows = grouped[run_key]
        gold = [row["y_true"] for row in rows]
        predicted = [row["y_pred"] for row in rows]
        metrics = compute_binary_metrics(gold, predicted)
        _assert_metric_equal(metrics, summary_row.get("test_metrics"), f"{run_key}.test_metrics")
        domains = sorted({str(row["domain"]) for row in rows})
        saved_by_domain = summary_row.get("test_metrics_by_domain")
        if not isinstance(saved_by_domain, dict) or set(saved_by_domain) != set(domains):
            raise NumericalAuditError(f"Per-domain metric schema differs: {run_key}")
        domain_metrics: dict[str, dict[str, object]] = {}
        for domain in domains:
            domain_rows = [row for row in rows if row["domain"] == domain]
            domain_result = compute_binary_metrics(
                [row["y_true"] for row in domain_rows],
                [row["y_pred"] for row in domain_rows],
            )
            _assert_metric_equal(
                domain_result, saved_by_domain[domain], f"{run_key}.domain.{domain}"
            )
            domain_metrics[domain] = domain_result
            comparison_rows.append(
                _comparison_row("domain", selected, domain_result, domain=domain)
            )
        run_results[run_key] = {"metrics": metrics, "by_domain": domain_metrics}
        score_by_run[run_key] = float(metrics["macro_f1"])
        comparison_rows.append(_comparison_row("run", selected, metrics))

    paired_differences = {
        str(seed): score_by_run[f"P1_seed{seed}"] - score_by_run[f"P0_seed{seed}"]
        for seed in (42, 43, 44)
    }
    contrast = summary.get("primary_contrast_descriptive")
    saved_differences = contrast.get("per_seed_differences") if isinstance(contrast, dict) else None
    differences_match = (
        isinstance(saved_differences, dict)
        and set(saved_differences) == set(paired_differences)
        and all(
            type(saved_differences[seed]) in (int, float)
            and not isinstance(saved_differences[seed], bool)
            and math.isclose(
                paired_differences[seed],
                float(saved_differences[seed]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for seed in paired_differences
        )
    )
    if (
        not isinstance(contrast, dict)
        or not differences_match
        or type(contrast.get("mean_difference")) not in (int, float)
        or not math.isclose(
            sum(paired_differences.values()) / 3,
            float(contrast["mean_difference"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise NumericalAuditError("Primary paired contrast differs from independently recomputed scores")

    family_rows: list[dict[str, object]] = []
    family_metrics: dict[str, dict[str, object]] = {}
    for configuration, keys in (
        ("S0", ["S0_seed42"]),
        ("P0", [f"P0_seed{seed}" for seed in (42, 43, 44)]),
        ("P1", [f"P1_seed{seed}" for seed in (42, 43, 44)]),
    ):
        fields = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        means = {
            field: statistics.mean(float(run_results[key]["metrics"][field]) for key in keys)
            for field in fields
        }
        deviations = {
            field: statistics.stdev(
                float(run_results[key]["metrics"][field]) for key in keys
            )
            if len(keys) > 1
            else None
            for field in fields
        }
        family_metrics[configuration] = {"mean": means, "sample_sd": deviations, "n": expected_n}
        family_row = {
            "row_type": "family_summary",
            "configuration": configuration,
            "run_key": "",
            "seed": "",
            "domain": "",
            "n": expected_n,
            "accuracy": means["accuracy"],
            "accuracy_sd": deviations["accuracy"] if deviations["accuracy"] is not None else "",
            "macro_precision": means["macro_precision"],
            "macro_precision_sd": deviations["macro_precision"] if deviations["macro_precision"] is not None else "",
            "macro_recall": means["macro_recall"],
            "macro_recall_sd": deviations["macro_recall"] if deviations["macro_recall"] is not None else "",
            "macro_f1": means["macro_f1"],
            "macro_f1_sd": deviations["macro_f1"] if deviations["macro_f1"] is not None else "",
            "f1_label_0": "",
            "f1_label_1": "",
            "tn": "",
            "fp": "",
            "fn": "",
            "tp": "",
        }
        family_rows.append(family_row)
    comparison_rows.extend(family_rows)

    evaluation_spec = protocol.get("evaluation")
    bootstrap_spec = evaluation_spec.get("bootstrap") if isinstance(evaluation_spec, dict) else None
    if not isinstance(bootstrap_spec, dict):
        raise NumericalAuditError("Locked protocol lacks paired-bootstrap settings")
    resamples = bootstrap_spec.get("resamples")
    random_seed = bootstrap_spec.get("random_seed")
    if type(resamples) is not int or type(random_seed) is not int:
        raise NumericalAuditError("Locked bootstrap count/seed is invalid")
    bootstrap_values = paired_bootstrap_differences(
        [row["y_true"] for row in grouped["P0_seed42"]],
        {
            key: [row["y_pred"] for row in grouped[key]]
            for key in run_order
            if key.startswith(("P0_", "P1_"))
        },
        resamples,
        random_seed,
    )
    ci_low, ci_high = (
        float(value)
        for value in np.quantile(bootstrap_values, [0.025, 0.975], method="linear")
    )
    audit: dict[str, object] = {
        "schema_version": 1,
        "status": "NUMERICAL_AUDIT_VERIFIED",
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "dataset_revision": revision,
        "selection_sha256": selection_sha256,
        "terminal_summary_sha256": summary_sha256,
        "terminal_predictions_sha256": predictions_sha256,
        "terminal_claim_sha256": claim_sha256,
        "terminal_outcome_sha256": _outcome_sha256,
        "terminal_evaluator_sha256": claim.get("terminal_evaluator_sha256"),
        "numerical_audit_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "numpy_version": numpy_version,
        "scikit_learn_version": selection.get("scikit_learn_version"),
        "test_rows": expected_n,
        "ordered_prediction_key_sha256": ordered_key_sha256,
        "run_order": list(run_order),
        "independent_metric_method": "pure-Python confusion counts; no evaluator metric functions imported",
        "family_summary": family_metrics,
        "primary_contrast": {
            "estimand": bootstrap_spec.get("estimand"),
            "per_seed_test_macro_f1_differences": paired_differences,
            "mean_test_macro_f1_difference": statistics.mean(paired_differences.values()),
            "bootstrap_resamples": resamples,
            "bootstrap_random_seed": random_seed,
            "bootstrap_ci_percentile_95": [ci_low, ci_high],
            "scope_limit": bootstrap_spec.get("scope_limit"),
            "interpretation": evaluation_spec.get("hypothesis_policy"),
        },
        "raw_review_text_saved": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    csv_payload = _csv_bytes(comparison_rows)
    bootstrap_csv = io.StringIO(newline="")
    writer = csv.writer(bootstrap_csv, lineterminator="\n")
    writer.writerow(["replicate", "paired_macro_f1_difference_p1_minus_p0"])
    writer.writerows((index, value) for index, value in enumerate(bootstrap_values, start=1))
    bootstrap_payload = bootstrap_csv.getvalue().encode("utf-8")
    stage_dir = Path(tempfile.mkdtemp(prefix=".numerical-audit-stage-", dir=terminal))
    try:
        comparison_sha256 = _write_new(stage_dir / "comparison.csv", csv_payload)
        bootstrap_sha256 = _write_new(
            stage_dir / "paired_bootstrap_differences.csv", bootstrap_payload
        )
        audit["comparison_csv_sha256"] = comparison_sha256
        audit["paired_bootstrap_csv_sha256"] = bootstrap_sha256
        encoded = (
            json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        audit_sha256 = _write_new(stage_dir / "numerical_audit.json", encoded)
        os.rename(stage_dir, output)
    except Exception:
        if stage_dir.exists():
            import shutil

            shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return {
        "audit": audit,
        "audit_sha256": audit_sha256,
        "comparison_csv_sha256": comparison_sha256,
        "paired_bootstrap_csv_sha256": bootstrap_sha256,
        "output_dir": str(output),
    }


def _comparison_row(
    row_type: str,
    selection_row: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    domain: str = "",
) -> dict[str, object]:
    confusion = metrics["confusion_matrix_labels_0_1"]
    per_class = metrics["per_class"]
    if not isinstance(confusion, list) or not isinstance(per_class, dict):
        raise NumericalAuditError("Independent metric result is malformed")
    return {
        "row_type": row_type,
        "configuration": selection_row["configuration"],
        "run_key": selection_row["run_key"],
        "seed": selection_row["seed"],
        "domain": domain,
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "accuracy_sd": "",
        "macro_precision": metrics["macro_precision"],
        "macro_precision_sd": "",
        "macro_recall": metrics["macro_recall"],
        "macro_recall_sd": "",
        "macro_f1": metrics["macro_f1"],
        "macro_f1_sd": "",
        "f1_label_0": per_class["0"]["f1"],
        "f1_label_1": per_class["1"]["f1"],
        "tn": confusion[0][0],
        "fp": confusion[0][1],
        "fn": confusion[1][0],
        "tp": confusion[1][1],
    }
