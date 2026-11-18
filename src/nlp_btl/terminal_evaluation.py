"""Selection-gated terminal evaluation for the locked seven-run matrix."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping, Sequence

from . import phobert, preprocessing
from .data import (
    DataIntegrityError,
    ReviewRecord,
    TerminalTestBundle,
    _sha256,
    load_terminal_test,
)
from .preprocessing import RDRSegmenter, UndertheseaSegmenter, preprocess_records
from .selection import (
    SelectionError,
    _read_json,
    _safe_artifact_path,
    _verify_metric_runtime,
    expected_run_keys,
    verify_selection_artifact,
)


@dataclass(frozen=True, slots=True)
class TerminalInput:
    """One commonly preprocessed test review without its gold label or raw text."""

    key: tuple[str, str, int]
    segmented_text: str
    domain: str


@dataclass(frozen=True, slots=True)
class TerminalPrediction:
    """A predicted class bound to a test source key."""

    key: tuple[str, str, int]
    y_pred: int


Predictor = Callable[
    [Mapping[str, object], Sequence[TerminalInput]], Sequence[TerminalPrediction]
]


class TerminalEvaluationError(DataIntegrityError):
    """Raised when terminal test predictions violate the locked evaluation contract."""


def _private_results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def _cohort_claim_paths(
    private_root: Path, selection: Mapping[str, object]
) -> tuple[Path, Path, str]:
    revision = selection.get("dataset_revision")
    manifest_sha256 = selection.get("manifest_sha256")
    if not isinstance(revision, str) or not isinstance(manifest_sha256, str):
        raise TerminalEvaluationError("Selection lacks a locked data-cohort identity")
    cohort_id = _sha256(
        f"viocd-terminal-cohort-v1\n{revision}\n{manifest_sha256}\n".encode("utf-8")
    )
    claim_dir = private_root / ".terminal_evaluation_claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    stem = claim_dir / f"{cohort_id}.json"
    return (
        stem.with_name(stem.stem + ".claim.json"),
        stem.with_name(stem.stem + ".outcome.json"),
        cohort_id,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _outcome_payload(
    status: str,
    stage: str,
    fetch_started: bool,
    fetch_returned_bytes: bool,
    predictor_backend: str,
    error: Exception | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "stage": stage,
        "test_fetch_started": fetch_started,
        "test_fetch_returned_bytes": fetch_returned_bytes,
        "predictor_backend": predictor_backend,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if error is not None:
        result["exception_type"] = type(error).__name__
    return result


class LockedArtifactPredictor:
    """Load the exact selected S0 or PhoBERT artifact before returning predictions."""

    backend = "LOCKED_ARTIFACTS"

    def __init__(
        self,
        protocol_path: str | Path,
        selection_path: str | Path,
        selection: Mapping[str, object],
    ) -> None:
        self.protocol_path = Path(protocol_path).resolve(strict=True)
        self.selection_path = Path(selection_path).resolve(strict=True)
        self.root = self.selection_path.parent.resolve(strict=True)
        self.preparation = self.protocol_path.parent
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.selection = selection
        runs = selection.get("runs")
        if not isinstance(runs, list) or any(not isinstance(row, dict) for row in runs):
            raise TerminalEvaluationError("Verified selection has an invalid run list")
        self.run_rows = {row["run_key"]: row for row in runs}
        try:
            import torch
        except Exception as exc:
            raise TerminalEvaluationError("Locked PhoBERT inference runtime is unavailable") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise TerminalEvaluationError("Locked PhoBERT terminal inference requires CUDA")
        self.torch = torch
        self.device = torch.device("cuda:0")
        try:
            tokenizer_audit = phobert._read_tokenizer_audit(self.preparation)
            self.tokenizer = preprocessing._load_pinned_tokenizer(
                self.preparation, tokenizer_audit, self.protocol
            )
        except Exception as exc:
            raise TerminalEvaluationError("Pinned PhoBERT tokenizer preflight failed") from exc
        self._current_run_key: str | None = None
        self._current_model = None

    def _verify_run(self, run_row: Mapping[str, object]) -> tuple[Path, dict[str, object]]:
        run_key = run_row.get("run_key")
        selected_row = self.run_rows.get(run_key)
        if selected_row != run_row:
            raise TerminalEvaluationError("Predictor received a run outside the verified selection")
        directory = run_row.get("run_directory")
        artifact_hashes = run_row.get("artifact_sha256")
        if not isinstance(directory, str) or not isinstance(artifact_hashes, dict):
            raise TerminalEvaluationError("Selected run artifact manifest is invalid")
        run_dir = _safe_artifact_path(self.root, directory, must_be_file=False)
        for relative, expected_hash in artifact_hashes.items():
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise TerminalEvaluationError("Selected run hash entry is invalid")
            artifact = _safe_artifact_path(run_dir, relative)
            if _sha256(artifact.read_bytes()) != expected_hash:
                raise TerminalEvaluationError("Selected run artifact changed after selection")
        metadata, _ = _read_json(run_dir / "run_metadata.json", "run_metadata.json")
        if metadata.get("run_id") != run_row.get("run_id"):
            raise TerminalEvaluationError("Selected run metadata identity changed")
        return run_dir, metadata

    def _load_phobert(self, run_key: str, run_dir: Path, metadata: Mapping[str, object]):
        if self._current_run_key == run_key and self._current_model is not None:
            return self._current_model
        self._current_model = None
        self._current_run_key = None
        gc.collect()
        self.torch.cuda.empty_cache()
        try:
            model, loading = phobert._build_pinned_model(
                self.preparation, self.protocol, int(metadata["seed"])
            )
            if loading.get("only_expected_head_differences") is not True:
                raise TerminalEvaluationError("Pinned PhoBERT base loading audit failed")
            phobert.configure_partial_finetuning(model)
            expected_binding = {
                name: metadata[name]
                for name in phobert.CHECKPOINT_BINDING_FIELDS
            }
            phobert.restore_trainable_state(
                model, run_dir / "best_checkpoint", expected_binding
            )
            model.to(self.device)
            model.eval()
        except TerminalEvaluationError:
            raise
        except Exception as exc:
            raise TerminalEvaluationError(
                f"Selected PhoBERT checkpoint restore failed for {run_key}"
            ) from exc
        self._current_run_key = run_key
        self._current_model = model
        return model

    def __call__(
        self,
        run_row: Mapping[str, object],
        inputs: Sequence[TerminalInput],
    ) -> tuple[TerminalPrediction, ...]:
        run_dir, metadata = self._verify_run(run_row)
        run_key = str(run_row["run_key"])
        configuration = str(run_row["configuration"])
        if configuration == "S0":
            self._current_model = None
            self._current_run_key = None
            self.torch.cuda.empty_cache()
            import joblib

            try:
                pipeline = joblib.load(_safe_artifact_path(run_dir, "pipeline.joblib"))
                predicted = pipeline.predict([row.segmented_text for row in inputs])
            except Exception as exc:
                raise TerminalEvaluationError("Selected S0 pipeline inference failed") from exc
            if len(predicted) != len(inputs):
                raise TerminalEvaluationError("Selected S0 pipeline returned the wrong row count")
            return tuple(
                TerminalPrediction(row.key, int(label))
                for row, label in zip(inputs, predicted, strict=True)
            )

        if configuration not in {"P0", "P1"}:
            raise TerminalEvaluationError("Selected model family is unsupported")
        model = self._load_phobert(run_key, run_dir, metadata)
        batch_size = int(self.protocol["training"]["micro_batch"])
        max_length = int(self.protocol["training"]["max_length"])
        results: list[TerminalPrediction] = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            try:
                encoded = preprocessing.tokenize_batch(
                    self.tokenizer, batch, max_length=max_length
                )
                if (
                    encoded["input_ids"].shape[1] > max_length
                    or encoded["input_ids"].shape[1] % 8
                ):
                    raise TerminalEvaluationError("Terminal tokenizer padding violates the locked contract")
                model_inputs = {
                    name: encoded[name].to(self.device, non_blocking=True)
                    for name in ("input_ids", "attention_mask")
                }
                with self.torch.inference_mode():
                    with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                        logits = model(**model_inputs).logits
                if not self.torch.isfinite(logits).all():
                    raise TerminalEvaluationError("Terminal model produced non-finite logits")
                labels = logits.argmax(dim=-1).detach().cpu().tolist()
            except TerminalEvaluationError:
                raise
            except Exception as exc:
                raise TerminalEvaluationError(f"Selected PhoBERT inference failed for {run_key}") from exc
            results.extend(
                TerminalPrediction(row.key, int(label))
                for row, label in zip(batch, labels, strict=True)
            )
        return tuple(results)


def _create_segmenter(
    segmenter_id: str, preparation_dir: Path, protocol: Mapping[str, object]
):
    if segmenter_id == "underthesea":
        return UndertheseaSegmenter()
    if segmenter_id != "rdr":
        raise TerminalEvaluationError("Requested segmenter is outside the locked protocol")
    try:
        lock = json.loads(
            (preparation_dir / "preparation_lock.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalEvaluationError("Pinned RDR preparation lock is unavailable") from exc
    preprocessing_spec = protocol.get("preprocessing")
    rdr_spec = (
        preprocessing_spec.get("rdr")
        if isinstance(preprocessing_spec, dict)
        else None
    )
    manifest_name = rdr_spec.get("assets_manifest") if isinstance(rdr_spec, dict) else None
    if not isinstance(lock, dict) or not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise TerminalEvaluationError("Pinned RDR asset contract is invalid")
    manifest, _ = preprocessing._read_locked_json(
        preparation_dir, lock, manifest_name
    )
    preprocessing._verify_rdr_assets(preparation_dir, manifest)
    preprocessing._verify_rdr_wrapper(preparation_dir, protocol)
    return RDRSegmenter(preparation_dir)


def _validate_predictions(
    records: Sequence[ReviewRecord],
    predictions: Sequence[TerminalPrediction],
    revision: str,
) -> dict[tuple[str, str, int], int]:
    if len(predictions) != len(records):
        raise TerminalEvaluationError("Predictor output count differs from the test population")
    expected = {record.key for record in records}
    observed: dict[tuple[str, str, int], int] = {}
    for row in predictions:
        if not isinstance(row, TerminalPrediction):
            raise TerminalEvaluationError("Predictor returned an invalid prediction row")
        if (
            not isinstance(row.key, tuple)
            or len(row.key) != 3
            or row.key[0] != revision
            or row.key[1] != "test"
            or type(row.key[2]) is not int
            or row.key not in expected
            or row.key in observed
            or type(row.y_pred) is not int
            or row.y_pred not in (0, 1)
        ):
            raise TerminalEvaluationError("Prediction keys or labels do not match the test contract")
        observed[row.key] = row.y_pred
    if observed.keys() != expected:
        raise TerminalEvaluationError("Predictions do not cover the exact terminal test keys")
    return observed


def _class_metrics(gold: Sequence[int], predicted: Sequence[int]) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(label): {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in (0, 1)
        },
        "confusion_matrix_labels_0_1": confusion_matrix(
            gold, predicted, labels=[0, 1]
        ).tolist(),
    }


def _metrics_by_domain(
    records: Sequence[ReviewRecord],
    predicted_by_key: Mapping[tuple[str, str, int], int],
) -> dict[str, dict[str, object]]:
    domains = sorted({record.domain for record in records})
    result: dict[str, dict[str, object]] = {}
    for domain in domains:
        rows = [record for record in records if record.domain == domain]
        result[domain] = _class_metrics(
            [record.label for record in rows],
            [predicted_by_key[record.key] for record in rows],
        )
    return result


def _write_create_only(path: Path, payload: Mapping[str, object]) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    digest = _sha256(encoded)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    with path.with_suffix(path.suffix + ".sha256").open(
        "x", encoding="ascii", newline="\n"
    ) as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def _build_locked_predictor(
    protocol_path: Path,
    selection_path: Path,
    selection: Mapping[str, object],
) -> Predictor:
    """Private factory seam used by production and replaced only in synthetic tests."""
    return LockedArtifactPredictor(protocol_path, selection_path, selection)


def evaluate_terminal_test(
    protocol_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict[str, object]:
    """Evaluate all seven locked runs once and persist private keyed evidence.

    Production always constructs the locked artifact predictor. Synthetic tests
    replace its private factory seam; callers cannot inject a predictor through
    this API. A global claim permits only one terminal attempt per locked cohort.
    """
    protocol_file = Path(protocol_path).resolve(strict=True)
    selection_file = Path(selection_path).resolve(strict=True)
    private_root = _private_results_root().resolve(strict=False)
    if not _is_within(selection_file.parent, private_root):
        raise TerminalEvaluationError("Selection must be stored under the private results root")
    output = Path(output_dir).resolve(strict=False)
    expected_output = (selection_file.parent / "terminal").resolve(strict=False)
    if output != expected_output or not _is_within(output, private_root):
        raise TerminalEvaluationError("Terminal output must use the selection's private terminal folder")
    if output.exists():
        raise TerminalEvaluationError("Terminal evaluation output is create-only")
    selection_relative = selection_file.parent.relative_to(private_root).as_posix()
    output_relative = output.relative_to(private_root).as_posix()

    try:
        selection, selection_sha256 = verify_selection_artifact(
            selection_file, protocol_file
        )
    except SelectionError as exc:
        raise TerminalEvaluationError("Selection verification failed before test access") from exc
    try:
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalEvaluationError("Verified protocol cannot be read") from exc
    if not isinstance(protocol, dict):
        raise TerminalEvaluationError("Verified protocol has an invalid schema")
    dataset = protocol.get("dataset")
    manifest_file = dataset.get("manifest_file") if isinstance(dataset, dict) else None
    if not isinstance(manifest_file, str) or Path(manifest_file).name != manifest_file:
        raise TerminalEvaluationError("Verified protocol lacks a locked data manifest")
    try:
        manifest_bytes = (protocol_file.parent / manifest_file).read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalEvaluationError("Locked data manifest is unavailable") from exc
    if (
        _sha256(manifest_bytes) != selection.get("manifest_sha256")
        or not isinstance(manifest, dict)
        or not isinstance(manifest.get("splits"), dict)
        or not isinstance(manifest["splits"].get("test"), dict)
    ):
        raise TerminalEvaluationError("Locked test manifest differs from the verified selection")
    test_prediction_binding_digest = manifest["splits"]["test"].get(
        "ordered_prediction_binding_digest_sha256"
    )
    if (
        not isinstance(test_prediction_binding_digest, str)
        or len(test_prediction_binding_digest) != 64
        or any(character not in "0123456789abcdef" for character in test_prediction_binding_digest)
    ):
        raise TerminalEvaluationError("Locked test manifest lacks a valid prediction binding digest")
    metric_version = _verify_metric_runtime(protocol)
    if selection.get("scikit_learn_version") != metric_version:
        raise TerminalEvaluationError("Selection metric runtime differs from terminal evaluation")

    predictor = _build_locked_predictor(protocol_file, selection_file, selection)
    is_locked_predictor = type(predictor) is LockedArtifactPredictor
    predictor_backend = (
        "LOCKED_ARTIFACTS" if is_locked_predictor else "SYNTHETIC_CALLBACK_TEST_ONLY"
    )
    if not callable(predictor):
        raise TerminalEvaluationError("Terminal predictor is unavailable")

    expected_runs = expected_run_keys(protocol)
    run_rows = selection.get("runs")
    if (
        not isinstance(run_rows, list)
        or len(run_rows) != len(expected_runs)
        or any(not isinstance(row, dict) for row in run_rows)
        or tuple(row.get("run_key") for row in run_rows if isinstance(row, dict))
        != expected_runs
    ):
        raise TerminalEvaluationError("Verified selection does not contain all declared runs")

    claim_path, outcome_path, cohort_id = _cohort_claim_paths(private_root, selection)
    if claim_path.exists() or outcome_path.exists():
        raise TerminalEvaluationError("This locked data cohort already has a terminal evaluation claim")
    terminal_evaluator_sha256 = _sha256(Path(__file__).resolve().read_bytes())
    claim_sha256 = _write_create_only(
        claim_path,
        {
            "schema_version": 1,
            "status": "EVALUATING",
            "cohort_id": cohort_id,
            "dataset_revision": selection["dataset_revision"],
            "manifest_sha256": selection["manifest_sha256"],
            "protocol_sha256": _sha256(protocol_file.read_bytes()),
            "selection_sha256": selection_sha256,
            "run_order": list(expected_runs),
            "test_prediction_binding_digest_sha256": test_prediction_binding_digest,
            "selection_directory": selection_relative,
            "output_directory": output_relative,
            "predictor_backend": predictor_backend,
            "terminal_evaluator_sha256": terminal_evaluator_sha256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )

    fetch_started = False
    fetch_returned_bytes = False
    stage_name = "test_fetch"
    stage_dir: Path | None = None

    def tracked_fetch(url: str) -> bytes:
        nonlocal fetch_started, fetch_returned_bytes
        fetch_started = True
        reader = fetch_bytes
        if reader is None:
            from .data import _fetch_bytes

            reader = _fetch_bytes
        payload = reader(url)
        fetch_returned_bytes = isinstance(payload, bytes)
        return payload

    try:
        test_bundle = load_terminal_test(
            protocol_file, selection_file, fetch_bytes=tracked_fetch
        )
        if test_bundle.selection_sha256 != selection_sha256:
            raise TerminalEvaluationError("Selection changed between preflight and test access")
        if (
            test_bundle.audit.get("ordered_prediction_binding_digest_sha256")
            != test_prediction_binding_digest
        ):
            raise TerminalEvaluationError("Loaded test rows differ from the claimed cohort binding")
        records = test_bundle.records
        gold_by_key = {record.key: record.label for record in records}
        record_by_key = {record.key: record for record in records}
        run_metrics: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        score_by_key: dict[str, float] = {}
        inputs_by_segmenter: dict[str, tuple[TerminalInput, ...]] = {}
        preparation_dir = protocol_file.parent
        stage_name = "preprocessing_and_prediction"

        for run_row in run_rows:
            if not isinstance(run_row, dict):
                raise TerminalEvaluationError("Selection run row is invalid")
            run_key = run_row["run_key"]
            configuration = run_row.get("configuration")
            segmenter_id = {
                "S0": "underthesea",
                "P0": "underthesea",
                "P1": "rdr",
            }.get(configuration)
            if segmenter_id is None:
                raise TerminalEvaluationError("Selected run has no locked preprocessing route")
            if segmenter_id not in inputs_by_segmenter:
                try:
                    segmenter = _create_segmenter(segmenter_id, preparation_dir, protocol)
                    processed = preprocess_records(records, segmenter)
                except Exception as exc:
                    raise TerminalEvaluationError(
                        f"Locked {segmenter_id} test preprocessing failed"
                    ) from exc
                if len(processed) != len(records) or any(
                    transformed.key != record.key
                    or transformed.label != record.label
                    or transformed.domain != record.domain
                    for transformed, record in zip(processed, records, strict=True)
                ):
                    raise TerminalEvaluationError("Shared preprocessing changed test row identity")
                inputs_by_segmenter[segmenter_id] = tuple(
                    TerminalInput(
                        key=row.key,
                        segmented_text=row.segmented_text,
                        domain=row.domain,
                    )
                    for row in processed
                )
            try:
                raw_predictions = predictor(run_row, inputs_by_segmenter[segmenter_id])
            except Exception as exc:
                raise TerminalEvaluationError(f"Prediction failed for declared run {run_key}") from exc
            predicted_by_key = _validate_predictions(records, raw_predictions, test_bundle.revision)
            keys_in_order = [record.key for record in records]
            gold = [gold_by_key[key] for key in keys_in_order]
            predicted = [predicted_by_key[key] for key in keys_in_order]
            metrics = _class_metrics(gold, predicted)
            score_by_key[str(run_key)] = float(metrics["macro_f1"])
            run_metrics.append(
                {
                    "run_key": run_key,
                    "configuration": run_row["configuration"],
                    "seed": run_row["seed"],
                    "best_epoch": run_row["best_epoch"],
                    "validation_macro_f1": run_row["validation_macro_f1"],
                    "test_metrics": metrics,
                    "test_metrics_by_domain": _metrics_by_domain(records, predicted_by_key),
                }
            )
            for key in keys_in_order:
                record = record_by_key[key]
                prediction_rows.append(
                    {
                        "run_key": run_key,
                        "key": list(key),
                        "domain": record.domain,
                        "y_true": gold_by_key[key],
                        "y_pred": predicted_by_key[key],
                    }
                )

        stage_name = "artifact_staging"
        paired_differences = {
            str(seed): score_by_key[f"P1_seed{seed}"] - score_by_key[f"P0_seed{seed}"]
            for seed in (42, 43, 44)
        }
        summary: dict[str, object] = {
            "schema_version": 1,
            "status": (
                "TERMINAL_EVALUATION_COMPLETE"
                if is_locked_predictor
                else "CALLBACK_FIXTURE_COMPLETE"
            ),
            "predictor_backend": (
                "LOCKED_ARTIFACTS" if is_locked_predictor else "SYNTHETIC_CALLBACK_TEST_ONLY"
            ),
            "protocol_id": test_bundle.protocol_id,
            "protocol_sha256": test_bundle.protocol_sha256,
            "manifest_sha256": test_bundle.manifest_sha256,
            "dataset_revision": test_bundle.revision,
            "scikit_learn_version": metric_version,
            "selection_sha256": test_bundle.selection_sha256,
            "selected_configuration": selection.get("selected_configuration"),
            "selected_demo_run": selection.get("selected_demo_run"),
            "test_source_sha256": test_bundle.source_sha256,
            "test_rows": len(records),
            "runs": run_metrics,
            "primary_contrast_descriptive": {
                "estimand": "mean of paired per-seed P1 minus P0 test Macro-F1",
                "per_seed_differences": paired_differences,
                "mean_difference": sum(paired_differences.values()) / 3,
                "interpretation": "Descriptive historical post-hoc evaluation; no independent test claim",
            },
            "terminal_loader_sha256": test_bundle.loader_sha256,
            "terminal_preprocessing_sha256": _sha256(
                Path(preprocessing.__file__).resolve().read_bytes()
            ),
            "terminal_evaluator_sha256": terminal_evaluator_sha256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_review_text_saved": False,
        }
        predictions_payload = {
            "schema_version": 1,
            "status": "TERMINAL_PREDICTIONS_SAVED",
            "protocol_sha256": test_bundle.protocol_sha256,
            "selection_sha256": test_bundle.selection_sha256,
            "test_source_sha256": test_bundle.source_sha256,
            "predictions": prediction_rows,
            "raw_review_text_saved": False,
        }
        stage_dir = Path(tempfile.mkdtemp(prefix=".terminal-stage-", dir=selection_file.parent))
        summary_sha256 = _write_create_only(stage_dir / "terminal_evaluation.json", summary)
        predictions_sha256 = _write_create_only(
            stage_dir / "test_predictions.json", predictions_payload
        )
        stage_name = "artifact_commit"
        os.rename(stage_dir, output)
        stage_dir = None
        _write_create_only(
            outcome_path,
            {
                **_outcome_payload(
                    "COMPLETE", "artifact_commit", True, True, predictor_backend
                ),
                "cohort_id": cohort_id,
                "claim_sha256": claim_sha256,
                "selection_sha256": selection_sha256,
                "summary_sha256": summary_sha256,
                "predictions_sha256": predictions_sha256,
                "output_directory": output_relative,
            },
        )
        return {
            "summary": summary,
            "summary_sha256": summary_sha256,
            "predictions_sha256": predictions_sha256,
            "output_dir": str(output),
            "outcome_path": str(outcome_path),
        }
    except Exception as exc:
        status = (
            "FAILED_BEFORE_TEST_FETCH"
            if not fetch_started
            else "FAILED_AFTER_TEST_FETCH"
            if fetch_returned_bytes
            else "TEST_FETCH_OUTCOME_UNKNOWN"
        )
        try:
            failure_receipt = _outcome_payload(
                status,
                stage_name,
                fetch_started,
                fetch_returned_bytes,
                predictor_backend,
                error=exc,
            )
            failure_receipt.update(
                {
                    "cohort_id": cohort_id,
                    "claim_sha256": claim_sha256,
                    "selection_sha256": selection_sha256,
                    "terminal_evaluator_sha256": terminal_evaluator_sha256,
                    "output_directory": output_relative,
                }
            )
            _write_create_only(
                outcome_path,
                failure_receipt,
            )
        except Exception as receipt_exc:
            raise TerminalEvaluationError(
                "Terminal attempt failed and its outcome receipt could not be written; claim remains"
            ) from receipt_exc
        raise
    finally:
        if stage_dir is not None and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
