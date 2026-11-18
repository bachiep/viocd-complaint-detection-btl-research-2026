"""Locked S0 TF-IDF/LinearSVC development path; no terminal-test access."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

from . import preprocessing
from .data import DataIntegrityError, DatasetBundle, ReviewRecord, _sha256
from .preprocessing import (
    UndertheseaSegmenter,
    preprocess_records,
)


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    """One validation prediction bound to a stable source key."""

    key: tuple[str, str, int]
    y_true: int
    y_pred: int


@dataclass(frozen=True, slots=True)
class S0DevelopmentResult:
    """In-memory S0 estimator and keyed validation predictions."""

    estimator: Any
    validation_predictions: tuple[BaselinePrediction, ...]
    dataset_revision: str
    protocol_sha256: str
    manifest_sha256: str
    baseline_code_sha256: str
    preprocessing_code_sha256: str
    scikit_learn_version: str
    train_rows: int
    validation_rows: int

    def audit_report(self) -> dict[str, object]:
        """Return run metadata without prediction keys, text or model state."""
        return {
            "schema_version": 1,
            "status": "PASS",
            "configuration": "S0",
            "dataset_revision": self.dataset_revision,
            "protocol_sha256": self.protocol_sha256,
            "manifest_sha256": self.manifest_sha256,
            "baseline_code_sha256": self.baseline_code_sha256,
            "preprocessing_code_sha256": self.preprocessing_code_sha256,
            "scikit_learn_version": self.scikit_learn_version,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "validation_prediction_rows": len(self.validation_predictions),
            "prediction_keys_validated": True,
            "fit_performed": True,
            "fit_data_split": "train",
            "test_split_available_to_runner": False,
            "raw_review_text_saved": False,
            "row_level_predictions_saved": False,
            "fit_scope": "train only",
        }


def _read_development_protocol(
    protocol_path: str | Path, bundle: DatasetBundle
) -> tuple[dict[str, object], str, str]:
    path = Path(protocol_path).resolve(strict=True)
    try:
        payload = path.read_bytes()
        protocol = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError("Cannot read the S0 development protocol") from exc
    protocol_sha256 = _sha256(payload)
    if protocol_sha256 != bundle.protocol_sha256:
        raise DataIntegrityError("S0 protocol differs from the verified dataset bundle")
    if not isinstance(protocol, dict):
        raise DataIntegrityError("S0 protocol has an invalid schema")
    dataset = protocol.get("dataset")
    if not isinstance(dataset, dict):
        raise DataIntegrityError("S0 protocol has no dataset identity")
    if dataset.get("revision") != bundle.revision:
        raise DataIntegrityError("S0 bundle revision differs from protocol")
    manifest_sha256 = dataset.get("manifest_sha256")
    if manifest_sha256 != bundle.manifest_sha256:
        raise DataIntegrityError("S0 bundle manifest differs from protocol")
    if protocol.get("protocol_id") != bundle.protocol_id:
        raise DataIntegrityError("S0 protocol identifier differs from dataset bundle")
    return protocol, protocol_sha256, str(manifest_sha256)


def _build_s0_pipeline(config: Mapping[str, object]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    vectorizer_config = config.get("vectorizer")
    classifier_config = config.get("classifier")
    if not isinstance(vectorizer_config, dict) or not isinstance(classifier_config, dict):
        raise DataIntegrityError("S0 model configuration is invalid")
    if set(vectorizer_config) != {
        "name",
        "tokenizer",
        "token_pattern",
        "lowercase",
        "ngram_range",
        "min_df",
        "max_df",
        "norm",
        "use_idf",
        "smooth_idf",
        "sublinear_tf",
        "max_features",
        "stop_words",
        "analyzer",
        "binary",
    }:
        raise DataIntegrityError("S0 vectorizer parameters differ from the supported protocol")
    if set(classifier_config) != {
        "name",
        "C",
        "class_weight",
        "dual",
        "fit_intercept",
        "intercept_scaling",
        "loss",
        "max_iter",
        "multi_class",
        "penalty",
        "random_state",
        "tol",
        "verbose",
    }:
        raise DataIntegrityError("S0 classifier parameters differ from the supported protocol")
    if config.get("repetitions") != 1 or config.get("fit_scope") != (
        "Pipeline fit on retained train only; no A/B/C re-selection in this fixed S0 protocol"
    ):
        raise DataIntegrityError("S0 fit scope differs from the locked protocol")
    if vectorizer_config.get("name") != "TfidfVectorizer":
        raise DataIntegrityError("S0 vectorizer differs from the locked protocol")
    if vectorizer_config.get("tokenizer") != "str.split":
        raise DataIntegrityError("S0 tokenizer differs from the locked protocol")
    if classifier_config.get("name") != "LinearSVC":
        raise DataIntegrityError("S0 classifier differs from the locked protocol")

    try:
        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=vectorizer_config["token_pattern"],
            lowercase=vectorizer_config["lowercase"],
            ngram_range=tuple(vectorizer_config["ngram_range"]),
            min_df=vectorizer_config["min_df"],
            max_df=vectorizer_config["max_df"],
            norm=vectorizer_config["norm"],
            use_idf=vectorizer_config["use_idf"],
            smooth_idf=vectorizer_config["smooth_idf"],
            sublinear_tf=vectorizer_config["sublinear_tf"],
            max_features=vectorizer_config["max_features"],
            stop_words=vectorizer_config["stop_words"],
            analyzer=vectorizer_config["analyzer"],
            binary=vectorizer_config["binary"],
        )
        classifier = LinearSVC(
            **{
                key: value
                for key, value in classifier_config.items()
                if key != "name"
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataIntegrityError("S0 parameters are invalid for the installed scikit-learn") from exc
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def validate_validation_predictions(
    records: Sequence[ReviewRecord],
    predictions: Sequence[BaselinePrediction],
) -> None:
    """Require a one-to-one validation prediction/key/label correspondence."""
    if not records or not predictions:
        raise DataIntegrityError("Validation prediction population is empty")
    if any(not isinstance(record, ReviewRecord) for record in records):
        raise DataIntegrityError("Validation population contains an invalid record")
    if any(record.split != "val" for record in records):
        raise DataIntegrityError("Prediction validation input contains a non-validation row")
    expected: dict[tuple[str, str, int], int] = {}
    for record in records:
        if record.key in expected:
            raise DataIntegrityError("Validation source keys are not unique")
        expected[record.key] = record.label

    observed: dict[tuple[str, str, int], BaselinePrediction] = {}
    for prediction in predictions:
        if not isinstance(prediction, BaselinePrediction):
            raise DataIntegrityError("Validation prediction artifact has an invalid row")
        if prediction.key in observed:
            raise DataIntegrityError("Validation prediction keys are not unique")
        observed[prediction.key] = prediction
    if observed.keys() != expected.keys():
        raise DataIntegrityError("Validation predictions do not match the expected source keys")
    for key, prediction in observed.items():
        if prediction.y_true != expected[key] or prediction.y_true not in (0, 1):
            raise DataIntegrityError("Validation prediction label differs from source label")
        if prediction.y_pred not in (0, 1):
            raise DataIntegrityError("S0 produced a label outside the binary task")


def run_s0_development(
    bundle: DatasetBundle,
    protocol_path: str | Path,
) -> S0DevelopmentResult:
    """Fit the fixed S0 pipeline on train and produce keyed validation predictions.

    The verified development bundle deliberately omits test. This function
    rejects any bundle that contains another split, so the terminal population
    cannot enter the development fit or prediction path.
    """
    if set(bundle.splits) != {"train", "val"}:
        raise DataIntegrityError("S0 development accepts only train and validation splits")
    train = bundle.splits["train"]
    validation = bundle.splits["val"]
    if not train or not validation:
        raise DataIntegrityError("S0 development requires non-empty train and validation splits")
    for split, records in (("train", train), ("val", validation)):
        if any(not isinstance(record, ReviewRecord) for record in records):
            raise DataIntegrityError(f"S0 {split} contains an invalid record")
        if any(record.split != split or record.revision != bundle.revision for record in records):
            raise DataIntegrityError(f"S0 {split} records differ from the verified bundle")
        if any(record.label not in (0, 1) for record in records):
            raise DataIntegrityError(f"S0 {split} contains a non-binary label")
        if len({record.key for record in records}) != len(records):
            raise DataIntegrityError(f"S0 {split} source keys are not unique")
    if len({record.label for record in train}) != 2:
        raise DataIntegrityError("S0 training split must contain both labels")

    protocol, protocol_sha256, manifest_sha256 = _read_development_protocol(
        protocol_path, bundle
    )
    baseline_config = protocol.get("baseline")
    environment = protocol.get("environment")
    if not isinstance(baseline_config, dict) or not isinstance(environment, dict):
        raise DataIntegrityError("S0 protocol configuration is missing")
    package_targets = environment.get("explicit_package_targets")
    if not isinstance(package_targets, dict):
        raise DataIntegrityError("S0 scikit-learn target is missing")

    from importlib.metadata import version

    actual_sklearn_version = version("scikit-learn")
    if actual_sklearn_version != package_targets.get("scikit-learn"):
        raise DataIntegrityError("scikit-learn version differs from locked protocol")

    segmenter = UndertheseaSegmenter()
    processed_train = preprocess_records(train, segmenter)
    processed_validation = preprocess_records(validation, segmenter)
    estimator = _build_s0_pipeline(baseline_config)
    try:
        estimator.fit(
            [record.segmented_text for record in processed_train],
            [record.label for record in processed_train],
        )
        predicted = estimator.predict(
            [record.segmented_text for record in processed_validation]
        )
    except Exception as exc:
        raise DataIntegrityError("S0 fit or validation prediction failed") from exc
    if len(predicted) != len(validation):
        raise DataIntegrityError("S0 validation prediction count differs from population")
    predictions = tuple(
        BaselinePrediction(key=record.key, y_true=record.label, y_pred=int(y_pred))
        for record, y_pred in zip(validation, predicted, strict=True)
    )
    validate_validation_predictions(validation, predictions)

    return S0DevelopmentResult(
        estimator=estimator,
        validation_predictions=predictions,
        dataset_revision=bundle.revision,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        baseline_code_sha256=_sha256(Path(__file__).resolve().read_bytes()),
        preprocessing_code_sha256=_sha256(
            Path(preprocessing.__file__).resolve().read_bytes()
        ),
        scikit_learn_version=actual_sklearn_version,
        train_rows=len(train),
        validation_rows=len(validation),
    )


def save_s0_development_artifacts(
    result: S0DevelopmentResult, output_dir: str | Path
) -> dict[str, object]:
    """Persist the already-fitted S0 pipeline and keyed validation evidence once."""
    import joblib
    from sklearn.metrics import f1_score

    if not isinstance(result, S0DevelopmentResult) or not result.validation_predictions:
        raise DataIntegrityError("S0 artifact input is not a completed development result")
    current_baseline_sha256 = _sha256(Path(__file__).resolve().read_bytes())
    current_preprocessing_sha256 = _sha256(Path(preprocessing.__file__).resolve().read_bytes())
    if (
        result.baseline_code_sha256 != current_baseline_sha256
        or result.preprocessing_code_sha256 != current_preprocessing_sha256
        or result.validation_rows != len(result.validation_predictions)
        or any(
            row.key[0] != result.dataset_revision
            or row.key[1] != "val"
            or type(row.key[2]) is not int
            or type(row.y_true) is not int
            or row.y_true not in (0, 1)
            or type(row.y_pred) is not int
            or row.y_pred not in (0, 1)
            for row in result.validation_predictions
        )
        or len({row.key for row in result.validation_predictions})
        != len(result.validation_predictions)
    ):
        raise DataIntegrityError("S0 artifact input no longer matches its validated fit and code")
    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)

    model_path = target / "pipeline.joblib"
    joblib.dump(result.estimator, model_path)
    prediction_payload = {
        "schema_version": 1,
        "configuration": "S0",
        "seed": 42,
        "split": "val",
        "predictions": [
            {"key": list(row.key), "y_true": row.y_true, "y_pred": row.y_pred}
            for row in result.validation_predictions
        ],
    }
    prediction_bytes = (
        json.dumps(prediction_payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    prediction_path = target / "validation_predictions.json"
    with prediction_path.open("xb") as stream:
        stream.write(prediction_bytes)
        stream.flush()
        os.fsync(stream.fileno())

    run_id = f"S0_seed42_{uuid.uuid4().hex[:12]}"
    metadata = result.audit_report()
    metadata.update(
        {
            "status": "FIT_COMPLETE",
            "run_id": run_id,
            "attempt_id": run_id,
            "seed": 42,
            "validation_macro_f1": float(
                f1_score(
                    [row.y_true for row in result.validation_predictions],
                    [row.y_pred for row in result.validation_predictions],
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            ),
            "pipeline_file": model_path.name,
            "pipeline_sha256": _sha256(model_path.read_bytes()),
            "validation_predictions_file": prediction_path.name,
            "validation_predictions_sha256": _sha256(prediction_bytes),
            "row_level_predictions_saved": True,
            "raw_review_text_saved": False,
            "test_split_available_to_runner": False,
        }
    )
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    with (target / "run_metadata.json").open("xb") as stream:
        stream.write(metadata_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    return metadata
