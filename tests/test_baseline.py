"""Synthetic tests for the locked S0 development baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from unittest import mock
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_btl.baseline import (
    BaselinePrediction,
    _build_s0_pipeline,
    run_s0_development,
    save_s0_development_artifacts,
    validate_validation_predictions,
)
from nlp_btl.data import DataIntegrityError, DatasetBundle, ReviewRecord


PROTOCOL_PATH = ROOT / "preparation" / "protocol.json"
PROTOCOL_BYTES = PROTOCOL_PATH.read_bytes()
PROTOCOL = json.loads(PROTOCOL_BYTES)


def review(split: str, row: int, text: str, label: int) -> ReviewRecord:
    return ReviewRecord(
        revision=PROTOCOL["dataset"]["revision"],
        split=split,
        source_row=row,
        text=text,
        label=label,
        domain="app",
    )


def dataset_bundle(splits: dict[str, tuple[ReviewRecord, ...]]) -> DatasetBundle:
    return DatasetBundle(
        protocol_id=PROTOCOL["protocol_id"],
        revision=PROTOCOL["dataset"]["revision"],
        manifest_sha256=PROTOCOL["dataset"]["manifest_sha256"],
        protocol_sha256=hashlib.sha256(PROTOCOL_BYTES).hexdigest(),
        loader_sha256="synthetic-loader-sha256",
        pandas_version="synthetic-pandas",
        splits=MappingProxyType(splits),
        audit=MappingProxyType({}),
    )


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = (
            review("train", 0, "Sản phẩm rất tốt chất lượng cao", 0),
            review("train", 1, "Sản phẩm tốt giao hàng nhanh", 0),
            review("train", 2, "Sản phẩm quá tệ chất lượng kém", 1),
            review("train", 3, "Sản phẩm rất tệ giao hàng chậm", 1),
        )
        self.validation = (
            review("val", 0, "Sản phẩm tốt validationonlytoken", 0),
            review("val", 1, "Sản phẩm tệ validationonlytoken", 1),
        )

    def test_s0_uses_protocol_parameters_and_train_only_vocabulary(self) -> None:
        bundle = dataset_bundle({"train": self.train, "val": self.validation})

        result = run_s0_development(bundle, PROTOCOL_PATH)

        self.assertEqual(result.train_rows, 4)
        self.assertEqual(result.validation_rows, 2)
        self.assertEqual(
            [prediction.key for prediction in result.validation_predictions],
            [record.key for record in self.validation],
        )
        validate_validation_predictions(self.validation, result.validation_predictions)
        vectorizer = result.estimator.named_steps["tfidf"]
        classifier = result.estimator.named_steps["classifier"]
        self.assertNotIn("validationonlytoken", vectorizer.vocabulary_)
        self.assertEqual(vectorizer.ngram_range, tuple(PROTOCOL["baseline"]["vectorizer"]["ngram_range"]))
        self.assertEqual(classifier.random_state, 42)
        self.assertEqual(result.scikit_learn_version, PROTOCOL["environment"]["explicit_package_targets"]["scikit-learn"])
        report = result.audit_report()
        self.assertTrue(report["fit_performed"])
        self.assertEqual(report["fit_data_split"], "train")
        self.assertFalse(report["test_split_available_to_runner"])
        self.assertFalse(report["raw_review_text_saved"])
        self.assertNotIn("validationonlytoken", json.dumps(report))

    def test_test_split_is_rejected_before_s0_fit(self) -> None:
        bundle = dataset_bundle(
            {
                "train": self.train,
                "val": self.validation,
                "test": (review("test", 0, "held out", 0),),
            }
        )

        with mock.patch("nlp_btl.baseline._build_s0_pipeline") as build_pipeline:
            with self.assertRaisesRegex(DataIntegrityError, "only train and validation"):
                run_s0_development(bundle, PROTOCOL_PATH)
            build_pipeline.assert_not_called()

    def test_changed_protocol_is_rejected_before_s0_fit(self) -> None:
        bundle = dataset_bundle({"train": self.train, "val": self.validation})
        with mock.patch(
            "nlp_btl.baseline.Path.read_bytes", return_value=PROTOCOL_BYTES + b"\n"
        ):
            with self.assertRaisesRegex(DataIntegrityError, "protocol differs"):
                run_s0_development(bundle, PROTOCOL_PATH)

    def test_validation_artifact_requires_exact_keys_and_gold_labels(self) -> None:
        expected = self.validation
        correct = (
            BaselinePrediction(expected[0].key, 0, 0),
            BaselinePrediction(expected[1].key, 1, 1),
        )
        validate_validation_predictions(expected, correct)
        with self.assertRaisesRegex(DataIntegrityError, "not unique"):
            validate_validation_predictions(expected, (correct[0], correct[0]))
        with self.assertRaisesRegex(DataIntegrityError, "source label"):
            validate_validation_predictions(
                expected,
                (BaselinePrediction(expected[0].key, 1, 0), correct[1]),
            )
        with self.assertRaisesRegex(DataIntegrityError, "outside the binary"):
            validate_validation_predictions(
                expected,
                (BaselinePrediction(expected[0].key, 0, 2), correct[1]),
            )

    def test_unsupported_vectorizer_parameter_fails_closed(self) -> None:
        configuration = json.loads(json.dumps(PROTOCOL["baseline"]))
        configuration["vectorizer"]["new_parameter"] = True

        with self.assertRaisesRegex(DataIntegrityError, "vectorizer parameters"):
            _build_s0_pipeline(configuration)

    def test_s0_artifacts_save_readable_model_and_keyed_validation_evidence(self) -> None:
        import joblib

        result = run_s0_development(
            dataset_bundle({"train": self.train, "val": self.validation}),
            PROTOCOL_PATH,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "S0_seed42"
            metadata = save_s0_development_artifacts(result, output)
            loaded = joblib.load(output / "pipeline.joblib")
            saved_predictions = json.loads(
                (output / "validation_predictions.json").read_text(encoding="utf-8")
            )

            self.assertEqual(metadata["status"], "FIT_COMPLETE")
            self.assertEqual(metadata["run_id"], metadata["attempt_id"])
            self.assertEqual(metadata["pipeline_sha256"], hashlib.sha256((output / "pipeline.joblib").read_bytes()).hexdigest())
            self.assertFalse(metadata["test_split_available_to_runner"])
            self.assertTrue(metadata["row_level_predictions_saved"])
            self.assertEqual(saved_predictions["predictions"], [
                {"key": list(row.key), "y_true": row.y_true, "y_pred": row.y_pred}
                for row in result.validation_predictions
            ])
            self.assertEqual(type(loaded).__name__, "Pipeline")


if __name__ == "__main__":
    unittest.main()
