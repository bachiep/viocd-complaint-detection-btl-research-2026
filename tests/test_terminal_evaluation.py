"""Synthetic tests for selection-gated terminal inference and output privacy."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nlp_btl.terminal_evaluation import (
    TerminalEvaluationError,
    TerminalPrediction,
    evaluate_terminal_test,
)
from nlp_btl.data import DataIntegrityError
from nlp_btl import terminal_evaluation
from selection_fixtures import TEST_CSV, build_selection_fixture


class TerminalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol_path, self.selection_path, self.experiment_dir, _ = build_selection_fixture(self.root)
        private_root_patch = mock.patch(
            "nlp_btl.terminal_evaluation._private_results_root",
            return_value=self.root,
        )
        private_root_patch.start()
        self.addCleanup(private_root_patch.stop)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _evaluate_with_predictor(self, predictor, output=None, fetch_bytes=None):
        class SyntheticPredictor:
            backend = "SYNTHETIC_CALLBACK_TEST_ONLY"

            def __call__(self, run_row, inputs):
                return predictor(run_row, inputs)

        with mock.patch(
            "nlp_btl.terminal_evaluation._build_locked_predictor",
            return_value=SyntheticPredictor(),
        ):
            return evaluate_terminal_test(
                self.protocol_path,
                self.selection_path,
                output or self.experiment_dir / "terminal",
                fetch_bytes=fetch_bytes,
            )

    def _global_outcome(self) -> dict[str, object]:
        outcomes = list((self.root / ".terminal_evaluation_claims").glob("*.outcome.json"))
        self.assertEqual(len(outcomes), 1)
        return json.loads(outcomes[0].read_text(encoding="utf-8"))

    def test_evaluates_every_run_with_keyed_predictions_and_no_saved_text(self) -> None:
        calls: list[str] = []
        segmented_inputs: list[str] = []
        original_selection = self.selection_path.read_bytes()

        def identity_segmenter(text: str) -> str:
            segmented_inputs.append(text)
            return text

        def predict(run_row, inputs):
            self.assertTrue(all(not hasattr(row, "label") for row in inputs))
            self.assertTrue(all(not hasattr(row, "text") for row in inputs))
            calls.append(run_row["run_key"])
            return tuple(
                TerminalPrediction(
                    row.key,
                    0 if row.segmented_text == "satisfactory" else 1,
                )
                for row in inputs
            )

        segmenter_routes: list[str] = []

        def create_segmenter(segmenter_id, _preparation_dir, _protocol):
            segmenter_routes.append(segmenter_id)
            return identity_segmenter

        with mock.patch(
            "nlp_btl.terminal_evaluation._create_segmenter",
            side_effect=create_segmenter,
        ):
            result = self._evaluate_with_predictor(
                predict,
                fetch_bytes=lambda _url: TEST_CSV,
            )

        self.assertEqual(len(calls), 7)
        self.assertEqual(segmenter_routes, ["underthesea", "rdr"])
        self.assertEqual(len(segmented_inputs), 4)
        self.assertEqual(result["summary"]["test_rows"], 2)
        self.assertEqual(result["summary"]["status"], "CALLBACK_FIXTURE_COMPLETE")
        self.assertEqual(
            result["summary"]["predictor_backend"], "SYNTHETIC_CALLBACK_TEST_ONLY"
        )
        self.assertEqual(
            result["summary"]["primary_contrast_descriptive"]["mean_difference"], 0.0
        )
        predictions_path = self.experiment_dir / "terminal" / "test_predictions.json"
        saved = predictions_path.read_text(encoding="utf-8")
        self.assertNotIn("satisfactory", saved)
        self.assertNotIn("broken", saved)
        self.assertEqual(saved.count('"run_key"'), 14)
        self.assertTrue(predictions_path.with_suffix(".json.sha256").is_file())
        self.assertEqual(self.selection_path.read_bytes(), original_selection)
        outcome = self._global_outcome()
        self.assertEqual(outcome["status"], "COMPLETE")
        self.assertEqual(outcome["predictor_backend"], "SYNTHETIC_CALLBACK_TEST_ONLY")
        self.assertEqual(outcome["selection_sha256"], result["summary"]["selection_sha256"])

    def test_duplicate_test_key_is_rejected(self) -> None:
        def duplicate(_run, inputs):
            return (
                TerminalPrediction(inputs[0].key, 0),
                TerminalPrediction(inputs[0].key, 1),
            )

        with self.assertRaisesRegex(TerminalEvaluationError, "keys or labels"):
            self._evaluate_with_predictor(
                duplicate,
                fetch_bytes=lambda _url: TEST_CSV,
            )

    def test_invalid_prediction_join_does_not_write_terminal_results(self) -> None:
        output = self.experiment_dir / "terminal"

        with self.assertRaisesRegex(TerminalEvaluationError, "output count"):
            self._evaluate_with_predictor(
                lambda _run, _inputs: (),
                output=output,
                fetch_bytes=lambda _url: TEST_CSV,
            )

        self.assertFalse(output.exists())
        outcome = self._global_outcome()
        self.assertEqual(outcome["status"], "FAILED_AFTER_TEST_FETCH")
        self.assertNotIn("exception_message", outcome)

    def test_failed_test_attempt_cannot_be_retried(self) -> None:
        fetch = mock.Mock(return_value=TEST_CSV)
        with self.assertRaises(DataIntegrityError):
            self._evaluate_with_predictor(
                lambda _run, _inputs: (),
                fetch_bytes=fetch,
            )
        self.assertEqual(fetch.call_count, 1)

        second_root = self.root / "second-selection"
        _second_protocol, second_selection, second_experiment, _ = build_selection_fixture(
            second_root
        )
        second_selection.write_bytes(self.selection_path.read_bytes())
        second_selection.with_suffix(".json.sha256").write_bytes(
            self.selection_path.with_suffix(".json.sha256").read_bytes()
        )
        self.assertEqual(second_selection.read_bytes(), self.selection_path.read_bytes())

        retry_fetch = mock.Mock(return_value=TEST_CSV)
        with mock.patch(
            "nlp_btl.terminal_evaluation._build_locked_predictor",
            return_value=type(
                "SyntheticPredictor",
                (),
                {
                    "backend": "SYNTHETIC_CALLBACK_TEST_ONLY",
                    "__call__": lambda _self, _run, inputs: tuple(
                        TerminalPrediction(row.key, 0) for row in inputs
                    ),
                },
            )(),
        ):
            with self.assertRaisesRegex(TerminalEvaluationError, "data cohort already"):
                evaluate_terminal_test(
                    _second_protocol,
                    second_selection,
                    second_experiment / "terminal",
                    fetch_bytes=retry_fetch,
                )
        retry_fetch.assert_not_called()

    def test_failed_fetch_is_recorded_and_selection_is_consumed(self) -> None:
        fetch = mock.Mock(side_effect=OSError("private fetch detail"))
        with self.assertRaises(DataIntegrityError):
            self._evaluate_with_predictor(
                lambda _run, _inputs: (),
                fetch_bytes=fetch,
            )
        outcome = self._global_outcome()
        self.assertEqual(outcome["status"], "TEST_FETCH_OUTCOME_UNKNOWN")
        self.assertNotIn("private fetch detail", json.dumps(outcome))

    def test_partial_artifact_write_never_publishes_summary(self) -> None:
        original_write = __import__(
            "nlp_btl.terminal_evaluation", fromlist=["_write_create_only"]
        )._write_create_only

        def fail_predictions(path, payload):
            if path.name == "test_predictions.json":
                raise OSError("injected staging failure")
            return original_write(path, payload)

        with mock.patch(
            "nlp_btl.terminal_evaluation._create_segmenter",
            return_value=lambda text: text,
        ), mock.patch(
            "nlp_btl.terminal_evaluation._write_create_only",
            side_effect=fail_predictions,
        ):
            with self.assertRaisesRegex(OSError, "injected staging failure"):
                self._evaluate_with_predictor(
                    lambda _run, inputs: tuple(
                        TerminalPrediction(
                            row.key, 0 if row.segmented_text == "satisfactory" else 1
                        )
                        for row in inputs
                    ),
                    fetch_bytes=lambda _url: TEST_CSV,
                )

        self.assertFalse((self.experiment_dir / "terminal").exists())
        self.assertEqual(
            list(self.experiment_dir.glob(".terminal-stage-*")), []
        )
        outcome = self._global_outcome()
        self.assertEqual(outcome["status"], "FAILED_AFTER_TEST_FETCH")

    def test_output_outside_private_selection_folder_is_rejected_before_fetch(self) -> None:
        fetch = mock.Mock(return_value=TEST_CSV)
        with self.assertRaisesRegex(TerminalEvaluationError, "selection's private"):
            evaluate_terminal_test(
                self.protocol_path,
                self.selection_path,
                self.root / "other-output",
                fetch_bytes=fetch,
            )
        fetch.assert_not_called()

    def test_metric_runtime_mismatch_is_rejected_before_claim_or_fetch(self) -> None:
        fetch = mock.Mock(return_value=TEST_CSV)
        with mock.patch(
            "nlp_btl.terminal_evaluation._verify_metric_runtime",
            return_value="unexpected-version",
        ):
            with self.assertRaisesRegex(TerminalEvaluationError, "metric runtime"):
                evaluate_terminal_test(
                    self.protocol_path,
                    self.selection_path,
                    self.experiment_dir / "terminal",
                    fetch_bytes=fetch,
                )
        fetch.assert_not_called()
        self.assertFalse((self.root / ".terminal_evaluation_claims").exists())

    def test_default_predictor_preflights_cuda_before_test_fetch(self) -> None:
        fetch = mock.Mock(return_value=TEST_CSV)
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(TerminalEvaluationError, "requires CUDA"):
                evaluate_terminal_test(
                    self.protocol_path,
                    self.selection_path,
                    self.experiment_dir / "terminal",
                    fetch_bytes=fetch,
                )
        fetch.assert_not_called()
        self.assertFalse((self.root / ".terminal_evaluation_claims").exists())

    def test_public_evaluator_has_no_predictor_injection_argument(self) -> None:
        self.assertNotIn("predictor", inspect.signature(evaluate_terminal_test).parameters)

    def test_forged_backend_label_cannot_claim_locked_inference(self) -> None:
        class ForgedPredictor:
            backend = "LOCKED_ARTIFACTS"

            def __call__(self, _run, inputs):
                return tuple(
                    TerminalPrediction(
                        row.key, 0 if row.segmented_text == "satisfactory" else 1
                    )
                    for row in inputs
                )

        with mock.patch(
            "nlp_btl.terminal_evaluation._build_locked_predictor",
            return_value=ForgedPredictor(),
        ), mock.patch(
            "nlp_btl.terminal_evaluation._create_segmenter",
            return_value=lambda text: text,
        ):
            result = evaluate_terminal_test(
                self.protocol_path,
                self.selection_path,
                self.experiment_dir / "terminal",
                fetch_bytes=lambda _url: TEST_CSV,
            )

        self.assertEqual(result["summary"]["status"], "CALLBACK_FIXTURE_COMPLETE")
        self.assertEqual(
            result["summary"]["predictor_backend"], "SYNTHETIC_CALLBACK_TEST_ONLY"
        )
        self.assertEqual(
            self._global_outcome()["predictor_backend"], "SYNTHETIC_CALLBACK_TEST_ONLY"
        )

    def test_invalid_selection_prevents_predictor_and_test_fetch(self) -> None:
        changed = self.experiment_dir / "runs" / "P0_seed42" / "training_history.json"
        changed.write_text("{}\n", encoding="utf-8")
        fetch = mock.Mock(return_value=TEST_CSV)
        predictor_factory = mock.Mock()

        with mock.patch(
            "nlp_btl.terminal_evaluation._build_locked_predictor",
            predictor_factory,
        ), self.assertRaises(TerminalEvaluationError):
            evaluate_terminal_test(
                self.protocol_path,
                self.selection_path,
                self.experiment_dir / "terminal",
                fetch_bytes=fetch,
            )

        fetch.assert_not_called()
        predictor_factory.assert_not_called()
        self.assertFalse((self.root / ".terminal_evaluation_claims").exists())


if __name__ == "__main__":
    unittest.main()
