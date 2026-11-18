"""Synthetic controls for independent metric and paired-bootstrap audit logic."""

from __future__ import annotations

import csv
import json
from importlib.metadata import version as installed_version
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nlp_btl import numerical_audit, terminal_evaluation
from nlp_btl.numerical_audit import (
    NumericalAuditError,
    _assert_metric_equal,
    _validate_prediction_rows,
    audit_terminal_artifacts,
    compute_binary_metrics,
    paired_bootstrap_differences,
)
from nlp_btl.terminal_evaluation import TerminalPrediction, evaluate_terminal_test
from nlp_btl.data import _ordered_prediction_binding_digest, _sha256
from selection_fixtures import TEST_CSV, build_selection_fixture


class NumericalAuditTests(unittest.TestCase):
    def test_independent_metrics_recompute_confusion_and_macro_scores(self) -> None:
        metrics = compute_binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_precision"], 0.5)
        self.assertEqual(metrics["macro_recall"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["confusion_matrix_labels_0_1"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["per_class"]["0"]["support"], 2)

    def test_metric_negative_control_rejects_modified_macro_f1(self) -> None:
        expected = compute_binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        altered = json.loads(json.dumps(expected))
        altered["macro_f1"] = 0.9
        with self.assertRaisesRegex(NumericalAuditError, "macro_f1"):
            _assert_metric_equal(expected, altered, "synthetic.run")

    def test_prediction_join_negative_controls_reject_duplicate_and_gold_mismatch(self) -> None:
        run_order = ["P0_seed42", "P1_seed42"]
        labels = [0, 0, 1, 1]
        rows = [
            {
                "run_key": run_key,
                "key": ["revision", "test", row_index],
                "domain": "app",
                "y_true": label,
                "y_pred": label,
            }
            for run_key in run_order
            for row_index, label in enumerate(labels)
        ]
        expected_binding = _ordered_prediction_binding_digest(
            (row["key"][1], row["key"][2], row["y_true"], row["domain"])
            for row in rows[:4]
        )
        _validate_prediction_rows(
            rows,
            run_order,
            "revision",
            4,
            {"0": 2, "1": 2},
            {"app": 4},
            expected_binding,
        )

        duplicate = json.loads(json.dumps(rows))
        duplicate[3]["key"] = duplicate[0]["key"]
        with self.assertRaisesRegex(NumericalAuditError, "Duplicate"):
            _validate_prediction_rows(
                duplicate,
                run_order,
                "revision",
                4,
                {"0": 2, "1": 2},
                {"app": 4},
                expected_binding,
            )

        changed_gold = json.loads(json.dumps(rows))
        changed_gold[4]["y_true"] = 1
        with self.assertRaisesRegex(NumericalAuditError, "gold labels"):
            _validate_prediction_rows(
                changed_gold,
                run_order,
                "revision",
                4,
                {"0": 2, "1": 2},
                {"app": 4},
                expected_binding,
            )

        permuted_gold = json.loads(json.dumps(rows))
        for run_index in range(len(run_order)):
            offset = run_index * len(labels)
            permuted_gold[offset]["y_true"], permuted_gold[offset + 2]["y_true"] = (
                permuted_gold[offset + 2]["y_true"],
                permuted_gold[offset]["y_true"],
            )
        with self.assertRaisesRegex(NumericalAuditError, "cohort binding"):
            _validate_prediction_rows(
                permuted_gold,
                run_order,
                "revision",
                4,
                {"0": 2, "1": 2},
                {"app": 4},
                expected_binding,
            )

        multi_domain_rows = json.loads(json.dumps(rows))
        for row in multi_domain_rows:
            row["domain"] = "app" if row["key"][2] < 2 else "mobile"
        expected_multi_domain_binding = _ordered_prediction_binding_digest(
            (row["key"][1], row["key"][2], row["y_true"], row["domain"])
            for row in multi_domain_rows[:4]
        )
        for run_index in range(len(run_order)):
            offset = run_index * len(labels)
            multi_domain_rows[offset]["domain"], multi_domain_rows[offset + 2]["domain"] = (
                multi_domain_rows[offset + 2]["domain"],
                multi_domain_rows[offset]["domain"],
            )
        with self.assertRaisesRegex(NumericalAuditError, "cohort binding"):
            _validate_prediction_rows(
                multi_domain_rows,
                run_order,
                "revision",
                4,
                {"0": 2, "1": 2},
                {"app": 2, "mobile": 2},
                expected_multi_domain_binding,
            )

    def test_paired_stratified_bootstrap_is_reproducible_and_uses_same_review_rows(self) -> None:
        gold = [0, 0, 1, 1]
        predictions: dict[str, list[int]] = {}
        for seed in (42, 43, 44):
            predictions[f"P0_seed{seed}"] = [0, 1, 1, 1]
            predictions[f"P1_seed{seed}"] = [0, 0, 1, 1]
        first = paired_bootstrap_differences(gold, predictions, 300, 20261116, batch_size=32)
        second = paired_bootstrap_differences(gold, predictions, 300, 20261116, batch_size=32)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 300)
        self.assertGreater(len(set(first)), 1)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in first))

    def test_audit_entrypoint_rejects_synthetic_terminal_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, selection, experiment, _ = build_selection_fixture(root)
            with mock.patch.object(terminal_evaluation, "_private_results_root", return_value=root), mock.patch.object(
                numerical_audit, "_private_results_root", return_value=root
            ), mock.patch.object(
                terminal_evaluation,
                "_build_locked_predictor",
                return_value=type(
                    "FixturePredictor",
                    (),
                    {
                        "backend": "LOCKED_ARTIFACTS",
                        "__call__": lambda _self, _run, inputs: tuple(
                            TerminalPrediction(row.key, 0) for row in inputs
                        ),
                    },
                )(),
            ), mock.patch.object(
                terminal_evaluation, "_create_segmenter", return_value=lambda text: text
            ):
                evaluate_terminal_test(
                    protocol,
                    selection,
                    experiment / "terminal",
                    fetch_bytes=lambda _url: TEST_CSV,
                )

            protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
            pinned_numpy = protocol_value["environment"]["explicit_package_targets"]["numpy"]

            def simulated_locked_version(package: str) -> str:
                return pinned_numpy if package == "numpy" else installed_version(package)

            with mock.patch.object(numerical_audit, "_private_results_root", return_value=root), mock.patch(
                "importlib.metadata.version", side_effect=simulated_locked_version
            ):
                with self.assertRaisesRegex(NumericalAuditError, "production run"):
                    audit_terminal_artifacts(
                        protocol,
                        selection,
                        experiment / "terminal",
                    )
            self.assertFalse((experiment / "terminal" / "analysis").exists())

    def test_audit_recomputes_hash_consistent_synthetic_terminal_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, selection, experiment, _ = build_selection_fixture(root)
            terminal = experiment / "terminal"

            class FixturePredictor:
                def __call__(self, _run, inputs):
                    return tuple(
                        TerminalPrediction(
                            row.key, 0 if row.segmented_text == "satisfactory" else 1
                        )
                        for row in inputs
                    )

            def write_bound_json(path: Path, value: dict[str, object]) -> str:
                payload = (
                    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                digest = _sha256(payload)
                path.write_bytes(payload)
                path.with_suffix(path.suffix + ".sha256").write_text(
                    digest + "\n", encoding="ascii"
                )
                return digest

            with mock.patch.object(terminal_evaluation, "_private_results_root", return_value=root), mock.patch.object(
                numerical_audit, "_private_results_root", return_value=root
            ), mock.patch.object(
                terminal_evaluation, "_build_locked_predictor", return_value=FixturePredictor()
            ), mock.patch.object(
                terminal_evaluation, "_create_segmenter", return_value=lambda text: text
            ):
                evaluate_terminal_test(
                    protocol,
                    selection,
                    terminal,
                    fetch_bytes=lambda _url: TEST_CSV,
                )

            summary_path = terminal / "terminal_evaluation.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "TERMINAL_EVALUATION_COMPLETE"
            summary["predictor_backend"] = "LOCKED_ARTIFACTS"
            summary_hash = write_bound_json(summary_path, summary)
            predictions_path = terminal / "test_predictions.json"
            predictions_hash = _sha256(predictions_path.read_bytes())
            claim_path = next((root / ".terminal_evaluation_claims").glob("*.claim.json"))
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim["predictor_backend"] = "LOCKED_ARTIFACTS"
            claim_hash = write_bound_json(claim_path, claim)
            outcome_path = claim_path.with_name(claim_path.name.replace(".claim.json", ".outcome.json"))
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcome.update(
                {
                    "predictor_backend": "LOCKED_ARTIFACTS",
                    "claim_sha256": claim_hash,
                    "summary_sha256": summary_hash,
                    "predictions_sha256": predictions_hash,
                }
            )
            write_bound_json(outcome_path, outcome)

            protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
            pinned_numpy = protocol_value["environment"]["explicit_package_targets"]["numpy"]

            def simulated_locked_version(package: str) -> str:
                return pinned_numpy if package == "numpy" else installed_version(package)

            with mock.patch.object(numerical_audit, "_private_results_root", return_value=root), mock.patch(
                "importlib.metadata.version", side_effect=simulated_locked_version
            ):
                result = audit_terminal_artifacts(protocol, selection, terminal)

            self.assertEqual(result["audit"]["status"], "NUMERICAL_AUDIT_VERIFIED")
            self.assertEqual(result["audit"]["test_rows"], 2)
            self.assertEqual(result["audit"]["primary_contrast"]["bootstrap_resamples"], 10000)
            self.assertEqual(result["audit"]["family_summary"]["P0"]["n"], 2)
            self.assertEqual(result["audit"]["family_summary"]["P0"]["sample_sd"]["macro_f1"], 0.0)
            self.assertEqual(
                result["audit"]["primary_contrast"]["bootstrap_ci_percentile_95"],
                [0.0, 0.0],
            )
            with (terminal / "analysis" / "comparison.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                comparison = list(csv.DictReader(stream))
            self.assertEqual(sum(row["row_type"] == "run" for row in comparison), 7)
            self.assertEqual(sum(row["row_type"] == "family_summary" for row in comparison), 3)
            self.assertTrue(
                (terminal / "analysis" / "paired_bootstrap_differences.csv.sha256").is_file()
            )

    def test_audit_rejects_rehashed_prediction_keys_outside_locked_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, selection, experiment, _ = build_selection_fixture(root)
            terminal = experiment / "terminal"

            class FixturePredictor:
                def __call__(self, _run, inputs):
                    return tuple(TerminalPrediction(row.key, 0) for row in inputs)

            def write_bound_json(path: Path, value: dict[str, object]) -> str:
                payload = (
                    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                digest = _sha256(payload)
                path.write_bytes(payload)
                path.with_suffix(path.suffix + ".sha256").write_text(
                    digest + "\n", encoding="ascii"
                )
                return digest

            with mock.patch.object(terminal_evaluation, "_private_results_root", return_value=root), mock.patch.object(
                numerical_audit, "_private_results_root", return_value=root
            ), mock.patch.object(
                terminal_evaluation, "_build_locked_predictor", return_value=FixturePredictor()
            ), mock.patch.object(
                terminal_evaluation, "_create_segmenter", return_value=lambda text: text
            ):
                evaluate_terminal_test(
                    protocol,
                    selection,
                    terminal,
                    fetch_bytes=lambda _url: TEST_CSV,
                )

            summary_path = terminal / "terminal_evaluation.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "TERMINAL_EVALUATION_COMPLETE"
            summary["predictor_backend"] = "LOCKED_ARTIFACTS"
            summary_hash = write_bound_json(summary_path, summary)

            claim_path = next((root / ".terminal_evaluation_claims").glob("*.claim.json"))
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim["predictor_backend"] = "LOCKED_ARTIFACTS"
            claim_hash = write_bound_json(claim_path, claim)
            outcome_path = claim_path.with_name(claim_path.name.replace(".claim.json", ".outcome.json"))
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            prediction_path = terminal / "test_predictions.json"
            prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            for row in prediction_payload["predictions"]:
                row["key"][2] += 1000
            predictions_hash = write_bound_json(prediction_path, prediction_payload)
            outcome.update(
                {
                    "predictor_backend": "LOCKED_ARTIFACTS",
                    "claim_sha256": claim_hash,
                    "summary_sha256": summary_hash,
                    "predictions_sha256": predictions_hash,
                }
            )
            write_bound_json(outcome_path, outcome)

            protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
            pinned_numpy = protocol_value["environment"]["explicit_package_targets"]["numpy"]

            def simulated_locked_version(package: str) -> str:
                return pinned_numpy if package == "numpy" else installed_version(package)

            with mock.patch.object(numerical_audit, "_private_results_root", return_value=root), mock.patch(
                "importlib.metadata.version", side_effect=simulated_locked_version
            ):
                with self.assertRaisesRegex(NumericalAuditError, "cohort binding"):
                    audit_terminal_artifacts(protocol, selection, terminal)


if __name__ == "__main__":
    unittest.main()
