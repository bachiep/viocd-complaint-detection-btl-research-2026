"""Synthetic tests for validation-only selection and the terminal boundary."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nlp_btl.data import DataIntegrityError, load_terminal_test
from nlp_btl.selection import SelectionError, create_selection_artifact, verify_selection_artifact
from selection_fixtures import TEST_CSV, VALIDATION, build_selection_fixture


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (
            self.protocol_path,
            self.selection_path,
            self.experiment_dir,
            self.validation,
        ) = build_selection_fixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_recomputes_exact_seven_run_matrix_and_earliest_epoch_tie(self) -> None:
        selection, digest = verify_selection_artifact(
            self.selection_path, self.protocol_path
        )

        self.assertEqual(len(selection["runs"]), 7)
        self.assertEqual(selection["selected_configuration"], "P1")
        self.assertEqual(selection["selected_demo_run"], "P1_seed42")
        by_key = {row["run_key"]: row for row in selection["runs"]}
        self.assertEqual(by_key["P1_seed42"]["best_epoch"], 2)
        self.assertEqual(by_key["P0_seed42"]["best_epoch"], 1)
        self.assertEqual(len(digest), 64)

    def test_family_tie_order_is_s0_then_p0_then_p1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, selection_path, _, _ = build_selection_fixture(root, "p0_p1_tie")
            selection, _ = verify_selection_artifact(selection_path, protocol)
            self.assertEqual(selection["selected_configuration"], "P0")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, selection_path, _, _ = build_selection_fixture(root, "s0_tie")
            selection, _ = verify_selection_artifact(selection_path, protocol)
            self.assertEqual(selection["selected_configuration"], "S0")

    def test_run_artifact_tampering_fails_before_terminal_fetch(self) -> None:
        changed = self.experiment_dir / "runs" / "P1_seed42" / "epoch_02_validation_predictions.json"
        changed.write_text("{}\n", encoding="utf-8")
        fetch = mock.Mock(return_value=TEST_CSV)

        with self.assertRaises(SelectionError):
            load_terminal_test(self.protocol_path, self.selection_path, fetch_bytes=fetch)

        fetch.assert_not_called()

    def test_development_artifact_containing_test_metric_fails_before_fetch(self) -> None:
        history_path = self.experiment_dir / "runs" / "P1_seed42" / "training_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history["test_macro_f1"] = 0.99
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fetch = mock.Mock(return_value=TEST_CSV)

        with self.assertRaisesRegex(SelectionError, "history contains terminal-test output"):
            load_terminal_test(self.protocol_path, self.selection_path, fetch_bytes=fetch)

        fetch.assert_not_called()

    def test_selection_file_tampering_is_rejected(self) -> None:
        raw = self.selection_path.read_bytes()
        self.selection_path.write_bytes(raw + b" ")

        with self.assertRaisesRegex(SelectionError, "digest"):
            verify_selection_artifact(self.selection_path, self.protocol_path)

    def test_rehashed_selection_with_stale_source_binding_is_rejected(self) -> None:
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        selection["source_sha256"]["selection.py"] = "0" * 64
        raw = (
            json.dumps(selection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.selection_path.write_bytes(raw)
        self.selection_path.with_suffix(".json.sha256").write_text(
            hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii"
        )

        with self.assertRaisesRegex(SelectionError, "stale or bound"):
            verify_selection_artifact(self.selection_path, self.protocol_path)

    def test_selection_is_create_only(self) -> None:
        run_dirs = {
            key: self.experiment_dir / "runs" / key
            for key in ("S0_seed42", "P0_seed42", "P1_seed42", "P1_seed43", "P0_seed43", "P0_seed44", "P1_seed44")
        }

        with self.assertRaisesRegex(SelectionError, "create-only"):
            create_selection_artifact(
                self.protocol_path,
                self.validation,
                run_dirs,
                self.selection_path,
            )

    def test_selection_rejects_partial_validation_population(self) -> None:
        run_dirs = {
            key: self.experiment_dir / "runs" / key
            for key in ("S0_seed42", "P0_seed42", "P1_seed42", "P1_seed43", "P0_seed43", "P0_seed44", "P1_seed44")
        }
        new_selection = self.experiment_dir / "partial_selection.json"

        with self.assertRaisesRegex(SelectionError, "locked cohort size"):
            create_selection_artifact(
                self.protocol_path,
                self.validation[:1],
                run_dirs,
                new_selection,
            )

        self.assertFalse(new_selection.exists())

    def test_terminal_test_loader_verifies_selection_before_fetching_test(self) -> None:
        events: list[str] = []
        from nlp_btl import selection as selection_module

        verify = selection_module.verify_selection_artifact

        def verify_spy(*args, **kwargs):
            result = verify(*args, **kwargs)
            events.append("selection_verified")
            return result

        def fetch(url: str) -> bytes:
            events.append("test_fetched")
            self.assertTrue(url.endswith("/test.csv"))
            return TEST_CSV

        with mock.patch.object(
            selection_module, "verify_selection_artifact", side_effect=verify_spy
        ):
            bundle = load_terminal_test(
                self.protocol_path, self.selection_path, fetch_bytes=fetch
            )

        self.assertEqual(events, ["selection_verified", "test_fetched"])
        self.assertEqual([row.label for row in bundle.records], [0, 1])
        self.assertEqual(bundle.audit_report()["test_rows"], 2)
        self.assertNotIn("key", bundle.audit_report())
        report = bundle.audit_report()
        self.assertFalse(report["raw_review_text_saved"])
        report_json = json.dumps(report).lower()
        self.assertNotIn("satisfactory", report_json)
        self.assertNotIn("broken", report_json)
        self.assertNotIn("source_row", report_json)


if __name__ == "__main__":
    unittest.main()
