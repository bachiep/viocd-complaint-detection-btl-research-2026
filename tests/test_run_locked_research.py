"""Pre-fit safety tests for the locked research orchestrator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from dataclasses import dataclass
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from preparation.scripts import run_locked_phobert_fit, run_locked_research


@dataclass(frozen=True)
class BundleFixture:
    protocol_id: str
    revision: str
    manifest_sha256: str
    protocol_sha256: str
    loader_sha256: str
    pandas_version: str
    splits: dict[str, tuple[object, ...]]
    audit: dict[str, dict[str, object]]


def make_bundle(audit_keys: tuple[str, ...] = ("train", "val", "test")) -> BundleFixture:
    return BundleFixture(
        protocol_id="protocol-v1",
        revision="dataset-revision-v1",
        manifest_sha256="manifest-sha256",
        protocol_sha256="protocol-sha256",
        loader_sha256="loader-sha256",
        pandas_version="3.0.0",
        splits={"train": ("train-record",), "val": ("validation-record",)},
        audit={
            key: {
                "source_sha256": f"{key}-sha256",
                "labels": {"0": 1, "1": 2},
                "domains": {"demo": 3},
            }
            for key in audit_keys
        },
    )


class RunLockedResearchPreflightTests(unittest.TestCase):
    def test_invalid_readiness_blocks_before_reserving_run_or_fitting_s0(self) -> None:
        run_order = [
            "P0_seed42",
            "P1_seed42",
            "P1_seed43",
            "P0_seed43",
            "P0_seed44",
            "P1_seed44",
        ]
        protocol_document = {"training": {"run_order": run_order}}
        expected_run_keys = ("S0_seed42", *run_order)
        bundle = SimpleNamespace(splits={"train": (), "val": ()}, revision="locked-revision")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation = root / "preparation"
            preparation.mkdir()
            protocol = preparation / "protocol.json"
            protocol.write_text(json.dumps(protocol_document), encoding="utf-8")
            manifest = preparation / "data_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            pre_fit_audit = root / "pre-fit-audit.json"
            readiness = root / "readiness.json"
            evidence = root / "evidence-manifest.json"
            run_root = root / "results" / "blocked-run"

            with (
                mock.patch.object(
                    run_locked_research,
                    "_validate_paths",
                    return_value=(root, preparation, protocol, pre_fit_audit, readiness, evidence, run_root),
                ),
                mock.patch.object(
                    run_locked_research,
                    "_validate_protocol_run_order",
                    return_value=protocol_document,
                ),
                mock.patch("nlp_btl.data.load_dataset", return_value=bundle),
                mock.patch("nlp_btl.selection.expected_run_keys", return_value=expected_run_keys),
                mock.patch(
                    "nlp_btl.phobert._verify_training_readiness",
                    side_effect=RuntimeError("readiness blocked"),
                ) as verify_readiness,
                mock.patch("nlp_btl.baseline.run_s0_development") as fit_s0,
            ):
                with self.assertRaisesRegex(RuntimeError, "readiness blocked"):
                    run_locked_research.run_locked_research(
                        str(protocol),
                        str(preparation),
                        str(pre_fit_audit),
                        str(readiness),
                        str(evidence),
                        "blocked-run",
                    )

            verify_readiness.assert_called_once_with(
                readiness,
                pre_fit_audit,
                preparation,
                protocol_document,
                bundle,
                hashlib.sha256(protocol.read_bytes()).hexdigest(),
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
                evidence,
            )
            fit_s0.assert_not_called()
            self.assertFalse(run_root.exists())

    def test_development_bundle_view_preserves_source_and_exposes_only_dev_audit(self) -> None:
        source_bundle = make_bundle()

        development_bundle = run_locked_research._development_bundle_view(source_bundle)

        self.assertIsNot(development_bundle, source_bundle)
        self.assertIs(development_bundle.splits, source_bundle.splits)
        self.assertEqual(set(source_bundle.audit), {"train", "val", "test"})
        self.assertEqual(set(development_bundle.audit), {"train", "val"})
        self.assertEqual(dict(development_bundle.audit), {
            "train": source_bundle.audit["train"],
            "val": source_bundle.audit["val"],
        })
        for field in (
            "protocol_id",
            "revision",
            "manifest_sha256",
            "protocol_sha256",
            "loader_sha256",
            "pandas_version",
        ):
            self.assertEqual(getattr(development_bundle, field), getattr(source_bundle, field))
        with self.assertRaises(TypeError):
            development_bundle.audit["test"] = source_bundle.audit["test"]
        with self.assertRaises(TypeError):
            development_bundle.audit["train"]["labels"]["0"] = 99

    def test_development_bundle_view_rejects_missing_or_extra_source_audits(self) -> None:
        for keys in (("train", "val"), ("train", "val", "test", "extra")):
            with self.subTest(keys=keys):
                with self.assertRaisesRegex(RuntimeError, "exactly train, validation, and test"):
                    run_locked_research._development_bundle_view(make_bundle(keys))

    def test_successful_gate_projects_full_audit_before_all_development_fits(self) -> None:
        run_order = [
            "P0_seed42",
            "P1_seed42",
            "P1_seed43",
            "P0_seed43",
            "P0_seed44",
            "P1_seed44",
        ]
        protocol_document = {"training": {"run_order": run_order}}
        expected_run_keys = ("S0_seed42", *run_order)
        bundle = make_bundle()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation = root / "preparation"
            preparation.mkdir()
            protocol = preparation / "protocol.json"
            protocol.write_text(json.dumps(protocol_document), encoding="utf-8")
            manifest = preparation / "data_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            pre_fit_audit = root / "pre-fit-audit.json"
            readiness = root / "readiness.json"
            evidence = root / "evidence-manifest.json"
            run_root = root / "results" / "successful-run"
            worker_dir = preparation / "scripts"
            worker_dir.mkdir()
            (worker_dir / "run_locked_phobert_fit.py").write_text("# mocked worker\n", encoding="utf-8")

            def save_fit_metadata(output_dir: Path) -> None:
                output_dir.mkdir(parents=True)
                (output_dir / "run_metadata.json").write_text(
                    json.dumps({"status": "FIT_COMPLETE"}), encoding="utf-8"
                )

            events: list[str] = []

            def verify_readiness(*_args: object) -> None:
                events.append("readiness")

            def fit_phobert(**kwargs: object) -> None:
                fit_bundle = kwargs["bundle"]
                self.assertEqual(set(fit_bundle.audit), {"train", "val"})
                events.append(f"{kwargs['config_id']}_seed{kwargs['seed']}")
                save_fit_metadata(kwargs["output_dir"])

            def launch_worker(command: list[str], cwd: Path) -> SimpleNamespace:
                self.assertEqual(cwd, root)
                configuration = command[command.index("--configuration") + 1]
                seed = command[command.index("--seed") + 1]
                output_dir = Path(command[command.index("--output-dir") + 1])
                events.append(f"{configuration}_seed{seed}")
                save_fit_metadata(output_dir)
                return SimpleNamespace(pid=1000 + len(events), wait=lambda: 0, terminate=lambda: None)

            def create_selection_artifact(**kwargs: object) -> None:
                events.append("selection")
                kwargs["selection_path"].write_text("{}", encoding="utf-8")

            with (
                mock.patch.object(
                    run_locked_research,
                    "_validate_paths",
                    return_value=(root, preparation, protocol, pre_fit_audit, readiness, evidence, run_root),
                ),
                mock.patch.object(
                    run_locked_research,
                    "_validate_protocol_run_order",
                    return_value=protocol_document,
                ),
                mock.patch("nlp_btl.data.load_dataset", return_value=bundle),
                mock.patch("nlp_btl.selection.expected_run_keys", return_value=expected_run_keys),
                mock.patch(
                    "nlp_btl.phobert._verify_training_readiness",
                    side_effect=verify_readiness,
                ) as verify_readiness_mock,
                mock.patch(
                    "nlp_btl.baseline.run_s0_development",
                    side_effect=lambda *_args: (events.append("S0_seed42") or object()),
                ) as fit_s0,
                mock.patch(
                    "nlp_btl.baseline.save_s0_development_artifacts",
                    side_effect=lambda _result, output_dir: save_fit_metadata(output_dir),
                ),
                mock.patch("nlp_btl.phobert.run_phobert_development", side_effect=fit_phobert) as fit_phobert_mock,
                mock.patch.object(
                    run_locked_research.subprocess,
                    "Popen",
                    side_effect=launch_worker,
                ) as worker_run_mock,
                mock.patch(
                    "nlp_btl.selection.create_selection_artifact",
                    side_effect=create_selection_artifact,
                ) as create_selection,
                mock.patch(
                    "nlp_btl.selection.verify_selection_artifact",
                    return_value=(
                        {
                            "status": "SELECTION_SAVED",
                            "protocol_id": "protocol-v1",
                            "protocol_sha256": "protocol-sha256",
                            "manifest_sha256": "manifest-sha256",
                            "selected_configuration": "P1",
                            "selected_demo_run": "P1_seed42",
                        },
                        "selection-sha256",
                    ),
                ),
            ):
                summary = run_locked_research.run_locked_research(
                    str(protocol),
                    str(preparation),
                    str(pre_fit_audit),
                    str(readiness),
                    str(evidence),
                    "successful-run",
                )

            verify_readiness_mock.assert_called_once()
            self.assertIs(verify_readiness_mock.call_args.args[4], bundle)
            s0_bundle = fit_s0.call_args.args[0]
            self.assertEqual(set(s0_bundle.audit), {"train", "val"})
            self.assertIs(s0_bundle.splits, bundle.splits)
            self.assertEqual(
                [call.kwargs["config_id"] + "_seed" + str(call.kwargs["seed"])
                 for call in fit_phobert_mock.call_args_list],
                [key for key in run_order if key.startswith("P0_")],
            )
            for call in fit_phobert_mock.call_args_list:
                self.assertIs(call.kwargs["bundle"], s0_bundle)
                self.assertEqual(set(call.kwargs["bundle"].audit), {"train", "val"})
            self.assertEqual(worker_run_mock.call_count, 3)
            self.assertEqual(events, ["readiness", "S0_seed42", *run_order, "selection"])
            self.assertEqual(create_selection.call_args.kwargs["validation_records"], bundle.splits["val"])
            self.assertEqual(summary["status"], "DEVELOPMENT_COMPLETE")
            self.assertTrue(run_root.is_dir())


class RunLockedPhoBertWorkerTests(unittest.TestCase):
    def test_worker_reloads_full_bundle_for_gate_then_fits_dev_only_view(self) -> None:
        run_order = [
            "P0_seed42",
            "P1_seed42",
            "P1_seed43",
            "P0_seed43",
            "P0_seed44",
            "P1_seed44",
        ]
        protocol_document = {"training": {"run_order": run_order}}
        bundle = make_bundle()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation = root / "preparation"
            preparation.mkdir()
            protocol = preparation / "protocol.json"
            protocol.write_text(json.dumps(protocol_document), encoding="utf-8")
            (preparation / "data_manifest.json").write_text("{}", encoding="utf-8")
            pre_fit_audit = root / "pre-fit-audit.json"
            readiness = root / "readiness.json"
            evidence = root / "evidence-manifest.json"
            for path in (pre_fit_audit, readiness, evidence):
                path.write_text("{}", encoding="utf-8")
            output = root / "results" / "worker-run" / "P1_seed43"
            output.parent.mkdir(parents=True)

            with (
                mock.patch.object(run_locked_phobert_fit, "_project_root", return_value=root),
                mock.patch("nlp_btl.data.load_dataset", return_value=bundle),
                mock.patch("nlp_btl.phobert._verify_training_readiness") as verify_readiness,
                mock.patch("nlp_btl.phobert.run_phobert_development") as fit_phobert,
            ):
                run_locked_phobert_fit.run_phobert_fit(
                    str(protocol),
                    str(preparation),
                    str(pre_fit_audit),
                    str(readiness),
                    str(evidence),
                    "P1",
                    43,
                    str(output),
                )

            self.assertIs(verify_readiness.call_args.args[4], bundle)
            fit_bundle = fit_phobert.call_args.kwargs["bundle"]
            self.assertIsNot(fit_bundle, bundle)
            self.assertIs(fit_bundle.splits, bundle.splits)
            self.assertEqual(set(bundle.audit), {"train", "val", "test"})
            self.assertEqual(set(fit_bundle.audit), {"train", "val"})
            self.assertEqual(fit_phobert.call_args.kwargs["config_id"], "P1")
            self.assertEqual(fit_phobert.call_args.kwargs["seed"], 43)


if __name__ == "__main__":
    unittest.main()
