from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import inspect
import json
import hashlib
import platform
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PUBLIC_SOURCE_SNAPSHOT = (ROOT / "PUBLIC_SOURCE_SNAPSHOT").is_file()
PRIVATE_PILOT_ASSETS_AVAILABLE = all(
    (ROOT / relative).is_file()
    for relative in (
        "preparation/preparation_lock.json",
        "preparation/tokenizer_input_audit.json",
        "preparation/assets/python_rdr/py_vncorenlp-0.1.4.tar.gz",
    )
)

from nlp_btl.data import DataIntegrityError, ReviewRecord
from nlp_btl.phobert import PhoBertTrainingError, _save_trainable_state
from nlp_btl.pilot import (
    _atomic_json,
    _perform_single_optimizer_step,
    _publish_pilot_success,
    _restore_pilot_checkpoint,
    _require_private_pilot_output,
    _safe_error_diagnostics,
    _select_pilot_records,
    run_cuda_pilot,
    verify_pilot_commit,
)


class _TinyClassifier:
    def __new__(cls):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)

            def forward(self, input_ids, attention_mask, labels):
                import torch.nn.functional as functional

                logits = self.linear(input_ids)
                return SimpleNamespace(logits=logits, loss=functional.cross_entropy(logits, labels))

        return Model()


class CudaPilotTests(unittest.TestCase):
    def test_batch_selection_is_source_ordered_and_train_only(self) -> None:
        records = tuple(
            ReviewRecord("rev", "train", row, f"synthetic {row}", row % 2, "app")
            for row in range(4)
        )
        selected = _select_pilot_records(records, 3)
        self.assertEqual([record.key[2] for record in selected], [0, 1, 2])
        with self.assertRaisesRegex(PhoBertTrainingError, "train split"):
            _select_pilot_records((records[0], ReviewRecord("rev", "val", 1, "x", 1, "app")), 2)

    def test_one_cpu_helper_step_checks_finite_gradients_and_parameter_update(self) -> None:
        import torch

        model = _TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = {
            "input_ids": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "attention_mask": torch.ones((2, 3)),
            "labels": torch.tensor([0, 1]),
        }
        outcome = _perform_single_optimizer_step(
            torch, model, optimizer, batch, torch.device("cpu"), max_grad_norm=1.0
        )
        self.assertTrue(outcome["minibatch_forward_backward_finite"])
        self.assertTrue(outcome["optimizer_step_succeeded"])
        self.assertGreater(outcome["changed_trainable_tensor_count"], 0)

    def test_pilot_checkpoint_roundtrip_restores_trainable_state(self) -> None:
        import torch

        model = _TinyClassifier()
        expected_binding = {"artifact_kind": "m1_08_cuda_pilot_trainable_state", "seed": 42}
        with TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoint"
            receipt = _save_trainable_state(model, checkpoint_dir, expected_binding)
            restored = _TinyClassifier()
            restored_receipt = _restore_pilot_checkpoint(
                torch, restored, checkpoint_dir, expected_binding
            )
            self.assertEqual(receipt["weights_sha256"], restored_receipt["weights_sha256"])
            for original, candidate in zip(model.parameters(), restored.parameters(), strict=True):
                self.assertTrue(torch.equal(original, candidate))

    def test_pilot_checkpoint_tamper_is_rejected(self) -> None:
        import torch

        model = _TinyClassifier()
        expected_binding = {"artifact_kind": "m1_08_cuda_pilot_trainable_state", "seed": 42}
        with TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoint"
            _save_trainable_state(model, checkpoint_dir, expected_binding)
            path = checkpoint_dir / "trainable_state.pt"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(PhoBertTrainingError, "weights hash mismatch"):
                _restore_pilot_checkpoint(torch, _TinyClassifier(), checkpoint_dir, expected_binding)

    @unittest.skipUnless(
        not PUBLIC_SOURCE_SNAPSHOT or PRIVATE_PILOT_ASSETS_AVAILABLE,
        "requires private preparation lock and tokenizer/RDR audit assets omitted from the public snapshot",
    )
    def test_cuda_unavailable_blocks_before_audit_or_dataset_load(self) -> None:
        private_root = ROOT / "results" / "m1-08"
        private_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=private_root) as temporary:
            output = Path(temporary) / "pilot"
            with (
                mock.patch("nlp_btl.pilot._runtime_preflight", return_value={"cuda_available": False}),
                mock.patch("nlp_btl.pilot.preprocessing._audit_full_preprocessing_with_segmenter", side_effect=AssertionError("audit must not run")),
                mock.patch("nlp_btl.pilot.load_dataset", side_effect=AssertionError("dataset must not load")),
            ):
                result = run_cuda_pilot(
                    ROOT / "preparation" / "protocol.json",
                    ROOT / "preparation",
                    output,
                )
            self.assertEqual(result.status, "BLOCKED")
            metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["failed_stage"], "cuda_preflight")
            self.assertFalse(metadata["test_split_available_to_pilot"])
            self.assertFalse((output / "pre_fit_audit.json").exists())


    @unittest.skipUnless(
        not PUBLIC_SOURCE_SNAPSHOT or PRIVATE_PILOT_ASSETS_AVAILABLE,
        "requires private preparation lock and tokenizer/RDR audit assets omitted from the public snapshot",
    )
    def test_pre_fit_os_failure_records_only_error_location(self) -> None:
        private_root = ROOT / "results" / "m1-08"
        private_root.mkdir(parents=True, exist_ok=True)
        runtime = {
            "cuda_available": True,
            "cuda_runtime": "13.0",
            "device": "cuda:0",
            "gpu_name": "Tesla T4",
            "device_count": 2,
            "gpu_total_memory_bytes": 15_636_037_632,
        }
        with TemporaryDirectory(dir=private_root) as temporary:
            output = Path(temporary) / "pilot"
            missing_path = "/kaggle/input/viocd/assets/vncorenlp/VnCoreNLP.jar"
            with (
                mock.patch("nlp_btl.pilot._runtime_preflight", return_value=runtime),
                mock.patch(
                    "nlp_btl.pilot.preprocessing._audit_full_preprocessing_with_segmenter",
                    side_effect=FileNotFoundError(2, "missing file", missing_path),
                ),
            ):
                result = run_cuda_pilot(
                    ROOT / "preparation" / "protocol.json",
                    ROOT / "preparation",
                    output,
                )
            metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "FAILED")
            self.assertEqual(metadata["failed_stage"], "pre_fit_audit")
            self.assertEqual(metadata["error_type"], "FileNotFoundError")
            self.assertEqual(
                metadata["error_diagnostics"],
                {"errno": 2, "filename": missing_path},
            )
            self.assertNotIn("error_message", metadata)
            self.assertFalse((output / "pre_fit_audit.json").exists())


    @unittest.skipUnless(
        not PUBLIC_SOURCE_SNAPSHOT or PRIVATE_PILOT_ASSETS_AVAILABLE,
        "requires private preparation lock and tokenizer/RDR audit assets omitted from the public snapshot",
    )
    def test_pilot_reuses_the_rdr_instance_returned_by_pre_fit_audit(self) -> None:
        import torch

        protocol_path = ROOT / "preparation" / "protocol.json"
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
        protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
        manifest_sha = protocol["dataset"]["manifest_sha256"]
        tokenizer_audit = json.loads(
            (ROOT / "preparation" / "tokenizer_input_audit.json").read_text(encoding="utf-8")
        )
        verified_packages = dict(protocol["environment"]["explicit_package_targets"])
        verified_packages.update(tokenizer_audit["packages"])
        runtime = {
            "cuda_available": True,
            "cuda_runtime": "13.0",
            "device": "cuda:0",
            "device_count": 2,
            "gpu_name": "Tesla T4",
            "gpu_total_memory_bytes": 15_636_037_632,
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        audit = {
            "status": "PASS",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "data_manifest_sha256": manifest_sha,
            "dataset_revision": protocol["dataset"]["revision"],
            "cross_split_collision_found": False,
            "test_used_only_for_pre_fit_integrity_audit": True,
            "raw_review_text_saved": False,
            "row_level_ids_saved": False,
            "token_id_sequences_saved": False,
            "model_weights_loaded": False,
            "training_performed": False,
            "runtime_packages": tokenizer_audit["packages"],
            "runtime": {
                "python_version": runtime["python"],
                "java_version": "21.0.12.1",
            },
        }
        segmenter = SimpleNamespace(import_origin_verified=True)
        records = tuple(
            ReviewRecord(
                protocol["dataset"]["revision"], "train", row,
                f"synthetic train review {row}", row % 2, "app",
            )
            for row in range(int(protocol["training"]["micro_batch"]))
        )
        bundle = SimpleNamespace(
            revision=protocol["dataset"]["revision"],
            manifest_sha256=manifest_sha,
            splits={"train": records},
        )
        encoded = {
            "input_ids": torch.ones((len(records), 8), dtype=torch.long),
            "attention_mask": torch.ones((len(records), 8), dtype=torch.long),
        }

        def fake_preprocess(selected, active_segmenter):
            self.assertIs(active_segmenter, segmenter)
            return tuple(
                SimpleNamespace(key=record.key, label=record.label, domain=record.domain)
                for record in selected
            )

        def fake_audit_worker(_protocol_path, output_path):
            Path(output_path).write_text(json.dumps(audit), encoding="utf-8")
            return audit, segmenter

        model_load_error = PhoBertTrainingError("Pinned PhoBERT load has unexpected weights outside the base LM head")
        model_load_error.safe_diagnostics = {
            "missing_key_count": 4,
            "unexpected_key_count": 2,
            "unexpected_key_namespace_counts": {
                "lm_head": 0,
                "roberta": 1,
                "classifier": 0,
                "other": 1,
                "non_string": 0,
            },
            # Synthetic architecture key for the persistence contract; not a v9 finding.
            "unexpected_roberta_keys": ["roberta.embeddings.position_ids"],
            "unexpected_key_names": ["private.raw.key"],
        }
        private_root = ROOT / "results" / "m1-08"
        private_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=private_root) as temporary:
            output = Path(temporary) / "pilot"
            with (
                mock.patch("nlp_btl.pilot._runtime_preflight", return_value=runtime),
                mock.patch(
                    "nlp_btl.pilot.preprocessing._audit_full_preprocessing_with_segmenter",
                    side_effect=fake_audit_worker,
                ) as audit_worker,
                mock.patch("nlp_btl.pilot.preprocessing.RDRSegmenter", side_effect=AssertionError("RDR must not be initialized twice")) as rdr_constructor,
                mock.patch("nlp_btl.pilot.load_dataset", return_value=bundle),
                mock.patch("nlp_btl.pilot._read_development_protocol", return_value=(protocol, protocol_sha, manifest_sha)),
                mock.patch("nlp_btl.pilot._read_tokenizer_audit", return_value=tokenizer_audit),
                mock.patch("nlp_btl.pilot._verify_runtime_packages", return_value=verified_packages),
                mock.patch("nlp_btl.pilot.preprocessing._java_runtime_version", return_value="21.0.12.1"),
                mock.patch("nlp_btl.pilot.preprocess_records", side_effect=fake_preprocess) as preprocess,
                mock.patch("nlp_btl.pilot.preprocessing._load_pinned_tokenizer", return_value=object()),
                mock.patch("nlp_btl.pilot.tokenize_batch", return_value=encoded),
                mock.patch("nlp_btl.pilot._build_pinned_model", side_effect=model_load_error),
            ):
                result = run_cuda_pilot(protocol_path, ROOT / "preparation", output)
                serialized_metadata = result.run_metadata_path.read_text(encoding="utf-8")
                metadata = json.loads(serialized_metadata)

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(metadata["failed_stage"], "pinned_model_loading", metadata)
        self.assertEqual(metadata["failed_substage"], "model_build")
        summary = metadata["error_diagnostics"]["model_loading_summary"]
        self.assertEqual(summary["unexpected_roberta_keys"], ["roberta.embeddings.position_ids"])
        self.assertNotIn("private.raw.key", serialized_metadata)
        self.assertNotIn('"unexpected_key_names":', serialized_metadata)
        self.assertNotIn("Pinned PhoBERT load has unexpected weights", serialized_metadata)
        audit_worker.assert_called_once()
        rdr_constructor.assert_not_called()
        preprocess.assert_called_once()
        self.assertIs(preprocess.call_args.args[1], segmenter)

    def test_safe_error_diagnostics_omits_non_os_exception_details(self) -> None:
        self.assertEqual(_safe_error_diagnostics(ValueError("review text must not be saved")), {})
        self.assertEqual(
            _safe_error_diagnostics(DataIntegrityError("RDR segmentation failed")),
            {"failure_code": "RDR_SEGMENTATION_FAILED"},
        )
        self.assertEqual(
            _safe_error_diagnostics(DataIntegrityError("review text must not be saved")),
            {},
        )
        model_error = PhoBertTrainingError("Cannot load the pinned PhoBERT base revision")
        model_error.__cause__ = FileNotFoundError(2, "private cause text", "/tmp/model-cache/weights")
        self.assertEqual(
            _safe_error_diagnostics(model_error),
            {
                "failure_code": "PHOBERT_BASE_LOAD_FAILED",
                "cause_type": "FileNotFoundError",
                "errno": 2,
                "filename": "/tmp/model-cache/weights",
            },
        )
        error = FileNotFoundError(2, "missing", "/tmp/tokenizer/vocab.txt")
        self.assertEqual(
            _safe_error_diagnostics(error),
            {"errno": 2, "filename": "/tmp/tokenizer/vocab.txt"},
        )
        malformed_os_error = OSError(5, "invalid path value", object())
        self.assertEqual(_safe_error_diagnostics(malformed_os_error), {"errno": 5})

        model_key_error = PhoBertTrainingError("Pinned PhoBERT load has unexpected weights outside the base LM head")
        model_key_error.safe_diagnostics = {
            "missing_key_count": 4,
            "unexpected_key_count": 2,
            "unexpected_key_namespace_counts": {
                "lm_head": 0,
                "roberta": 1,
                "classifier": 0,
                "other": 1,
                "non_string": 0,
            },
            # Synthetic architecture key for sanitizer coverage; not a v9 observation.
            "unexpected_roberta_keys": ["roberta.embeddings.position_ids"],
            "unexpected_key_names": ["private.weight.name"],
        }
        self.assertEqual(
            _safe_error_diagnostics(model_key_error),
            {
                "failure_code": "PHOBERT_UNEXPECTED_WEIGHTS",
                "model_loading_summary": {
                    "missing_key_count": 4,
                    "unexpected_key_count": 2,
                    "unexpected_key_namespace_counts": {
                        "lm_head": 0,
                        "roberta": 1,
                        "classifier": 0,
                        "other": 1,
                        "non_string": 0,
                    },
                    "unexpected_roberta_keys": ["roberta.embeddings.position_ids"],
                },
            },
        )
        malformed_model_diagnostics = (
            {
                "missing_key_count": 4,
                "unexpected_key_count": 2,
                "unexpected_key_namespace_counts": {
                    "lm_head": 0,
                    "roberta": 1,
                    "classifier": 0,
                    "other": 0,
                    "non_string": 0,
                },
            },
            {
                "missing_key_count": True,
                "unexpected_key_count": 0,
                "unexpected_key_namespace_counts": {
                    "lm_head": 0,
                    "roberta": 0,
                    "classifier": 0,
                    "other": 0,
                    "non_string": 0,
                },
            },
        )
        for safe_diagnostics in malformed_model_diagnostics:
            with self.subTest(safe_diagnostics=safe_diagnostics):
                malformed = PhoBertTrainingError("Pinned PhoBERT load has unexpected weights outside the base LM head")
                malformed.safe_diagnostics = safe_diagnostics
                self.assertEqual(
                    _safe_error_diagnostics(malformed),
                    {"failure_code": "PHOBERT_UNEXPECTED_WEIGHTS"},
                )

        invalid_key_lists = (
            ["private.raw.key"],
            ["roberta.invalid/key"],
            ["roberta." + "a" * 193],
            ["roberta.a", "roberta.a"],
            ["roberta.b", "roberta.a"],
            [f"roberta.layer.{index}.weight" for index in range(9)],
        )
        for names in invalid_key_lists:
            with self.subTest(names=names):
                invalid_names = PhoBertTrainingError("Pinned PhoBERT load has unexpected weights outside the base LM head")
                invalid_names.safe_diagnostics = {
                    "missing_key_count": 4,
                    "unexpected_key_count": len(names),
                    "unexpected_key_namespace_counts": {
                        "lm_head": 0,
                        "roberta": len(names),
                        "classifier": 0,
                        "other": 0,
                        "non_string": 0,
                    },
                    "unexpected_roberta_keys": names,
                }
                sanitized = _safe_error_diagnostics(invalid_names)["model_loading_summary"]
                self.assertNotIn("unexpected_roberta_keys", sanitized)
                for name in names:
                    self.assertNotIn(name, repr(sanitized))

        count_mismatch = PhoBertTrainingError("Pinned PhoBERT load has unexpected weights outside the base LM head")
        count_mismatch.safe_diagnostics = {
            "missing_key_count": 4,
            "unexpected_key_count": 2,
            "unexpected_key_namespace_counts": {
                "lm_head": 0,
                "roberta": 2,
                "classifier": 0,
                "other": 0,
                "non_string": 0,
            },
            "unexpected_roberta_keys": ["roberta.embeddings.position_ids"],
        }
        count_summary = _safe_error_diagnostics(count_mismatch)["model_loading_summary"]
        self.assertNotIn("unexpected_roberta_keys", count_summary)


    def test_pilot_output_must_stay_in_private_results_root(self) -> None:
        with TemporaryDirectory(dir=ROOT / "results") as temporary:
            with self.assertRaisesRegex(PhoBertTrainingError, "private results/m1-08"):
                _require_private_pilot_output(Path(temporary) / "outside")

    def test_success_commit_binds_receipt_metadata_audit_and_checkpoint(self) -> None:
        private_root = ROOT / "results" / "m1-08"
        private_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=private_root) as temporary:
            output = Path(temporary)
            checkpoint = output / "checkpoint"
            checkpoint.mkdir()
            (output / "pre_fit_audit.json").write_text("{}\n", encoding="utf-8")
            weights = checkpoint / "trainable_state.pt"
            weights.write_bytes(b"checkpoint")
            weight_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            checkpoint_meta = {"weights_sha256": weight_sha}
            checkpoint_path = checkpoint / "checkpoint.json"
            checkpoint_path.write_text(json.dumps(checkpoint_meta), encoding="utf-8")
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            binding = {
                "configuration": "P1",
                "seed": 42,
                "protocol_sha256": "protocol-sha",
                "dataset_revision": "revision",
                "manifest_sha256": "manifest-sha",
                "preparation_lock_sha256": "lock-sha",
                "pre_fit_audit_sha256": hashlib.sha256((output / "pre_fit_audit.json").read_bytes()).hexdigest(),
                "pilot_code_sha256": "pilot-code-sha",
                "trainer_sha256": "trainer-sha",
                "preprocessing_sha256": "preprocessing-sha",
            }
            restore_path = output / "restore_verification.json"
            restore_path.write_text(json.dumps({
                "schema_version": 1,
                "status": "PASS",
                "phase": "M1-08_CUDA_PILOT_ONLY",
                "checkpoint_restore_match": True,
                **binding,
                "checkpoint_metadata_sha256": checkpoint_sha,
                "checkpoint_weights_sha256": weight_sha,
            }), encoding="utf-8")
            restore_sha = hashlib.sha256(restore_path.read_bytes()).hexdigest()
            metadata = {"status": "CREATED", "completed": False}
            _atomic_json(output / "run_metadata.json", metadata)
            receipt = {
                "status": "PASS",
                "phase": "M1-08_CUDA_PILOT_ONLY",
                **binding,
                "pre_fit_audit_sha256": hashlib.sha256((output / "pre_fit_audit.json").read_bytes()).hexdigest(),
                "restore_verification_sha256": restore_sha,
                "pilot": {"checkpoint_restore_match": True},
                "checkpoint": {"metadata_sha256": checkpoint_sha, "weights_sha256": weight_sha},
            }
            _publish_pilot_success(output, metadata, receipt)
            verified = verify_pilot_commit(output)
            self.assertEqual(verified["status"], "PASS")
            restore = json.loads(restore_path.read_text(encoding="utf-8"))
            restore["dataset_revision"] = "different-revision"
            restore_path.write_text(json.dumps(restore), encoding="utf-8")
            receipt_path = output / "pilot_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["restore_verification_sha256"] = hashlib.sha256(restore_path.read_bytes()).hexdigest()
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            metadata_path = output / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["pilot_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            checkpoint_bytes = (checkpoint / "checkpoint.json").read_bytes()
            weights_bytes = weights.read_bytes()
            (output / "pilot_commit.json").write_text(json.dumps({
                "schema_version": 1,
                "status": "COMMITTED",
                "pilot_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "run_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "pre_fit_audit_sha256": hashlib.sha256((output / "pre_fit_audit.json").read_bytes()).hexdigest(),
                "checkpoint_metadata_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "checkpoint_weights_sha256": hashlib.sha256(weights_bytes).hexdigest(),
                "restore_verification_sha256": hashlib.sha256(restore_path.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            with self.assertRaisesRegex(PhoBertTrainingError, "restore verification binding differs"):
                verify_pilot_commit(output)
            (output / "run_metadata.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PhoBertTrainingError, "commit marker hashes"):
                verify_pilot_commit(output)

    def test_failed_finalization_removes_uncommitted_pass_receipt(self) -> None:
        private_root = ROOT / "results" / "m1-08"
        private_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=private_root) as temporary:
            output = Path(temporary)
            checkpoint = output / "checkpoint"
            checkpoint.mkdir()
            (output / "pre_fit_audit.json").write_text("{}\n", encoding="utf-8")
            restore_path = output / "restore_verification.json"
            restore_path.write_text(json.dumps({"status": "PASS", "checkpoint_restore_match": True}), encoding="utf-8")
            restore_sha = hashlib.sha256(restore_path.read_bytes()).hexdigest()
            weights = checkpoint / "trainable_state.pt"
            weights.write_bytes(b"checkpoint")
            weight_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            (checkpoint / "checkpoint.json").write_text(json.dumps({"weights_sha256": weight_sha}), encoding="utf-8")
            metadata = {"status": "CREATED", "completed": False}
            _atomic_json(output / "run_metadata.json", metadata)
            receipt = {
                "status": "PASS",
                "phase": "M1-08_CUDA_PILOT_ONLY",
                "pre_fit_audit_sha256": hashlib.sha256((output / "pre_fit_audit.json").read_bytes()).hexdigest(),
                "restore_verification_sha256": restore_sha,
                "pilot": {"checkpoint_restore_match": True},
                "checkpoint": {"weights_sha256": weight_sha},
            }
            real_writer = _atomic_json
            failed_once = {"value": False}

            def fail_final_metadata(path, payload, *, create_only=False):
                if Path(path).name == "run_metadata.json" and payload.get("status") == "PASS" and not failed_once["value"]:
                    failed_once["value"] = True
                    raise OSError("injected final metadata write failure")
                return real_writer(path, payload, create_only=create_only)

            with mock.patch("nlp_btl.pilot._atomic_json", side_effect=fail_final_metadata):
                with self.assertRaisesRegex(OSError, "injected"):
                    _publish_pilot_success(output, metadata, receipt)
            self.assertFalse((output / "pilot_receipt.json").exists())
            self.assertFalse((output / "pilot_commit.json").exists())

    def test_pilot_api_does_not_accept_training_readiness_receipt(self) -> None:
        self.assertNotIn("readiness_receipt_path", inspect.signature(run_cuda_pilot).parameters)


if __name__ == "__main__":
    unittest.main()
