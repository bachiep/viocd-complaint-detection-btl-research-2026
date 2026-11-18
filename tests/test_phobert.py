"""Synthetic contract tests for the CUDA-only PhoBERT development runner."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from unittest import mock
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_btl.phobert import (
    CHECKPOINT_BINDING_FIELDS,
    M1_09_SOURCE_BUNDLE_ALLOWLIST,
    PhoBertTrainingError,
    _amp_step_succeeded,
    _assert_locked_fit,
    _verify_cuda_pilot_runtime,
    _normalize_accumulated_gradients_,
    _validate_model_loading_info,
    frozen_base_state_sha256,
    run_phobert_development,
    _save_trainable_state,
    _should_select_epoch,
    _validate_development_bundle,
    _verify_training_readiness,
    _load_authorized_m1_09_reviewer,
    _validate_m1_09_kernel_source,
    _validate_m1_09_evidence_file_inventory,
    _validate_m1_09_preparation_root,
    _validate_m1_09_source_bundle,
    _validate_submitted_pilot_notebook,
    _validate_submitted_notebook_matches_staging,
    configure_partial_finetuning,
    restore_trainable_state,
    set_partial_finetuning_train_mode,
    validate_phobert_predictions,
    PhoBertPrediction,
    _verify_runtime_packages,
)
from nlp_btl import data as data_module
from nlp_btl.data import DatasetBundle, ReviewRecord, _sha256
from nlp_btl.pilot import _publish_pilot_success


PROTOCOL_PATH = ROOT / "preparation" / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(values))


class TinyRoberta(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([TinyLayer() for _ in range(12)])


class TinyPhoBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.roberta = TinyRoberta()
        self.classifier = nn.Linear(2, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.roberta.embeddings(values)
        for layer in self.roberta.encoder.layer:
            values = layer(values)
        return self.classifier(values)


def tiny_model() -> TinyPhoBert:
    torch.manual_seed(7)
    model = TinyPhoBert()
    configure_partial_finetuning(model)
    return model


def complete_checkpoint_binding(model: nn.Module) -> dict[str, object]:
    binding = {
        "configuration": "P0",
        "seed": 42,
        "dataset_revision": PROTOCOL["dataset"]["revision"],
        "manifest_sha256": PROTOCOL["dataset"]["manifest_sha256"],
        "protocol_sha256": "synthetic-protocol-sha256",
        "preparation_lock_sha256": "synthetic-preparation-lock-sha256",
        "pre_fit_audit_sha256": "synthetic-pre-fit-audit-sha256",
        "readiness_receipt_sha256": "synthetic-readiness-sha256",
        "model_id": PROTOCOL["model"]["id"],
        "model_revision": PROTOCOL["model"]["revision"],
        "model_config_sha256": PROTOCOL["model"]["config_sha256"],
        "frozen_base_state_sha256": frozen_base_state_sha256(model),
        "segmenter_id": "underthesea",
        "tokenizer_revision": PROTOCOL["model"]["revision"],
        "tokenizer_audit_sha256": "synthetic-tokenizer-audit-sha256",
        "max_length": 128,
        "trainable_scope": "encoder_layers_10_11_and_classifier",
        "trainer_sha256": "synthetic-trainer-sha256",
        "preprocessing_sha256": "synthetic-preprocessing-sha256",
    }
    assert set(binding) == CHECKPOINT_BINDING_FIELDS
    return binding


def write_completed_attempt(
    run_dir: Path, checkpoint: Path, binding: dict[str, object], receipt: dict[str, object]
) -> None:
    checkpoint_metadata = checkpoint / "checkpoint.json"
    weights = checkpoint / "trainable_state.pt"
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                **binding,
                "status": "FIT_COMPLETE",
                "selected_epoch": receipt["selected_epoch"],
                "validation_macro_f1": receipt["validation_macro_f1"],
                "checkpoint_metadata_sha256": hashlib.sha256(checkpoint_metadata.read_bytes()).hexdigest(),
                "checkpoint_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


class PhoBertContractTests(unittest.TestCase):
    def test_submitted_pilot_notebook_requires_one_live_locked_call(self) -> None:
        notebook = {
            "nbformat": 4,
            "cells": [{
                "cell_type": "code",
                "source": [
                    "from nlp_btl.pilot import run_cuda_pilot\n",
                    'pilot_result = run_cuda_pilot(protocol, preparation, output, config_id="P1", seed=42)\n',
                ],
            }],
        }
        _validate_submitted_pilot_notebook(notebook)

        comment_only = copy.deepcopy(notebook)
        comment_only["cells"][0]["source"] = [
            "from nlp_btl.pilot import run_cuda_pilot\n",
            '# run_cuda_pilot(config_id="P1", seed=42)\n',
        ]
        with self.assertRaisesRegex(PhoBertTrainingError, "exactly one live top-level pilot call"):
            _validate_submitted_pilot_notebook(comment_only)

        nested_call = copy.deepcopy(notebook)
        nested_call["cells"][0]["source"] = [
            "from nlp_btl.pilot import run_cuda_pilot\n",
            'if False:\n    run_cuda_pilot(config_id="P1", seed=42)\n',
        ]
        with self.assertRaisesRegex(PhoBertTrainingError, "exactly one live top-level pilot call"):
            _validate_submitted_pilot_notebook(nested_call)

        wrong_seed = copy.deepcopy(notebook)
        wrong_seed["cells"][0]["source"][1] = 'run_cuda_pilot(protocol, preparation, output, config_id="P1", seed=43)\n'
        with self.assertRaisesRegex(PhoBertTrainingError, "locked P1/seed42 pilot"):
            _validate_submitted_pilot_notebook(wrong_seed)

        research_call = copy.deepcopy(notebook)
        research_call["cells"][0]["source"].append("run_phobert_development()\n")
        with self.assertRaisesRegex(PhoBertTrainingError, "must not reference the research fit runner"):
            _validate_submitted_pilot_notebook(research_call)

        missing_arguments = copy.deepcopy(notebook)
        missing_arguments["cells"][0]["source"][1] = 'run_cuda_pilot(config_id="P1", seed=42)\n'
        with self.assertRaisesRegex(PhoBertTrainingError, "exactly one live top-level pilot call"):
            _validate_submitted_pilot_notebook(missing_arguments)

        fake_local_binding = copy.deepcopy(notebook)
        fake_local_binding["cells"][0]["source"].extend(
            [
                "def run_cuda_pilot(*args, **kwargs):\n",
                "    return None\n",
            ]
        )
        with self.assertRaisesRegex(PhoBertTrainingError, "must not shadow the production pilot binding"):
            _validate_submitted_pilot_notebook(fake_local_binding)

        imported_fake_binding = copy.deepcopy(notebook)
        imported_fake_binding["cells"][0]["source"].append("from fake_module import run_cuda_pilot\n")
        with self.assertRaisesRegex(PhoBertTrainingError, "ambiguous pilot import binding"):
            _validate_submitted_pilot_notebook(imported_fake_binding)

        replaced_code = copy.deepcopy(notebook)
        replaced_code["cells"][0]["source"].extend(
            [
                "def fake_pilot(*args, **kwargs):\n",
                "    return None\n",
                "run_cuda_pilot.__code__ = fake_pilot.__code__\n",
            ]
        )
        with self.assertRaisesRegex(PhoBertTrainingError, "may not mutate or reflectively replace"):
            _validate_submitted_pilot_notebook(replaced_code)

        replaced_module_attribute = copy.deepcopy(notebook)
        replaced_module_attribute["cells"][0]["source"].extend(
            [
                "import nlp_btl.pilot\n",
                "nlp_btl.pilot.run_cuda_pilot = lambda *args, **kwargs: None\n",
            ]
        )
        with self.assertRaisesRegex(PhoBertTrainingError, "may not mutate or reflectively replace"):
            _validate_submitted_pilot_notebook(replaced_module_attribute)

        for mutation in (
            "vars()['SOURCE_ROOT'] = None\n",
            "import builtins\nbuiltins.vars()['EXPECTED_FILES'] = []\n",
            "from builtins import vars as reflect\nreflect()['EXPECTED_FILE_HASHES'] = {}\n",
            "__builtins__['vars']()['SOURCE_FILE_NAMES'] = {}\n",
            "import inspect\ninspect.currentframe().__getattribute__('f_globals')['SOURCE_ROOT'] = None\n",
            "import builtins\nobject.__getattribute__(builtins, '__dict__')['vars']()['SOURCE_ROOT'] = None\n",
            "import builtins\nobject.__getattribute__(builtins, 'vars')()['SOURCE_ROOT'] = None\n",
        ):
            reflected_binding = copy.deepcopy(notebook)
            reflected_binding["cells"][0]["source"].append(mutation)
            with self.subTest(mutation=mutation.splitlines()[-1]):
                with self.assertRaisesRegex(PhoBertTrainingError, "may not use dynamic code or import execution"):
                    _validate_submitted_pilot_notebook(reflected_binding)

        aliased_research = copy.deepcopy(notebook)
        aliased_research["cells"][0]["source"].append(
            "from nlp_btl.phobert import run_phobert_development as fit\n"
        )
        aliased_research["cells"][0]["source"].append("fit()\n")
        with self.assertRaisesRegex(PhoBertTrainingError, "must not reference the research fit runner"):
            _validate_submitted_pilot_notebook(aliased_research)

    def test_m1_09_notebook_code_must_match_staged_notebook_ast(self) -> None:
        staged = json.loads(
            (ROOT / "staging/kernel/m1_08_cuda_pilot.ipynb").read_text(encoding="utf-8")
        )
        _validate_submitted_notebook_matches_staging(staged, staged)

        changed = copy.deepcopy(staged)
        changed["cells"][0]["source"].append("pilot_ref = run_cuda_pilot\n")
        changed["cells"][0]["source"].append("fake = lambda *args, **kwargs: None\n")
        changed["cells"][0]["source"].append("pilot_ref.__code__ = fake.__code__\n")
        with self.assertRaisesRegex(PhoBertTrainingError, "differs from the reviewed staged pilot"):
            _validate_submitted_notebook_matches_staging(changed, staged)

        object_setattr = copy.deepcopy(staged)
        object_setattr["cells"][0]["source"].append(
            'object.__setattr__(run_cuda_pilot, "__code__", fake.__code__)\n'
        )
        with self.assertRaisesRegex(PhoBertTrainingError, "dynamic code or import execution"):
            _validate_submitted_pilot_notebook(object_setattr)

    def test_m1_09_kernel_must_attach_exactly_one_reviewed_dataset(self) -> None:
        source_bundle = {"dataset_id": "example-owner/viocd-source"}
        metadata = {
            "dataset_sources": ["example-owner/viocd-source"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        _validate_m1_09_kernel_source(metadata, source_bundle)
        for sources in (
            ["example-owner/viocd-source", "someone/extra"],
            ["example-owner/viocd-source", "example-owner/viocd-source"],
            [],
        ):
            with self.subTest(sources=sources):
                with self.assertRaisesRegex(PhoBertTrainingError, "exactly the reviewed source bundle"):
                    _validate_m1_09_kernel_source({**metadata, "dataset_sources": sources}, source_bundle)
        with self.assertRaisesRegex(PhoBertTrainingError, "exactly the reviewed source bundle"):
            _validate_m1_09_kernel_source(
                {**metadata, "competition_sources": ["some-competition"]}, source_bundle
            )

    def test_m1_09_evidence_hash_map_must_be_exact(self) -> None:
        expected = ("kernel_metadata.json", "submitted_source.ipynb")
        _validate_m1_09_evidence_file_inventory(
            {"kernel_metadata.json": "a" * 64, "submitted_source.ipynb": "b" * 64},
            expected,
        )
        for files in (
            {"kernel_metadata.json": "a" * 64},
            {
                "kernel_metadata.json": "a" * 64,
                "submitted_source.ipynb": "b" * 64,
                "data/reviews.csv": "c" * 64,
            },
        ):
            with self.subTest(files=files):
                with self.assertRaisesRegex(PhoBertTrainingError, "missing or unreviewed file entries"):
                    _validate_m1_09_evidence_file_inventory(files, expected)

    def test_m1_09_reviewer_registry_requires_one_authorized_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preparation = Path(temporary)
            registry_path = preparation / "m1_09_trusted_reviewers.json"
            registry = {
                "schema_version": 2,
                "status": "CONFIGURED",
                "reviewers": [
                    {
                        "reviewer_id": "reviewer-01",
                        "reviewer_role": "independent QA",
                        "reviewer_kind": "codex_subagent",
                        "authorization_basis": "user requested an independent reviewer",
                    }
                ],
            }
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            reviewer = _load_authorized_m1_09_reviewer(
                preparation, "reviewer-01", "independent QA"
            )
            self.assertEqual(reviewer, registry["reviewers"][0])

            for malformed_version in (True, 2.0):
                registry["schema_version"] = malformed_version
                registry_path.write_text(json.dumps(registry), encoding="utf-8")
                with self.subTest(schema_version=malformed_version):
                    self.assertIsNone(
                        _load_authorized_m1_09_reviewer(
                            preparation, "reviewer-01", "independent QA"
                        )
                    )
            registry["schema_version"] = 2

            registry["reviewers"].append(dict(registry["reviewers"][0]))
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            self.assertIsNone(
                _load_authorized_m1_09_reviewer(
                    preparation, "reviewer-01", "independent QA"
                )
            )
            self.assertIsNone(
                _load_authorized_m1_09_reviewer(
                    preparation, "unlisted", "independent QA"
                )
            )

    def test_m1_09_source_bundle_rejects_extra_or_stale_files(self) -> None:
        source_root = ROOT.resolve()
        files = {
            relative: _sha256(source_root.joinpath(*Path(relative).parts).read_bytes())
            for relative in M1_09_SOURCE_BUNDLE_ALLOWLIST
        }
        current_hashes = {
            "trainer_sha256": "a" * 64,
            "reviewer_trust_anchor_sha256": files["preparation/m1_09_trusted_reviewers.json"],
        }
        bundle = {
            "schema_version": 1,
            "source_hashes": current_hashes,
            "files": files,
        }
        _validate_m1_09_source_bundle(bundle, current_hashes, source_root)

        for malformed_version in (True, 1.0):
            malformed = copy.deepcopy(bundle)
            malformed["schema_version"] = malformed_version
            with self.subTest(schema_version=malformed_version):
                with self.assertRaisesRegex(PhoBertTrainingError, "manifest schema is invalid"):
                    _validate_m1_09_source_bundle(malformed, current_hashes, source_root)

        with_extra = copy.deepcopy(bundle)
        with_extra["files"]["data/reviews.csv"] = "b" * 64
        with self.assertRaisesRegex(PhoBertTrainingError, "missing or unreviewed files"):
            _validate_m1_09_source_bundle(with_extra, current_hashes, source_root)

        stale = copy.deepcopy(bundle)
        stale["files"]["src/nlp_btl/pilot.py"] = "0" * 64
        with self.assertRaisesRegex(PhoBertTrainingError, "exact current source files"):
            _validate_m1_09_source_bundle(stale, current_hashes, source_root)

        alternate_trust = dict(current_hashes)
        alternate_trust["reviewer_trust_anchor_sha256"] = "f" * 64
        attacker_bundle = copy.deepcopy(bundle)
        attacker_bundle["source_hashes"] = alternate_trust
        with self.assertRaisesRegex(PhoBertTrainingError, "trust anchor differs"):
            _validate_m1_09_source_bundle(attacker_bundle, alternate_trust, source_root)

    def test_m1_09_evidence_rejects_an_alternate_preparation_root(self) -> None:
        self.assertEqual(
            _validate_m1_09_preparation_root(ROOT / "preparation"),
            (ROOT / "preparation").resolve(),
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            with self.assertRaisesRegex(PhoBertTrainingError, "reviewed checkout root"):
                _validate_m1_09_preparation_root(temporary)

    def test_locked_configuration_rejects_changed_schedule_and_allows_declared_seed(self) -> None:
        configuration, segmenter = _assert_locked_fit(PROTOCOL, "P1", 43)
        self.assertEqual(configuration["id"], "P1")
        self.assertEqual(segmenter, "vncorenlp_rdr")

        changed = copy.deepcopy(PROTOCOL)
        changed["training"]["gradient_accumulation"] = 2
        with self.assertRaisesRegex(PhoBertTrainingError, "parameters differ"):
            _assert_locked_fit(changed, "P0", 42)
        with self.assertRaisesRegex(PhoBertTrainingError, "Seed is not declared"):
            _assert_locked_fit(PROTOCOL, "P0", 99)

    def test_trainable_scope_and_dropout_modes_match_two_last_layers(self) -> None:
        model = TinyPhoBert()
        names = configure_partial_finetuning(model)
        self.assertTrue(names)
        self.assertTrue(all(
            name.startswith(("roberta.encoder.layer.10.", "roberta.encoder.layer.11.", "classifier."))
            for name in names
        ))
        self.assertTrue(all(not p.requires_grad for p in model.roberta.embeddings.parameters()))
        self.assertTrue(all(not p.requires_grad for layer in model.roberta.encoder.layer[:10] for p in layer.parameters()))
        self.assertTrue(all(p.requires_grad for layer in model.roberta.encoder.layer[10:] for p in layer.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

        set_partial_finetuning_train_mode(model)
        self.assertFalse(model.roberta.embeddings.training)
        self.assertTrue(all(not layer.training for layer in model.roberta.encoder.layer[:10]))
        self.assertTrue(all(layer.training for layer in model.roberta.encoder.layer[10:]))
        self.assertTrue(model.classifier.training)

    def test_wrong_layer_count_fails_closed(self) -> None:
        model = TinyPhoBert()
        model.roberta.encoder.layer = nn.ModuleList([TinyLayer() for _ in range(11)])
        with self.assertRaisesRegex(PhoBertTrainingError, "12-layer"):
            configure_partial_finetuning(model)

    def test_model_loading_allows_only_fresh_classifier_and_pinned_unused_weights(self) -> None:
        model = TinyPhoBert()
        expected_missing = {
            name for name in model.state_dict() if name.startswith("classifier.")
        }
        expected_unused_pooler = {
            "roberta.pooler.dense.bias",
            "roberta.pooler.dense.weight",
        }
        summary = _validate_model_loading_info(
            model,
            {
                "missing_keys": sorted(expected_missing),
                "unexpected_keys": ["lm_head.dense.weight", *sorted(expected_unused_pooler)],
                "mismatched_keys": [],
                "error_msgs": [],
            },
        )
        self.assertTrue(summary["only_expected_head_differences"])
        self.assertEqual(summary["missing_classifier_tensor_count"], len(expected_missing))
        self.assertEqual(summary["unexpected_base_lm_head_tensor_count"], 1)
        self.assertEqual(summary["expected_unused_pooler_tensor_count"], 2)
        for pooler_key in sorted(expected_unused_pooler):
            with self.subTest(pooler_key=pooler_key):
                subset_summary = _validate_model_loading_info(
                    model,
                    {
                        "missing_keys": sorted(expected_missing),
                        "unexpected_keys": [pooler_key],
                    },
                )
                self.assertEqual(subset_summary["expected_unused_pooler_tensor_count"], 1)
        for invalid in (
            {"missing_keys": sorted(expected_missing | {"roberta.embeddings.0.weight"}), "unexpected_keys": []},
            {"missing_keys": sorted(expected_missing), "unexpected_keys": [], "mismatched_keys": [("roberta.weight", (1,), (2,))]},
        ):
            with self.assertRaises(PhoBertTrainingError):
                _validate_model_loading_info(model, invalid)

        with self.assertRaises(PhoBertTrainingError) as caught:
            _validate_model_loading_info(
                model,
                {
                    "missing_keys": sorted(expected_missing),
                    "unexpected_keys": [
                        *sorted(expected_unused_pooler),
                        "roberta.encoder.layer.0.weight",
                        "private.raw.key",
                    ],
                },
            )
        self.assertEqual(
            caught.exception.safe_diagnostics,
            {
                "missing_key_count": len(expected_missing),
                "unexpected_key_count": 4,
                "unexpected_key_namespace_counts": {
                    "lm_head": 0,
                    "roberta": 3,
                    "classifier": 0,
                    "other": 1,
                    "non_string": 0,
                },
                "unexpected_roberta_keys": [
                    "roberta.encoder.layer.0.weight",
                    "roberta.pooler.dense.bias",
                    "roberta.pooler.dense.weight",
                ],
            },
        )
        self.assertNotIn("private.raw.key", repr(caught.exception.safe_diagnostics))

        malformed = ["roberta.invalid/key"]
        with self.assertRaises(PhoBertTrainingError) as malformed_caught:
            _validate_model_loading_info(
                model,
                {"missing_keys": sorted(expected_missing), "unexpected_keys": malformed},
            )
        self.assertNotIn("unexpected_roberta_keys", malformed_caught.exception.safe_diagnostics)
        self.assertEqual(
            malformed_caught.exception.safe_diagnostics["unexpected_key_namespace_counts"]["roberta"],
            1,
        )
        self.assertNotIn("roberta.invalid/key", repr(malformed_caught.exception.safe_diagnostics))

        overflow = [f"roberta.encoder.layer.{index}.weight" for index in range(9)]
        with self.assertRaises(PhoBertTrainingError) as overflow_caught:
            _validate_model_loading_info(
                model,
                {"missing_keys": sorted(expected_missing), "unexpected_keys": overflow},
            )
        self.assertNotIn("unexpected_roberta_keys", overflow_caught.exception.safe_diagnostics)
        self.assertEqual(
            overflow_caught.exception.safe_diagnostics["unexpected_key_namespace_counts"]["roberta"],
            9,
        )

    def test_accumulation_normalization_weights_final_partial_group_by_samples(self) -> None:
        parameter = nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        # Two microbatches: the first represents two examples with mean grad 1;
        # the final partial batch represents one example with mean grad 3.
        (parameter * 1.0 * 2).backward()
        (parameter * 3.0 * 1).backward()
        _normalize_accumulated_gradients_(optimizer, sample_count=3)
        self.assertAlmostEqual(float(parameter.grad), 5.0 / 3.0, places=6)
        with self.assertRaisesRegex(PhoBertTrainingError, "no examples"):
            _normalize_accumulated_gradients_(optimizer, sample_count=0)

    def test_amp_scale_drop_is_skip_and_exact_tie_keeps_first_epoch(self) -> None:
        self.assertTrue(_amp_step_succeeded(128.0, 128.0))
        self.assertTrue(_amp_step_succeeded(128.0, 256.0))
        self.assertFalse(_amp_step_succeeded(128.0, 64.0))
        self.assertTrue(_should_select_epoch(None, 0.5))
        self.assertFalse(_should_select_epoch(0.5, 0.5))
        self.assertTrue(_should_select_epoch(0.5, 0.500001))

    def test_checkpoint_restore_reproduces_fixture_logits_and_checks_binding(self) -> None:
        model = tiny_model().eval()
        values = torch.tensor([[0.25, -0.75], [1.0, 0.5]])
        before = model(values).detach().clone()
        binding = complete_checkpoint_binding(model)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "attempt"
            run_dir.mkdir()
            checkpoint = run_dir / "best_checkpoint"
            checkpoint_receipt = _save_trainable_state(
                model,
                checkpoint,
                {**binding, "selected_epoch": 2, "validation_macro_f1": 0.75},
            )
            write_completed_attempt(run_dir, checkpoint, binding, checkpoint_receipt)
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(1.0)
            receipt = restore_trainable_state(model, checkpoint, binding)
            after = model(values).detach()
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(receipt["storage_scope"], "trainable_parameters_only; reload pinned base before restore")
            with self.assertRaisesRegex(PhoBertTrainingError, "complete checkpoint identity"):
                restore_trainable_state(model, checkpoint, {})
            with self.assertRaisesRegex(PhoBertTrainingError, "Parent run identity mismatch"):
                restore_trainable_state(model, checkpoint, {**binding, "seed": 43})
            wrong_base = tiny_model()
            with torch.no_grad():
                wrong_base.roberta.embeddings[0].weight[0, 0].add_(1.0)
            with self.assertRaisesRegex(PhoBertTrainingError, "frozen base differs"):
                restore_trainable_state(wrong_base, checkpoint, binding)

    def test_checkpoint_tampering_is_detected(self) -> None:
        model = tiny_model()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "attempt"
            run_dir.mkdir()
            checkpoint = run_dir / "checkpoint"
            binding = complete_checkpoint_binding(model)
            receipt = _save_trainable_state(
                model,
                checkpoint,
                {**binding, "selected_epoch": 1, "validation_macro_f1": 0.5},
            )
            write_completed_attempt(run_dir, checkpoint, binding, receipt)
            weights = checkpoint / receipt["weights_file"]
            weights.write_bytes(weights.read_bytes() + b"tamper")
            with self.assertRaisesRegex(PhoBertTrainingError, "weight hash mismatch"):
                restore_trainable_state(model, checkpoint, binding)

    def test_checkpoint_from_failed_attempt_cannot_be_restored(self) -> None:
        model = tiny_model()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "attempt"
            run_dir.mkdir()
            checkpoint = run_dir / "checkpoint"
            binding = complete_checkpoint_binding(model)
            receipt = _save_trainable_state(
                model,
                checkpoint,
                {**binding, "selected_epoch": 1, "validation_macro_f1": 0.5},
            )
            write_completed_attempt(run_dir, checkpoint, binding, receipt)
            failed = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            failed["status"] = "FAILED"
            (run_dir / "run_metadata.json").write_text(json.dumps(failed), encoding="utf-8")
            with self.assertRaisesRegex(PhoBertTrainingError, "incomplete or failed"):
                restore_trainable_state(model, checkpoint, binding)

    def test_validation_predictions_require_exact_validation_keys_and_labels(self) -> None:
        records = (
            ReviewRecord("rev", "val", 0, "private text", 0, "app"),
            ReviewRecord("rev", "val", 1, "private text", 1, "mobile"),
        )
        predictions = (
            PhoBertPrediction(records[0].key, 0, 0),
            PhoBertPrediction(records[1].key, 1, 1),
        )
        validate_phobert_predictions(records, predictions)
        with self.assertRaisesRegex(PhoBertTrainingError, "source keys"):
            validate_phobert_predictions(records, predictions[:1])
        with self.assertRaisesRegex(PhoBertTrainingError, "gold label"):
            validate_phobert_predictions(records, (PhoBertPrediction(records[0].key, 1, 0), predictions[1]))

    def test_development_bundle_rejects_terminal_test_split(self) -> None:
        class Bundle:
            splits = {"train": (), "val": (), "test": ()}

        with self.assertRaisesRegex(PhoBertTrainingError, "only train and validation"):
            _validate_development_bundle(Bundle(), PROTOCOL, {})

    def test_development_bundle_records_are_bound_to_trusted_membership(self) -> None:
        revision = PROTOCOL["dataset"]["revision"]
        train = (
            ReviewRecord(revision, "train", 0, "good product", 0, "app"),
            ReviewRecord(revision, "train", 1, "bad product", 1, "mobile"),
        )
        validation = (
            ReviewRecord(revision, "val", 0, "good service", 0, "app"),
            ReviewRecord(revision, "val", 1, "bad service", 1, "mobile"),
        )

        def split_contract(split: str, records: tuple[ReviewRecord, ...]) -> dict[str, object]:
            membership = "".join(
                f"{split}\t{record.source_row}\t{record.label}\t{record.domain}\t"
                f"{_sha256(record.text.encode('utf-8'))}\n"
                for record in sorted(records, key=lambda row: row.source_row)
            ).encode("utf-8")
            return {
                "retained_rows": len(records),
                "source_sha256": f"{split}-source-sha",
                "source_bytes": 1,
                "ordered_membership_digest_sha256": _sha256(membership),
            }

        manifest = {"splits": {"train": split_contract("train", train), "val": split_contract("val", validation)}}
        bundle = DatasetBundle(
            PROTOCOL["protocol_id"],
            revision,
            "manifest-sha",
            "protocol-sha",
            _sha256(Path(data_module.__file__).read_bytes()),
            "synthetic-pandas",
            MappingProxyType({"train": train, "val": validation}),
            MappingProxyType({
                name: MappingProxyType(dict(values)) for name, values in manifest["splits"].items()
            }),
        )
        _validate_development_bundle(bundle, PROTOCOL, manifest)
        tampered_train = (
            train[0],
            ReviewRecord(revision, "train", 1, "fabricated complaint", 1, "mobile"),
        )
        tampered_bundle = DatasetBundle(
            bundle.protocol_id,
            bundle.revision,
            bundle.manifest_sha256,
            bundle.protocol_sha256,
            bundle.loader_sha256,
            bundle.pandas_version,
            MappingProxyType({"train": tampered_train, "val": validation}),
            bundle.audit,
        )
        with self.assertRaisesRegex(PhoBertTrainingError, "trusted membership"):
            _validate_development_bundle(tampered_bundle, PROTOCOL, manifest)

    def test_runner_records_blocked_attempt_when_cuda_is_unavailable(self) -> None:
        revision = PROTOCOL["dataset"]["revision"]
        train = (
            ReviewRecord(revision, "train", 0, "synthetic positive", 0, "app"),
            ReviewRecord(revision, "train", 1, "synthetic complaint", 1, "mobile"),
        )
        validation = (
            ReviewRecord(revision, "val", 0, "synthetic positive validation", 0, "app"),
            ReviewRecord(revision, "val", 1, "synthetic complaint validation", 1, "mobile"),
        )
        bundle = DatasetBundle(
            protocol_id=PROTOCOL["protocol_id"],
            revision=revision,
            manifest_sha256=PROTOCOL["dataset"]["manifest_sha256"],
            protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            loader_sha256="synthetic-loader",
            pandas_version="synthetic-pandas",
            splits=MappingProxyType({"train": train, "val": validation}),
            audit=MappingProxyType({}),
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            output = Path(temporary) / "attempt"
            with (
                mock.patch("nlp_btl.phobert._validate_development_bundle"),
                self.assertRaisesRegex(PhoBertTrainingError, "must stay under private results/") ,
            ):
                run_phobert_development(
                    bundle,
                    PROTOCOL_PATH,
                    "P0",
                    42,
                    ROOT / "preparation",
                    Path(temporary) / "prefit.json",
                    Path(temporary) / "readiness.json",
                    ROOT / "outside-results-escape-test" / "attempt",
                    Path(temporary) / "evidence_manifest.json",
                )
            with (
                mock.patch("nlp_btl.phobert._validate_development_bundle"),
                mock.patch(
                    "nlp_btl.phobert._verify_training_readiness",
                    return_value=({}, {}, "readiness-sha", "audit-sha"),
                ),
                mock.patch("torch.cuda.is_available", return_value=False),
            ):
                with self.assertRaisesRegex(PhoBertTrainingError, "requires an available CUDA GPU"):
                    run_phobert_development(
                        bundle,
                        PROTOCOL_PATH,
                        "P0",
                        42,
                        ROOT / "preparation",
                        Path(temporary) / "prefit.json",
                        Path(temporary) / "readiness.json",
                        output,
                        Path(temporary) / "evidence_manifest.json",
                    )
            run_metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(run_metadata["status"], "BLOCKED")
            self.assertFalse(run_metadata["test_split_available_to_runner"])

    def test_runner_cannot_fit_without_gate_b_readiness_receipt(self) -> None:
        revision = PROTOCOL["dataset"]["revision"]
        train = (
            ReviewRecord(revision, "train", 0, "synthetic positive", 0, "app"),
            ReviewRecord(revision, "train", 1, "synthetic complaint", 1, "mobile"),
        )
        validation = (
            ReviewRecord(revision, "val", 0, "synthetic positive validation", 0, "app"),
            ReviewRecord(revision, "val", 1, "synthetic complaint validation", 1, "mobile"),
        )
        bundle = DatasetBundle(
            protocol_id=PROTOCOL["protocol_id"],
            revision=revision,
            manifest_sha256=PROTOCOL["dataset"]["manifest_sha256"],
            protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            loader_sha256="synthetic-loader",
            pandas_version="synthetic-pandas",
            splits=MappingProxyType({"train": train, "val": validation}),
            audit=MappingProxyType({}),
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            output = Path(temporary) / "attempt"
            with (
                mock.patch("nlp_btl.phobert._validate_development_bundle"),
                mock.patch("torch.cuda.is_available", side_effect=AssertionError("CUDA must not be queried")),
            ):
                with self.assertRaisesRegex(PhoBertTrainingError, "pre-fit audit receipt is unavailable"):
                    run_phobert_development(
                        bundle,
                        PROTOCOL_PATH,
                        "P0",
                        42,
                        ROOT / "preparation",
                        Path(temporary) / "missing-prefit.json",
                        Path(temporary) / "missing-readiness.json",
                        output,
                        Path(temporary) / "missing-evidence.json",
                    )
            run_metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(run_metadata["status"], "BLOCKED")
            self.assertEqual(run_metadata["failed_stage"], "training_readiness")

    def test_fit_runtime_checks_protocol_and_tokenizer_audit_package_versions(self) -> None:
        tokenizer_audit = {
            "packages": {
                "transformers": "4.51.3",
                "tokenizers": "0.21.4",
                "huggingface_hub": "0.30.2",
            }
        }
        expected = dict(PROTOCOL["environment"]["explicit_package_targets"])
        expected.update(tokenizer_audit["packages"])

        with mock.patch("importlib.metadata.version", side_effect=expected.__getitem__):
            self.assertEqual(_verify_runtime_packages(PROTOCOL, tokenizer_audit), expected)

        for package in ("numpy", "huggingface_hub"):
            changed = dict(expected)
            changed[package] = "unexpected-version"
            with (
                mock.patch("importlib.metadata.version", side_effect=changed.__getitem__),
                self.assertRaisesRegex(PhoBertTrainingError, "differ from the locked protocol/audit"),
            ):
                _verify_runtime_packages(PROTOCOL, tokenizer_audit)

    def test_fit_cuda_resources_must_match_the_passed_pilot(self) -> None:
        pilot = {
            "torch": "2.11.0+cu130",
            "cuda_runtime": "13.0",
            "gpu_name": "Tesla T4",
            "device_count": 1,
            "gpu_total_memory_bytes": 16_083_123_200,
        }
        _verify_cuda_pilot_runtime(dict(pilot), pilot)

        for field in ("device_count", "gpu_total_memory_bytes"):
            changed_runtime = dict(pilot)
            changed_runtime[field] -= 1
            with self.assertRaisesRegex(PhoBertTrainingError, "GPU capacity differs from the passed pilot"):
                _verify_cuda_pilot_runtime(changed_runtime, pilot)

    def test_readiness_gate_accepts_bound_fixture_and_rejects_stale_or_colliding_audit(self) -> None:
        import nlp_btl.phobert as phobert_module
        import nlp_btl.preprocessing as preprocessing_module

        preparation = ROOT / "preparation"
        lock = json.loads((preparation / "preparation_lock.json").read_text(encoding="utf-8"))
        segmentation_lock = json.loads((preparation / "segmentation_audit.json").read_text(encoding="utf-8"))
        tokenizer_audit = json.loads((preparation / "tokenizer_input_audit.json").read_text(encoding="utf-8"))
        protocol_sha = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
        manifest_sha = PROTOCOL["dataset"]["manifest_sha256"]
        revision = PROTOCOL["dataset"]["revision"]
        runtime_packages = tokenizer_audit["packages"]
        fake_assets = {"assets/vncorenlp/test-fixture": "asset-sha"}
        fake_wrapper = {
            "sdist_sha256": PROTOCOL["preprocessing"]["rdr"]["wrapper_sdist_sha256"],
            "installed_source_sha256": {"test-fixture": "wrapper-sha"},
            "installed_sources_match_sdist": True,
        }
        audit = {
            "schema_version": 1,
            "status": "PASS",
            "protocol_id": PROTOCOL["protocol_id"],
            "protocol_sha256": protocol_sha,
            "data_manifest_sha256": manifest_sha,
            "dataset_revision": revision,
            "segmentation_audit_sha256": lock["hashes"]["segmentation_audit.json"],
            "tokenizer_input_audit_sha256": lock["hashes"]["tokenizer_input_audit.json"],
            "preprocessing_code_sha256": _sha256(Path(preprocessing_module.__file__).read_bytes()),
            "cross_split_collision_found": False,
            "test_used_only_for_pre_fit_integrity_audit": True,
            "raw_review_text_saved": False,
            "row_level_ids_saved": False,
            "token_id_sequences_saved": False,
            "model_weights_loaded": False,
            "training_performed": False,
            "rdr_import_origin_verified": True,
            "rdr_asset_sha256": fake_assets,
            "rdr_wrapper": fake_wrapper,
            "runtime_packages": runtime_packages,
            "max_length": 128,
            "runtime": {"python_version": "3.13.fixture", "java_version": "21.fixture"},
            "segmentation": {
                arm: {
                    "matches_locked_audit": True,
                    "splits": segmentation_lock["results"][arm]["splits"],
                }
                for arm in ("underthesea", "vncorenlp_rdr")
            },
            "tokenizer_inputs": {
                config_id: {
                    **tokenizer_audit["configurations"][config_id],
                    "matches_locked_audit": True,
                }
                for config_id in ("P0", "P1")
            },
        }
        bundle = DatasetBundle(
            PROTOCOL["protocol_id"],
            revision,
            manifest_sha,
            protocol_sha,
            "synthetic-loader",
            "synthetic-pandas",
            MappingProxyType({"train": (), "val": ()}),
            MappingProxyType({}),
        )
        reviewer_patch = mock.patch(
            "nlp_btl.phobert._load_authorized_m1_09_reviewer",
            return_value={
                "reviewer_id": "test-reviewer-01",
                "reviewer_role": "independent QA",
                "reviewer_kind": "codex_subagent",
                "authorization_basis": "synthetic test fixture",
            },
        )
        reviewer_patch.start()
        self.addCleanup(reviewer_patch.stop)
        pilot = {
            "status": "PASS",
            "cuda_available": True,
            "minibatch_forward_backward_finite": True,
            "trainable_scope_verified": True,
            "optimizer_step_succeeded": True,
            "checkpoint_restore_match": True,
            "torch": "torch.fixture",
            "cuda_runtime": "cuda.fixture",
            "gpu_name": "gpu.fixture",
            "device_count": 1,
            "gpu_total_memory_bytes": 16_083_123_200,
        }

        def write_pair(
            folder: Path,
            name: str,
            audit_data: dict[str, object],
            pilot_identity_overrides: dict[str, object] | None = None,
            pilot_files_mutation: str | None = None,
        ):
            audit_path = folder / f"{name}-audit.json"
            audit_path.write_text(json.dumps(audit_data), encoding="utf-8")
            evidence_root = folder / f"{name}-evidence"
            evidence_root.mkdir()
            pilot_dir = evidence_root / "pilot_output"
            checkpoint_dir = pilot_dir / "checkpoint"
            checkpoint_dir.mkdir(parents=True)
            (pilot_dir / "pre_fit_audit.json").write_bytes(audit_path.read_bytes())
            weights_path = checkpoint_dir / "trainable_state.pt"
            weights_path.write_bytes(b"synthetic checkpoint weights")
            weights_sha = _sha256(weights_path.read_bytes())
            checkpoint_path = checkpoint_dir / "checkpoint.json"
            checkpoint_path.write_text(json.dumps({"weights_sha256": weights_sha}), encoding="utf-8")
            restore = {
                "schema_version": 1,
                "status": "PASS",
                "phase": "M1-08_CUDA_PILOT_ONLY",
                "checkpoint_restore_match": True,
                "protocol_id": PROTOCOL["protocol_id"],
                "configuration": "P1",
                "seed": 42,
                "protocol_sha256": protocol_sha,
                "dataset_revision": revision,
                "manifest_sha256": manifest_sha,
                "preparation_lock_sha256": _sha256((preparation / "preparation_lock.json").read_bytes()),
                "pre_fit_audit_sha256": _sha256(audit_path.read_bytes()),
                "pilot_code_sha256": _sha256(Path(phobert_module.__file__).with_name("pilot.py").read_bytes()),
                "trainer_sha256": _sha256(Path(phobert_module.__file__).read_bytes()),
                "preprocessing_sha256": _sha256(Path(preprocessing_module.__file__).read_bytes()),
                "checkpoint_metadata_sha256": _sha256(checkpoint_path.read_bytes()),
                "checkpoint_weights_sha256": weights_sha,
            }
            if pilot_identity_overrides:
                restore.update(pilot_identity_overrides)
            restore_path = pilot_dir / "restore_verification.json"
            restore_path.write_text(json.dumps(restore), encoding="utf-8")
            restore_sha = _sha256(restore_path.read_bytes())
            pilot_receipt = {
                "schema_version": 1,
                "status": "PASS",
                "phase": "M1-08_CUDA_PILOT_ONLY",
                "protocol_id": restore["protocol_id"],
                "configuration": "P1",
                "seed": 42,
                "protocol_sha256": protocol_sha,
                "dataset_revision": revision,
                "manifest_sha256": manifest_sha,
                "preparation_lock_sha256": restore["preparation_lock_sha256"],
                "pre_fit_audit_sha256": restore["pre_fit_audit_sha256"],
                "pilot_code_sha256": restore["pilot_code_sha256"],
                "trainer_sha256": restore["trainer_sha256"],
                "preprocessing_sha256": restore["preprocessing_sha256"],
                "restore_verification_sha256": restore_sha,
                "pilot": pilot,
                "checkpoint": {
                    "metadata_sha256": restore["checkpoint_metadata_sha256"],
                    "weights_sha256": weights_sha,
                },
            }
            from nlp_btl.pilot import _atomic_json
            run_metadata_path = pilot_dir / "run_metadata.json"
            _atomic_json(run_metadata_path, {"status": "CREATED", "completed": False})
            _publish_pilot_success(pilot_dir, json.loads(run_metadata_path.read_text(encoding="utf-8")), pilot_receipt)

            source_hashes = {
                "protocol_sha256": _sha256((preparation / "protocol.json").read_bytes()),
                "manifest_sha256": _sha256((preparation / "data_manifest.json").read_bytes()),
                "preparation_lock_sha256": _sha256((preparation / "preparation_lock.json").read_bytes()),
                "trainer_sha256": _sha256(Path(phobert_module.__file__).read_bytes()),
                "pilot_code_sha256": _sha256(Path(phobert_module.__file__).with_name("pilot.py").read_bytes()),
                "preprocessing_sha256": _sha256(Path(preprocessing_module.__file__).read_bytes()),
                "reviewer_trust_anchor_sha256": _sha256((preparation / "m1_09_trusted_reviewers.json").read_bytes()),
                "pilot_notebook_source_sha256": _validate_submitted_pilot_notebook(
                    json.loads((ROOT / "staging/kernel/m1_08_cuda_pilot.ipynb").read_text(encoding="utf-8"))
                ),
            }
            source_files = {
                "src/nlp_btl/phobert.py": source_hashes["trainer_sha256"],
                "src/nlp_btl/pilot.py": source_hashes["pilot_code_sha256"],
                "src/nlp_btl/preprocessing.py": source_hashes["preprocessing_sha256"],
                "preparation/protocol.json": source_hashes["protocol_sha256"],
                "preparation/data_manifest.json": source_hashes["manifest_sha256"],
                "preparation/preparation_lock.json": source_hashes["preparation_lock_sha256"],
                "preparation/m1_09_trusted_reviewers.json": source_hashes["reviewer_trust_anchor_sha256"],
            }
            source_files.update(
                {
                    relative: _sha256((ROOT / Path(*relative.split("/"))).read_bytes())
                    for relative in M1_09_SOURCE_BUNDLE_ALLOWLIST
                }
            )
            bundle_id = "example-owner/viocd-segmentation-source-fixture"
            source_bundle = {
                "schema_version": 1,
                "status": "PASS",
                "dataset_id": bundle_id,
                "source_hashes": source_hashes,
                "files": source_files,
            }
            notebook = json.loads(
                (ROOT / "staging/kernel/m1_08_cuda_pilot.ipynb").read_text(encoding="utf-8")
            )
            artifacts = {
                "kaggle_execution_log.txt": json.dumps({
                    "schema_version": 1,
                    "status": "PASS",
                    "phase": "M1-08_CUDA_PILOT_ONLY",
                    "source_hashes": source_hashes,
                    "pilot_files": {},
                }).encode("utf-8"),
                "submitted_source.ipynb": json.dumps(notebook).encode("utf-8"),
                "kernel_metadata.json": json.dumps({
                    "id": "example-owner/fixture-kernel",
                    "code_file": "fixture-kernel.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_tpu": False,
                    "enable_internet": True,
                    "machine_shape": "NvidiaTeslaT4",
                    "dataset_sources": [bundle_id],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
                }).encode("utf-8"),
                "source_bundle_manifest.json": json.dumps(source_bundle).encode("utf-8"),
                "review_transcript.md": b"Independent reviewer transcript for the synthetic fixture.\n",
            }
            files = {}
            for relative, content in artifacts.items():
                target = evidence_root / relative
                target.write_bytes(content)
                files[relative] = _sha256(content)
            for relative in (
                "pilot_commit.json",
                "pilot_receipt.json",
                "run_metadata.json",
                "pre_fit_audit.json",
                "restore_verification.json",
                "checkpoint/checkpoint.json",
                "checkpoint/trainable_state.pt",
            ):
                target = pilot_dir / relative
                files[f"pilot_output/{relative}"] = _sha256(target.read_bytes())
            commit_sha = files["pilot_output/pilot_commit.json"]
            receipt_sha = files["pilot_output/pilot_receipt.json"]
            log_path = evidence_root / "kaggle_execution_log.txt"
            execution_log = json.loads(log_path.read_text(encoding="utf-8"))
            pilot_files = {
                "pilot_commit.json": commit_sha,
                "pilot_receipt.json": receipt_sha,
                "run_metadata.json": files["pilot_output/run_metadata.json"],
                "pre_fit_audit.json": files["pilot_output/pre_fit_audit.json"],
                "restore_verification.json": files["pilot_output/restore_verification.json"],
                "checkpoint/checkpoint.json": files["pilot_output/checkpoint/checkpoint.json"],
                "checkpoint/trainable_state.pt": files["pilot_output/checkpoint/trainable_state.pt"],
            }
            if pilot_files_mutation == "missing":
                pilot_files.pop("pilot_commit.json")
            elif pilot_files_mutation == "extra":
                pilot_files["unexpected.json"] = "0" * 64
            elif pilot_files_mutation == "wrong_hash":
                pilot_files["pilot_commit.json"] = "0" * 64
            elif pilot_files_mutation is not None:
                raise AssertionError(f"unknown pilot_files mutation: {pilot_files_mutation}")
            execution_log["pilot_files"] = pilot_files
            log_path.write_text(json.dumps(execution_log), encoding="utf-8")
            files["kaggle_execution_log.txt"] = _sha256(log_path.read_bytes())
            execution_receipt = {
                "schema_version": 1,
                "status": "PASS",
                "kaggle_execution": {
                    "ref": "example-owner/fixture-kernel",
                    "version": 1,
                    "status": "COMPLETE",
                },
                "kernel_metadata_sha256": files["kernel_metadata.json"],
                "submitted_source_sha256": files["submitted_source.ipynb"],
                "execution_log_sha256": files["kaggle_execution_log.txt"],
                "source_bundle_manifest_sha256": files["source_bundle_manifest.json"],
                "pilot_commit_sha256": commit_sha,
                "pilot_receipt_sha256": receipt_sha,
                "restore_verification_sha256": files["pilot_output/restore_verification.json"],
            }
            execution_receipt_path = evidence_root / "kaggle_execution_receipt.json"
            execution_receipt_path.write_text(json.dumps(execution_receipt), encoding="utf-8")
            files["kaggle_execution_receipt.json"] = _sha256(execution_receipt_path.read_bytes())
            evidence = {
                "schema_version": 1,
                "status": "PASS",
                "phase": "M1-09_GATE_B_EVIDENCE",
                "kaggle_execution": {"ref": "example-owner/fixture-kernel", "version": 1, "status": "COMPLETE"},
                "pilot_artifact_relative_path": "pilot_output",
                "source_hashes": source_hashes,
                "files": files,
                "pilot_commit_sha256": commit_sha,
                "pilot_receipt_sha256": receipt_sha,
                "restore_verification_sha256": files["pilot_output/restore_verification.json"],
            }
            reviewed_at_utc = "2026-11-17T12:00:00+00:00"
            review_basis_sha256 = phobert_module._m1_09_review_basis_sha256(evidence)
            reviewer_report = {
                "schema_version": 1,
                "status": "PASS",
                "phase": "M1-09_REVIEW_ATTESTATION",
                "reviewer_id": "test-reviewer-01",
                "reviewer_role": "independent QA",
                "reviewed_at_utc": reviewed_at_utc,
                "review_transcript_file": "review_transcript.md",
                "review_transcript_sha256": files["review_transcript.md"],
                "review_basis_sha256": review_basis_sha256,
                "kaggle_execution": evidence["kaggle_execution"],
                "source_hashes": source_hashes,
                "reviewed_files": {
                    relative: digest
                    for relative, digest in files.items()
                    if relative not in {"review_transcript.md", "review_attestation.json"}
                },
                "pilot_commit_sha256": commit_sha,
                "pilot_receipt_sha256": receipt_sha,
                "restore_verification_sha256": files["pilot_output/restore_verification.json"],
                "checks_passed": [
                    "kernel_ref_version_status",
                    "version_bound_execution_receipt",
                    "submitted_notebook_source",
                    "source_bundle_hashes",
                    "runtime_gpu_cuda_java_rdr",
                    "train_only_optimizer_step",
                    "checkpoint_restore_match",
                    "exact_output_inventory_and_hashes",
                    "no_raw_review_text_or_row_ids",
                ],
                "blocking_findings": [],
                "non_blocking_findings": [],
                "scope_limits": [
                    "Review identity is procedural and transcript-backed; this attestation is not cryptographically signed.",
                    "The reviewer did not rerun the Kaggle training job or independently reproduce the checkpoint restore.",
                ],
            }
            report_path = evidence_root / "review_attestation.json"
            report_path.write_text(json.dumps(reviewer_report), encoding="utf-8")
            report_sha256 = _sha256(report_path.read_bytes())
            files["review_attestation.json"] = report_sha256
            evidence["files"] = files
            evidence["independent_review"] = {
                "status": "PASS",
                "reviewer_id": "test-reviewer-01",
                "reviewer_role": "independent QA",
                "reviewed_at_utc": reviewed_at_utc,
                "report_file": "review_attestation.json",
                "report_sha256": report_sha256,
                "transcript_file": "review_transcript.md",
                "transcript_sha256": files["review_transcript.md"],
                "review_basis_sha256": review_basis_sha256,
            }
            evidence_path = evidence_root / "evidence_manifest.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            readiness = {
                "schema_version": 1,
                "status": "TRAINING_READY",
                "protocol_id": PROTOCOL["protocol_id"],
                "protocol_sha256": protocol_sha,
                "dataset_revision": revision,
                "manifest_sha256": manifest_sha,
                "model_id": PROTOCOL["model"]["id"],
                "model_revision": PROTOCOL["model"]["revision"],
                "model_config_sha256": PROTOCOL["model"]["config_sha256"],
                "preparation_lock_sha256": _sha256((preparation / "preparation_lock.json").read_bytes()),
                "trainer_sha256": _sha256(Path(phobert_module.__file__).read_bytes()),
                "pilot_code_sha256": _sha256(Path(phobert_module.__file__).with_name("pilot.py").read_bytes()),
                "preprocessing_sha256": _sha256(Path(preprocessing_module.__file__).read_bytes()),
                "pre_fit_audit_sha256": _sha256(audit_path.read_bytes()),
                "evidence_manifest_sha256": _sha256(evidence_path.read_bytes()),
                "review_attestation_sha256": report_sha256,
                "review_transcript_sha256": files["review_transcript.md"],
                "pilot_commit_sha256": commit_sha,
                "pilot_receipt_sha256": receipt_sha,
                "restore_verification_sha256": files["pilot_output/restore_verification.json"],
                "pilot": pilot,
                "runtime": {**audit_data["runtime"], "packages": audit_data["runtime_packages"]},
            }
            readiness_path = folder / f"{name}-readiness.json"
            readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
            return audit_path, readiness_path, evidence_path

        private_root = ROOT / "results" / "m1-08"
        private_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private_root) as temporary:
            folder = Path(temporary)
            audit_path, readiness_path, evidence_path = write_pair(folder, "valid", audit)
            with (
                mock.patch("nlp_btl.phobert._verify_rdr_assets", return_value=fake_assets),
                mock.patch("nlp_btl.phobert._verify_rdr_wrapper", return_value=fake_wrapper),
            ):
                readiness, verified_audit, _, audit_sha = _verify_training_readiness(
                    readiness_path,
                    audit_path,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    evidence_path,
                )
            self.assertEqual(readiness["status"], "TRAINING_READY")
            self.assertEqual(verified_audit["status"], "PASS")
            self.assertEqual(audit_sha, _sha256(audit_path.read_bytes()))

            for mutation in ("missing", "extra", "wrong_hash"):
                with self.subTest(pilot_files_mutation=mutation):
                    _, malformed_readiness, malformed_evidence = write_pair(
                        folder,
                        f"pilot-files-{mutation}",
                        audit,
                        pilot_files_mutation=mutation,
                    )
                    with self.assertRaisesRegex(
                        PhoBertTrainingError,
                        "execution log does not identify the committed pilot artifacts",
                    ):
                        _verify_training_readiness(
                            malformed_readiness,
                            malformed_evidence.parent / "pilot_output/pre_fit_audit.json",
                            preparation,
                            PROTOCOL,
                            bundle,
                            protocol_sha,
                            manifest_sha,
                            malformed_evidence,
                        )

            readiness_bytes = readiness_path.read_bytes()
            for malformed_version in (True, 1.0):
                malformed_readiness = json.loads(readiness_bytes)
                malformed_readiness["schema_version"] = malformed_version
                readiness_path.write_text(json.dumps(malformed_readiness), encoding="utf-8")
                with self.subTest(readiness_schema_version=malformed_version):
                    with self.assertRaisesRegex(PhoBertTrainingError, "schema version must be an integer"):
                        _verify_training_readiness(
                            readiness_path,
                            audit_path,
                            preparation,
                            PROTOCOL,
                            bundle,
                            protocol_sha,
                            manifest_sha,
                            evidence_path,
                        )
            readiness_path.write_bytes(readiness_bytes)

            unbound_readiness = json.loads(readiness_bytes)
            unbound_readiness["review_attestation_sha256"] = "0" * 64
            readiness_path.write_text(json.dumps(unbound_readiness), encoding="utf-8")
            with self.assertRaisesRegex(PhoBertTrainingError, "readiness receipt is missing or differs"):
                _verify_training_readiness(
                    readiness_path,
                    audit_path,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    evidence_path,
                )
            readiness_path.write_bytes(readiness_bytes)

            evidence_bytes = evidence_path.read_bytes()
            for malformed_version in (True, 1.0):
                malformed_evidence = json.loads(evidence_bytes)
                malformed_evidence["schema_version"] = malformed_version
                evidence_path.write_text(json.dumps(malformed_evidence), encoding="utf-8")
                malformed_readiness = json.loads(readiness_bytes)
                malformed_readiness["evidence_manifest_sha256"] = _sha256(evidence_path.read_bytes())
                readiness_path.write_text(json.dumps(malformed_readiness), encoding="utf-8")
                with self.subTest(evidence_schema_version=malformed_version):
                    with self.assertRaisesRegex(PhoBertTrainingError, "evidence manifest is not passing"):
                        _verify_training_readiness(
                            readiness_path,
                            audit_path,
                            preparation,
                            PROTOCOL,
                            bundle,
                            protocol_sha,
                            manifest_sha,
                            evidence_path,
                        )
            evidence_path.write_bytes(evidence_bytes)
            readiness_path.write_bytes(readiness_bytes)

            mismatched_execution = json.loads(evidence_bytes)
            mismatched_execution["kaggle_execution"]["version"] = 2
            evidence_path.write_text(json.dumps(mismatched_execution), encoding="utf-8")
            mismatched_readiness = json.loads(readiness_bytes)
            mismatched_readiness["evidence_manifest_sha256"] = _sha256(evidence_path.read_bytes())
            readiness_path.write_text(json.dumps(mismatched_readiness), encoding="utf-8")
            with self.assertRaisesRegex(PhoBertTrainingError, "execution receipt does not bind the claimed version"):
                _verify_training_readiness(
                    readiness_path,
                    audit_path,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    evidence_path,
                )
            evidence_path.write_bytes(evidence_bytes)
            readiness_path.write_bytes(readiness_bytes)

            report_path = evidence_path.parent / "review_attestation.json"
            report_bytes = report_path.read_bytes()
            for malformed_version in (True, 1.0):
                malformed_report = json.loads(report_bytes)
                malformed_report["schema_version"] = malformed_version
                report_path.write_text(json.dumps(malformed_report), encoding="utf-8")
                malformed_evidence = json.loads(evidence_bytes)
                report_sha256 = _sha256(report_path.read_bytes())
                malformed_evidence["files"]["review_attestation.json"] = report_sha256
                malformed_evidence["independent_review"]["report_sha256"] = report_sha256
                evidence_path.write_text(json.dumps(malformed_evidence), encoding="utf-8")
                malformed_readiness = json.loads(readiness_bytes)
                malformed_readiness["evidence_manifest_sha256"] = _sha256(evidence_path.read_bytes())
                readiness_path.write_text(json.dumps(malformed_readiness), encoding="utf-8")
                with self.subTest(reviewer_report_schema_version=malformed_version):
                    with self.assertRaisesRegex(PhoBertTrainingError, "reviewer report differs from the evidence"):
                        _verify_training_readiness(
                            readiness_path,
                            audit_path,
                            preparation,
                            PROTOCOL,
                            bundle,
                            protocol_sha,
                            manifest_sha,
                            evidence_path,
                        )
            report_path.write_bytes(report_bytes)
            evidence_path.write_bytes(evidence_bytes)
            readiness_path.write_bytes(readiness_bytes)

            forged_evidence = json.loads(evidence_bytes)
            forged_evidence["independent_review"]["reviewed_at_utc"] = "2026-11-17T12:01:00+00:00"
            evidence_path.write_text(json.dumps(forged_evidence), encoding="utf-8")
            forged_readiness = json.loads(readiness_bytes)
            forged_readiness["evidence_manifest_sha256"] = _sha256(evidence_path.read_bytes())
            readiness_path.write_text(json.dumps(forged_readiness), encoding="utf-8")
            with self.assertRaisesRegex(PhoBertTrainingError, "reviewer report differs from the evidence"):
                _verify_training_readiness(
                    readiness_path,
                    audit_path,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    evidence_path,
                )
            evidence_path.write_bytes(evidence_bytes)
            readiness_path.write_bytes(readiness_bytes)

            with mock.patch("nlp_btl.phobert._load_authorized_m1_09_reviewer", return_value=None):
                with self.assertRaisesRegex(PhoBertTrainingError, "reviewer is not authorized in the pinned reviewer registry"):
                    _verify_training_readiness(
                        readiness_path,
                        audit_path,
                        preparation,
                        PROTOCOL,
                        bundle,
                        protocol_sha,
                        manifest_sha,
                        evidence_path,
                    )

            execution_log = evidence_path.parent / "kaggle_execution_log.txt"
            original_log = execution_log.read_bytes()
            execution_log.write_bytes(original_log + b"tampered\n")
            with self.assertRaisesRegex(PhoBertTrainingError, "evidence file is missing or its SHA-256 differs"):
                _verify_training_readiness(
                    readiness_path,
                    audit_path,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    evidence_path,
                )
            execution_log.write_bytes(original_log)

            stale_pilot_audit, stale_pilot_ready, stale_pilot_evidence = write_pair(
                folder,
                "stale-pilot",
                audit,
                pilot_identity_overrides={"pilot_code_sha256": "0" * 64},
            )
            with self.assertRaisesRegex(PhoBertTrainingError, "pilot receipt is stale"):
                _verify_training_readiness(
                    stale_pilot_ready,
                    stale_pilot_audit,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    stale_pilot_evidence,
                )

            stale = copy.deepcopy(audit)
            stale["preprocessing_code_sha256"] = "stale-code"
            stale_audit, stale_ready, stale_evidence = write_pair(folder, "stale", stale)
            with self.assertRaisesRegex(PhoBertTrainingError, "(?i)pre-fit audit receipt is missing"):
                _verify_training_readiness(
                    stale_ready, stale_audit, preparation, PROTOCOL, bundle, protocol_sha, manifest_sha, stale_evidence
                )

            for malformed_version in (True, 1.0):
                malformed_schema = copy.deepcopy(audit)
                malformed_schema["schema_version"] = malformed_version
                malformed_audit, malformed_ready, malformed_evidence = write_pair(
                    folder, f"malformed-schema-{malformed_version}", malformed_schema
                )
                with self.subTest(audit_schema_version=malformed_version):
                    with self.assertRaisesRegex(PhoBertTrainingError, "(?i)pre-fit audit receipt is missing"):
                        _verify_training_readiness(
                            malformed_ready,
                            malformed_audit,
                            preparation,
                            PROTOCOL,
                            bundle,
                            protocol_sha,
                            manifest_sha,
                            malformed_evidence,
                        )

            collision = copy.deepcopy(audit)
            collision["cross_split_collision_found"] = True
            collision_audit, collision_ready, collision_evidence = write_pair(folder, "collision", collision)
            with self.assertRaisesRegex(PhoBertTrainingError, "(?i)pre-fit audit receipt is missing"):
                _verify_training_readiness(
                    collision_ready, collision_audit, preparation, PROTOCOL, bundle, protocol_sha, manifest_sha, collision_evidence
                )

            missing_pair = copy.deepcopy(audit)
            del missing_pair["tokenizer_inputs"]["P0"]["cross_split_collisions"]["truncated"]["val__test"]
            missing_pair_audit, missing_pair_ready, missing_pair_evidence = write_pair(folder, "missing-pair", missing_pair)
            with self.assertRaisesRegex(PhoBertTrainingError, "differs from locked segmentation/tokenizer"):
                _verify_training_readiness(
                    missing_pair_ready,
                    missing_pair_audit,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    missing_pair_evidence,
                )

            missing_split = copy.deepcopy(audit)
            missing_split["segmentation"]["underthesea"]["splits"].pop()
            missing_split_audit, missing_split_ready, missing_split_evidence = write_pair(folder, "missing-split", missing_split)
            with self.assertRaisesRegex(PhoBertTrainingError, "differs from locked segmentation/tokenizer"):
                _verify_training_readiness(
                    missing_split_ready,
                    missing_split_audit,
                    preparation,
                    PROTOCOL,
                    bundle,
                    protocol_sha,
                    manifest_sha,
                    missing_split_evidence,
                )

            with (
                mock.patch("nlp_btl.phobert._verify_rdr_assets", return_value={"tampered": "asset"}),
                mock.patch("nlp_btl.phobert._verify_rdr_wrapper", return_value=fake_wrapper),
            ):
                with self.assertRaisesRegex(PhoBertTrainingError, "RDR assets or wrapper differ"):
                    _verify_training_readiness(
                        readiness_path,
                        audit_path,
                        preparation,
                        PROTOCOL,
                        bundle,
                        protocol_sha,
                        manifest_sha,
                        evidence_path,
                    )


if __name__ == "__main__":
    unittest.main()
