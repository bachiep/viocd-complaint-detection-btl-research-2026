"""Locked PhoBERT partial-fine-tuning development path.

The real fit path is CUDA-only. Small helper seams are intentionally usable by
CPU tests, but they do not provide a CPU training fallback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

from . import preprocessing
from . import data as data_module
from .baseline import _read_development_protocol
from .data import DataIntegrityError, DatasetBundle, ReviewRecord, _sha256, normalize_review
from .preprocessing import (
    ProcessedReview,
    RDRSegmenter,
    UndertheseaSegmenter,
    _load_pinned_tokenizer,
    _read_locked_json,
    _verify_rdr_assets,
    _verify_rdr_wrapper,
    preprocess_records,
    tokenize_batch,
)


class PhoBertTrainingError(DataIntegrityError):
    """Raised when the locked PhoBERT development contract cannot be met."""


class PhoBertTrainingBlocked(PhoBertTrainingError):
    """Raised when a required runtime gate has not been passed."""


_SAFE_ROBERTA_KEY_NAME_PATTERN = re.compile(
    r"roberta(?:\.[A-Za-z0-9_]+)+\Z", flags=re.ASCII
)
_SAFE_ROBERTA_KEY_NAME_MAX_COUNT = 8
_SAFE_ROBERTA_KEY_NAME_MAX_LENGTH = 192


M1_09_SOURCE_BUNDLE_ALLOWLIST = frozenset(
    {
        "src/nlp_btl/__init__.py",
        "src/nlp_btl/baseline.py",
        "src/nlp_btl/data.py",
        "src/nlp_btl/phobert.py",
        "src/nlp_btl/pilot.py",
        "src/nlp_btl/preprocessing.py",
        "preparation/data_manifest.json",
        "preparation/m1_09_trusted_reviewers.json",
        "preparation/preparation_lock.json",
        "preparation/protocol.json",
        "preparation/rdr_assets_verification.json",
        "preparation/segmentation_audit.json",
        "preparation/tokenizer_input_audit.json",
        "preparation/assets/python_rdr/py_vncorenlp-0.1.4.tar.gz",
        "preparation/assets/python_rdr/py_vncorenlp/__init__.py",
        "preparation/assets/python_rdr/py_vncorenlp/vncorenlp.py",
        "THIRD_PARTY_NOTICES.md",
        "requirements-gate-b.txt",
    }
)


CHECKPOINT_BINDING_FIELDS = frozenset(
    {
        "configuration",
        "seed",
        "dataset_revision",
        "manifest_sha256",
        "protocol_sha256",
        "preparation_lock_sha256",
        "pre_fit_audit_sha256",
        "readiness_receipt_sha256",
        "model_id",
        "model_revision",
        "model_config_sha256",
        "frozen_base_state_sha256",
        "segmenter_id",
        "tokenizer_revision",
        "tokenizer_audit_sha256",
        "max_length",
        "trainable_scope",
        "trainer_sha256",
        "preprocessing_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class PhoBertPrediction:
    """One source-keyed validation prediction; review text is never retained."""

    key: tuple[str, str, int]
    y_true: int
    y_pred: int


@dataclass(frozen=True, slots=True)
class PhoBertDevelopmentResult:
    """Selected validation checkpoint and its non-text audit metadata."""

    config_id: str
    seed: int
    selected_epoch: int
    validation_macro_f1: float
    validation_predictions: tuple[PhoBertPrediction, ...]
    checkpoint_dir: Path
    history: tuple[Mapping[str, object], ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    protocol_sha256: str
    manifest_sha256: str
    model_revision: str
    segmenter_id: str


def _assert_locked_fit(
    protocol: Mapping[str, object], config_id: str, seed: int
) -> tuple[dict[str, object], str]:
    matrix = protocol.get("experiment_matrix")
    if not isinstance(matrix, list):
        raise PhoBertTrainingError("Experiment matrix is missing")
    configuration = next(
        (entry for entry in matrix if isinstance(entry, dict) and entry.get("id") == config_id),
        None,
    )
    if not isinstance(configuration, dict) or configuration.get("kind") != "phobert":
        raise PhoBertTrainingError("Only locked PhoBERT configurations P0 and P1 are supported")
    if seed not in configuration.get("seeds", []):
        raise PhoBertTrainingError("Seed is not declared for the selected configuration")
    expected_segmenter = {"P0": "underthesea", "P1": "vncorenlp_rdr"}.get(config_id)
    if configuration.get("segmentation") != expected_segmenter:
        raise PhoBertTrainingError("Configuration segmentation differs from the locked protocol")
    if configuration.get("trainable_scope") != "last_two_encoder_layers_and_classifier":
        raise PhoBertTrainingError("Trainable scope differs from the locked protocol")

    training = protocol.get("training")
    model = protocol.get("model")
    if not isinstance(training, dict) or not isinstance(model, dict):
        raise PhoBertTrainingError("Locked model or training settings are missing")
    optimizer = training.get("optimizer")
    schedule = training.get("schedule")
    dataloader = training.get("dataloader")
    if not isinstance(optimizer, dict) or not isinstance(schedule, dict) or not isinstance(dataloader, dict):
        raise PhoBertTrainingError("Locked optimizer, schedule, or loader settings are invalid")

    exact = (
        model.get("id") == "vinai/phobert-base"
        and model.get("num_labels") == 2
        and model.get("tokenizer_use_fast") is False
        and training.get("epochs") == 3
        and training.get("max_length") == 128
        and training.get("micro_batch") == 8
        and training.get("gradient_accumulation") == 4
        and training.get("nominal_effective_batch") == 32
        and training.get("gradient_normalization")
        == "sample_weighted_actual_accumulation_group_size_including_final_partial_group"
        and training.get("learning_rate") == 3e-5
        and optimizer.get("name") == "torch.optim.AdamW"
        and optimizer.get("weight_decay") == 0.01
        and optimizer.get("fused") is True
        and optimizer.get("betas") == [0.9, 0.999]
        and optimizer.get("eps") == 1e-8
        and optimizer.get("amsgrad") is False
        and schedule.get("name") == "linear_with_warmup"
        and schedule.get("warmup_steps") == "floor(0.1 * total_optimizer_steps)"
        and schedule.get("steps_per_epoch") == "ceil(number_of_microbatches / 4)"
        and schedule.get("advance_after") == "successful AMP optimizer step only"
        and training.get("max_grad_norm") == 1.0
        and training.get("precision") == "CUDA autocast float16 with torch.amp.GradScaler('cuda')"
        and training.get("allow_tf32") is True
        and training.get("threads") == 4
        and dataloader.get("shuffle_train") is True
        and dataloader.get("shuffle_val_test") is False
        and dataloader.get("num_workers") == 0
        and dataloader.get("pin_memory") is True
        and dataloader.get("padding") == "dynamic"
        and dataloader.get("pad_to_multiple_of") == 8
    )
    if not exact:
        raise PhoBertTrainingError("PhoBERT fit parameters differ from the locked protocol")
    return configuration, expected_segmenter or ""


def _validate_development_bundle(
    bundle: DatasetBundle,
    protocol: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    if set(bundle.splits) != {"train", "val"}:
        raise PhoBertTrainingError("PhoBERT development accepts only train and validation splits")
    if bundle.loader_sha256 != _sha256(Path(data_module.__file__).resolve().read_bytes()):
        raise PhoBertTrainingError("Dataset bundle was not produced by the current verified loader")
    manifest_splits = manifest.get("splits")
    domains = protocol.get("dataset", {}).get("domains")
    if not isinstance(manifest_splits, dict) or not isinstance(domains, list):
        raise PhoBertTrainingError("Locked dataset membership manifest is invalid")
    if set(bundle.audit) != {"train", "val"}:
        raise PhoBertTrainingError("Dataset bundle audit does not contain the development splits")
    train = bundle.splits["train"]
    validation = bundle.splits["val"]
    if not train or not validation:
        raise PhoBertTrainingError("PhoBERT development requires non-empty train and validation splits")
    for split, records in (("train", train), ("val", validation)):
        if any(not isinstance(record, ReviewRecord) for record in records):
            raise PhoBertTrainingError(f"PhoBERT {split} contains an invalid record")
        if any(record.split != split or record.revision != bundle.revision for record in records):
            raise PhoBertTrainingError(f"PhoBERT {split} records differ from the verified bundle")
        if any(record.label not in (0, 1) for record in records):
            raise PhoBertTrainingError(f"PhoBERT {split} contains a non-binary label")
        if len({record.key for record in records}) != len(records):
            raise PhoBertTrainingError(f"PhoBERT {split} source keys are not unique")
        expected_split = manifest_splits.get(split)
        audit = bundle.audit.get(split)
        if not isinstance(expected_split, dict) or not isinstance(audit, Mapping):
            raise PhoBertTrainingError(f"PhoBERT {split} has no trusted membership contract")
        if len(records) != expected_split.get("retained_rows"):
            raise PhoBertTrainingError(f"PhoBERT {split} row count differs from the manifest")
        for field in (
            "source_sha256",
            "source_bytes",
            "retained_rows",
            "ordered_membership_digest_sha256",
        ):
            if audit.get(field) != expected_split.get(field):
                raise PhoBertTrainingError(f"PhoBERT {split} loader audit differs from manifest: {field}")
        membership = "".join(
            f"{split}\t{record.source_row}\t{record.label}\t{record.domain}\t"
            f"{_sha256(record.text.encode('utf-8'))}\n"
            for record in sorted(records, key=lambda item: item.source_row)
        ).encode("utf-8")
        if _sha256(membership) != expected_split.get("ordered_membership_digest_sha256"):
            raise PhoBertTrainingError(f"PhoBERT {split} records differ from trusted membership")
        if any(
            record.domain not in domains or normalize_review(record.text) != record.text
            for record in records
        ):
            raise PhoBertTrainingError(f"PhoBERT {split} text/domain differs from data contract")
    if len({record.label for record in train}) != 2:
        raise PhoBertTrainingError("PhoBERT training split must contain both labels")


def configure_partial_finetuning(model: Any) -> tuple[str, ...]:
    """Freeze embeddings/encoder 0–9 and enable encoder 10–11/classifier."""
    roberta = getattr(model, "roberta", None)
    embeddings = getattr(roberta, "embeddings", None)
    encoder = getattr(roberta, "encoder", None)
    layers = getattr(encoder, "layer", None)
    classifier = getattr(model, "classifier", None)
    if embeddings is None or layers is None or classifier is None or len(layers) != 12:
        raise PhoBertTrainingError("Model architecture is not the locked 12-layer PhoBERT classifier")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers[10:12]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)

    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not names or any(
        not (name.startswith("roberta.encoder.layer.10.")
             or name.startswith("roberta.encoder.layer.11.")
             or name.startswith("classifier."))
        for name in names
    ):
        raise PhoBertTrainingError("Trainable parameter scope differs from the locked two-layer scope")
    if not any(name.startswith("roberta.encoder.layer.10.") for name in names):
        raise PhoBertTrainingError("Encoder layer 10 has no trainable parameters")
    if not any(name.startswith("roberta.encoder.layer.11.") for name in names):
        raise PhoBertTrainingError("Encoder layer 11 has no trainable parameters")
    if not any(name.startswith("classifier.") for name in names):
        raise PhoBertTrainingError("Classification head has no trainable parameters")
    return names


def frozen_base_state_sha256(model: Any) -> str:
    """Fingerprint frozen parameters and buffers to bind restore to the same base."""
    digest = hashlib.sha256()
    frozen_count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        _update_tensor_digest(digest, name, parameter.detach())
        frozen_count += 1
    for name, buffer in model.named_buffers():
        _update_tensor_digest(digest, f"buffer:{name}", buffer.detach())
    if frozen_count == 0:
        raise PhoBertTrainingError("Cannot fingerprint a model with no frozen base parameters")
    return digest.hexdigest()


def _update_tensor_digest(digest: Any, name: str, tensor: Any) -> None:
    import torch

    value = tensor.to(device="cpu").contiguous()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())


def set_partial_finetuning_train_mode(model: Any) -> None:
    """Enable training only for the two unfrozen blocks and classifier."""
    model.train()
    roberta = model.roberta
    roberta.embeddings.eval()
    for layer in roberta.encoder.layer[:10]:
        layer.eval()


def _normalize_accumulated_gradients_(optimizer: Any, sample_count: int) -> None:
    """Convert summed mean-loss gradients to a sample-weighted group mean."""
    if sample_count <= 0:
        raise PhoBertTrainingError("Gradient accumulation group has no examples")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.div_(sample_count)


def _amp_step_succeeded(previous_scale: float, updated_scale: float) -> bool:
    """GradScaler lowers its scale when it skips an optimizer step for overflow."""
    return updated_scale >= previous_scale


def _should_select_epoch(best_score: float | None, candidate_score: float) -> bool:
    """Strict comparison preserves the earliest epoch on an exact tie."""
    return best_score is None or candidate_score > best_score


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _save_trainable_state(
    model: Any, checkpoint_dir: Path, metadata: Mapping[str, object]
) -> dict[str, object]:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "trainable_state.pt"
    metadata_path = checkpoint_dir / "checkpoint.json"
    if weights_path.exists() != metadata_path.exists():
        raise PhoBertTrainingError("Checkpoint directory has an incomplete prior write")
    state = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise PhoBertTrainingError("Cannot save an empty trainable checkpoint")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=checkpoint_dir, prefix=".trainable_state.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, weights_path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    receipt = dict(metadata)
    receipt.update(
        {
            "schema_version": 1,
            "storage_scope": "trainable_parameters_only; reload pinned base before restore",
            "trainable_parameter_names": sorted(state),
            "trainable_parameter_count": sum(value.numel() for value in state.values()),
            "weights_file": weights_path.name,
            "weights_sha256": _sha256(weights_path.read_bytes()),
        }
    )
    _write_json_atomic(metadata_path, receipt)
    return receipt


def restore_trainable_state(
    model: Any, checkpoint_dir: str | Path, expected_binding: Mapping[str, object]
) -> dict[str, object]:
    """Restore trainable tensors into a freshly loaded, protocol-pinned base."""
    import torch

    if set(expected_binding) != CHECKPOINT_BINDING_FIELDS:
        raise PhoBertTrainingError("Restore requires the complete checkpoint identity binding")
    directory = Path(checkpoint_dir).resolve(strict=True)
    try:
        run_metadata_bytes = (directory.parent / "run_metadata.json").read_bytes()
        run_metadata = json.loads(run_metadata_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Checkpoint has no valid parent run metadata") from exc
    if not isinstance(run_metadata, dict) or run_metadata.get("status") != "FIT_COMPLETE":
        raise PhoBertTrainingError("Cannot restore a checkpoint from an incomplete or failed attempt")
    for key in CHECKPOINT_BINDING_FIELDS:
        if run_metadata.get(key) != expected_binding.get(key):
            raise PhoBertTrainingError(f"Parent run identity mismatch: {key}")
    actual_frozen_base_sha256 = frozen_base_state_sha256(model)
    if actual_frozen_base_sha256 != expected_binding["frozen_base_state_sha256"]:
        raise PhoBertTrainingError("Restore model frozen base differs from the checkpoint identity")
    try:
        checkpoint_metadata_bytes = (directory / "checkpoint.json").read_bytes()
        receipt = json.loads(checkpoint_metadata_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Cannot read PhoBERT checkpoint metadata") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise PhoBertTrainingError("PhoBERT checkpoint metadata schema is invalid")
    if run_metadata.get("checkpoint_metadata_sha256") != _sha256(checkpoint_metadata_bytes):
        raise PhoBertTrainingError("Parent run checkpoint metadata hash mismatch")
    if run_metadata.get("selected_epoch") != receipt.get("selected_epoch"):
        raise PhoBertTrainingError("Parent run and checkpoint selected epochs differ")
    if run_metadata.get("validation_macro_f1") != receipt.get("validation_macro_f1"):
        raise PhoBertTrainingError("Parent run and checkpoint validation scores differ")
    for key, expected in expected_binding.items():
        if receipt.get(key) != expected:
            raise PhoBertTrainingError(f"PhoBERT checkpoint binding mismatch: {key}")
    if receipt.get("frozen_base_state_sha256") != actual_frozen_base_sha256:
        raise PhoBertTrainingError("Checkpoint frozen base fingerprint differs from the restore model")

    weights_path = directory / "trainable_state.pt"
    if receipt.get("weights_file") != weights_path.name:
        raise PhoBertTrainingError("PhoBERT checkpoint weight filename is invalid")
    try:
        weights_bytes = weights_path.read_bytes()
    except OSError as exc:
        raise PhoBertTrainingError("PhoBERT checkpoint weights are missing") from exc
    if _sha256(weights_bytes) != receipt.get("weights_sha256"):
        raise PhoBertTrainingError("PhoBERT checkpoint weight hash mismatch")
    if run_metadata.get("checkpoint_weights_sha256") != receipt.get("weights_sha256"):
        raise PhoBertTrainingError("Parent run checkpoint weight hash mismatch")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PhoBertTrainingError("PhoBERT checkpoint weights cannot be safely loaded") from exc
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise PhoBertTrainingError("PhoBERT checkpoint state schema is invalid")
    if receipt.get("storage_scope") != "trainable_parameters_only; reload pinned base before restore":
        raise PhoBertTrainingError("PhoBERT checkpoint storage scope is invalid")

    parameters = dict(model.named_parameters())
    trainable_names = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != trainable_names or sorted(state) != receipt.get("trainable_parameter_names"):
        raise PhoBertTrainingError("Checkpoint trainable parameter set differs from the fresh model")
    if sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)) != receipt.get(
        "trainable_parameter_count"
    ):
        raise PhoBertTrainingError("Checkpoint trainable parameter count is invalid")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise PhoBertTrainingError(f"Checkpoint tensor shape or dtype mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device))
    return receipt


def validate_phobert_predictions(
    records: Sequence[ReviewRecord], predictions: Sequence[PhoBertPrediction]
) -> None:
    if not records or not predictions:
        raise PhoBertTrainingError("PhoBERT validation prediction population is empty")
    expected: dict[tuple[str, str, int], int] = {}
    for record in records:
        if not isinstance(record, ReviewRecord) or record.split != "val":
            raise PhoBertTrainingError("PhoBERT prediction input contains a non-validation row")
        if record.key in expected:
            raise PhoBertTrainingError("PhoBERT validation source keys are not unique")
        expected[record.key] = record.label
    observed: dict[tuple[str, str, int], PhoBertPrediction] = {}
    for prediction in predictions:
        if not isinstance(prediction, PhoBertPrediction) or prediction.key in observed:
            raise PhoBertTrainingError("PhoBERT validation predictions have an invalid or duplicate key")
        observed[prediction.key] = prediction
    if observed.keys() != expected.keys():
        raise PhoBertTrainingError("PhoBERT validation predictions do not match source keys")
    for key, prediction in observed.items():
        if prediction.y_true != expected[key] or prediction.y_true not in (0, 1):
            raise PhoBertTrainingError("PhoBERT validation gold label differs from source")
        if prediction.y_pred not in (0, 1):
            raise PhoBertTrainingError("PhoBERT prediction is outside the binary label set")


def _read_tokenizer_audit(preparation: Path) -> dict[str, object]:
    try:
        lock = json.loads((preparation / "preparation_lock.json").read_text(encoding="utf-8"))
        audit_bytes = (preparation / "tokenizer_input_audit.json").read_bytes()
        audit = json.loads(audit_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Pinned tokenizer audit or preparation lock is unavailable") from exc
    hashes = lock.get("hashes") if isinstance(lock, dict) else None
    if not isinstance(hashes, dict) or hashes.get("tokenizer_input_audit.json") != _sha256(audit_bytes):
        raise PhoBertTrainingError("Tokenizer input audit differs from the preparation lock")
    model = audit.get("model_assets") if isinstance(audit, dict) else None
    if not isinstance(model, list):
        raise PhoBertTrainingError("Pinned tokenizer asset manifest is invalid")
    return audit


def _verify_preparation_lock(preparation: Path, protocol_path: Path, protocol_sha256: str) -> None:
    try:
        lock = json.loads((preparation / "preparation_lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Preparation lock is unavailable") from exc
    hashes = lock.get("hashes") if isinstance(lock, dict) else None
    if (
        protocol_path.parent != preparation
        or protocol_path.name != "protocol.json"
        or not isinstance(hashes, dict)
        or hashes.get("protocol.json") != protocol_sha256
    ):
        raise PhoBertTrainingError("Protocol differs from the frozen preparation lock")


def _read_receipt(path: str | Path, name: str) -> tuple[dict[str, object], str]:
    try:
        receipt_path = Path(path).resolve(strict=True)
        payload = receipt_path.read_bytes()
        receipt = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingBlocked(f"Required {name} receipt is unavailable or invalid") from exc
    if not isinstance(receipt, dict):
        raise PhoBertTrainingBlocked(f"Required {name} receipt has an invalid schema")
    return receipt, _sha256(payload)


def _json_receipt_matches(observed: object, expected: object) -> bool:
    try:
        observed_json = json.dumps(observed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return False
    return observed_json == expected_json


def _validate_submitted_pilot_notebook(notebook: object) -> str:
    """Validate one live P1/seed42 pilot call and return the notebook AST digest."""
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise PhoBertTrainingBlocked("M1-09 submitted notebook is not a valid Jupyter notebook")
    code_cells = [cell for cell in notebook["cells"] if isinstance(cell, dict) and cell.get("cell_type") == "code"]
    if not code_cells or any(
        not isinstance(cell.get("source"), (str, list))
        or isinstance(cell.get("source"), list)
        and any(not isinstance(line, str) for line in cell["source"])
        for cell in code_cells
    ):
        raise PhoBertTrainingBlocked("M1-09 submitted notebook has no valid code cells")
    notebook_source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in code_cells
    )
    try:
        module = ast.parse(notebook_source, filename="<submitted Kaggle notebook>")
    except SyntaxError as exc:
        raise PhoBertTrainingBlocked("M1-09 submitted pilot notebook has a syntax error") from exc

    pilot_imports = [
        statement for statement in module.body
        if isinstance(statement, ast.ImportFrom) and statement.module == "nlp_btl.pilot"
    ]
    if (
        len(pilot_imports) != 1
        or len(pilot_imports[0].names) != 1
        or pilot_imports[0].names[0].name != "run_cuda_pilot"
        or pilot_imports[0].names[0].asname is not None
    ):
        raise PhoBertTrainingBlocked("M1-09 notebook must import the production run_cuda_pilot binding directly")
    pilot_bindings = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            pilot_bindings.extend(
                alias for alias in node.names
                if (alias.asname or alias.name) == "run_cuda_pilot"
            )
        elif isinstance(node, ast.Import):
            pilot_bindings.extend(
                alias for alias in node.names
                if (alias.asname or alias.name.split(".", 1)[0]) == "run_cuda_pilot"
            )
    if len(pilot_bindings) != 1 or pilot_bindings[0] is not pilot_imports[0].names[0]:
        raise PhoBertTrainingBlocked("M1-09 notebook has an ambiguous pilot import binding")

    research_runner_referenced = any(
        (isinstance(node, ast.Name) and node.id == "run_phobert_development")
        or (isinstance(node, ast.Attribute) and node.attr == "run_phobert_development")
        or (
            isinstance(node, ast.alias)
            and node.name in {"run_phobert_development", "*"}
        )
        for node in ast.walk(module)
    )
    dynamic_execution_called = any(
        (
            isinstance(node, ast.alias)
            and node.name in {
                "eval", "exec", "compile", "getattr", "setattr", "delattr",
                "globals", "locals", "vars", "__import__",
            }
        )
        or isinstance(node, ast.Name) and node.id == "__builtins__"
        or isinstance(node, ast.Attribute)
        and node.attr in {"__dict__", "__getattribute__", "_getframe", "f_globals", "f_locals"}
        or isinstance(node, ast.Constant)
        and node.value in {
            "__builtins__", "__dict__", "__getattribute__", "_getframe", "f_globals", "f_locals",
            "vars", "globals", "locals", "DATASET_ID", "SOURCE_ROOT", "EXPECTED_FILES",
            "EXPECTED_FILE_HASHES", "SOURCE_FILE_NAMES",
        }
        or (
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id in {
                    "eval", "exec", "compile", "getattr", "setattr", "delattr",
                    "globals", "locals", "vars", "__import__",
                }
                or isinstance(node.func, ast.Attribute)
                and node.func.attr in {
                    "import_module", "__import__", "patch", "__setattr__", "__delattr__",
                    "__getattribute__", "vars",
                }
            )
        )
        for node in ast.walk(module)
    )
    def target_root_name(node: ast.AST) -> str | None:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    pilot_binding_mutated = any(
        (
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and target_root_name(node) == "run_cuda_pilot"
        )
        or (
            isinstance(node, ast.Attribute)
            and node.attr == "run_cuda_pilot"
            and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        or (
            isinstance(node, ast.Constant)
            and node.value in {"run_cuda_pilot", "nlp_btl.pilot"}
        )
        for node in ast.walk(module)
    )
    if research_runner_referenced:
        raise PhoBertTrainingBlocked("M1-09 submitted notebook must not reference the research fit runner")
    if dynamic_execution_called:
        raise PhoBertTrainingBlocked("M1-09 submitted notebook may not use dynamic code or import execution")
    if pilot_binding_mutated:
        raise PhoBertTrainingBlocked("M1-09 notebook may not mutate or reflectively replace the production pilot binding")
    if any(
        (
            isinstance(node, ast.Name)
            and node.id == "run_cuda_pilot"
            and isinstance(node.ctx, ast.Store)
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "run_cuda_pilot"
        )
        or (isinstance(node, ast.MatchAs) and node.name == "run_cuda_pilot")
        or (isinstance(node, ast.ExceptHandler) and node.name == "run_cuda_pilot")
        for node in ast.walk(module)
    ):
        raise PhoBertTrainingBlocked("M1-09 notebook must not shadow the production pilot binding")

    def call_name(call: ast.Call) -> str | None:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    pilot_calls = [
        node for node in ast.walk(module)
        if isinstance(node, ast.Call) and call_name(node) == "run_cuda_pilot"
    ]
    top_level_pilot_calls = []
    for statement in module.body:
        value = (
            statement.value
            if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign))
            else None
        )
        if isinstance(value, ast.Call) and call_name(value) == "run_cuda_pilot":
            top_level_pilot_calls.append(value)
    if (
        len(pilot_calls) != 1
        or len(top_level_pilot_calls) != 1
        or pilot_calls[0] is not top_level_pilot_calls[0]
        or not isinstance(pilot_calls[0].func, ast.Name)
        or pilot_calls[0].func.id != "run_cuda_pilot"
        or len(pilot_calls[0].args) != 3
        or any(isinstance(argument, ast.Starred) for argument in pilot_calls[0].args)
        or any(keyword.arg is None for keyword in pilot_calls[0].keywords)
        or {keyword.arg for keyword in pilot_calls[0].keywords} != {"config_id", "seed"}
        or len(pilot_calls[0].keywords) != 2
    ):
        raise PhoBertTrainingBlocked("M1-09 notebook must contain exactly one live top-level pilot call")
    keywords = {keyword.arg: keyword.value for keyword in pilot_calls[0].keywords if keyword.arg is not None}
    if (
        not isinstance(keywords.get("config_id"), ast.Constant)
        or keywords["config_id"].value != "P1"
        or not isinstance(keywords.get("seed"), ast.Constant)
        or type(keywords["seed"].value) is not int
        or keywords["seed"].value != 42
    ):
        raise PhoBertTrainingBlocked("M1-09 submitted notebook does not call the locked P1/seed42 pilot")
    canonical_ast = ast.dump(module, annotate_fields=True, include_attributes=False)
    return _sha256(canonical_ast.encode("utf-8"))


def _validate_submitted_notebook_matches_staging(
    notebook: object,
    staged_notebook: object,
) -> str:
    submitted_digest = _validate_submitted_pilot_notebook(notebook)
    staged_digest = _validate_submitted_pilot_notebook(staged_notebook)
    if submitted_digest != staged_digest:
        raise PhoBertTrainingBlocked("M1-09 submitted notebook code differs from the reviewed staged pilot")
    return submitted_digest


def _validate_m1_09_kernel_source(kernel_metadata: object, source_bundle: object) -> None:
    if not isinstance(kernel_metadata, dict) or not isinstance(source_bundle, dict):
        raise PhoBertTrainingBlocked("M1-09 Kaggle kernel or source-bundle metadata is invalid")
    dataset_id = source_bundle.get("dataset_id")
    if (
        not isinstance(dataset_id, str)
        or kernel_metadata.get("dataset_sources") != [dataset_id]
        or any(kernel_metadata.get(field) != [] for field in (
            "competition_sources",
            "kernel_sources",
            "model_sources",
        ))
    ):
        raise PhoBertTrainingBlocked("M1-09 kernel metadata must attach exactly the reviewed source bundle")


def _validate_m1_09_evidence_file_inventory(files: object, required_files: Sequence[str]) -> None:
    if not isinstance(files, dict) or set(files) != set(required_files):
        raise PhoBertTrainingBlocked("M1-09 evidence manifest has missing or unreviewed file entries")


def _validate_m1_09_source_bundle(
    source_bundle: object,
    current_hashes: Mapping[str, str],
    source_root: Path,
) -> None:
    if (
        not isinstance(source_bundle, dict)
        or type(source_bundle.get("schema_version")) is not int
        or source_bundle.get("schema_version") != 1
    ):
        raise PhoBertTrainingBlocked("M1-09 source-bundle manifest schema is invalid")
    source_files = source_bundle.get("files")
    if not isinstance(source_files, dict) or set(source_files) != M1_09_SOURCE_BUNDLE_ALLOWLIST:
        raise PhoBertTrainingBlocked("M1-09 source bundle has missing or unreviewed files")
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in source_files.values()
    ):
        raise PhoBertTrainingBlocked("M1-09 source-bundle file manifest is invalid")
    expected_source_files: dict[str, str] = {}
    try:
        for relative in M1_09_SOURCE_BUNDLE_ALLOWLIST:
            target = source_root.joinpath(*PurePosixPath(relative).parts).resolve(strict=True)
            target.relative_to(source_root)
            if not target.is_file():
                raise OSError("source-bundle entry is not a file")
            expected_source_files[relative] = _sha256(target.read_bytes())
    except (OSError, ValueError) as exc:
        raise PhoBertTrainingBlocked("M1-09 allowlisted source file is missing or unsafe") from exc
    if source_bundle.get("source_hashes") != dict(current_hashes) or source_files != expected_source_files:
        raise PhoBertTrainingBlocked("M1-09 source-bundle manifest is not bound to the exact current source files")
    if current_hashes.get("reviewer_trust_anchor_sha256") != source_files[
        "preparation/m1_09_trusted_reviewers.json"
    ]:
        raise PhoBertTrainingBlocked("M1-09 reviewer trust anchor differs from the allowlisted source file")


def _validate_m1_09_preparation_root(preparation: str | Path) -> Path:
    expected = Path(__file__).resolve().parents[2] / "preparation"
    try:
        expected = expected.resolve(strict=True)
        supplied = Path(preparation).resolve(strict=True)
    except OSError as exc:
        raise PhoBertTrainingBlocked("M1-09 preparation directory is unavailable") from exc
    if supplied != expected:
        raise PhoBertTrainingBlocked("M1-09 preparation directory must be the reviewed checkout root")
    return expected


def _load_authorized_m1_09_reviewer(
    preparation: Path,
    reviewer_id: str,
    reviewer_role: str,
) -> dict[str, object] | None:
    """Resolve an authorized independent reviewer from the pinned registry."""
    trust_path = preparation / "m1_09_trusted_reviewers.json"
    try:
        trust = json.loads(trust_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(trust, dict):
        return None
    entries = trust.get("reviewers")
    if (
        type(trust.get("schema_version")) is not int
        or trust.get("schema_version") != 2
        or trust.get("status") != "CONFIGURED"
        or not isinstance(entries, list)
    ):
        return None
    matches = [
        entry for entry in entries
        if isinstance(entry, dict)
        and entry.get("reviewer_id") == reviewer_id
        and entry.get("reviewer_role") == reviewer_role
        and entry.get("reviewer_kind") == "codex_subagent"
    ]
    if len(matches) != 1:
        return None
    authorization_basis = matches[0].get("authorization_basis")
    if not isinstance(authorization_basis, str) or not authorization_basis.strip():
        return None
    return matches[0]


def _m1_09_review_basis_sha256(evidence: Mapping[str, object]) -> str:
    """Hash review inputs while excluding the reviewer report and transcript."""
    basis = json.loads(json.dumps(evidence))
    basis.pop("independent_review", None)
    files = basis.get("files")
    if not isinstance(files, dict):
        raise PhoBertTrainingBlocked("M1-09 evidence file map is invalid")
    files.pop("review_attestation.json", None)
    files.pop("review_transcript.md", None)
    canonical = json.dumps(
        basis, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256(canonical)


def _has_zero_cross_split_collisions(
    audit: Mapping[str, object],
    locked_segmentation: Mapping[str, object],
    locked_tokenizer: Mapping[str, object],
) -> bool:
    segmentation = audit.get("segmentation")
    token_inputs = audit.get("tokenizer_inputs")
    locked_segmentation_results = locked_segmentation.get("results")
    locked_tokenizer_configs = locked_tokenizer.get("configurations")
    if (
        not isinstance(segmentation, dict)
        or not isinstance(token_inputs, dict)
        or not isinstance(locked_segmentation_results, dict)
        or not isinstance(locked_tokenizer_configs, dict)
    ):
        return False
    expected_splits = {"train", "val", "test"}
    expected_pairs = {"train__val", "train__test", "val__test"}
    for arm in ("underthesea", "vncorenlp_rdr"):
        result = segmentation.get(arm)
        rows = result.get("splits") if isinstance(result, dict) else None
        locked_result = locked_segmentation_results.get(arm)
        locked_rows = locked_result.get("splits") if isinstance(locked_result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("matches_locked_audit") is not True
            or not isinstance(rows, list)
            or not isinstance(locked_rows, list)
            or len(rows) != len(expected_splits)
            or {row.get("split") for row in rows if isinstance(row, dict)} != expected_splits
            or not _json_receipt_matches(rows, locked_rows)
        ):
            return False
        for row in rows:
            if not isinstance(row, dict) or any(
                type(row.get(field)) is not int or row[field] != 0
                for field in ("empty_output_rows", "within_split_duplicate_rows", "cross_split_overlap_rows")
            ):
                return False
    for config_id in ("P0", "P1"):
        result = token_inputs.get(config_id)
        collisions = result.get("cross_split_collisions") if isinstance(result, dict) else None
        locked_config = locked_tokenizer_configs.get(config_id)
        observed_config = (
            {key: value for key, value in result.items() if key != "matches_locked_audit"}
            if isinstance(result, dict)
            else None
        )
        if (
            not isinstance(result, dict)
            or result.get("matches_locked_audit") is not True
            or not isinstance(collisions, dict)
            or set(collisions) != {"full", "truncated"}
            or not isinstance(locked_config, dict)
            or not _json_receipt_matches(observed_config, locked_config)
        ):
            return False
        for representation in collisions.values():
            if not isinstance(representation, dict) or set(representation) != expected_pairs:
                return False
            if any(
                not isinstance(pair, dict)
                or type(pair.get("shared_unique_input_sequences")) is not int
                or pair["shared_unique_input_sequences"] != 0
                for pair in representation.values()
            ):
                return False
    return True


def _verify_m1_09_evidence_manifest(
    evidence_manifest_path: str | Path,
    preparation: Path,
    protocol_id: str,
    protocol_sha256: str,
    manifest_sha256: str,
    dataset_revision: str,
    pre_fit_audit_sha256: str,
) -> dict[str, object]:
    """Verify reviewed Kaggle evidence, reviewer attestation, and pilot hashes."""
    preparation = _validate_m1_09_preparation_root(preparation)
    evidence_path = Path(evidence_manifest_path).resolve(strict=True)
    evidence_root = evidence_path.parent.resolve(strict=True)
    private_root = (Path(__file__).resolve().parents[2] / "results" / "m1-08").resolve()
    try:
        evidence_root.relative_to(private_root)
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingBlocked("M1-09 evidence manifest is unavailable or outside private results") from exc
    if not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int or any(
        evidence.get(key) != value
        for key, value in {
            "schema_version": 1,
            "status": "PASS",
            "phase": "M1-09_GATE_B_EVIDENCE",
        }.items()
    ):
        raise PhoBertTrainingBlocked("M1-09 evidence manifest is not passing")
    kernel = evidence.get("kaggle_execution")
    review = evidence.get("independent_review")
    if not isinstance(kernel, dict) or (
        set(kernel) != {"ref", "version", "status"}
        or not isinstance(kernel.get("ref"), str)
        or not kernel["ref"]
        or type(kernel.get("version")) is not int
        or kernel["version"] <= 0
        or kernel.get("status") != "COMPLETE"
    ):
        raise PhoBertTrainingBlocked("M1-09 evidence lacks a completed Kaggle kernel version")
    if not isinstance(review, dict) or (
        set(review) != {
            "status",
            "reviewer_role",
            "reviewer_id",
            "reviewed_at_utc",
            "report_file",
            "report_sha256",
            "transcript_file",
            "transcript_sha256",
            "review_basis_sha256",
        }
        or review.get("status") != "PASS"
        or not isinstance(review.get("reviewer_role"), str)
        or not review["reviewer_role"]
        or not isinstance(review.get("reviewer_id"), str)
        or not review["reviewer_id"]
        or not isinstance(review.get("reviewed_at_utc"), str)
        or review.get("report_file") != "review_attestation.json"
        or review.get("transcript_file") != "review_transcript.md"
        or not isinstance(review.get("report_sha256"), str)
        or not isinstance(review.get("transcript_sha256"), str)
        or not isinstance(review.get("review_basis_sha256"), str)
    ):
        raise PhoBertTrainingBlocked("M1-09 evidence lacks an identified independent reviewer")

    pilot_relative = evidence.get("pilot_artifact_relative_path")
    if not isinstance(pilot_relative, str) or "\\" in pilot_relative:
        raise PhoBertTrainingBlocked("M1-09 pilot artifact path is invalid")
    pilot_parts = PurePosixPath(pilot_relative)
    if pilot_parts.is_absolute() or not pilot_parts.parts or any(
        part in {"", ".", ".."} for part in pilot_parts.parts
    ):
        raise PhoBertTrainingBlocked("M1-09 pilot artifact path escapes the evidence directory")
    pilot_dir = (evidence_root / Path(*pilot_parts.parts)).resolve(strict=True)
    try:
        pilot_dir.relative_to(private_root)
        pilot_dir.relative_to(evidence_root)
    except ValueError as exc:
        raise PhoBertTrainingBlocked("M1-09 pilot artifacts are outside the private evidence bundle") from exc

    required_files = (
        "kaggle_execution_log.txt",
        "kaggle_execution_receipt.json",
        "submitted_source.ipynb",
        "kernel_metadata.json",
        "source_bundle_manifest.json",
        "review_attestation.json",
        "review_transcript.md",
        f"{pilot_relative}/pilot_commit.json",
        f"{pilot_relative}/pilot_receipt.json",
        f"{pilot_relative}/run_metadata.json",
        f"{pilot_relative}/pre_fit_audit.json",
        f"{pilot_relative}/restore_verification.json",
        f"{pilot_relative}/checkpoint/checkpoint.json",
        f"{pilot_relative}/checkpoint/trainable_state.pt",
    )
    files = evidence.get("files")
    _validate_m1_09_evidence_file_inventory(files, required_files)
    verified_files: dict[str, str] = {}
    for relative in required_files:
        digest = files.get(relative)
        posix = PurePosixPath(relative)
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or posix.is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise PhoBertTrainingBlocked("M1-09 evidence manifest has an invalid required file entry")
        target = (evidence_root / Path(*posix.parts)).resolve(strict=True)
        try:
            target.relative_to(evidence_root)
        except ValueError as exc:
            raise PhoBertTrainingBlocked("M1-09 evidence file path escapes its bundle") from exc
        if not target.is_file() or _sha256(target.read_bytes()) != digest:
            raise PhoBertTrainingBlocked("M1-09 evidence file is missing or its SHA-256 differs")
        verified_files[relative] = digest
    # The private bundle is intentionally exact: unlisted files cannot silently
    # become part of the review scope after the hashes were prepared.
    actual_paths = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(required_files) | {"evidence_manifest.json"}:
        raise PhoBertTrainingBlocked("M1-09 evidence bundle contains missing or unreviewed files")

    try:
        kernel_metadata = json.loads((evidence_root / "kernel_metadata.json").read_text(encoding="utf-8"))
        source_bundle = json.loads((evidence_root / "source_bundle_manifest.json").read_text(encoding="utf-8"))
        execution_log = json.loads((evidence_root / "kaggle_execution_log.txt").read_text(encoding="utf-8"))
        execution_receipt = json.loads(
            (evidence_root / "kaggle_execution_receipt.json").read_text(encoding="utf-8")
        )
        notebook = json.loads((evidence_root / "submitted_source.ipynb").read_text(encoding="utf-8"))
        reviewer_report = json.loads((evidence_root / "review_attestation.json").read_text(encoding="utf-8"))
        reviewer_transcript = (evidence_root / "review_transcript.md").read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingBlocked(
            "M1-09 Kaggle metadata, execution receipt, source manifest, log, notebook or reviewer report is invalid"
        ) from exc
    expected_remote_code_file = f"{PurePosixPath(kernel['ref']).name}.ipynb"
    if not isinstance(kernel_metadata, dict) or (
        kernel_metadata.get("id") != kernel.get("ref")
        or kernel_metadata.get("language") != "python"
        or kernel_metadata.get("kernel_type") != "notebook"
        or kernel_metadata.get("is_private") is not True
        or kernel_metadata.get("enable_gpu") is not True
        or kernel_metadata.get("enable_tpu") is not False
        or kernel_metadata.get("enable_internet") is not True
        or kernel_metadata.get("machine_shape") != "NvidiaTeslaT4"
        # Kaggle's remote pull metadata names the notebook after the kernel slug.
        or kernel_metadata.get("code_file") != expected_remote_code_file
    ):
        raise PhoBertTrainingBlocked("M1-09 Kaggle metadata does not match the claimed private GPU kernel")
    if (
        not isinstance(source_bundle, dict)
        or type(source_bundle.get("schema_version")) is not int
        or source_bundle.get("schema_version") != 1
    ):
        raise PhoBertTrainingBlocked("M1-09 source-bundle manifest schema is invalid")
    _validate_m1_09_kernel_source(kernel_metadata, source_bundle)
    source_root = Path(__file__).resolve().parents[2]
    try:
        staged_notebook = json.loads(
            (source_root / "staging" / "kernel" / "m1_08_cuda_pilot.ipynb").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingBlocked("M1-09 reviewed pilot notebook staging is unavailable") from exc
    submitted_notebook_source_sha256 = _validate_submitted_notebook_matches_staging(
        notebook, staged_notebook
    )
    if not isinstance(execution_log, dict) or (
        type(execution_log.get("schema_version")) is not int
        or execution_log.get("schema_version") != 1
        or execution_log.get("status") != "PASS"
        or execution_log.get("phase") != "M1-08_CUDA_PILOT_ONLY"
    ):
        raise PhoBertTrainingBlocked("M1-09 Kaggle execution log does not record a passing pilot")

    current_hashes = {
        "protocol_sha256": _sha256((preparation / "protocol.json").read_bytes()),
        "manifest_sha256": _sha256((preparation / "data_manifest.json").read_bytes()),
        "preparation_lock_sha256": _sha256((preparation / "preparation_lock.json").read_bytes()),
        "trainer_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "pilot_code_sha256": _sha256(Path(__file__).with_name("pilot.py").read_bytes()),
        "preprocessing_sha256": _sha256(Path(preprocessing.__file__).resolve().read_bytes()),
        "reviewer_trust_anchor_sha256": _sha256((preparation / "m1_09_trusted_reviewers.json").read_bytes()),
        "pilot_notebook_source_sha256": submitted_notebook_source_sha256,
    }
    if current_hashes["protocol_sha256"] != protocol_sha256 or current_hashes["manifest_sha256"] != manifest_sha256:
        raise PhoBertTrainingBlocked("M1-09 source bindings differ from the locked protocol or manifest")
    source_hashes = evidence.get("source_hashes")
    if not isinstance(source_hashes, dict) or any(
        source_hashes.get(key) != value for key, value in current_hashes.items()
    ):
        raise PhoBertTrainingBlocked("M1-09 source bundle hashes differ from the current implementation")

    _validate_m1_09_source_bundle(source_bundle, current_hashes, source_root)
    if execution_log.get("source_hashes") != current_hashes:
        raise PhoBertTrainingBlocked("M1-09 execution log was produced from different source files")

    from .pilot import verify_pilot_commit

    pilot_receipt = verify_pilot_commit(pilot_dir)
    commit_bytes = (pilot_dir / "pilot_commit.json").read_bytes()
    receipt_bytes = (pilot_dir / "pilot_receipt.json").read_bytes()
    restore_bytes = (pilot_dir / "restore_verification.json").read_bytes()
    if evidence.get("pilot_commit_sha256") != _sha256(commit_bytes):
        raise PhoBertTrainingBlocked("M1-09 evidence does not bind the pilot commit marker")
    if evidence.get("pilot_receipt_sha256") != _sha256(receipt_bytes):
        raise PhoBertTrainingBlocked("M1-09 evidence does not bind the pilot receipt")
    if evidence.get("restore_verification_sha256") != _sha256(restore_bytes):
        raise PhoBertTrainingBlocked("M1-09 evidence does not bind restore verification")
    expected_pilot_identity = {
        "configuration": "P1",
        "seed": 42,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "dataset_revision": dataset_revision,
        "manifest_sha256": manifest_sha256,
        "preparation_lock_sha256": current_hashes["preparation_lock_sha256"],
        "pre_fit_audit_sha256": pre_fit_audit_sha256,
        "pilot_code_sha256": current_hashes["pilot_code_sha256"],
        "trainer_sha256": current_hashes["trainer_sha256"],
        "preprocessing_sha256": current_hashes["preprocessing_sha256"],
    }
    if any(pilot_receipt.get(key) != value for key, value in expected_pilot_identity.items()):
        raise PhoBertTrainingBlocked("M1-08 pilot receipt is stale or differs from the locked preparation/audit")
    expected_pilot_files = {
        "pilot_commit.json": _sha256(commit_bytes),
        "pilot_receipt.json": _sha256(receipt_bytes),
        "run_metadata.json": verified_files[f"{pilot_relative}/run_metadata.json"],
        "pre_fit_audit.json": verified_files[f"{pilot_relative}/pre_fit_audit.json"],
        "restore_verification.json": _sha256(restore_bytes),
        "checkpoint/checkpoint.json": verified_files[f"{pilot_relative}/checkpoint/checkpoint.json"],
        "checkpoint/trainable_state.pt": verified_files[f"{pilot_relative}/checkpoint/trainable_state.pt"],
    }
    if execution_log.get("pilot_files") != expected_pilot_files:
        raise PhoBertTrainingBlocked("M1-09 execution log does not identify the committed pilot artifacts")

    expected_execution_receipt = {
        "schema_version": 1,
        "status": "PASS",
        "kaggle_execution": kernel,
        "kernel_metadata_sha256": verified_files["kernel_metadata.json"],
        "submitted_source_sha256": verified_files["submitted_source.ipynb"],
        "execution_log_sha256": verified_files["kaggle_execution_log.txt"],
        "source_bundle_manifest_sha256": verified_files["source_bundle_manifest.json"],
        "pilot_commit_sha256": _sha256(commit_bytes),
        "pilot_receipt_sha256": _sha256(receipt_bytes),
        "restore_verification_sha256": _sha256(restore_bytes),
    }
    if (
        not isinstance(execution_receipt, dict)
        or type(execution_receipt.get("schema_version")) is not int
        or execution_receipt != expected_execution_receipt
    ):
        raise PhoBertTrainingBlocked(
            "M1-09 Kaggle execution receipt does not bind the claimed version to the exact remote artifacts"
        )

    try:
        reviewed_at = datetime.fromisoformat(review["reviewed_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhoBertTrainingBlocked("M1-09 reviewer attestation timestamp is invalid") from exc
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() != timezone.utc.utcoffset(reviewed_at):
        raise PhoBertTrainingBlocked("M1-09 reviewer attestation timestamp must be UTC")
    authorized_reviewer = _load_authorized_m1_09_reviewer(
        preparation, review["reviewer_id"], review["reviewer_role"]
    )
    if authorized_reviewer is None:
        raise PhoBertTrainingBlocked("M1-09 reviewer is not authorized in the pinned reviewer registry")
    review_basis_sha256 = _m1_09_review_basis_sha256(evidence)
    reviewed_files = {
        relative: digest for relative, digest in verified_files.items()
        if relative not in {"review_attestation.json", "review_transcript.md"}
    }
    expected_review_report = {
        "schema_version": 1,
        "status": "PASS",
        "phase": "M1-09_REVIEW_ATTESTATION",
        "reviewer_id": review["reviewer_id"],
        "reviewer_role": review["reviewer_role"],
        "reviewed_at_utc": review["reviewed_at_utc"],
        "review_transcript_file": "review_transcript.md",
        "review_transcript_sha256": verified_files["review_transcript.md"],
        "review_basis_sha256": review_basis_sha256,
        "kaggle_execution": kernel,
        "source_hashes": current_hashes,
        "reviewed_files": reviewed_files,
        "pilot_commit_sha256": _sha256(commit_bytes),
        "pilot_receipt_sha256": _sha256(receipt_bytes),
        "restore_verification_sha256": _sha256(restore_bytes),
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
        "non_blocking_findings": reviewer_report.get("non_blocking_findings")
        if isinstance(reviewer_report, dict)
        else None,
        "scope_limits": [
            "Review identity is procedural and transcript-backed; this attestation is not cryptographically signed.",
            "The reviewer did not rerun the Kaggle training job or independently reproduce the checkpoint restore.",
        ],
    }
    non_blocking_findings = (
        reviewer_report.get("non_blocking_findings")
        if isinstance(reviewer_report, dict)
        else None
    )
    if (
        not isinstance(reviewer_report, dict)
        or type(reviewer_report.get("schema_version")) is not int
        or not isinstance(non_blocking_findings, list)
        or any(not isinstance(item, str) or not item.strip() for item in non_blocking_findings)
        or not reviewer_transcript.strip()
        or reviewer_report != expected_review_report
    ):
        raise PhoBertTrainingBlocked("M1-09 reviewer report differs from the evidence or required review scope")
    if (
        review.get("report_sha256") != verified_files.get("review_attestation.json")
        or review.get("transcript_sha256") != verified_files.get("review_transcript.md")
        or review.get("review_basis_sha256") != review_basis_sha256
    ):
        raise PhoBertTrainingBlocked("M1-09 reviewer report hash or review-basis binding is invalid")
    return {
        "evidence": evidence,
        "evidence_sha256": _sha256(evidence_bytes),
        "pilot_receipt": pilot_receipt,
        "pilot_dir": pilot_dir,
        "pilot_commit_sha256": _sha256(commit_bytes),
        "pilot_receipt_sha256": _sha256(receipt_bytes),
        "restore_verification_sha256": _sha256(restore_bytes),
        "review_attestation_sha256": verified_files["review_attestation.json"],
        "review_transcript_sha256": verified_files["review_transcript.md"],
        "source_hashes": current_hashes,
    }


def _verify_training_readiness(
    readiness_path: str | Path,
    pre_fit_audit_path: str | Path,
    preparation: Path,
    protocol: Mapping[str, object],
    bundle: DatasetBundle,
    protocol_sha256: str,
    manifest_sha256: str,
    evidence_manifest_path: str | Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    """Require M1-09 evidence and bind the research runner to the committed pilot."""
    audit, audit_sha256 = _read_receipt(pre_fit_audit_path, "pre-fit audit")
    readiness, readiness_sha256 = _read_receipt(readiness_path, "training readiness")
    if type(readiness.get("schema_version")) is not int:
        raise PhoBertTrainingBlocked("Training readiness schema version must be an integer")
    evidence_bundle = _verify_m1_09_evidence_manifest(
        evidence_manifest_path,
        preparation,
        str(protocol.get("protocol_id")),
        protocol_sha256,
        manifest_sha256,
        bundle.revision,
        audit_sha256,
    )
    # The lock file is itself the authority for hashes of the frozen M0 inputs;
    # it is not self-hashed in that manifest, so read it and verify its members.
    try:
        lock = json.loads((preparation / "preparation_lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Preparation lock is unavailable") from exc
    if not isinstance(lock, dict) or not isinstance(lock.get("hashes"), dict):
        raise PhoBertTrainingError("Preparation lock schema is invalid")
    lock_hashes = lock["hashes"]
    preparation_lock_sha256 = _sha256((preparation / "preparation_lock.json").read_bytes())
    current_trainer_sha256 = _sha256(Path(__file__).resolve().read_bytes())
    current_preprocessing_sha256 = _sha256(Path(preprocessing.__file__).resolve().read_bytes())

    expected_top_level = {
        "schema_version": 1,
        "status": "TRAINING_READY",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256,
        "dataset_revision": bundle.revision,
        "manifest_sha256": manifest_sha256,
        "model_id": protocol["model"]["id"],
        "model_revision": protocol["model"]["revision"],
        "model_config_sha256": protocol["model"]["config_sha256"],
        "preparation_lock_sha256": preparation_lock_sha256,
        "trainer_sha256": current_trainer_sha256,
        "pilot_code_sha256": _sha256(Path(__file__).with_name("pilot.py").read_bytes()),
        "preprocessing_sha256": current_preprocessing_sha256,
        "pre_fit_audit_sha256": audit_sha256,
        "evidence_manifest_sha256": evidence_bundle["evidence_sha256"],
        "review_attestation_sha256": evidence_bundle["review_attestation_sha256"],
        "review_transcript_sha256": evidence_bundle["review_transcript_sha256"],
        "pilot_commit_sha256": evidence_bundle["pilot_commit_sha256"],
        "pilot_receipt_sha256": evidence_bundle["pilot_receipt_sha256"],
        "restore_verification_sha256": evidence_bundle["restore_verification_sha256"],
    }
    if any(readiness.get(key) != value for key, value in expected_top_level.items()):
        raise PhoBertTrainingBlocked("M1-09 readiness receipt is missing or differs from current inputs/code")

    pilot = readiness.get("pilot")
    required_pilot = {
        "status": "PASS",
        "cuda_available": True,
        "minibatch_forward_backward_finite": True,
            "trainable_scope_verified": True,
        "optimizer_step_succeeded": True,
        "checkpoint_restore_match": True,
    }
    pilot_boolean_fields = (
        "cuda_available",
        "minibatch_forward_backward_finite",
        "trainable_scope_verified",
        "optimizer_step_succeeded",
        "checkpoint_restore_match",
    )
    if (
        not isinstance(pilot, dict)
        or any(type(pilot.get(field)) is not bool for field in pilot_boolean_fields)
        or any(pilot.get(key) != value for key, value in required_pilot.items())
    ):
        raise PhoBertTrainingBlocked("M1-08 CUDA pilot or checkpoint restore has not passed")
    observed_pilot = evidence_bundle["pilot_receipt"].get("pilot")
    if (
        not isinstance(observed_pilot, dict)
        or any(type(observed_pilot.get(field)) is not bool for field in pilot_boolean_fields)
        or pilot != observed_pilot
    ):
        raise PhoBertTrainingBlocked("TRAINING_READY pilot claims differ from the committed pilot receipt")
    if readiness.get("pilot_code_sha256") != evidence_bundle["source_hashes"]["pilot_code_sha256"]:
        raise PhoBertTrainingBlocked("TRAINING_READY is not bound to the current pilot implementation")

    expected_audit = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256,
        "data_manifest_sha256": manifest_sha256,
        "dataset_revision": bundle.revision,
        "segmentation_audit_sha256": lock_hashes.get("segmentation_audit.json"),
        "tokenizer_input_audit_sha256": lock_hashes.get("tokenizer_input_audit.json"),
        "preprocessing_code_sha256": current_preprocessing_sha256,
        "cross_split_collision_found": False,
        "test_used_only_for_pre_fit_integrity_audit": True,
        "raw_review_text_saved": False,
        "row_level_ids_saved": False,
        "token_id_sequences_saved": False,
        "model_weights_loaded": False,
        "training_performed": False,
        "rdr_import_origin_verified": True,
    }
    audit_boolean_fields = (
        "cross_split_collision_found",
        "test_used_only_for_pre_fit_integrity_audit",
        "raw_review_text_saved",
        "row_level_ids_saved",
        "token_id_sequences_saved",
        "model_weights_loaded",
        "training_performed",
        "rdr_import_origin_verified",
    )
    if (
        type(audit.get("schema_version")) is not int
        or any(type(audit.get(field)) is not bool for field in audit_boolean_fields)
        or any(audit.get(key) != value for key, value in expected_audit.items())
    ):
        raise PhoBertTrainingBlocked("Pre-fit audit receipt is missing, stale, or non-passing")
    rdr_manifest, _ = _read_locked_json(preparation, lock, "rdr_assets_verification.json")
    segmentation_audit, _ = _read_locked_json(preparation, lock, "segmentation_audit.json")
    tokenizer_audit = _read_tokenizer_audit(preparation)
    if not _has_zero_cross_split_collisions(audit, segmentation_audit, tokenizer_audit):
        raise PhoBertTrainingBlocked("Runtime pre-fit audit differs from locked segmentation/tokenizer results")
    actual_rdr_assets = _verify_rdr_assets(preparation, rdr_manifest)
    actual_wrapper = _verify_rdr_wrapper(preparation, protocol)
    if audit.get("rdr_asset_sha256") != actual_rdr_assets or audit.get("rdr_wrapper") != actual_wrapper:
        raise PhoBertTrainingBlocked("RDR assets or wrapper differ from the runtime audit")
    if rdr_manifest.get("revision") != protocol["preprocessing"]["rdr"]["revision"]:
        raise PhoBertTrainingBlocked("RDR asset revision differs from the locked protocol")
    expected_packages = tokenizer_audit.get("packages")
    observed_packages = audit.get("runtime_packages")
    if not isinstance(expected_packages, dict) or not isinstance(observed_packages, dict):
        raise PhoBertTrainingBlocked("Pre-fit audit lacks runtime package versions")
    package_names = (
        "underthesea", "transformers", "tokenizers", "huggingface_hub", "py_vncorenlp", "pyjnius"
    )
    if any(observed_packages.get(name) != expected_packages.get(name) for name in package_names):
        raise PhoBertTrainingBlocked("Pre-fit audit runtime package versions differ from the lock")
    if audit.get("max_length") != protocol["training"]["max_length"]:
        raise PhoBertTrainingBlocked("Pre-fit audit maximum length differs from the protocol")
    if not isinstance(audit.get("runtime"), dict) or not all(
        isinstance(audit["runtime"].get(name), str) and audit["runtime"][name]
        for name in ("python_version", "java_version")
    ):
        raise PhoBertTrainingBlocked("Pre-fit audit lacks observed Python/Java runtime versions")
    readiness_runtime = readiness.get("runtime")
    if not isinstance(readiness_runtime, dict) or any(
        readiness_runtime.get(key) != audit["runtime"].get(key)
        for key in ("python_version", "java_version")
    ) or readiness_runtime.get("packages") != observed_packages:
        raise PhoBertTrainingBlocked("Readiness receipt and pre-fit audit runtime do not match")
    if not all(
        isinstance(pilot.get(field), str) and pilot[field]
        for field in ("torch", "cuda_runtime", "gpu_name")
    ):
        raise PhoBertTrainingBlocked("CUDA pilot receipt lacks observed GPU/runtime identity")
    if any(
        type(pilot.get(field)) is not int or pilot[field] <= 0
        for field in ("device_count", "gpu_total_memory_bytes")
    ):
        raise PhoBertTrainingBlocked("CUDA pilot receipt lacks observed GPU resource capacity")
    return readiness, audit, readiness_sha256, audit_sha256


def _build_pinned_model(preparation: Path, protocol: Mapping[str, object], seed: int):
    from importlib.metadata import version
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed

    targets = protocol["environment"]["explicit_package_targets"]
    for package in ("transformers", "tokenizers"):
        if version(package) != targets[package]:
            raise PhoBertTrainingError(f"Locked package version mismatch: {package}")
    model_spec = protocol["model"]
    config_path = (preparation / "assets" / "phobert_tokenizer" / "config.json").resolve(strict=True)
    if _sha256(config_path.read_bytes()) != model_spec["config_sha256"]:
        raise PhoBertTrainingError("PhoBERT config hash differs from the locked model")
    set_seed(seed)
    try:
        config = AutoConfig.from_pretrained(str(config_path.parent), local_files_only=True, num_labels=2)
        defaults = model_spec["config_defaults_verified"]
        if any(getattr(config, name, object()) != value for name, value in defaults.items()):
            raise PhoBertTrainingError("PhoBERT model config defaults differ from the locked model")
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_spec["id"],
            revision=model_spec["revision"],
            config=config,
            output_loading_info=True,
        )
        loading_summary = _validate_model_loading_info(model, loading_info)
        if getattr(model, "classifier", None) is None:
            raise PhoBertTrainingError("Loaded PhoBERT classifier has no classification head")
        # Explicitly reinitialize the task head so every fit starts from the
        # pinned base and a seed-specific fresh head, even if an upstream
        # checkpoint happens to contain classifier tensors.
        model.classifier.apply(model._init_weights)
    except PhoBertTrainingError:
        raise
    except Exception as exc:
        raise PhoBertTrainingError("Cannot load the pinned PhoBERT base revision") from exc
    return model, loading_summary


def _validate_model_loading_info(model: Any, loading_info: Mapping[str, object]) -> dict[str, object]:
    """Allow only a fresh classifier and known-unused pretrained heads to differ.

    The compatibility field ``only_expected_head_differences`` covers the new
    classifier, ignored LM head, and exact unused checkpoint pooler tensors.
    """
    if not isinstance(loading_info, Mapping):
        raise PhoBertTrainingError("Transformers returned no model-loading diagnostics")
    expected_missing = {
        name for name in model.state_dict() if name.startswith("classifier.")
    }
    missing = set(loading_info.get("missing_keys", []))
    unexpected = set(loading_info.get("unexpected_keys", []))
    mismatched = loading_info.get("mismatched_keys", [])
    errors = loading_info.get("error_msgs", [])
    expected_unused_pooler = {
        "roberta.pooler.dense.bias",
        "roberta.pooler.dense.weight",
    }
    unexpected_pooler = unexpected & expected_unused_pooler
    unexpected_lm_head = {
        name for name in unexpected
        if isinstance(name, str) and name.startswith("lm_head.")
    }
    if not expected_missing or missing != expected_missing:
        raise PhoBertTrainingError("Pinned PhoBERT load has missing weights outside the fresh task head")
    if unexpected - unexpected_pooler - unexpected_lm_head:
        namespace_counts = {"lm_head": 0, "roberta": 0, "classifier": 0, "other": 0, "non_string": 0}
        for name in unexpected:
            if not isinstance(name, str):
                namespace_counts["non_string"] += 1
                continue
            namespace = name.partition(".")[0]
            namespace_counts[namespace if namespace in {"lm_head", "roberta", "classifier"} else "other"] += 1
        error = PhoBertTrainingError("Pinned PhoBERT load has unexpected weights outside the base LM head")
        # Keep fixed namespace counts and a bounded, syntax-checked list for the
        # pinned model's roberta namespace; never persist other parameter names
        # or arbitrary strings returned by the model loader.
        safe_diagnostics = {
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
            "unexpected_key_namespace_counts": namespace_counts,
        }
        roberta_keys = sorted(
            name for name in unexpected
            if type(name) is str and name.partition(".")[0] == "roberta"
        )
        if len(roberta_keys) <= _SAFE_ROBERTA_KEY_NAME_MAX_COUNT and all(
            len(name) <= _SAFE_ROBERTA_KEY_NAME_MAX_LENGTH
            and _SAFE_ROBERTA_KEY_NAME_PATTERN.fullmatch(name)
            for name in roberta_keys
        ):
            safe_diagnostics["unexpected_roberta_keys"] = roberta_keys
        error.safe_diagnostics = safe_diagnostics
        raise error
    if mismatched or errors:
        raise PhoBertTrainingError("Pinned PhoBERT load reported mismatched weights or errors")
    return {
        "missing_classifier_tensor_count": len(missing),
        "unexpected_base_lm_head_tensor_count": len(unexpected_lm_head),
        "expected_unused_pooler_tensor_count": len(unexpected_pooler),
        "mismatched_tensor_count": 0,
        "loading_error_count": 0,
        "only_expected_head_differences": True,
    }


def _make_loaders(
    torch: Any,
    tokenizer: Any,
    train_records: Sequence[ProcessedReview],
    validation_records: Sequence[ProcessedReview],
    training: Mapping[str, object],
    seed: int,
):
    from torch.utils.data import DataLoader

    micro_batch = int(training["micro_batch"])
    max_length = int(training["max_length"])

    def collate(records: Sequence[ProcessedReview]) -> dict[str, object]:
        encoded = tokenize_batch(tokenizer, records, max_length=max_length)
        encoded["labels"] = torch.tensor([record.label for record in records], dtype=torch.long)
        encoded["keys"] = tuple(record.key for record in records)
        if encoded["input_ids"].shape[1] > max_length or encoded["input_ids"].shape[1] % 8:
            raise PhoBertTrainingError("Dynamic tokenizer padding violates the locked length contract")
        return encoded

    generator = torch.Generator()
    generator.manual_seed(seed)
    shared = {
        "batch_size": micro_batch,
        "num_workers": int(training["dataloader"]["num_workers"]),
        "pin_memory": bool(training["dataloader"]["pin_memory"]),
        "collate_fn": collate,
    }
    train_loader = DataLoader(
        train_records, shuffle=True, generator=generator, **shared
    )
    validation_loader = DataLoader(
        validation_records, shuffle=False, **shared
    )
    return train_loader, validation_loader


def _validate_epoch(
    torch: Any, model: Any, loader: Any, device: Any, autocast: Any
) -> tuple[float, tuple[PhoBertPrediction, ...]]:
    from sklearn.metrics import f1_score

    model.eval()
    gold: list[int] = []
    predicted: list[int] = []
    keys: list[tuple[str, str, int]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                name: batch[name].to(device, non_blocking=True)
                for name in ("input_ids", "attention_mask")
            }
            with autocast():
                logits = model(**inputs).logits
            if not torch.isfinite(logits).all():
                raise PhoBertTrainingError("Validation produced non-finite logits")
            predicted.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            gold.extend(batch["labels"].tolist())
            keys.extend(batch["keys"])
    predictions = tuple(
        PhoBertPrediction(key=key, y_true=int(y_true), y_pred=int(y_pred))
        for key, y_true, y_pred in zip(keys, gold, predicted, strict=True)
    )
    score = float(f1_score(gold, predicted, labels=[0, 1], average="macro", zero_division=0))
    return score, predictions


def _write_validation_predictions(
    path: Path,
    rows: Sequence[PhoBertPrediction],
    config_id: str,
    seed: int,
    epoch: int,
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "split": "val",
            "configuration": config_id,
            "seed": seed,
            "epoch": epoch,
            "predictions": [
                {"key": list(row.key), "y_true": row.y_true, "y_pred": row.y_pred}
                for row in rows
            ],
        },
    )


def _runtime_preflight(torch: Any) -> dict[str, object]:
    available = bool(torch.cuda.is_available())
    runtime = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device": "cuda:0" if available else None,
    }
    if not available:
        return runtime
    device_name = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)
    runtime.update(
        {
            "gpu_name": device_name,
            "gpu_total_memory_bytes": int(properties.total_memory),
        }
    )
    return runtime


def _verify_cuda_pilot_runtime(runtime: Mapping[str, object], pilot: Mapping[str, object]) -> None:
    """Require the fit allocation to match the passed CUDA pilot and capacity."""
    fields = ("torch", "cuda_runtime", "gpu_name", "device_count", "gpu_total_memory_bytes")
    if any(runtime.get(field) != pilot.get(field) for field in fields):
        raise PhoBertTrainingBlocked("Current CUDA runtime or GPU capacity differs from the passed pilot")


def _verify_runtime_packages(
    protocol: Mapping[str, object], tokenizer_audit: Mapping[str, object]
) -> dict[str, str]:
    """Recheck every protocol-pinned package immediately before preprocessing/fit."""
    from importlib.metadata import PackageNotFoundError, version

    environment = protocol.get("environment")
    protocol_packages = environment.get("explicit_package_targets") if isinstance(environment, dict) else None
    audit_packages = tokenizer_audit.get("packages")
    if not isinstance(protocol_packages, dict) or not isinstance(audit_packages, dict):
        raise PhoBertTrainingBlocked("Locked package versions are unavailable")

    expected = dict(protocol_packages)
    for package, target in audit_packages.items():
        if package in expected and expected[package] != target:
            raise PhoBertTrainingBlocked("Protocol and tokenizer audit package locks disagree")
        expected[package] = target

    observed: dict[str, str] = {}
    for package, target in expected.items():
        if not isinstance(package, str) or not package or not isinstance(target, str) or not target:
            raise PhoBertTrainingBlocked("Locked package version entry is invalid")
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise PhoBertTrainingBlocked("A protocol-pinned package is not installed") from exc
        if installed != target:
            raise PhoBertTrainingBlocked("Installed package versions differ from the locked protocol/audit")
        observed[package] = installed
    return observed


def run_phobert_development(
    bundle: DatasetBundle,
    protocol_path: str | Path,
    config_id: str,
    seed: int,
    preparation_dir: str | Path,
    pre_fit_audit_path: str | Path,
    readiness_receipt_path: str | Path,
    output_dir: str | Path,
    evidence_manifest_path: str | Path,
) -> PhoBertDevelopmentResult:
    """Fit one locked P0/P1 seed on train, select by validation, and save best.

    This runner deliberately accepts only a verified train/validation bundle.
    Its run directory must be new; failed attempts retain a small metadata file
    and are never silently overwritten.
    """
    protocol, protocol_sha256, manifest_sha256 = _read_development_protocol(
        protocol_path, bundle
    )
    _, segmenter_id = _assert_locked_fit(protocol, config_id, seed)

    preparation = Path(preparation_dir).resolve(strict=True)
    protocol_file = Path(protocol_path).resolve(strict=True)
    _verify_preparation_lock(preparation, protocol_file, protocol_sha256)
    try:
        lock = json.loads((preparation / "preparation_lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Preparation lock is unavailable") from exc
    manifest, _ = _read_locked_json(preparation, lock, "data_manifest.json")
    _validate_development_bundle(bundle, protocol, manifest)

    private_results_root = (Path(__file__).resolve().parents[2] / "results").resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(private_results_root)
    except ValueError as exc:
        raise PhoBertTrainingError("PhoBERT result directory must stay under private results/") from exc
    if output == private_results_root:
        raise PhoBertTrainingError("PhoBERT attempt output cannot be the private results root")
    if output.exists():
        raise PhoBertTrainingError("PhoBERT attempt output directory already exists")
    if output == preparation or preparation in output.parents:
        raise PhoBertTrainingError("PhoBERT result directory cannot be inside preparation assets")

    training = protocol["training"]
    model_spec = protocol["model"]
    preparation_lock_sha256 = _sha256(
        (preparation / "preparation_lock.json").read_bytes()
    )
    output.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "status": "CREATED",
        "phase": "development",
        "run_id": output.parent.name,
        "attempt_id": output.name,
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "configuration": config_id,
        "seed": seed,
        "dataset_revision": bundle.revision,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "preparation_lock_sha256": preparation_lock_sha256,
        "model_id": model_spec["id"],
        "model_revision": model_spec["revision"],
        "model_config_sha256": model_spec["config_sha256"],
        "segmenter_id": segmenter_id,
        "tokenizer_revision": model_spec["revision"],
        "max_length": int(training["max_length"]),
        "trainable_scope": "encoder_layers_10_11_and_classifier",
        "runtime": None,
        "pre_fit_audit_sha256": None,
        "readiness_receipt_sha256": None,
        "loader_sha256": bundle.loader_sha256,
        "trainer_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "preprocessing_sha256": _sha256(Path(preprocessing.__file__).resolve().read_bytes()),
        "train_rows": None,
        "validation_rows": None,
        "test_split_available_to_runner": False,
        "raw_review_text_saved": False,
    }
    _write_json_atomic(output / "run_metadata.json", metadata)
    stage = "training_readiness"
    try:
        readiness, pre_fit_audit, readiness_sha256, pre_fit_audit_sha256 = _verify_training_readiness(
            readiness_receipt_path,
            pre_fit_audit_path,
            preparation,
            protocol,
            bundle,
            protocol_sha256,
            manifest_sha256,
            evidence_manifest_path,
        )
        import torch

        runtime = _runtime_preflight(torch)
        metadata["runtime"] = runtime
        if not runtime["cuda_available"]:
            raise PhoBertTrainingBlocked("PhoBERT research fit requires an available CUDA GPU")
        pilot = readiness["pilot"]
        _verify_cuda_pilot_runtime(runtime, pilot)
        from .preprocessing import _java_runtime_version

        java_version = _java_runtime_version()
        readiness_runtime = readiness["runtime"]
        if (
            java_version != readiness_runtime.get("java_version")
            or runtime["python"] != readiness_runtime.get("python_version")
        ):
            raise PhoBertTrainingBlocked("Current Python/Java runtime differs from the passed pre-fit audit")
        runtime["java_version"] = java_version
        metadata.update(
            {
                "runtime": runtime,
                "pre_fit_audit_sha256": pre_fit_audit_sha256,
                "readiness_receipt_sha256": readiness_sha256,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        audit = _read_tokenizer_audit(preparation)
        package_versions = _verify_runtime_packages(protocol, audit)
        runtime["packages"] = package_versions
        metadata["tokenizer_audit_sha256"] = _sha256(
            (preparation / "tokenizer_input_audit.json").read_bytes()
        )
        metadata["model_config_sha256_observed"] = _sha256(
            (preparation / "assets" / "phobert_tokenizer" / "config.json").read_bytes()
        )
        tokenizer = _load_pinned_tokenizer(preparation, audit, protocol)
        segmenter = UndertheseaSegmenter() if config_id == "P0" else RDRSegmenter(preparation)
        processed_train = preprocess_records(bundle.splits["train"], segmenter)
        processed_validation = preprocess_records(bundle.splits["val"], segmenter)
        metadata.update(
            {
                "status": "INPUTS_VERIFIED",
                "runtime": runtime,
                "train_rows": len(processed_train),
                "validation_rows": len(processed_validation),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        torch.set_num_threads(int(training["threads"]))
        torch.backends.cuda.matmul.allow_tf32 = bool(training["allow_tf32"])
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(training["allow_tf32"])
        _write_json_atomic(output / "run_metadata.json", metadata)

        stage = "model_initialization"
        metadata.update(
            {
                "status": "DEVELOPING",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        _write_json_atomic(output / "run_metadata.json", metadata)
        model, model_loading_summary = _build_pinned_model(preparation, protocol, seed)
        metadata["model_loading"] = model_loading_summary
        _write_json_atomic(output / "run_metadata.json", metadata)
        trainable_names = configure_partial_finetuning(model)
        base_state_sha256 = frozen_base_state_sha256(model)
        metadata["frozen_base_state_sha256"] = base_state_sha256
        _write_json_atomic(output / "run_metadata.json", metadata)
        trainable_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        model.to(torch.device("cuda:0"))
        train_loader, validation_loader = _make_loaders(
            torch, tokenizer, processed_train, processed_validation, training, seed
        )
        if len(train_loader) == 0 or len(validation_loader) == 0:
            raise PhoBertTrainingError("PhoBERT train or validation loader is empty")

        stage = "optimizer_initialization"
        optimizer_config = training["optimizer"]
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(training["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
            amsgrad=bool(optimizer_config["amsgrad"]),
            fused=bool(optimizer_config["fused"]),
        )
        for group in optimizer.param_groups:
            if (
                group["lr"] != training["learning_rate"]
                or group["weight_decay"] != optimizer_config["weight_decay"]
                or tuple(group["betas"]) != tuple(optimizer_config["betas"])
                or group["eps"] != optimizer_config["eps"]
                or group["amsgrad"] != optimizer_config["amsgrad"]
                or group.get("fused") is not True
            ):
                raise PhoBertTrainingError("Instantiated AdamW settings differ from the locked protocol")

        accumulation = int(training["gradient_accumulation"])
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
        total_optimizer_steps = optimizer_steps_per_epoch * int(training["epochs"])
        warmup_steps = math.floor(0.1 * total_optimizer_steps)
        try:
            from transformers import get_linear_schedule_with_warmup
        except Exception as exc:
            raise PhoBertTrainingError("Locked Transformers scheduler is unavailable") from exc
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps
        )
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)

        stage = "training"
        best_score: float | None = None
        best_epoch = 0
        best_predictions: tuple[PhoBertPrediction, ...] = ()
        history: list[dict[str, object]] = []
        checkpoint_binding = {
            "configuration": config_id,
            "seed": seed,
            "dataset_revision": bundle.revision,
            "manifest_sha256": manifest_sha256,
            "protocol_sha256": protocol_sha256,
            "preparation_lock_sha256": preparation_lock_sha256,
            "pre_fit_audit_sha256": pre_fit_audit_sha256,
            "readiness_receipt_sha256": readiness_sha256,
            "model_id": model_spec["id"],
            "model_revision": model_spec["revision"],
            "model_config_sha256": model_spec["config_sha256"],
            "frozen_base_state_sha256": base_state_sha256,
            "segmenter_id": segmenter_id,
            "tokenizer_revision": model_spec["revision"],
            "tokenizer_audit_sha256": metadata["tokenizer_audit_sha256"],
            "max_length": int(training["max_length"]),
            "trainable_scope": "encoder_layers_10_11_and_classifier",
        }
        for epoch in range(1, int(training["epochs"]) + 1):
            stage = f"epoch_{epoch}_training"
            set_partial_finetuning_train_mode(model)
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
            successful_steps = 0
            overflow_steps = 0
            weighted_loss = 0.0
            seen_samples = 0
            for batch_index, batch in enumerate(train_loader):
                labels = batch["labels"].to("cuda:0", non_blocking=True)
                inputs = {
                    name: batch[name].to("cuda:0", non_blocking=True)
                    for name in ("input_ids", "attention_mask")
                }
                with autocast():
                    output_batch = model(**inputs, labels=labels)
                    loss = output_batch.loss
                if not torch.isfinite(loss):
                    raise PhoBertTrainingError(f"Non-finite training loss at epoch {epoch}")
                batch_size = int(labels.shape[0])
                scaler.scale(loss * batch_size).backward()
                accumulated_samples += batch_size
                seen_samples += batch_size
                weighted_loss += float(loss.detach().float().cpu()) * batch_size

                is_boundary = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == len(train_loader)
                )
                if not is_boundary:
                    continue
                scaler.unscale_(optimizer)
                _normalize_accumulated_gradients_(optimizer, accumulated_samples)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=float(training["max_grad_norm"]),
                )
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if _amp_step_succeeded(previous_scale, float(scaler.get_scale())):
                    scheduler.step()
                    successful_steps += 1
                else:
                    overflow_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0
            if successful_steps == 0:
                raise PhoBertTrainingError(f"No successful AMP optimizer step in epoch {epoch}")

            stage = f"epoch_{epoch}_validation"
            validation_score, predictions = _validate_epoch(
                torch, model, validation_loader, torch.device("cuda:0"), autocast
            )
            validate_phobert_predictions(bundle.splits["val"], predictions)
            _write_validation_predictions(
                output / f"epoch_{epoch:02d}_validation_predictions.json",
                predictions,
                config_id,
                seed,
                epoch,
            )
            selected = _should_select_epoch(best_score, validation_score)
            if selected:
                metadata_for_checkpoint = {
                    **checkpoint_binding,
                    "selected_epoch": epoch,
                    "validation_macro_f1": validation_score,
                    "trainer_sha256": _sha256(Path(__file__).resolve().read_bytes()),
                    "preprocessing_sha256": _sha256(Path(preprocessing.__file__).resolve().read_bytes()),
                }
                _save_trainable_state(model, output / "best_checkpoint", metadata_for_checkpoint)
                best_score = validation_score
                best_epoch = epoch
                best_predictions = predictions
            row = {
                "epoch": epoch,
                "train_loss": weighted_loss / seen_samples,
                "validation_macro_f1": validation_score,
                "optimizer_steps_succeeded": successful_steps,
                "amp_overflow_steps": overflow_steps,
                "learning_rate_after_epoch": float(optimizer.param_groups[0]["lr"]),
                "checkpoint_saved": selected,
                "checkpoint_became_best": selected,
            }
            history.append(row)
            _write_json_atomic(
                output / "training_history.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "attempt_id": metadata["attempt_id"],
                    "configuration": config_id,
                    "seed": seed,
                    "epochs": history,
                },
            )

        if best_epoch == 0 or best_score is None:
            raise PhoBertTrainingError("No validation-selected checkpoint was produced")
        validate_phobert_predictions(bundle.splits["val"], best_predictions)
        checkpoint_dir = output / "best_checkpoint"
        checkpoint_metadata_path = checkpoint_dir / "checkpoint.json"
        checkpoint_weights_path = checkpoint_dir / "trainable_state.pt"
        metadata.update(
            {
                "status": "FIT_COMPLETE",
                "selected_epoch": best_epoch,
                "validation_macro_f1": best_score,
                "checkpoint_metadata_sha256": _sha256(checkpoint_metadata_path.read_bytes()),
                "checkpoint_weights_sha256": _sha256(checkpoint_weights_path.read_bytes()),
                "trainable_parameter_count": trainable_count,
                "trainable_parameter_name_sha256": _sha256("\n".join(trainable_names).encode()),
                "validation_prediction_rows": len(best_predictions),
                "raw_review_text_saved": False,
                "row_level_validation_predictions_saved": True,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        _write_json_atomic(output / "run_metadata.json", metadata)
        return PhoBertDevelopmentResult(
            config_id=config_id,
            seed=seed,
            selected_epoch=best_epoch,
            validation_macro_f1=best_score,
            validation_predictions=best_predictions,
            checkpoint_dir=checkpoint_dir,
            history=tuple(history),
            trainable_parameter_names=trainable_names,
            trainable_parameter_count=trainable_count,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
            model_revision=model_spec["revision"],
            segmenter_id=segmenter_id,
        )
    except Exception as exc:
        metadata.update(
            {
                "status": "BLOCKED" if isinstance(exc, PhoBertTrainingBlocked) else "FAILED",
                "failed_stage": stage,
                "exception_type": type(exc).__name__,
                "failure_detail_saved": False,
                "block_reason_code": type(exc).__name__ if isinstance(exc, PhoBertTrainingBlocked) else None,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "last_completed_epoch": len(locals().get("history", [])),
            }
        )
        if "epoch" in locals():
            metadata["failed_epoch"] = epoch
        if "overflow_steps" in locals():
            metadata["failed_epoch_amp_overflow_steps"] = overflow_steps
        _write_json_atomic(output / "run_metadata.json", metadata)
        raise
