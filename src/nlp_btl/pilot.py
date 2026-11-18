"""One-batch, train-only CUDA pilot used to establish M1-08 readiness evidence.

This path deliberately does not accept a TRAINING_READY receipt. It is isolated
from the six-run development runner and cannot select a model or inspect test
predictions. Test bytes may be consumed by the locked pre-fit integrity audit;
they are never returned to this pilot's model-input path.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

from . import preprocessing
from .baseline import _read_development_protocol
from .data import DataIntegrityError, ReviewRecord, _sha256, load_dataset
from .phobert import (
    _SAFE_ROBERTA_KEY_NAME_MAX_COUNT,
    _SAFE_ROBERTA_KEY_NAME_MAX_LENGTH,
    _SAFE_ROBERTA_KEY_NAME_PATTERN,
    PhoBertTrainingBlocked,
    PhoBertTrainingError,
    _amp_step_succeeded,
    _assert_locked_fit,
    _build_pinned_model,
    _read_tokenizer_audit,
    _runtime_preflight,
    _save_trainable_state,
    _verify_preparation_lock,
    _verify_runtime_packages,
    configure_partial_finetuning,
    frozen_base_state_sha256,
    set_partial_finetuning_train_mode,
)
from .preprocessing import preprocess_records, tokenize_batch


@dataclass(frozen=True, slots=True)
class CudaPilotResult:
    """Status and private output paths from one M1-08 pilot attempt."""

    status: str
    output_dir: Path
    run_metadata_path: Path
    pilot_receipt_path: Path | None


_SAFE_FAILURE_CODES = {
    "Pilot requires a complete positive train microbatch": "INVALID_TRAIN_MICROBATCH",
    "Pilot model inputs must come from the train split": "NONTRAIN_PILOT_INPUT",
    "Pilot train microbatch has duplicate source keys": "DUPLICATE_PILOT_KEYS",
    "RDR Python dependencies differ from locked protocol": "RDR_PACKAGE_VERSION_MISMATCH",
    "Cannot initialize pinned RDR segmenter": "RDR_INITIALIZATION_FAILED",
    "RDR segmentation failed": "RDR_SEGMENTATION_FAILED",
    "RDR returned an invalid segmentation result": "RDR_INVALID_RESULT",
    "Review is empty after normalization": "EMPTY_NORMALIZED_REVIEW",
    "Segmenter returned an empty or non-text result": "INVALID_SEGMENTER_OUTPUT",
    "Pinned PhoBERT tokenizer packages are unavailable": "TOKENIZER_PACKAGES_UNAVAILABLE",
    "Tokenizer package version mismatch: transformers": "TRANSFORMERS_VERSION_MISMATCH",
    "Tokenizer package version mismatch: tokenizers": "TOKENIZERS_VERSION_MISMATCH",
    "Tokenizer package version mismatch: huggingface_hub": "HF_HUB_VERSION_MISMATCH",
    "Tokenizer asset manifest is invalid": "TOKENIZER_ASSET_MANIFEST_INVALID",
    "Tokenizer asset manifest entry is invalid": "TOKENIZER_ASSET_ENTRY_INVALID",
    "Tokenizer asset path escapes preparation directory": "TOKENIZER_ASSET_PATH_INVALID",
    "Cannot load pinned PhoBERT tokenizer locally": "TOKENIZER_LOAD_FAILED",
    "PhoBERT tokenizer class differs from locked audit": "TOKENIZER_CLASS_MISMATCH",
    "Cannot tokenize an empty batch": "EMPTY_TRAIN_BATCH",
    "max_length differs from locked PhoBERT protocol": "TOKENIZER_MAX_LENGTH_MISMATCH",
    "Locked package version mismatch: transformers": "TRANSFORMERS_PIN_MISMATCH",
    "Locked package version mismatch: tokenizers": "TOKENIZERS_PIN_MISMATCH",
    "PhoBERT config hash differs from the locked model": "PHOBERT_CONFIG_HASH_MISMATCH",
    "PhoBERT model config defaults differ from the locked model": "PHOBERT_CONFIG_DEFAULTS_MISMATCH",
    "Loaded PhoBERT classifier has no classification head": "PHOBERT_CLASSIFIER_MISSING",
    "Cannot load the pinned PhoBERT base revision": "PHOBERT_BASE_LOAD_FAILED",
    "Transformers returned no model-loading diagnostics": "MODEL_LOADING_DIAGNOSTICS_MISSING",
    "Pinned PhoBERT load has missing weights outside the fresh task head": "PHOBERT_MISSING_BASE_WEIGHTS",
    "Pinned PhoBERT load has unexpected weights outside the base LM head": "PHOBERT_UNEXPECTED_WEIGHTS",
    "Pinned PhoBERT load reported mismatched weights or errors": "PHOBERT_WEIGHT_MISMATCH",
}


def _select_pilot_records(
    train_records: Sequence[ReviewRecord], batch_size: int
) -> tuple[ReviewRecord, ...]:
    """Select one deterministic source-ordered microbatch from train only."""
    if type(batch_size) is not int or batch_size <= 0 or len(train_records) < batch_size:
        raise PhoBertTrainingError("Pilot requires a complete positive train microbatch")
    selected = tuple(train_records[:batch_size])
    keys = tuple(record.key for record in selected)
    if any(key[1] != "train" for key in keys):
        raise PhoBertTrainingError("Pilot model inputs must come from the train split")
    if len(set(keys)) != batch_size:
        raise PhoBertTrainingError("Pilot train microbatch has duplicate source keys")
    return selected


def _perform_single_optimizer_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    batch: Mapping[str, Any],
    device: Any,
    max_grad_norm: float,
) -> dict[str, object]:
    """Run one finite forward/backward/optimizer step; CPU branch is test-only."""
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable:
        raise PhoBertTrainingError("Pilot model has no trainable parameters")
    before = tuple(parameter.detach().clone() for parameter in trainable)
    inputs = {
        "input_ids": batch["input_ids"].to(device, non_blocking=device.type == "cuda"),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=device.type == "cuda"),
        "labels": batch["labels"].to(device, non_blocking=device.type == "cuda"),
    }
    optimizer.zero_grad(set_to_none=True)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    with amp:
        output = model(**inputs)
    loss = getattr(output, "loss", None)
    logits = getattr(output, "logits", None)
    if loss is None or logits is None or not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
        raise PhoBertTrainingError("Pilot forward pass produced missing or non-finite loss/logits")
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    gradients = tuple(parameter.grad for parameter in trainable)
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise PhoBertTrainingError("Pilot backward pass produced missing or non-finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    if not torch.isfinite(norm):
        raise PhoBertTrainingError("Pilot gradient norm is non-finite")
    if scaler is None:
        optimizer.step()
        step_succeeded = True
    else:
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        step_succeeded = _amp_step_succeeded(scale_before, float(scaler.get_scale()))
    changed = sum(
        int(not torch.equal(original, parameter.detach()))
        for original, parameter in zip(before, trainable, strict=True)
    )
    if not step_succeeded or changed == 0:
        raise PhoBertTrainingBlocked("Pilot optimizer step did not update trainable parameters")
    return {
        "loss": float(loss.detach().float().cpu().item()),
        "gradient_norm": float(norm.detach().float().cpu().item()),
        "trainable_tensor_count": len(trainable),
        "changed_trainable_tensor_count": changed,
        "optimizer_step_succeeded": True,
        "minibatch_forward_backward_finite": True,
    }


def _atomic_json(path: Path, payload: Mapping[str, object], *, create_only: bool = False) -> None:
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
        if create_only and path.exists():
            raise PhoBertTrainingError("Pilot receipt already exists")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _safe_error_diagnostics(exc: Exception) -> dict[str, object]:
    """Capture allowlisted failure codes and OS locations, never arbitrary exception text."""
    details: dict[str, object] = {}
    if isinstance(exc, DataIntegrityError):
        failure_code = _SAFE_FAILURE_CODES.get(str(exc))
        if failure_code is not None:
            details["failure_code"] = failure_code
            cause = exc.__cause__
            if cause is not None:
                details["cause_type"] = type(cause).__name__
    model_diagnostics = getattr(exc, "safe_diagnostics", None)
    if isinstance(model_diagnostics, Mapping):
        missing_count = model_diagnostics.get("missing_key_count")
        unexpected_count = model_diagnostics.get("unexpected_key_count")
        namespaces = model_diagnostics.get("unexpected_key_namespace_counts")
        expected_namespaces = ("lm_head", "roberta", "classifier", "other", "non_string")
        if (
            type(missing_count) is int
            and 0 <= missing_count <= 1_000_000
            and type(unexpected_count) is int
            and 0 <= unexpected_count <= 1_000_000
            and isinstance(namespaces, Mapping)
            and set(namespaces) == set(expected_namespaces)
            and all(type(namespaces[key]) is int and 0 <= namespaces[key] <= 1_000_000 for key in expected_namespaces)
            and sum(namespaces.values()) == unexpected_count
        ):
            summary = {
                "missing_key_count": missing_count,
                "unexpected_key_count": unexpected_count,
                "unexpected_key_namespace_counts": {key: namespaces[key] for key in expected_namespaces},
            }
            roberta_names = model_diagnostics.get("unexpected_roberta_keys")
            if (
                type(roberta_names) is list
                and len(roberta_names) <= _SAFE_ROBERTA_KEY_NAME_MAX_COUNT
                and len(roberta_names) == namespaces["roberta"]
                and all(
                    type(name) is str
                    and len(name) <= _SAFE_ROBERTA_KEY_NAME_MAX_LENGTH
                    and _SAFE_ROBERTA_KEY_NAME_PATTERN.fullmatch(name)
                    for name in roberta_names
                )
                and roberta_names == sorted(set(roberta_names))
            ):
                summary["unexpected_roberta_keys"] = list(roberta_names)
            details["model_loading_summary"] = summary
    os_error = exc if isinstance(exc, OSError) else exc.__cause__
    if isinstance(os_error, OSError):
        if type(os_error.errno) is int:
            details["errno"] = os_error.errno
        for field in ("filename", "filename2"):
            value = getattr(os_error, field, None)
            if isinstance(value, (str, bytes)):
                try:
                    details[field] = os.fsdecode(value)[:1024]
                except (TypeError, UnicodeError):
                    continue
    return details


def _restore_pilot_checkpoint(
    torch: Any,
    model: Any,
    checkpoint_dir: Path,
    expected_binding: Mapping[str, object],
) -> dict[str, object]:
    """Verify and restore a pilot-only checkpoint without claiming FIT_COMPLETE."""
    receipt_path = checkpoint_dir / "checkpoint.json"
    weights_path = checkpoint_dir / "trainable_state.pt"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        weights = weights_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Pilot checkpoint is incomplete") from exc
    if receipt.get("artifact_kind") != "m1_08_cuda_pilot_trainable_state":
        raise PhoBertTrainingError("Checkpoint is not a pilot-only artifact")
    for key, value in expected_binding.items():
        if receipt.get(key) != value:
            raise PhoBertTrainingError(f"Pilot checkpoint binding differs: {key}")
    if _sha256(weights) != receipt.get("weights_sha256"):
        raise PhoBertTrainingError("Pilot checkpoint weights hash mismatch")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PhoBertTrainingError("Pilot checkpoint cannot be loaded safely") from exc
    parameters = dict(model.named_parameters())
    expected_names = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if not isinstance(state, dict) or set(state) != expected_names:
        raise PhoBertTrainingError("Pilot checkpoint trainable scope differs from the fresh model")
    if sorted(state) != receipt.get("trainable_parameter_names"):
        raise PhoBertTrainingError("Pilot checkpoint parameter receipt differs from weights")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise PhoBertTrainingError("Pilot checkpoint tensor shape or dtype differs")
            parameter.copy_(value.to(device=parameter.device))
    return receipt


def _private_pilot_root() -> Path:
    """Return the ignored, private M1-08 result root for this checkout/runtime."""
    return Path(__file__).resolve().parents[2] / "results" / "m1-08"


def _require_private_pilot_output(output_dir: str | Path) -> Path:
    root = _private_pilot_root().resolve()
    output = Path(output_dir).resolve()
    if output == root or root not in output.parents:
        raise PhoBertTrainingError("Pilot artifacts must stay under the private results/m1-08 directory")
    return output


def verify_pilot_commit(output_dir: str | Path) -> dict[str, object]:
    """Accept only a complete pilot bundle whose receipt, metadata and weights agree."""
    output = _require_private_pilot_output(Path(output_dir).resolve(strict=True))
    try:
        marker_bytes = (output / "pilot_commit.json").read_bytes()
        receipt_bytes = (output / "pilot_receipt.json").read_bytes()
        metadata_bytes = (output / "run_metadata.json").read_bytes()
        audit_bytes = (output / "pre_fit_audit.json").read_bytes()
        checkpoint_bytes = (output / "checkpoint" / "checkpoint.json").read_bytes()
        weights_bytes = (output / "checkpoint" / "trainable_state.pt").read_bytes()
        restore_bytes = (output / "restore_verification.json").read_bytes()
        marker = json.loads(marker_bytes)
        receipt = json.loads(receipt_bytes)
        metadata = json.loads(metadata_bytes)
        checkpoint = json.loads(checkpoint_bytes)
        restore = json.loads(restore_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoBertTrainingError("Pilot commit bundle is incomplete or invalid") from exc
    expected = {
        "schema_version": 1,
        "status": "COMMITTED",
        "pilot_receipt_sha256": _sha256(receipt_bytes),
        "run_metadata_sha256": _sha256(metadata_bytes),
        "pre_fit_audit_sha256": _sha256(audit_bytes),
        "checkpoint_metadata_sha256": _sha256(checkpoint_bytes),
        "checkpoint_weights_sha256": _sha256(weights_bytes),
        "restore_verification_sha256": _sha256(restore_bytes),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise PhoBertTrainingError("Pilot commit marker hashes do not match the bundle")
    if receipt.get("status") != "PASS" or receipt.get("phase") != "M1-08_CUDA_PILOT_ONLY":
        raise PhoBertTrainingError("Pilot receipt is not a passing M1-08 artifact")
    if (
        restore.get("schema_version") != 1
        or restore.get("status") != "PASS"
        or restore.get("phase") != "M1-08_CUDA_PILOT_ONLY"
        or restore.get("checkpoint_restore_match") is not True
    ):
        raise PhoBertTrainingError("Pilot restore verification is missing or non-passing")
    if receipt.get("restore_verification_sha256") != expected["restore_verification_sha256"]:
        raise PhoBertTrainingError("Pilot receipt does not bind restore verification")
    restore_binding_fields = (
        "configuration",
        "seed",
        "protocol_sha256",
        "dataset_revision",
        "manifest_sha256",
        "preparation_lock_sha256",
        "pre_fit_audit_sha256",
        "pilot_code_sha256",
        "trainer_sha256",
        "preprocessing_sha256",
    )
    if any(restore.get(key) != receipt.get(key) for key in restore_binding_fields):
        raise PhoBertTrainingError("Pilot restore verification binding differs from the receipt")
    if metadata.get("status") != "PASS" or metadata.get("completed") is not True:
        raise PhoBertTrainingError("Pilot run metadata is not complete")
    if metadata.get("pilot_receipt_sha256") != expected["pilot_receipt_sha256"] or (
        metadata.get("pilot_commit_file") != "pilot_commit.json"
    ):
        raise PhoBertTrainingError("Pilot metadata does not bind the committed receipt")
    if receipt.get("pre_fit_audit_sha256") != expected["pre_fit_audit_sha256"]:
        raise PhoBertTrainingError("Pilot receipt does not bind the pre-fit audit")
    if receipt.get("checkpoint", {}).get("weights_sha256") != checkpoint.get("weights_sha256"):
        raise PhoBertTrainingError("Pilot receipt and checkpoint metadata differ")
    if checkpoint.get("weights_sha256") != expected["checkpoint_weights_sha256"]:
        raise PhoBertTrainingError("Pilot checkpoint weight hash differs from commit marker")
    if (
        receipt.get("checkpoint", {}).get("metadata_sha256") != expected["checkpoint_metadata_sha256"]
        or restore.get("checkpoint_metadata_sha256") != expected["checkpoint_metadata_sha256"]
        or restore.get("checkpoint_weights_sha256") != expected["checkpoint_weights_sha256"]
        or receipt.get("pilot", {}).get("checkpoint_restore_match") is not True
    ):
        raise PhoBertTrainingError("Pilot restore verification differs from the committed checkpoint")
    return receipt


def _publish_pilot_success(
    output: Path,
    metadata: dict[str, object],
    receipt: Mapping[str, object],
) -> Path:
    """Publish PASS artifacts with a final marker; incomplete bundles never verify."""
    receipt_path = output / "pilot_receipt.json"
    metadata_path = output / "run_metadata.json"
    marker_path = output / "pilot_commit.json"
    try:
        _atomic_json(receipt_path, receipt, create_only=True)
        metadata.update({
            "status": "PASS",
            "completed": True,
            "pilot_receipt_sha256": _sha256(receipt_path.read_bytes()),
            "pre_fit_audit_sha256": receipt["pre_fit_audit_sha256"],
            "checkpoint_weights_sha256": receipt["checkpoint"]["weights_sha256"],
            "checkpoint_restore_match": receipt["pilot"]["checkpoint_restore_match"],
            "pilot_commit_file": marker_path.name,
        })
        _atomic_json(metadata_path, metadata)
        marker = {
            "schema_version": 1,
            "status": "COMMITTED",
            "pilot_receipt_sha256": _sha256(receipt_path.read_bytes()),
            "run_metadata_sha256": _sha256(metadata_path.read_bytes()),
            "pre_fit_audit_sha256": _sha256((output / "pre_fit_audit.json").read_bytes()),
            "checkpoint_metadata_sha256": _sha256((output / "checkpoint" / "checkpoint.json").read_bytes()),
            "checkpoint_weights_sha256": _sha256((output / "checkpoint" / "trainable_state.pt").read_bytes()),
            "restore_verification_sha256": _sha256((output / "restore_verification.json").read_bytes()),
        }
        _atomic_json(marker_path, marker, create_only=True)
        verify_pilot_commit(output)
        return marker_path
    except Exception:
        for path in (receipt_path, marker_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def run_cuda_pilot(
    protocol_path: str | Path,
    preparation_dir: str | Path,
    output_dir: str | Path,
    *,
    config_id: str = "P1",
    seed: int = 42,
) -> CudaPilotResult:
    """Audit inputs and run one CUDA optimizer step on P1 train records only.

    There is intentionally no readiness-receipt argument. This is the bounded
    M1-08 smoke path that supplies evidence for M1-09; it does not enable a
    development run or produce validation/test metrics.
    """
    protocol_file = Path(protocol_path).resolve(strict=True)
    preparation = Path(preparation_dir).resolve(strict=True)
    if protocol_file.parent != preparation:
        raise PhoBertTrainingError("Pilot protocol must be inside its locked preparation directory")
    protocol_sha = _sha256(protocol_file.read_bytes())
    protocol = json.loads(protocol_file.read_bytes())
    if config_id != "P1" or seed != 42:
        raise PhoBertTrainingError("Gate-B pilot is fixed to P1/seed42 to exercise the RDR path")
    _assert_locked_fit(protocol, config_id, seed)
    _verify_preparation_lock(preparation, protocol_file, protocol_sha)

    output = _require_private_pilot_output(output_dir)
    if output == preparation or preparation in output.parents:
        raise PhoBertTrainingError("Pilot output cannot be written inside locked preparation")
    if output.exists():
        raise PhoBertTrainingError("Pilot attempt directory already exists; preserve it and choose a new ID")
    output.mkdir(parents=True, exist_ok=False)
    run_metadata_path = output / "run_metadata.json"
    pilot_receipt_path = output / "pilot_receipt.json"
    metadata: dict[str, object] = {
        "schema_version": 1,
        "status": "CREATED",
        "phase": "M1-08_CUDA_PILOT_ONLY",
        "configuration": config_id,
        "seed": seed,
        "protocol_sha256": protocol_sha,
        "preparation_lock_sha256": _sha256((preparation / "preparation_lock.json").read_bytes()),
        "pilot_code_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "trainer_sha256": _sha256(Path(__file__).with_name("phobert.py").read_bytes()),
        "preprocessing_sha256": _sha256(Path(preprocessing.__file__).read_bytes()),
        "validation_used": False,
        "test_split_available_to_pilot": False,
        "raw_review_text_saved": False,
        "row_level_ids_saved": False,
    }
    _atomic_json(run_metadata_path, metadata)
    started = time.perf_counter()
    stage = "cuda_preflight"
    substage: str | None = None
    try:
        import torch

        runtime = _runtime_preflight(torch)
        metadata["runtime"] = runtime
        if not runtime["cuda_available"]:
            raise PhoBertTrainingBlocked("M1-08 pilot requires an available CUDA GPU")
        if not isinstance(runtime.get("cuda_runtime"), str) or not runtime["cuda_runtime"]:
            raise PhoBertTrainingBlocked("M1-08 pilot lacks a concrete CUDA runtime version")
        if not isinstance(runtime.get("gpu_name"), str) or not runtime["gpu_name"] or (
            type(runtime.get("device_count")) is not int or runtime["device_count"] <= 0
        ) or (type(runtime.get("gpu_total_memory_bytes")) is not int or runtime["gpu_total_memory_bytes"] <= 0):
            raise PhoBertTrainingBlocked("M1-08 pilot lacks concrete GPU identity or capacity")
        stage = "pre_fit_audit"
        pre_fit_audit_path = output / "pre_fit_audit.json"
        audit, segmenter = preprocessing._audit_full_preprocessing_with_segmenter(
            protocol_file, pre_fit_audit_path
        )
        required_audit = {
            "status": "PASS",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "data_manifest_sha256": protocol["dataset"]["manifest_sha256"],
            "cross_split_collision_found": False,
            "test_used_only_for_pre_fit_integrity_audit": True,
            "raw_review_text_saved": False,
            "row_level_ids_saved": False,
            "token_id_sequences_saved": False,
            "model_weights_loaded": False,
            "training_performed": False,
        }
        if any(audit.get(key) != expected for key, expected in required_audit.items()):
            raise PhoBertTrainingBlocked("Pilot pre-fit audit is incomplete or stale")
        pre_fit_audit_sha = _sha256(pre_fit_audit_path.read_bytes())

        stage = "source_and_package_preflight"
        bundle = load_dataset(protocol_file)
        current_protocol, current_protocol_sha, manifest_sha = _read_development_protocol(protocol_file, bundle)
        if current_protocol_sha != protocol_sha or manifest_sha != bundle.manifest_sha256:
            raise PhoBertTrainingBlocked("Protocol or cohort changed during the pilot")
        if audit.get("dataset_revision") != bundle.revision or audit.get("data_manifest_sha256") != bundle.manifest_sha256:
            raise PhoBertTrainingBlocked("Pilot pre-fit audit and loaded development cohort differ")
        tokenizer_audit = _read_tokenizer_audit(preparation)
        verified_protocol_packages = _verify_runtime_packages(current_protocol, tokenizer_audit)
        audit_packages = tokenizer_audit.get("packages")
        observed_audit_packages = audit.get("runtime_packages")
        expected_audit_packages = {
            name: verified_protocol_packages[name]
            for name in audit_packages
            if name in verified_protocol_packages
        } if isinstance(audit_packages, dict) else None
        from .preprocessing import _java_runtime_version

        java_version = _java_runtime_version()
        audit_runtime = audit.get("runtime")
        if not isinstance(audit_runtime, dict) or (
            audit_runtime.get("python_version") != runtime.get("python")
            or audit_runtime.get("java_version") != java_version
        ):
            raise PhoBertTrainingBlocked("Pilot Python/Java runtime differs from the verified pre-fit audit")
        if observed_audit_packages != expected_audit_packages:
            raise PhoBertTrainingBlocked("Pilot audit package versions differ from the current verified packages")
        runtime.update({
            "java_version": java_version,
            "packages": observed_audit_packages,
            "verified_protocol_packages": verified_protocol_packages,
        })
        metadata.update({
            "dataset_revision": bundle.revision,
            "manifest_sha256": bundle.manifest_sha256,
            "pre_fit_audit_sha256": pre_fit_audit_sha,
            "runtime": runtime,
        })
        stage = "train_batch_preprocessing"
        training = current_protocol["training"]
        substage = "record_selection"
        selected = _select_pilot_records(bundle.splits["train"], int(training["micro_batch"]))
        substage = "rdr_import_origin_verification"
        if not segmenter.import_origin_verified:
            raise PhoBertTrainingBlocked("Pilot RDR wrapper import origin is not verified")
        substage = "train_record_segmentation"
        processed = preprocess_records(selected, segmenter)
        if any(record.key[1] != "train" for record in processed):
            raise PhoBertTrainingError("Segmented pilot inputs escaped the train split")
        substage = "tokenizer_loading"
        tokenizer = preprocessing._load_pinned_tokenizer(preparation, tokenizer_audit, current_protocol)
        substage = "train_batch_tokenization"
        encoded = tokenize_batch(tokenizer, processed, max_length=int(training["max_length"]))
        if encoded["input_ids"].shape[1] > int(training["max_length"]) or encoded["input_ids"].shape[1] % 8:
            raise PhoBertTrainingError("Pilot tokenizer padding violates the locked input contract")
        substage = "tensor_batch_construction"
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([record.label for record in processed], dtype=torch.long),
        }

        stage = "pinned_model_loading"
        substage = "model_build"
        model, loading_summary = _build_pinned_model(preparation, current_protocol, seed)
        trainable_names = configure_partial_finetuning(model)
        expected_prefixes = ("roberta.encoder.layer.10.", "roberta.encoder.layer.11.", "classifier.")
        if any(not name.startswith(expected_prefixes) for name in trainable_names):
            raise PhoBertTrainingError("Pilot model trainable parameter scope differs from P1 protocol")
        base_sha = frozen_base_state_sha256(model)
        device = torch.device("cuda:0")
        model.to(device)
        set_partial_finetuning_train_mode(model)
        torch.set_num_threads(int(training["threads"]))
        torch.backends.cuda.matmul.allow_tf32 = bool(training["allow_tf32"])
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(training["allow_tf32"])
        optimizer_spec = training["optimizer"]
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(training["learning_rate"]),
            weight_decay=float(optimizer_spec["weight_decay"]),
            betas=tuple(optimizer_spec["betas"]),
            eps=float(optimizer_spec["eps"]),
            amsgrad=bool(optimizer_spec["amsgrad"]),
            fused=bool(optimizer_spec["fused"]),
        )
        group = optimizer.param_groups[0]
        if (group["lr"] != training["learning_rate"]
            or group["weight_decay"] != optimizer_spec["weight_decay"]
            or tuple(group["betas"]) != tuple(optimizer_spec["betas"])
            or group["eps"] != optimizer_spec["eps"]
            or group["amsgrad"] != optimizer_spec["amsgrad"]
            or group.get("fused") is not True):
            raise PhoBertTrainingError("Pilot AdamW settings differ from the locked protocol")

        stage = "single_train_only_step"
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        step = _perform_single_optimizer_step(
            torch, model, optimizer, batch, device, float(training["max_grad_norm"])
        )
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        if peak_allocated <= 0:
            raise PhoBertTrainingBlocked("Pilot did not report CUDA memory allocation")
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            expected_logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.detach().float().cpu()
        pilot_binding = {
            "artifact_kind": "m1_08_cuda_pilot_trainable_state",
            "configuration": config_id,
            "seed": seed,
            "dataset_revision": bundle.revision,
            "manifest_sha256": bundle.manifest_sha256,
            "protocol_sha256": protocol_sha,
            "preparation_lock_sha256": metadata["preparation_lock_sha256"],
            "pre_fit_audit_sha256": pre_fit_audit_sha,
            "model_id": current_protocol["model"]["id"],
            "model_revision": current_protocol["model"]["revision"],
            "model_config_sha256": current_protocol["model"]["config_sha256"],
            "frozen_base_state_sha256": base_sha,
            "segmenter_id": "rdr",
            "tokenizer_revision": current_protocol["model"]["revision"],
            "tokenizer_audit_sha256": _sha256((preparation / "tokenizer_input_audit.json").read_bytes()),
            "max_length": int(training["max_length"]),
            "trainable_scope": "encoder_layers_10_11_and_classifier",
            "pilot_code_sha256": metadata["pilot_code_sha256"],
            "trainer_sha256": metadata["trainer_sha256"],
            "preprocessing_sha256": metadata["preprocessing_sha256"],
        }
        checkpoint_dir = output / "checkpoint"
        checkpoint = _save_trainable_state(model, checkpoint_dir, pilot_binding)
        del optimizer, model
        torch.cuda.empty_cache()

        stage = "checkpoint_restore"
        restore_started = time.perf_counter()
        restored, restored_loading_summary = _build_pinned_model(preparation, current_protocol, seed)
        restored_names = configure_partial_finetuning(restored)
        if restored_names != trainable_names:
            raise PhoBertTrainingError("Fresh pilot restore model has a different trainable scope")
        restored_base_sha = frozen_base_state_sha256(restored)
        if restored_base_sha != base_sha:
            raise PhoBertTrainingError("Fresh pilot restore model has a different frozen base")
        restored.to(device)
        checkpoint_receipt = _restore_pilot_checkpoint(torch, restored, checkpoint_dir, pilot_binding)
        restored.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            restored_logits = restored(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.detach().float().cpu()
        difference = float((expected_logits - restored_logits).abs().max().item())
        restore_match = bool(torch.allclose(expected_logits, restored_logits, rtol=1e-3, atol=1e-3))
        if not restore_match:
            raise PhoBertTrainingError("Pilot checkpoint restore logits differ from the saved model")
        torch.cuda.synchronize(device)
        restore_seconds = time.perf_counter() - restore_started

        stage = "finalize_receipt"
        pilot_summary = {
            "status": "PASS",
            "cuda_available": True,
            "minibatch_forward_backward_finite": step["minibatch_forward_backward_finite"],
            "trainable_scope_verified": True,
            "optimizer_step_succeeded": step["optimizer_step_succeeded"],
            "checkpoint_restore_match": restore_match,
            "torch": str(runtime["torch"]),
            "cuda_runtime": str(runtime["cuda_runtime"]),
            "gpu_name": str(runtime["gpu_name"]),
            "device_count": int(runtime["device_count"]),
            "gpu_total_memory_bytes": int(runtime["gpu_total_memory_bytes"]),
            "peak_allocated_memory_bytes": peak_allocated,
            "pilot_batch_size": len(selected),
            "single_step_elapsed_seconds": step_seconds,
            "checkpoint_restore_elapsed_seconds": restore_seconds,
            "training_split": "train",
            "validation_used": False,
            "test_used_for_model_input_or_selection": False,
        }
        restore_verification = {
            "schema_version": 1,
            "status": "PASS",
            "phase": "M1-08_CUDA_PILOT_ONLY",
            "checkpoint_restore_match": restore_match,
            "configuration": config_id,
            "seed": seed,
            "protocol_sha256": protocol_sha,
            "dataset_revision": bundle.revision,
            "manifest_sha256": bundle.manifest_sha256,
            "preparation_lock_sha256": metadata["preparation_lock_sha256"],
            "pre_fit_audit_sha256": pre_fit_audit_sha,
            "pilot_code_sha256": metadata["pilot_code_sha256"],
            "trainer_sha256": metadata["trainer_sha256"],
            "preprocessing_sha256": metadata["preprocessing_sha256"],
            "frozen_base_state_sha256": base_sha,
            "checkpoint_metadata_sha256": _sha256((checkpoint_dir / "checkpoint.json").read_bytes()),
            "checkpoint_weights_sha256": checkpoint["weights_sha256"],
            "restore_max_abs_logit_difference": difference,
            "rtol": 1e-3,
            "atol": 1e-3,
            "raw_review_text_saved": False,
            "row_level_ids_saved": False,
        }
        restore_verification_path = output / "restore_verification.json"
        _atomic_json(restore_verification_path, restore_verification, create_only=True)
        restore_verification_sha = _sha256(restore_verification_path.read_bytes())
        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "phase": "M1-08_CUDA_PILOT_ONLY",
            "configuration": config_id,
            "seed": seed,
            "protocol_id": current_protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "dataset_revision": bundle.revision,
            "manifest_sha256": bundle.manifest_sha256,
            "preparation_lock_sha256": metadata["preparation_lock_sha256"],
            "pre_fit_audit_sha256": pre_fit_audit_sha,
            "pilot_code_sha256": metadata["pilot_code_sha256"],
            "trainer_sha256": metadata["trainer_sha256"],
            "preprocessing_sha256": metadata["preprocessing_sha256"],
            "runtime": {
                "python_version": str(runtime["python"]),
                "java_version": java_version,
                "packages": observed_audit_packages,
                "verified_protocol_packages": verified_protocol_packages,
            },
            "pilot": pilot_summary,
            "step": step,
            "model_loading": loading_summary,
            "restore_model_loading": restored_loading_summary,
            "restore_verification_sha256": restore_verification_sha,
            "checkpoint": {
                "metadata_sha256": _sha256((checkpoint_dir / "checkpoint.json").read_bytes()),
                "weights_sha256": checkpoint["weights_sha256"],
                "trainable_parameter_count": checkpoint["trainable_parameter_count"],
                "changed_trainable_tensor_count": step["changed_trainable_tensor_count"],
                "restore_max_abs_logit_difference": difference,
            },
            "test_split_available_to_pilot": False,
            "test_used_only_for_pre_fit_integrity_audit": True,
            "raw_review_text_saved": False,
            "row_level_ids_saved": False,
            "token_id_sequences_saved": False,
            "validation_predictions_saved": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _publish_pilot_success(output, metadata, receipt)
        return CudaPilotResult("PASS", output, run_metadata_path, pilot_receipt_path)
    except Exception as exc:
        status = "BLOCKED" if isinstance(exc, PhoBertTrainingBlocked) else "FAILED"
        metadata.update({
            "status": status,
            "completed": False,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error_diagnostics": _safe_error_diagnostics(exc),
            "elapsed_seconds": time.perf_counter() - started,
        })
        if substage is not None:
            metadata["failed_substage"] = substage
        _atomic_json(run_metadata_path, metadata)
        return CudaPilotResult(status, output, run_metadata_path, None)
