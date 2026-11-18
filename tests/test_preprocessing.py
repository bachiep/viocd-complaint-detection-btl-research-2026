"""Synthetic tests for shared preprocessing and representation audits."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_btl.data import DataIntegrityError, ReviewRecord
from nlp_btl.preprocessing import (
    ProcessedReview,
    _load_pinned_tokenizer,
    _tokenizer_report,
    _verify_rdr_assets,
    _verify_rdr_import_origin,
    _verify_rdr_wrapper,
    _validate_preprocessing_contract,
    _write_new_audit_receipt,
    audit_full_preprocessing,
    preprocess_records,
    preprocess_review,
    tokenize_batch,
)


def record(split: str, source_row: int, text: str, label: int = 0) -> ReviewRecord:
    return ReviewRecord(
        revision="synthetic-revision",
        split=split,
        source_row=source_row,
        text=text,
        label=label,
        domain="app",
    )


class PreprocessingTests(unittest.TestCase):
    def test_public_audit_returns_only_aggregate_report(self) -> None:
        report = {"status": "PASS", "row_level_ids_saved": False}
        segmenter = object()
        with mock.patch(
            "nlp_btl.preprocessing._audit_full_preprocessing_with_segmenter",
            return_value=(report, segmenter),
        ) as audit_worker:
            result = audit_full_preprocessing("protocol.json", "audit.json")
        audit_worker.assert_called_once_with("protocol.json", "audit.json", None)
        self.assertEqual(result, report)
        self.assertIsNot(result, segmenter)

    def test_pinned_tokenizer_uses_installed_packages_without_vendor_tree(self) -> None:
        packages = {
            "transformers": "4.51.3",
            "tokenizers": "0.21.4",
            "huggingface_hub": "0.30.2",
        }
        captured: dict[str, object] = {}

        class FakeTokenizer:
            pass

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                captured["path"] = path
                captured["kwargs"] = kwargs
                return FakeTokenizer()

        with tempfile.TemporaryDirectory() as temporary:
            preparation = Path(temporary) / "preparation"
            tokenizer_dir = preparation / "assets" / "phobert_tokenizer"
            tokenizer_dir.mkdir(parents=True)
            config = tokenizer_dir / "config.json"
            payload = b"{}\n"
            config.write_bytes(payload)
            vendor_tree = preparation / "assets" / "python_tokenizer"
            audit = {
                "model_revision": "pinned-revision",
                "tokenizer_class": "FakeTokenizer",
                "packages": packages,
                "model_assets": [{
                    "relative_path": "assets/phobert_tokenizer/config.json",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }],
            }
            protocol = {"model": {"revision": "pinned-revision"}}
            fake_transformers = SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
            clean_sys_path = [entry for entry in sys.path if Path(entry or ".") != vendor_tree]
            with (
                mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
                mock.patch("importlib.metadata.version", side_effect=packages.__getitem__),
                mock.patch.object(sys, "path", clean_sys_path),
            ):
                tokenizer = _load_pinned_tokenizer(preparation, audit, protocol)

            self.assertIsInstance(tokenizer, FakeTokenizer)
            self.assertEqual(captured["path"], str(tokenizer_dir.resolve()))
            self.assertEqual(captured["kwargs"], {"use_fast": False, "local_files_only": True})
            self.assertFalse(vendor_tree.exists())
            self.assertNotIn(str(vendor_tree.resolve()), sys.path)

    def test_audit_refuses_locked_preparation_output_and_receipt_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation = root / "preparation"
            results = root / "results"
            preparation.mkdir()
            results.mkdir()
            protocol = preparation / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(DataIntegrityError, "inside locked preparation"):
                audit_full_preprocessing(protocol, preparation / "audit.json")
            existing = results / "audit.json"
            existing.write_text("prior receipt", encoding="utf-8")
            with self.assertRaisesRegex(DataIntegrityError, "already exists"):
                audit_full_preprocessing(protocol, existing)
            with self.assertRaisesRegex(DataIntegrityError, "already exists"):
                _write_new_audit_receipt(preparation, existing, {"status": "PASS"})
            self.assertEqual(existing.read_text(encoding="utf-8"), "prior receipt")

    def test_locked_preprocessing_contract_accepts_exact_runtime_plan(self) -> None:
        protocol = {
            "preprocessing": {
                "normalization": [
                    "Unicode NFC",
                    "lowercase",
                    "collapse whitespace with str.split then single-space join",
                ],
                "stopword_removal": False,
                "stemming_or_lemmatization": False,
                "underthesea": {
                    "version": "9.5.0",
                    "call": "word_tokenize(normalized_review, format='text')",
                },
                "rdr": {
                    "wrapper_version": "0.1.4",
                    "pyjnius_version": "1.9.0",
                    "annotators": ["wseg"],
                },
            },
            "model": {
                "id": "vinai/phobert-base",
                "revision": "pinned-revision",
                "tokenizer_use_fast": False,
            },
            "environment": {
                "explicit_package_targets": {
                    "transformers": "4.51.3",
                    "tokenizers": "0.21.4",
                    "underthesea": "9.5.0",
                    "py_vncorenlp": "0.1.4",
                    "pyjnius": "1.9.0",
                }
            },
        }
        audit = {
            "model": "vinai/phobert-base",
            "model_revision": "pinned-revision",
            "use_fast": False,
            "packages": dict(protocol["environment"]["explicit_package_targets"]),
        }

        _validate_preprocessing_contract(protocol, audit)
        audit["use_fast"] = True
        with self.assertRaisesRegex(DataIntegrityError, "fast-mode"):
            _validate_preprocessing_contract(protocol, audit)

    def test_normalization_precedes_segmenter_and_preserves_source_fields(self) -> None:
        received: list[str] = []

        def segmenter(text: str) -> str:
            received.append(text)
            return f"SEG<{text}>"

        source = record("train", 7, "  SẢN   PHẨM  TỐT  ", label=1)
        processed = preprocess_records((source,), segmenter)[0]

        self.assertEqual(received, ["sản phẩm tốt"])
        self.assertEqual(
            processed,
            ProcessedReview(
                key=("synthetic-revision", "train", 7),
                segmented_text="SEG<sản phẩm tốt>",
                label=1,
                domain="app",
            ),
        )

    def test_blank_and_non_string_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(DataIntegrityError, "empty after normalization"):
            preprocess_review("  \t\n ", lambda text: text)
        with self.assertRaisesRegex(DataIntegrityError, "must be strings"):
            preprocess_review(None, lambda text: text)  # type: ignore[arg-type]

    def test_empty_or_non_text_segmenter_output_fails(self) -> None:
        with self.assertRaisesRegex(DataIntegrityError, "empty or non-text"):
            preprocess_review("hàng kém", lambda text: "  \n")
        with self.assertRaisesRegex(DataIntegrityError, "empty or non-text"):
            preprocess_review("hàng kém", lambda text: None)  # type: ignore[return-value]

    def test_tokenizer_receives_locked_batch_arguments(self) -> None:
        captured: dict[str, object] = {}

        def tokenizer(texts: list[str], **kwargs):
            captured["texts"] = texts
            captured.update(kwargs)
            return {"input_ids": [[1, 2]]}

        rows = (
            ProcessedReview(("r", "train", 0), "x y", 0, "app"),
            ProcessedReview(("r", "train", 1), "z", 1, "mobile"),
        )
        result = tokenize_batch(tokenizer, rows)

        self.assertEqual(result, {"input_ids": [[1, 2]]})
        self.assertEqual(captured, {
            "texts": ["x y", "z"],
            "truncation": True,
            "max_length": 128,
            "padding": True,
            "pad_to_multiple_of": 8,
            "return_tensors": "pt",
        })

    def test_empty_batch_and_unlocked_length_are_rejected(self) -> None:
        with self.assertRaisesRegex(DataIntegrityError, "empty batch"):
            tokenize_batch(lambda *args, **kwargs: {}, ())
        row = ProcessedReview(("r", "train", 0), "x", 0, "app")
        with self.assertRaisesRegex(DataIntegrityError, "max_length"):
            tokenize_batch(lambda *args, **kwargs: {}, (row,), max_length=127)

    def test_tokenizer_audit_detects_train_truncation_duplicate_without_leakage(self) -> None:
        train = (record("train", 0, "first"), record("train", 1, "second"))
        val = (record("val", 0, "validation"),)
        test = (record("test", 0, "terminal"),)
        records = {"test": test, "val": val, "train": train}
        segmented = {
            "test": ("test unique",),
            "val": ("validation unique",),
            "train": ("long first", "long second"),
        }
        shared_prefix = list(range(128))

        def fake_tokenizer(
            texts, truncation, max_length=None, padding=None, return_attention_mask=None
        ):
            sequences = {
                "test unique": [900],
                "validation unique": [901],
                "long first": shared_prefix + [1000],
                "long second": shared_prefix + [1001],
            }
            if truncation:
                sequences = {key: value[:max_length] for key, value in sequences.items()}
            return {"input_ids": [sequences[text] for text in texts]}

        result = _tokenizer_report(fake_tokenizer, records, segmented, 128)

        self.assertEqual(result["truncation_counts"], {"test": 0, "val": 0, "train": 2})
        self.assertEqual(result["within_split_duplicate_rows"]["full"]["train"], 0)
        self.assertEqual(result["within_split_duplicate_rows"]["truncated"]["train"], 1)
        self.assertEqual(
            result["within_split_duplicate_group_labels"]["truncated"]["train"],
            {
                "duplicate_groups": 1,
                "rows_in_duplicate_groups": 2,
                "conflicting_label_groups": 0,
                "rows_in_conflicting_label_groups": 0,
            },
        )
        self.assertTrue(
            all(
                collision["shared_unique_input_sequences"] == 0
                for representation in result["cross_split_collisions"].values()
                for collision in representation.values()
            )
        )
        self.assertNotIn("long first", repr(result))
        self.assertNotIn("1000", repr(result))

    def test_tokenizer_audit_exposes_exact_cross_split_collision(self) -> None:
        records = {
            "test": (record("test", 0, "test"),),
            "val": (record("val", 0, "validation"),),
            "train": (record("train", 0, "training"),),
        }
        segmented = {split: (split,) for split in records}

        def colliding_tokenizer(
            texts, truncation, max_length=None, padding=None, return_attention_mask=None
        ):
            return {
                "input_ids": [
                    [11, 12] if text != "test" else [13, 14] for text in texts
                ]
            }

        result = _tokenizer_report(colliding_tokenizer, records, segmented, 128)

        self.assertEqual(
            result["cross_split_collisions"]["full"]["train__val"],
            {"shared_unique_input_sequences": 1, "left_rows": 1, "right_rows": 1},
        )

    def test_rdr_asset_manifest_detects_tampered_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "assets" / "rdr.bin"
            path.parent.mkdir()
            path.write_bytes(b"pinned")
            manifest = {
                "artifacts": [
                    {
                        "relative_path": "assets/rdr.bin",
                        "bytes": 6,
                        "sha256": hashlib.sha256(b"pinned").hexdigest(),
                    }
                ]
            }

            self.assertEqual(
                _verify_rdr_assets(root, manifest),
                {"assets/rdr.bin": hashlib.sha256(b"pinned").hexdigest()},
            )
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(DataIntegrityError, "asset hash mismatch"):
                _verify_rdr_assets(root, manifest)

    def test_rdr_wrapper_source_must_match_pinned_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper_root = root / "assets" / "python_rdr"
            package_root = wrapper_root / "py_vncorenlp"
            package_root.mkdir(parents=True)
            sources = {
                "py_vncorenlp/__init__.py": b"from .vncorenlp import VnCoreNLP\n",
                "py_vncorenlp/vncorenlp.py": b"class VnCoreNLP: pass\n",
            }
            for relative_path, payload in sources.items():
                (wrapper_root / relative_path).write_bytes(payload)
            archive_buffer = io.BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
                for relative_path, payload in sources.items():
                    member = tarfile.TarInfo(f"py_vncorenlp-0.1.4/{relative_path}")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            archive_payload = archive_buffer.getvalue()
            (wrapper_root / "py_vncorenlp-0.1.4.tar.gz").write_bytes(archive_payload)
            protocol = {
                "preprocessing": {
                    "rdr": {
                        "wrapper_version": "0.1.4",
                        "wrapper_sdist_sha256": hashlib.sha256(archive_payload).hexdigest(),
                    }
                }
            }

            verified = _verify_rdr_wrapper(root, protocol)
            self.assertTrue(verified["installed_sources_match_sdist"])
            (package_root / "vncorenlp.py").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(DataIntegrityError, "differs"):
                _verify_rdr_wrapper(root, protocol)

    def test_rdr_import_origin_must_be_the_verified_target_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "python_rdr" / "py_vncorenlp"
            package.mkdir(parents=True)
            init_file = package / "__init__.py"
            implementation_file = package / "vncorenlp.py"
            init_file.touch()
            implementation_file.touch()
            module = SimpleNamespace(
                __file__=str(init_file),
                vncorenlp=SimpleNamespace(__file__=str(implementation_file)),
            )
            self.assertTrue(_verify_rdr_import_origin(module, package.parent))
            other_file = root / "other" / "__init__.py"
            other_file.parent.mkdir()
            other_file.touch()
            module.__file__ = str(other_file)
            with self.assertRaisesRegex(DataIntegrityError, "does not match"):
                _verify_rdr_import_origin(module, package.parent)


if __name__ == "__main__":
    unittest.main()
