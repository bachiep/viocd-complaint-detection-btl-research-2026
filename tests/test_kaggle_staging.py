from __future__ import annotations

import hashlib
import copy
from pathlib import Path
import tempfile
import unittest

from preparation.scripts.check_kaggle_pilot_staging import (
    validate_flat_upload,
    validate_notebook_matches_generator,
    validate_notebook_source_binding,
)
from preparation.scripts.kaggle_staging_helpers import upload_filename
from preparation.scripts.stage_kaggle_pilot import SOURCE_HASH_FILE_BINDINGS, write_notebook


class KaggleFlatUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name)
        self.contents = {
            "src/nlp_btl/pilot.py": b"pilot source\n",
            "preparation/protocol.json": b"{}\n",
        }
        self.source_hashes = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in self.contents.items()
        }
        self.allowed_files = frozenset(self.contents)
        (self.upload_dir / "dataset-metadata.json").write_text("{}\n", encoding="utf-8")
        manifest_bytes = b"{}\n"
        (self.upload_dir / "source_bundle_manifest.json").write_bytes(manifest_bytes)
        (self.upload_dir / "source_bundle_manifest.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "\n",
            encoding="ascii",
        )
        for relative, payload in self.contents.items():
            (self.upload_dir / upload_filename(relative)).write_bytes(payload)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_accepts_exact_flat_payload_with_matching_hashes(self) -> None:
        validate_flat_upload(self.upload_dir, self.source_hashes, self.allowed_files)

    def test_uploaded_names_are_unique_flat_and_archive_free(self) -> None:
        names = {upload_filename(name) for name in self.allowed_files}

        self.assertEqual(len(names), len(self.allowed_files))
        self.assertTrue(all(Path(name).name == name for name in names))
        self.assertTrue(all(not name.endswith((".zip", ".tar", ".gz")) for name in names))

    def test_rejects_missing_or_extra_upload_entries(self) -> None:
        missing_file = self.upload_dir / upload_filename("src/nlp_btl/pilot.py")
        missing_file.unlink()
        with self.assertRaisesRegex(ValueError, "missing, duplicate, extra"):
            validate_flat_upload(self.upload_dir, self.source_hashes, self.allowed_files)

        missing_file.write_bytes(self.contents["src/nlp_btl/pilot.py"])
        (self.upload_dir / "unexpected.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing, duplicate, extra"):
            validate_flat_upload(self.upload_dir, self.source_hashes, self.allowed_files)

    def test_rejects_non_file_upload_entries(self) -> None:
        (self.upload_dir / "unexpected_dir").mkdir()

        with self.assertRaisesRegex(ValueError, "missing, duplicate, extra"):
            validate_flat_upload(self.upload_dir, self.source_hashes, self.allowed_files)

    def test_rejects_content_hash_mismatch(self) -> None:
        (self.upload_dir / upload_filename("src/nlp_btl/pilot.py")).write_bytes(b"changed\n")

        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            validate_flat_upload(self.upload_dir, self.source_hashes, self.allowed_files)

    def test_rejects_source_manifest_checksum_mismatch(self) -> None:
        (self.upload_dir / "source_bundle_manifest.json").write_text('{"changed": true}\n', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "source manifest checksum mismatch"):
            validate_flat_upload(self.upload_dir, self.source_hashes, self.allowed_files)

    def test_rejects_manifest_with_incomplete_file_map(self) -> None:
        incomplete_hashes = {"src/nlp_btl/pilot.py": self.source_hashes["src/nlp_btl/pilot.py"]}

        with self.assertRaisesRegex(ValueError, "exact reviewed upload inventory"):
            validate_flat_upload(self.upload_dir, incomplete_hashes, self.allowed_files)

    def test_upload_name_rejects_absolute_and_parent_paths(self) -> None:
        for relative in ("/absolute/path.py", "../outside.py", "src/../../outside.py"):
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(ValueError, "Unsafe source path"):
                    upload_filename(relative)

    def notebook_source_cell(
        self,
        dataset_id: str = "owner/dataset",
        input_root: str = "/kaggle/input",
        mount_roots: list[str] | None = None,
    ) -> str:
        names = {relative: upload_filename(relative) for relative in self.allowed_files}
        owner, slug = dataset_id.split("/", 1)
        if mount_roots is None:
            mount_roots = [
                f"/kaggle/input/{slug}",
                f"/kaggle/input/{owner}/{slug}",
                f"/kaggle/input/datasets/{owner}/{slug}",
            ]
        expected_source_hashes = {
            key: self.source_hashes[path]
            for key, path in sorted(SOURCE_HASH_FILE_BINDINGS.items())
            if path in self.source_hashes
        }
        expected_source_hash_keys = sorted([*expected_source_hashes, "pilot_notebook_source_sha256"])
        return "\n".join(
            (
                f"DATASET_ID = {dataset_id!r}",
                f"INPUT_ROOT = Path({input_root!r})",
                f"EXPECTED_MOUNT_ROOTS = {mount_roots!r}",
                f"EXPECTED_FILES = {sorted(self.allowed_files)!r}",
                f"EXPECTED_FILE_HASHES = {self.source_hashes!r}",
                f"SOURCE_FILE_NAMES = {names!r}",
                f"EXPECTED_SOURCE_HASHES = {expected_source_hashes!r}",
                f"EXPECTED_SOURCE_HASH_KEYS = {expected_source_hash_keys!r}",
            )
        )

    def test_notebook_source_map_binds_dataset_id_root_files_and_hashes(self) -> None:
        validate_notebook_source_binding(
            self.notebook_source_cell(),
            "owner/dataset",
            self.source_hashes,
            self.allowed_files,
        )

    def test_notebook_source_binding_rejects_source_hash_literal_mismatch(self) -> None:
        expected_source_hashes = {
            key: self.source_hashes[path]
            for key, path in sorted(SOURCE_HASH_FILE_BINDINGS.items())
            if path in self.source_hashes
        }
        wrong_source_hashes = dict(expected_source_hashes)
        wrong_source_hashes["protocol_sha256"] = "0" * 64
        cell = self.notebook_source_cell().replace(
            f"EXPECTED_SOURCE_HASHES = {expected_source_hashes!r}",
            f"EXPECTED_SOURCE_HASHES = {wrong_source_hashes!r}",
        )

        with self.assertRaisesRegex(ValueError, "differs from the reviewed manifest"):
            validate_notebook_source_binding(cell, "owner/dataset", self.source_hashes, self.allowed_files)

    def test_notebook_source_binding_rejects_dataset_id_or_mount_path_mismatch(self) -> None:
        for dataset_id, input_root, mount_roots in (
            ("owner/other-dataset", "/kaggle/input", None),
            ("owner/dataset", "/tmp/other-input", None),
            ("owner/dataset", "/kaggle/input", ["/kaggle/input/unrelated/dataset"]),
        ):
            with self.subTest(dataset_id=dataset_id, input_root=input_root, mount_roots=mount_roots):
                with self.assertRaisesRegex(ValueError, "differs from the reviewed manifest"):
                    validate_notebook_source_binding(
                        self.notebook_source_cell(dataset_id, input_root, mount_roots),
                        "owner/dataset",
                        self.source_hashes,
                        self.allowed_files,
                    )

    def test_notebook_source_binding_rejects_reflective_mutation_after_literals(self) -> None:
        mutations = (
            "vars()['SOURCE_ROOT'] = Path('/kaggle/input/other')\n",
            "import inspect\ninspect.currentframe().__getattribute__('f_globals')['SOURCE_ROOT'] = None\n",
            "import builtins\nobject.__getattribute__(builtins, '__dict__')['vars']()['SOURCE_ROOT'] = None\n",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.splitlines()[-1]):
                mutated_cell = self.notebook_source_cell() + "\n" + mutation
                with self.assertRaisesRegex(ValueError, "dynamic reflection"):
                    validate_notebook_source_binding(
                        mutated_cell,
                        "owner/dataset",
                        self.source_hashes,
                        self.allowed_files,
                    )

    def test_notebook_must_match_deterministic_reviewed_generator(self) -> None:
        names = {relative: upload_filename(relative) for relative in self.allowed_files}
        generated = write_notebook(self.source_hashes, names)
        validate_notebook_matches_generator(generated, self.source_hashes, self.allowed_files)
        pilot_cell = "".join(generated["cells"][2]["source"])
        self.assertIn('"source_mount": SOURCE_ROOT.as_posix()', pilot_cell)
        setup_cell = "".join(generated["cells"][0]["source"])
        self.assertIn('source_bundle.get("status") != "STAGED"', setup_cell)
        self.assertIn('type(source_bundle.get("schema_version")) is not int', setup_cell)
        self.assertIn('set(source_hashes) != set(EXPECTED_SOURCE_HASH_KEYS)', setup_cell)
        self.assertIn('source_hashes.get(name) != digest', setup_cell)
        self.assertIn('manifest_sha256_path = SOURCE_ROOT / "source_bundle_manifest.sha256"', setup_cell)
        self.assertIn('expected_manifest_sha256 != actual_manifest_sha256', setup_cell)

        mutated = copy.deepcopy(generated)
        mutated["cells"][0]["source"].append(
            "object.__getattribute__(builtins, '__dict__')['vars']()['SOURCE_ROOT'] = None\n"
        )
        with self.assertRaisesRegex(ValueError, "deterministic reviewed generator"):
            validate_notebook_matches_generator(mutated, self.source_hashes, self.allowed_files)


if __name__ == "__main__":
    unittest.main()
