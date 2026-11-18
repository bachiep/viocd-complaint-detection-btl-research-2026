from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preparation" / "scripts"))

import prepare_kaggle_terminal_runtime as runtime


class TerminalRuntimePackageTargetTests(unittest.TestCase):
    def test_runtime_install_targets_cover_locked_tokenizer_packages(self) -> None:
        preparation = ROOT / "preparation"
        protocol = json.loads((preparation / "protocol.json").read_text(encoding="utf-8"))
        tokenizer_audit = json.loads(
            (preparation / "tokenizer_input_audit.json").read_text(encoding="utf-8")
        )
        locked_tokenizer_targets = tokenizer_audit["packages"]

        runtime_targets = runtime.resolve_package_targets(protocol, tokenizer_audit)

        for name, version in locked_tokenizer_targets.items():
            self.assertEqual(runtime_targets[name], version)
        self.assertEqual(runtime_targets["huggingface_hub"], "0.30.2")

    def test_conflicting_runtime_and_tokenizer_pins_fail_closed(self) -> None:
        protocol = {
            "environment": {
                "explicit_package_targets": {"transformers": "4.51.4"}
            }
        }
        tokenizer_audit = {"packages": {"transformers": "4.51.3"}}

        with self.assertRaisesRegex(ValueError, "Conflicting pin"):
            runtime.resolve_package_targets(protocol, tokenizer_audit)


if __name__ == "__main__":
    unittest.main()
