"""Regression checks for portable, line-ending-independent demo integrity pins."""

from pathlib import Path
import tempfile
import unittest

from demo.app import sha256_text_file


class DemoIntegrityTests(unittest.TestCase):
    def test_text_hash_normalizes_line_endings_but_detects_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "source.py"
            path.write_bytes(b"alpha\nbeta\n")
            lf_digest = sha256_text_file(path)

            path.write_bytes(b"alpha\r\nbeta\r\n")
            crlf_digest = sha256_text_file(path)

            path.write_bytes(b"alpha\rbeta\r")
            cr_digest = sha256_text_file(path)

            path.write_bytes(b"alpha\r\nBETA\r\n")
            changed_digest = sha256_text_file(path)

        self.assertEqual(lf_digest, crlf_digest)
        self.assertEqual(lf_digest, cr_digest)
        self.assertNotEqual(lf_digest, changed_digest)


if __name__ == "__main__":
    unittest.main()
