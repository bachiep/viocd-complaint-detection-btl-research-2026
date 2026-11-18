# Public source release audit

**Scope:** curated source-only candidate `staging/public_release_20261118`, prepared 18/11/2026. This is an exposure and application-boundary review for the proposed public code snapshot, not a penetration test of Kaggle, Hugging Face, or GitHub.

## Checks and results

| Check | Result | Evidence / boundary |
|---|---|---|
| File allowlist and inventory | PASS | 40 tracked source, protocol, test, demo, notice, ignore-rule, line-ending, and hash-manifest files; no symlinks or large binary payloads. The SHA-256 manifest covers the 39 other tracked files. Added `preparation/scripts/audit_eda_lengths.py`, an aggregate-only audit that requires private manifests/RDR assets not included here. |
| Credential-pattern scan | PASS | No matches for GitHub, Hugging Face, AWS, Google, OpenAI-style keys, or literal credential assignments; scanner did not print candidate values |
| Private data and model artifacts | PASS | No CSV/TSV/JSONL/Parquet, pickle/joblib, safetensors, PT/PTH, or checkpoint files included |
| Demo network boundary | PASS | Demo binds to `127.0.0.1`, limits input size, does not persist request text, and is not configured as a public service. A clean checkout from the staged Git index loaded the pinned model and passed 5/5 synthetic API checks |
| Third-party materials | PASS WITH LIMIT | Notices are included; the original dataset and trained weights are omitted. Dataset use remains restricted to the group-confirmed academic/coursework scope |
| Source license | OPEN | No student-group license choice is included. Public visibility alone does not grant reuse rights |

The Kaggle account prefix in synthetic M1-09 test fixtures was replaced with `example-owner` in this public copy. It does not change the tested source-binding behavior.

## Portability regression

The demo now hashes UTF-8 protocol/source text after normalizing CRLF and CR line endings to LF. Binary model artifacts and the saved selection/run metadata remain pinned by their original byte hashes; their recorded source/protocol hashes remain unchanged as provenance. The regression test confirms equivalent LF/CRLF/CR content hashes identically and that a content change is rejected. In a clean checkout materialized from the staged Git index, only the three pinned private demo artifacts were added after their hashes matched. Under Windows Python 3.13.7, the regression test passed 1/1, the pinned model loaded, and the synthetic API verifier passed 5/5 checks; no review text was saved. The public GitHub head has not yet been updated or re-verified.

## Findings

No actionable security vulnerability was identified in the scoped release candidate, so no CVSS score or code patch is applicable. This does not certify the entire project or external services as vulnerability-free.

## Release decision

The candidate is suitable for a **source-only public repository** after the final inventory/hash check. Keep the dataset, private manifests, Kaggle credentials, trained model, checkpoints, row-level outputs, and report draft outside the public repository. Give an instructor access to private Kaggle assets separately if the course requires execution access. Do not add a license until the student group selects one.
