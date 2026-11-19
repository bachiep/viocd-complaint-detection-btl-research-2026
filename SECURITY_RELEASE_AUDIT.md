# Public source release audit

**Scope:** curated source-only candidate `staging/public_release_20261118`, prepared 18/11/2026, with a test-scope update on 20/11/2026. This is an exposure and application-boundary review for the proposed public code snapshot, not a penetration test of Kaggle, Hugging Face, or GitHub.

## Checks and results

| Check | Result | Evidence / boundary |
|---|---|---|
| File allowlist and inventory | PASS | 43 tracked source, protocol, test, demo, notice, ignore-rule, pytest-configuration, source-snapshot marker, test-dependency file, line-ending, and hash-manifest files; no symlinks or large binary payloads. The SHA-256 manifest covers the 42 other tracked files. Added `preparation/scripts/audit_eda_lengths.py`, an aggregate-only audit that requires private manifests/RDR assets not included here. |
| Credential-pattern scan | PASS | No matches for GitHub, Hugging Face, AWS, Google, OpenAI-style keys, or literal credential assignments; scanner did not print candidate values |
| Private data and model artifacts | PASS | No CSV/TSV/JSONL/Parquet, pickle/joblib, safetensors, PT/PTH, or checkpoint files included |
| Demo network boundary | PASS | Demo binds to `127.0.0.1`, limits input size, does not persist request text, and is not configured as a public service. Public commit `05a5e2d` loaded the pinned model in a fresh GitHub clone and passed 5/5 synthetic API checks; Edge reported no console/HTTP errors and `/favicon.ico` returned 204 |
| Third-party materials | PASS WITH LIMIT | Notices are included; the original dataset and trained weights are omitted. Dataset use remains restricted to the group-confirmed academic/coursework scope |
| Source license | OPEN | No student-group license choice is included. Public visibility alone does not grant reuse rights |
| Public pytest command | PASS WITH EXPLICIT SCOPE | `python -m pytest -q` passed 87 tests and 24 subtests, with 11 explicit skips in the marked source-only snapshot. Three modules skip collection when omitted preparation scripts are unavailable; eight cases in PhoBERT/pilot tests skip only when private readiness assets or the staged notebook are absent. Without the snapshot marker, missing assets fail visibly. The full 115-test/48-subtest run is verified only in the private project workspace; README states this scope |

The Kaggle account prefix in synthetic M1-09 test fixtures was replaced with `example-owner` in this public copy. It does not change the tested source-binding behavior.

## Portability regression

The demo now hashes UTF-8 protocol/source text after normalizing CRLF and CR line endings to LF. Binary model artifacts and the saved selection/run metadata remain pinned by their original byte hashes; their recorded source/protocol hashes remain unchanged as provenance. The regression test confirms equivalent LF/CRLF/CR content hashes identically and that a content change is rejected. Public commit `05a5e2d` was checked from a fresh GitHub clone. After the three pinned private demo artifacts matched, the local clone passed 1/1 regression test and 5/5 API checks under Windows Python 3.13.7; Edge captured both synthetic predictions with no page, console, or HTTP errors, and the favicon returned an empty 204. No review text was saved.

## Findings

No actionable security vulnerability was identified in the scoped release candidate, so no CVSS score or code patch is applicable. This does not certify the entire project or external services as vulnerability-free.

## Release decision

The candidate is suitable for a **source-only public repository** after the final inventory/hash check. Keep the dataset, private manifests, Kaggle credentials, trained model, checkpoints, row-level outputs, and report draft outside the public repository. Give an instructor access to private Kaggle assets separately if the course requires execution access. Do not add a license until the student group selects one.
