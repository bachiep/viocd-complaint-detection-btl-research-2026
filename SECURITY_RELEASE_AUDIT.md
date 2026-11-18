# Public source release audit

**Scope:** curated source-only candidate `staging/public_release_20261118`, prepared 18/11/2026. This is an exposure and application-boundary review for the proposed public code snapshot, not a penetration test of Kaggle, Hugging Face, or GitHub.

## Checks and results

| Check | Result | Evidence / boundary |
|---|---|---|
| File allowlist and inventory | PASS | 38 source, protocol, test, demo, notice, ignore-rule, line-ending, and hash-manifest files; no symlinks or large binary payloads |
| Credential-pattern scan | PASS | No matches for GitHub, Hugging Face, AWS, Google, OpenAI-style keys, or literal credential assignments; scanner did not print candidate values |
| Private data and model artifacts | PASS | No CSV/TSV/JSONL/Parquet, pickle/joblib, safetensors, PT/PTH, or checkpoint files included |
| Demo network boundary | PASS | Demo binds to `127.0.0.1`, limits input size, does not persist request text, and is not configured as a public service; the project runtime receipt records 5/5 synthetic API checks |
| Third-party materials | PASS WITH LIMIT | Notices are included; the original dataset and trained weights are omitted. Dataset use remains restricted to the group-confirmed academic/coursework scope |
| Source license | OPEN | No student-group license choice is included. Public visibility alone does not grant reuse rights |

The Kaggle account prefix in synthetic M1-09 test fixtures was replaced with `example-owner` in this public copy. It does not change the tested source-binding behavior.

## Findings

No actionable security vulnerability was identified in the scoped release candidate, so no CVSS score or code patch is applicable. This does not certify the entire project or external services as vulnerability-free.

## Release decision

The candidate is suitable for a **source-only public repository** after the final inventory/hash check. Keep the dataset, private manifests, Kaggle credentials, trained model, checkpoints, row-level outputs, and report draft outside the public repository. Give an instructor access to private Kaggle assets separately if the course requires execution access. Do not add a license until the student group selects one.
