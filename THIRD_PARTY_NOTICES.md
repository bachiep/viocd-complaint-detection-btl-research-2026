# Third-party software and model notices

This file records provenance for components used by the private course-work
pilot. It does not relicense project code or third-party material. Public
redistribution remains out of scope until each included artifact and its
license obligations have been checked for the exact version being shared.

| Component | Pinned use in this project | Source and license evidence |
|---|---|---|
| VnCoreNLP / RDR | JAR and word-segmentation assets fetched at the commit and SHA-256 values in `preparation/rdr_assets_verification.json` | The upstream `LICENSE.md` says GNU GPL version 3 or, at the recipient's option, any later version: <https://github.com/vncorenlp/VnCoreNLP/blob/master/LICENSE.md>. The project downloads these assets at runtime rather than storing them in this source bundle. |
| `py_vncorenlp` | Version 0.1.4 source distribution, pinned by `preparation/protocol.json`; the matching source files are checked against the sdist | The PyPI sdist metadata declares MIT, while the upstream repository identifies itself as Apache-2.0 and contains `LICENSE.txt`: <https://pypi.org/project/py-vncorenlp/0.1.4/> and <https://github.com/thelinhbkhn2014/VnCoreNLP_Wrapper>. This metadata discrepancy is unresolved; preserve the exact sdist and do not publish this bundle until reviewed. |
| PhoBERT base | `vinai/phobert-base` at the immutable revision in `preparation/protocol.json` | The model card identifies the model as MIT-licensed: <https://huggingface.co/vinai/phobert-base>. Weights are downloaded at runtime and are not included in the source bundle. |

The Kaggle source dataset is intended to remain private. Its Kaggle metadata
uses `other` so that the upload does not assert CC0 or another public license.
