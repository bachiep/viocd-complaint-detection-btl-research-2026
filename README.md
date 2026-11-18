# Vietnamese product review complaint detection

Course project: **Building and evaluating a Vietnamese complaint detection system for product reviews**.

This repository is a source-only snapshot for code review and coursework. It contains no review corpus, row-level predictions, checkpoints, Kaggle credentials, or trained model weights. The private Kaggle runs and datasets must be shared separately by the project owner when access is appropriate.

## Task and experiment

The task uses the complaint / non-complaint labels in UIT-ViOCD. This is distinct from sentiment polarity and star rating. The pinned data snapshot is `25e1851683ebc407d28e5a2d514b7ae0eadd4e4b`; the retained cohort has 5,247 reviews across four domains.

The experiment compares:

- **S0:** TF-IDF word unigrams/bigrams with LinearSVC.
- **P0:** partially fine-tuned PhoBERT with Underthesea word segmentation.
- **P1:** the same PhoBERT training protocol with VnCoreNLP/RDR word segmentation.

Selection was made on validation data: S0 Macro-F1 0.92070, P0 mean 0.91648, and P1 mean 0.91223. On the historically exposed test split, descriptive Macro-F1 was 0.91744, 0.92542 ± 0.00107, and 0.91936 ± 0.00732, respectively. The paired P1 − P0 difference was −0.00607 (95% bootstrap interval [−0.01521, 0.00307]); this interval includes zero and supports neither superiority nor equivalence. The test result is not an independent confirmation and was not used to change model selection.

## Contents

- `src/nlp_btl/`: data, preprocessing, baseline, PhoBERT training, selection, terminal evaluation, and numerical audit code.
- `preparation/scripts/audit_eda_lengths.py`: aggregate-only descriptive length/token audit used to build the report's EDA table; it does not fit or select a model.
- `tests/`: synthetic and contract-focused tests; they do not include source review text.
- `preparation/protocol.json`: locked experiment protocol. Private data manifests and split files are not included.
- `demo/`: local demo source and its pinned runtime requirements. The trained S0 artifact is intentionally omitted, so the demo cannot be run from this public snapshot alone.
- `THIRD_PARTY_NOTICES.md`: notices for bundled or referenced software.
- `RELEASE_MANIFEST.sha256`: hashes for the released files; text files use LF line endings across platforms.

## Runtime recorded for the completed Kaggle runs

Python 3.13.16; Tesla T4 × 2; CUDA 13.0; PyTorch 2.11.0+cu130; Transformers 4.51.3; tokenizers 0.21.4; NumPy 2.1.3; pandas 2.3.3; scikit-learn 1.8.0; Underthesea 9.5.0; py_vncorenlp 0.1.4; pyjnius 1.9.0; Java 21.0.12.1.

An exact rerun requires access to the private Kaggle kernels and input datasets plus the pinned data snapshot. See the project owner for the appropriate course-review access. Do not place raw reviews, credential files, checkpoints, or per-row predictions in this repository.

The aggregate EDA script requires the group's private `data_manifest.json`, preparation lock, tokenizer audit receipt and RDR assets, plus authorized access to the pinned source CSV files and the package versions listed in the script. Those inputs are intentionally omitted from this repository. It writes only split-level aggregate distributions to `results/eda/viocd_length_token_audit.json`; it refuses to overwrite an existing output and does not persist review text, row IDs, token IDs or per-review measurements. The local report documents the resulting source hashes and cohort counts.

## Data use

The project team confirmed use of UIT-ViOCD for coursework and academic research. That confirmation does not grant permission to redistribute the dataset. Obtain the dataset from its [original Hugging Face page](https://huggingface.co/datasets/tarudesu/ViOCD) and check its current access and license terms before using it outside the stated coursework context.

## License

No source-code license has been selected by the student group. Public visibility of this repository does not grant additional reuse rights. Third-party datasets and models remain subject to their own terms.
