# PeerBERT

Experiments for the MS thesis **"Explainable Classification of AI-Generated Peer Reviews Using
Transformer Models"**. PeerBERT is `answerdotai/ModernBERT-base` fine-tuned separately on:

| Dataset | Task | Source |
|---|---|---|
| [IntelLabs/AI-Peer-Review-Detection-Benchmark](https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark) | binary: human (0) / AI (1) | CSV files, loaded one by one with `huggingface_hub` |
| [Sem-Detect/ML_Conferences-Peer-Reviews](https://huggingface.co/datasets/Sem-Detect/ML_Conferences-Peer-Reviews) | 3-class: human (0) / ai (1) / rewrite (2) | config `reviews`, Parquet |

The two datasets are never mixed. Only the real files and their real labels are used:
- No data is generated, and nothing about the datasets is hard-coded.
- Columns, labels and counts are read from the files.
- Missing optional columns are reported, never invented.
- `tests/` uses small hand-written examples only to check code paths; they are never
  written to `outputs/`.

## What has been run

| Step | Status |
|---|---|
| Code tests (`pytest tests`): small hand-written examples and tiny random models, CPU only | pass on Python 3.13 with transformers 4.48.3 and 5.20, pandas 2.2 and 3.0; every file compiles on Python 3.10/3.11 |
| Real-data loading, audit, training, evaluation, baselines | **not run yet**: needs Colab/Kaggle with Hugging Face access and a GPU |

Every result file is produced by the scripts below. `scripts/09_build_report.py` only reports
artefacts that exist and marks everything else as **NOT RUN**.

## Running on Google Colab (recommended)

Open `notebooks/kaggle_runner.ipynb` in Colab, select a T4 GPU, and run the cells top to bottom.
- Outputs are written to `MyDrive/PeerBERT_outputs`.
- After a disconnect, run cells 1-4 again and continue. Finished steps are skipped and
  training resumes from its last epoch.
- The same notebook also runs on Kaggle, where outputs go to `/kaggle/working`.

If training runs out of GPU memory, edit `configs/train.yaml`:
1. Lower `training.batch_size` and raise `grad_accum`.
2. Or set `training.gradient_checkpointing: true`.

## Pipeline

```bash
export PRAD_WORK_DIR=/kaggle/working        # or /content on Colab; outputs go here
pip install -q -r requirements.txt          # torch is preinstalled on Colab/Kaggle
python -m pytest -q tests                   # toy-data tests

python scripts/00_inspect_datasets.py --include-semdetect   # read-only inspection
python scripts/01_run_audit.py                              # leakage audit only
python scripts/02_prepare_splits.py                         # audit + paper-level manifests
python scripts/02_prepare_splits.py --verify-only           # re-check manifests offline

python scripts/03_train_intellabs.py --smoke                # 1-epoch pipeline check
python scripts/03_train_intellabs.py                        # 3 seeds (train.yaml)
python scripts/04_train_semdetect.py
python scripts/05_evaluate.py --dataset intellabs
python scripts/05_evaluate.py --dataset semdetect

python scripts/06_run_baselines.py --dataset intellabs
python scripts/06_run_baselines.py --dataset semdetect
python scripts/07_run_ablation.py --dataset intellabs       # validation-based selection
python scripts/08_cross_dataset.py --source intellabs       # no retraining
python scripts/08_cross_dataset.py --source semdetect
python scripts/10_explain.py --dataset intellabs           # XAI (attention + Integrated Gradients)
python scripts/10_explain.py --dataset semdetect
python scripts/09_build_report.py                           # run last: collects everything
```

Every heavy step is restart-safe: loaded subsets are cached as parquet, training resumes from
`last_state.pt` and finished seeds are skipped (`done.json`), and predictions/baseline scores
are cached. Train on a single GPU (`--gpu`, sets `CUDA_VISIBLE_DEVICES` before torch loads).

## Data protocol (`src/data`)

- **IntelLabs**: `calibration` (the dataset card's "Training") is the train pool, `test` is the
  official test, and `extended` is optional and used for robustness only. Subsets are never merged.
  - Labels come from the `label` column, never from folder names. Each file is checked for:
    label agrees with the `custom_id` prefix, one human and one AI row per `review-id`,
    an integer `PaperId`, and conference/year matching the filename.
  - Human reviews repeated in every generator file are kept once; every dropped copy is
    logged, and copies with different text are counted.
  - Text = allow-listed narrative fields joined by a blank line (`configs/config.yaml`).
    Scores, flags and `code_of_conduct` (label-leaking) are never used.
  - A fail-late schema guard collects all unknown narrative-looking columns and raises once.
- **Sem-Detect**: the official train/test split is kept, and validation is cut from train papers.
  - The test split is not deduplicated; duplicates are only flagged.
  - `og_review_id` is metadata only.
- **Paper key** = `conference_year_paperid` (composite); the naive id is reported for information.
- **Splits are by paper only**. Compute caps keep whole papers. `test_baseline` is nested in the
  model test set, and all methods are compared on it.
- **Leakage audit** runs on the full partitions before caps and blocks training on FAIL. It checks:
  - paper overlap
  - review-pair integrity
  - duplicate ids
  - exact and near-duplicate text
  - distributions
  - length-only AUROC (overall and per generator)
- Every dropped row is listed in `outputs/manifests/<ds>/exclusions.parquet`.

## PeerBERT (`src/models`)

- Custom single-GPU PyTorch loop (no HF Trainer): AMP (bf16 if supported, else fp16 +
  GradScaler), gradient accumulation, non-finite-loss guard, early stopping and best checkpoint
  on validation Macro-F1, resume, truncation-rate reporting. `reference_compile=False`.
- Loss `L = L_cls + alpha * L_cons`:
  - `L_cls` is one of: unweighted CE, class-weighted CE (w_c = N / (K n_c), train split only),
    weighted CE with label smoothing 0.05, or weighted focal loss (gamma 2).
  - `L_cons` is the R-Drop symmetric KL between two dropout passes.
  - ModernBERT's default dropout is 0.0, so dropout is set to 0.1. The trainer asserts that the
    two train-mode passes differ.
- Final configuration: 3 seeds, mean +/- std. Ablations are selected on validation only.
- The ablation also includes the cost-sensitive margin loss of CoCoDet (Chen et al., ACL 2026)
  on a cosine classification head (`csm_cosine`):
  - margins m_base = m_cost = 0.25 and scale s = 30, as in that paper;
  - the cost margin targets human <-> AI confusions.
  CoCoDet itself is not a baseline.

## Baselines (`src/baselines`, `configs/baselines.yaml`)

| Baseline | Type | Notes |
|---|---|---|
| LogRank | zero-shot | -mean log-rank (1-indexed), shared token-stats LM |
| GLTR | trained classifier | rank-bucket fractions + logistic regression on a capped train sample |
| DetectLLM-LRR | zero-shot | mean NLL / mean log-rank (+epsilon), official-code formula |
| MFD | trained classifier | 4 LM features + LR ("MFD") or 4-16-16-16-1 MLP ("MFD-M"); preprint, disclosed |
| Fast-DetectGPT | zero-shot | analytic criterion verified against the official repo; surrogate = token-stats LM |
| Binoculars | zero-shot | formula verified against the official repo; smaller same-tokenizer pair on 16 GB GPUs (flagged) |
| RADAR | pretrained detector | `TrustSafeAI/RADAR-Vicuna-7B`; label order UNVERIFIED, so no index is assumed: the AI output is chosen on validation only |
| DNA-GPT | zero-shot | paper BScore with a local regenerator LM (deviation, disclosed) |

Thresholds are calibrated on validation only. On Sem-Detect, baselines see human/ai rows only.
For rewrite rows, only a descriptive "fraction flagged as AI" is reported.

## Evaluation (`src/evaluation`)

- **Metrics**: Accuracy, Precision, Recall, F1, Macro/Weighted-F1, ROC-AUC, PR-AUC, confusion
  matrix, per-class metrics, AI TPR/FPR, and TPR at 1/5/10% FPR (only when n_neg supports it).
  Positive class = AI.
- **Thresholds**: rows at 0.5 and at the validation-calibrated threshold.
  Sem-Detect is reported natively as 3-class and as a human/ai binary view.
- **Statistics**: paper-level cluster bootstrap CIs, paired tests on shared rows, McNemar, and
  Holm correction.
- **Error analysis**: FP/FN tables and per-class / per-conference-year breakdowns.
- **Length check**: metrics per review-length tercile (word count), to show performance does not
  depend on review length.
- **Outputs**: `outputs/results/{intellabs,sem_detect}/`, including `comparison_table.{csv,json}`.

## Explainability (`src/xai`, `scripts/10_explain.py`)

No SHAP or LIME. Reviews are drawn from the real `test_baseline` predictions of one seed:
- The seed is the one with the median validation Macro-F1.
- Up to N reviews are taken per (true class, correct/incorrect) group, with a fixed seed.

Each review is explained for the model's predicted class with:
- **Attention** from an eager-attention copy: last layer, mean over layers, and rollout.
  The pooling (`cls`/`mean`) is read from the model config.
- **Integrated Gradients** (Captum) on the token embeddings. The baseline replaces content
  tokens with pad tokens (special tokens kept). Attributions are signed, and the
  completeness gap (`delta`) is reported.
- **Faithfulness** (ERASER): comprehensiveness and sufficiency at 10% and 20%, compared
  with a random ranking. Spearman agreement between |IG| and attention is also reported.

Outputs:
- `explanations.jsonl`
- `summary.json`
- `top_tokens.csv`: highest mean IG tokens per class
- `examples.html`: colour-coded reviews
