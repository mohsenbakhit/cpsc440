# Introduction-Only Baseline Models Guide

This file explains the introduction-only baseline workflow in [../../scripts/train_baselines_intro_only.py](../../scripts/train_baselines_intro_only.py), including which features are used, which models are trained, how evaluation is run, and how to read outputs.

## Purpose

The script trains baseline classifiers for bill passage prediction using only features available at or near introduction time from [../../data/normalized/bills.csv](../../data/normalized/bills.csv).

This version is intended to reduce downstream process leakage and give a more realistic early-prediction baseline.

## Input Data

The script expects normalized schema from [../../scripts/normalize.py](../../scripts/normalize.py), documented in [../../data/NORMALIZATION.md](../../data/NORMALIZATION.md).

Target:

- passed (0 or 1)

Feature groups in introduction-only mode:

- Numeric metadata: year, title_word_count, description_word_count, month_introduced, parliament_number, session_number, reinstated.
- Categorical metadata: bill_type, bill_type_raw, chamber, party.
- Text: combined text blob from title plus description.

Excluded to avoid process leakage:

- Reading-stage indicators, voting rollup features, committee-derived features, and other post-introduction process signals.

## Models Trained

The script trains 7 baseline models:

1. dummy_stratified
- Dummy baseline with stratified random predictions.

2. logreg_l2_meta
- Logistic regression with L2 regularization on metadata.
- Uses class_weight=balanced.

3. logreg_l1_meta
- Logistic regression with L1 regularization on metadata.
- Uses class_weight=balanced.

4. random_forest_meta
- Random forest on metadata.
- Uses class_weight=balanced_subsample.

5. logreg_l2_meta_text
- Logistic regression on metadata plus TF-IDF text.

6. linear_svm_meta_text
- Linear SVM on metadata plus TF-IDF text.

7. naive_bayes_text
- Multinomial Naive Bayes on text TF-IDF only.

## Evaluation Splits

The script evaluates four splits:

1. temporal_us
- Train on earlier US bills, test on later US bills.

2. temporal_canada
- Train on earlier Canada bills, test on later Canada bills.

3. transfer_us_to_canada
- Train on all US bills, test on all Canada bills.

4. transfer_canada_to_us
- Train on all Canada bills, test on all US bills.

## Metrics Reported

For each split and model, the script reports:

- pr_auc (primary ranking metric)
- roc_auc
- f1
- precision
- recall
- balanced_accuracy
- predicted_positive_rate
- fit_seconds
- n_train, n_test, train_positive_rate, test_positive_rate
- train_source_mix, test_source_mix

## Output Files

Running the intro-only script writes:

- [../results_raw/baseline_metrics_intro_only.csv](../results_raw/baseline_metrics_intro_only.csv)
- [../results_raw/baseline_metrics_intro_only.json](../results_raw/baseline_metrics_intro_only.json)

## Common Commands

Run from repo root:

```bash
python scripts/train_baselines_intro_only.py
```

Fast, parallel run:

```bash
python scripts/train_baselines_intro_only.py --fast-mode --parallel-evals --jobs -1
```

Other useful flags:

- --input PATH: override input CSV path.
- --outdir PATH: override output directory.
- --min-train-size N: skip splits with too little training data.
- --no-progress-bar: disable tqdm progress display.
- --show-warnings: show sklearn warnings.

Notes:

- --jobs -1 uses all logical CPU cores.
- With --parallel-evals, model-level parallelism is used and per-model threading is reduced to avoid oversubscription.
- --fast-mode reduces TF-IDF dimensionality and n-gram complexity for faster turnaround.

## How To Compare With Process-Aware Baselines

Use this file together with process-aware results from [../results_raw/baseline_metrics.csv](../results_raw/baseline_metrics.csv).

Suggested comparison flow:

1. Compare temporal split PR-AUC and recall to measure expected performance drop without leakage-prone features.
2. Compare transfer splits to check cross-country robustness.
3. Inspect model rank changes across the two runs to identify which models depended most on process features.
