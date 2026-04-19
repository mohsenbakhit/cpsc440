# Baseline Models Guide

This file explains what [scripts/train_baselines.py](scripts/train_baselines.py) does, which models it trains, how evaluation is set up, and how to read the results files.

## Purpose

The script trains interpretable baseline classifiers for bill passage prediction using the normalized dataset at [data/normalized/bills.csv](data/normalized/bills.csv).

It is designed to answer two questions:

1. How well do baseline models work within each country over time?
2. How well do models transfer across countries?

## Input Data

The script expects the schema produced by [scripts/normalize.py](scripts/normalize.py), documented in [data/NORMALIZATION.md](data/NORMALIZATION.md).

Target:

- passed (0 or 1)

Feature groups:

- Numeric metadata: year, title and description word counts, month introduced, process features, and source-specific count features.
- Categorical metadata: bill_type, bill_type_raw, chamber, party.
- Text: a combined text blob built from title plus description.

## Models Trained

The script builds these models:

1. dummy_stratified
- Dummy baseline with stratified random predictions.

2. logreg_l2_meta
- Logistic regression with L2 penalty on metadata.
- Class weighting enabled.

3. logreg_l1_meta
- Logistic regression with L1 penalty on metadata.
- Class weighting enabled.

4. random_forest_meta
- Random forest on metadata.
- Uses balanced_subsample class weighting.

5. logreg_l2_meta_text
- Logistic regression with L2 penalty on metadata plus TF-IDF text.

6. linear_svm_meta_text
- Linear SVM on metadata plus TF-IDF text.

7. naive_bayes_text
- Multinomial Naive Bayes using text TF-IDF features only.

## Evaluation Splits

The script evaluates four splits:

1. temporal_us
- Train on early US bills, test on later US bills.

2. temporal_canada
- Train on early Canada bills, test on later Canada bills.

3. transfer_us_to_canada
- Train on US, test on Canada.

4. transfer_canada_to_us
- Train on Canada, test on US.

## Metrics Reported

For each split and model, the script saves:

- pr_auc (primary metric)
- roc_auc
- f1
- precision
- recall
- balanced_accuracy
- predicted_positive_rate
- fit_seconds
- train and test sizes, source mix, and class rates

## Output Files

Running the script writes:

- [data/normalized/baseline_metrics.csv](data/normalized/baseline_metrics.csv)
- [data/normalized/baseline_metrics.json](data/normalized/baseline_metrics.json)

## Common Commands

From repo root:

```bash
python scripts/train_baselines.py --fast-mode --parallel-evals --jobs -1
```

Notes:

- Use jobs -1 to use all CPU cores.
- Do not use jobs -2 because values below 1 get clamped to 1.
- Use no-progress-bar if you prefer line-by-line updates over tqdm.

## What Current Results Show

Based on [data/normalized/baseline_metrics.csv](data/normalized/baseline_metrics.csv), the best model for each split is:

| Split | Best Model | PR-AUC | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| `temporal_canada` | `logreg_l1_meta` | 1.000 | 1.000 | 1.000 |
| `temporal_us` | `random_forest_meta` | 0.745 | 0.998 | 0.641 |
| `transfer_canada_to_us` | `random_forest_meta` | 0.087 | 0.805 | 0.000 |
| `transfer_us_to_canada` | `naive_bayes_text` | 0.188 | 0.587 | 0.000 |

Interpretation:

- The US temporal baseline is strong and looks plausible for an imbalanced classification task.
- Cross-country transfer is weak in both directions, which supports the project idea that learned signals do not transfer cleanly across legislative systems.
- The perfect Canada temporal score is a warning sign that downstream process features are carrying a lot of outcome information.