# Late Fusion BERT Transfer Report

## Data

This experiment uses the unified bill table in [../../data/normalized/bills.csv](../../data/normalized/bills.csv) plus full bill text reconstructed from Canada and US raw text sources. The task is binary prediction of bill passage (`passed`).

Transfer evaluation follows chronological splits per country:
- train/validation/test on Canada
- train/validation/test on US
- two transfer directions: Canada -> US and US -> Canada

## Model Architecture

Notebook: [../../notebooks/late_fusion_bert.ipynb](../../notebooks/late_fusion_bert.ipynb)

Architecture components:
- frozen Legal-BERT text encoder (`nlpaueb/legal-bert-base-uncased`)
- categorical embeddings for transfer-safe metadata (`bill_type`, `chamber`)
- numeric projection for intro-time numeric metadata (`title_word_count`, `description_word_count`)
- fused MLP classifier head (nonlinear late fusion)

## Design Choices

- Chronological splits to reduce temporal leakage.
- Train-country-only metadata fitting (category maps and numeric normalization) for transfer safety.
- Weighted BCEWithLogits loss for class imbalance.
- Validation-derived decision threshold (best F1) applied to in-domain and transfer tests.
- Token-level perturbation summaries enabled; permutation metadata importance optional.

## Results

Source report: [../results_raw/late_fusion_transfer_report.json](../results_raw/late_fusion_transfer_report.json)

| Train | Test | Setting | Accuracy | Balanced Acc | PR-AUC | ROC-AUC | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Canada | Canada | in_country_test | 0.7126 | 0.7122 | 0.1157 | 0.7749 | 0.1704 |
| Canada | US | transfer_test | 0.6759 | 0.4922 | 0.0099 | 0.4762 | 0.0148 |
| US | US | in_country_test | 0.9571 | 0.6146 | 0.0354 | 0.7731 | 0.0899 |
| US | Canada | transfer_test | 0.9199 | 0.5204 | 0.0603 | 0.5265 | 0.0806 |

## Interpretation

- In-domain performance is consistently better than cross-country transfer.
- Canada -> US transfer is weak (near-random ranking by PR-AUC/ROC-AUC).
- US -> Canada transfer is better than Canada -> US, but still materially below in-domain quality.
- Accuracy remains high in some settings because of class imbalance; PR-AUC and balanced accuracy are more informative here.
