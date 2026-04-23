# Text+Tabular BERT Logistic Fusion Transfer Report

## Data

This experiment uses [../../data/normalized/bills.csv](../../data/normalized/bills.csv) plus reconstructed full text from Canada/US bill text corpora. The prediction target is bill passage (`passed`) with country-separated chronological splits.

Transfer protocol:
- Canada-trained model tested on Canada and US
- US-trained model tested on US and Canada

## Model Architecture

Notebook: [../../notebooks/text_tabular_bert_logistic_transfer.ipynb](../../notebooks/text_tabular_bert_logistic_transfer.ipynb)

Architecture components:
- frozen Legal-BERT text encoder (`nlpaueb/legal-bert-base-uncased`)
- categorical embeddings (`bill_type`, `chamber`)
- numeric projection (`year`, `title_word_count`, `description_word_count`, `month_introduced`)
- logistic fusion head: a single linear layer on the fused representation (no MLP hidden layers)

## Design Choices

- Transfer-safe preprocessing fitted on train-country only.
- Unknown categories mapped to UNK at transfer time.
- Weighted BCEWithLogits for imbalance.
- Validation F1 threshold selected per train-country run.
- Coefficient-based feature importance exported for interpretability.

Importance outputs:
- [../results_raw/text_tabular_logistic_feature_importance_canada.json](../results_raw/text_tabular_logistic_feature_importance_canada.json)
- [../results_raw/text_tabular_logistic_feature_importance_us.json](../results_raw/text_tabular_logistic_feature_importance_us.json)

## Results

Source report: [../results_raw/text_tabular_logistic_transfer_report.json](../results_raw/text_tabular_logistic_transfer_report.json)

| Train | Test | Setting | Accuracy | Balanced Acc | PR-AUC | ROC-AUC | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Canada | Canada | in_country_test | 0.8967 | 0.7515 | 0.2663 | 0.8515 | 0.3226 |
| Canada | US | transfer_test | 0.7197 | 0.5143 | 0.0086 | 0.5066 | 0.0170 |
| US | US | in_country_test | 0.9777 | 0.5809 | 0.0567 | 0.7963 | 0.1125 |
| US | Canada | transfer_test | 0.7611 | 0.5835 | 0.1011 | 0.6085 | 0.1192 |

## Interpretation

- The logistic fusion variant improves in-domain Canada metrics compared with the MLP late-fusion run.
- Canada -> US transfer remains very weak by PR-AUC despite acceptable accuracy.
- US -> Canada transfer is stronger than in the MLP late-fusion run (notably better recall-oriented behavior).
- This model offers better interpretability via linear fusion coefficients while preserving text+tabular fusion.
