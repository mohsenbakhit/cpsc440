# Text-Only BERT Transfer Report

## Data

This experiment uses the unified bill table in [../../data/normalized/bills.csv](../../data/normalized/bills.csv) and reconstructed bill text from Canada/US corpora. The prediction target is bill passage (`passed`).

Transfer protocol:
- Canada-trained model tested on Canada and US
- US-trained model tested on US and Canada

## Model Architecture

Notebook: [../../notebooks/text_only_bert_transfer.ipynb](../../notebooks/text_only_bert_transfer.ipynb)

Architecture components:
- frozen Legal-BERT encoder (`nlpaueb/legal-bert-base-uncased`)
- text input built from title + description/long_description + full text
- pooled text representation into a small MLP classifier head
- no tabular fusion features

## Design Choices

- Country-separated chronological splits for transfer evaluation.
- Validation-threshold selection (best F1) per train-country run.
- Weighted BCEWithLogits for class imbalance.
- Chunked tokenization with bounded chunks for runtime control.
- Focus on text signal only to compare against fusion variants.

## Results

Source report: [../results_raw/text_only_bert_transfer_report.json](../results_raw/text_only_bert_transfer_report.json)

| Train | Test | Setting | Accuracy | Balanced Acc | PR-AUC | ROC-AUC | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Canada | Canada | in_country_test | 0.8644 | 0.7914 | 0.4951 | 0.8526 | 0.3032 |
| Canada | US | transfer_test | 0.7883 | 0.5379 | 0.0116 | 0.5024 | 0.0208 |
| US | US | in_country_test | 0.9868 | 0.5910 | 0.0542 | 0.6946 | 0.1848 |
| US | Canada | transfer_test | 0.9585 | 0.5000 | 0.1023 | 0.7542 | 0.0000 |

## Interpretation

- Strongest result is Canada in-domain by PR-AUC and balanced accuracy.
- Canada -> US transfer is very weak (near-random ranking quality).
- US -> Canada shows reasonable ranking (ROC-AUC/PR-AUC), but thresholded classification collapses (F1 = 0), indicating calibration/threshold mismatch under shift.
- Accuracy alone is misleading in these imbalanced settings; PR-AUC and balanced accuracy are the primary indicators.