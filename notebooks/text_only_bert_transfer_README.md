# Text-Only BERT Bill Transfer Classifier

This notebook trains a text-only Legal-BERT model to predict whether a bill passes, then evaluates how well the model transfers across countries.

## Model

- Base encoder: `nlpaueb/legal-bert-base-uncased`
- Input text: concatenation of bill title, description or long_description, and full bill text
- Architecture: frozen BERT encoder + small MLP classification head
- Training objective: weighted binary cross entropy
- Threshold selection: best F1 threshold on the validation split

The notebook is text-only, so it does not use any tabular metadata features such as sponsor, chamber, or bill type.

## Process

1. Load normalized bill metadata from `data/normalized/bills.csv`.
2. Load raw bill text for Canada and the US.
3. Build a `model_text` field from title, description, and full text.
4. Split each country chronologically into train, validation, and test sets.
5. Train two models:
   - Canada train -> Canada test and US test
   - US train -> US test and Canada test
6. Save the final evaluation table to `data/normalized/text_only_bert_transfer_report.json`.

## Results

The final report shows that the model performs much better in-domain than cross-country.

| Train Country | Test Country | Setting | Accuracy | Balanced Accuracy | PR-AUC | ROC-AUC | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Canada | Canada | in_country_test | 0.8644 | 0.7914 | 0.4951 | 0.8526 | 0.3032 |
| Canada | US | transfer_test | 0.7883 | 0.5379 | 0.0116 | 0.5024 | 0.0208 |
| US | US | in_country_test | 0.9868 | 0.5910 | 0.0542 | 0.6946 | 0.1848 |
| US | Canada | transfer_test | 0.9585 | 0.5000 | 0.1023 | 0.7542 | 0.0000 |

## Interpretation

- Canada -> Canada is the strongest result, with the best PR-AUC and balanced accuracy among the four settings.
- Canada -> US performs poorly, with PR-AUC close to zero, which suggests weak cross-country generalization.
- US -> Canada transfers better than Canada -> US in ranking terms, but the classification threshold does not produce useful positive predictions.
- Raw accuracy is not very informative here because the data is imbalanced, so PR-AUC, balanced accuracy, and F1 are more useful.

## Output Files

- Notebook: `notebooks/text_only_bert_transfer.ipynb`
- Report: `data/normalized/text_only_bert_transfer_report.json`

## Notes

The notebook currently uses `MAX_CHUNKS = 1`, which makes it faster but also means long bills are truncated to the first 256-token window of text.