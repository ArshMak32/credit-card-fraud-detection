# Fraud Detection: Cost-Sensitive Temporal Benchmark

Portfolio version 2. See `MODEL_CARD.md`, tests and included `*results.json` reports for evidence and limits.

## Mac setup
Python 3.11 or 3.12:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```
The app displays included evaluation artifacts. It does not retrain on every interaction. Open the local URL shown in Terminal. Stop with Control+C.

## Reproduce

```bash
python download_data.py
python evaluate.py --data data/creditcard.csv --output real_results.json
```
For a fast synthetic smoke run: `python evaluate.py --output demo_results.json`. Do not present demo metrics as real-data results.

## Design
Consecutive 60/20/20 split after exact deduplication; Time excluded as a predictor. Compare a prior baseline, weighted logistic regression, random forest and histogram gradient boosting. Both model and decision threshold minimize validation cost only. Evaluate the frozen winner once on the final time block. Report AP, ROC AUC, precision, recall, Brier score, test confusion matrix, bootstrap intervals and permutation importance. Bootstrap intervals assume IID rows and may understate temporal uncertainty.

Cost defaults: missed fraud 25 units; false alarm 1 unit. These are illustrative assumptions, not measured bank losses. Change with `--miss-cost` and `--false-alarm-cost`; each rerun then represents a new experiment, not independent test evidence. No calibration, production monitoring or independent bank validation is claimed.

Data: ULB credit-card benchmark, mirrored as OpenML data ID 1597. https://www.openml.org/d/1597 . Dataset redistribution is omitted; check source terms before redistributing it. The original anonymous PCA features were generated upstream, so preprocessing chronology cannot be fully audited.
