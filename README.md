# Case-Studies-in-Data-Science-COSC2669-COSC2186
Two machine learning models (SVM classification, Gradient Boosting regression) built on real banking datasets for an RMIT Data Science case study — predicting term-deposit subscription and credit card limits.

# Individual Task 1: Part 1 — Data Science Case Study

Supplementary code repository for Individual Task 1 (COSC2669/COSC2816, RMIT University),
built around a real Westpac 2027 Data, Digital and AI Graduate Program (AI Science pathway) job listing.

Full report, appendix, and pseudo-code are in the submitted PDF — this repository provides the
runnable scripts and notebooks referenced there.

## Models

**Model 1 — SVM (`model1_bank_marketing_svm.py` / `.ipynb`)**
Predicts term-deposit subscription from client profile and campaign-contact history using the
UCI Bank Marketing dataset, withholding the leaky `duration` feature per the dataset's own documentation.

**Model 2 — Gradient Boosting Regressor (`model2_credit_card_gbr.py` / `.ipynb`)**
Predicts a cardholder's credit limit from account usage and repayment behaviour using the Kaggle
Credit Card Dataset for Clustering, benchmarked against a Linear Regression baseline.

## Data

Download the datasets separately and place them in a `data/` folder:
- `data/bank-additional-full.csv` — [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- `data/CC GENERAL.csv` — [Kaggle Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

## Running

```bash
pip install pandas numpy scikit-learn matplotlib
python3 model1_bank_marketing_svm.py
python3 model2_credit_card_gbr.py
```

Each script prints evaluation metrics, a confusion matrix or feature importances, and an interpretation
of the results, matching the analysis in the report.

## Files

- `model1_bank_marketing_svm.py` / `.ipynb` — Model 1
- `model2_credit_card_gbr.py` / `.ipynb` — Model 2
- `make_figures.py` — regenerates the four report figures
- `model1_output.txt` / `model2_output.txt` — captured console output
