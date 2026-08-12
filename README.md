# Blackwoods Customer Decision Intelligence

Reproducible case study for an RMIT data-science assignment, tailored to the Blackwoods Junior Data Scientist role. The project investigates two decision horizons:

1. Which established customers are most likely to purchase again in the next 100 days?
2. Which web sessions are most likely to end in a purchase?

The two public datasets are analysed separately because they represent different organisations, units and identifiers. The work compares a class-weighted regularised logistic regression with a class-weighted RBF support vector machine. The SVM is the model not used in the author's previous assessed coursework.

## Data Sources

- [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), DOI: 10.24432/C5BW33, CC BY 4.0.
- [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), DOI: 10.24432/C5F88Q, CC BY 4.0.

Raw data is not committed. `src/run_analysis.py` downloads the official UCI archives automatically when the expected files are absent.

## Reproduce the Analysis

Python 3.12 was used for the verified run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_analysis.py
```

The run uses seed 42 and writes processed features, model metrics, bootstrap intervals, paired comparisons, lift tables and figures into `data/processed/` and `outputs/`.

To compile the report after setting the user-owned Overleaf and GitHub links in `report/overleaf/00_macros.tex`:

```bash
cd report/overleaf
latexmk -pdf -interaction=nonstopmode -halt-on-error 00_main.tex
```

## Evaluation Design

- Stratified 75:25 train/holdout split.
- Five-fold stratified training-only grid search, optimising average precision.
- Out-of-fold training scores select the F1 threshold.
- Untouched holdout set provides final metrics.
- 1,000 bootstrap resamples provide 95% intervals.
- Paired bootstrap compares SVM and logistic average precision.
- Top-20% precision, lift and captured positives translate ranking into resource-constrained decisions.
- Permutation importance is calculated on holdout data with 25 repeats.

## Headline Results

| Outcome | Model | Average precision | ROC AUC | F1 | Top-20% lift |
|---|---|---:|---:|---:|---:|
| Future repeat purchase | Logistic regression | 0.810 | 0.736 | 0.748 | 1.52× |
| Future repeat purchase | RBF SVM | 0.814 | 0.736 | 0.748 | 1.54× |
| Session conversion | Logistic regression | 0.663 | 0.904 | 0.647 | 3.63× |
| Session conversion | RBF SVM | 0.670 | 0.904 | 0.649 | 3.69× |

The paired SVM-minus-logistic average-precision intervals include zero for both outcomes, so the nonlinear model does not demonstrate a reliable gain. Logistic regression is therefore recommended as the explainable pilot baseline.

## Repository Structure

```text
data/                  Data provenance and generated feature tables
outputs/figures/       Publication-ready charts
outputs/tables/        Metrics, uncertainty, lift and insight tables
report/overleaf/       Exact LaTeX template source and embedded job listing
src/run_analysis.py    End-to-end reproducible analysis
src/create_job_ad_archive.py
video/                 Executive slide deck and own-voice recording script
submission_checklist.md
```

## Scope and Responsible Use

These historical, overseas UCI datasets are proxies; they are not Blackwoods data. The results demonstrate a decision-support method and must not be interpreted as evidence about contemporary Blackwoods customers. Before operational use, rebuild and validate on current Australian industrial-supply data, confirm feature availability at decision time, calibrate displayed probabilities, quantify decision costs, review privacy, test subgroup errors and monitor drift.

## Reproducibility Notes

- The report cites every external source in ACM format.
- Data-source and methodological citations also appear as comments near the relevant code.
- The assignment PDF contains a transparent Condition 3 generative-AI attribution; the required separate declaration form must still be submitted through Canvas.
