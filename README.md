# Food Waste Prediction and Meal-Planning Optimization for Event Catering

A decision-support system for event caterers, built as the PODS (AD23532) mini project. Its pipeline is

**Booking Information -> Waste Prediction -> Risk Classification -> Preparation Recommendation -> Donation Alert**

| Review | Scope | Deliverable |
|---|---|---|
| Review 1 | Problem statement, dataset | `PODS-MINI PROJECT(Review 1).pdf` |
| Review 2 | Data quality, preprocessing, EDA | `Review2_Report.pdf`, `notebooks/Review2_Preprocessing_EDA.ipynb` |
| Review 3 | Model building, tuning, comparison, final product | `Review3_Report.pdf` / `.docx`, `notebooks/Review3_Model_Building_and_Comparison.ipynb`, `app/streamlit_app.py` |

## Dataset

- **Dataset:** "Food Wastage Data in Restaurant" (Kaggle)
- **Raw file:** `data/raw/food_wastage_data.csv` (1,782 records x 11 columns, already in the repo; no Kaggle login needed)

## Project structure

```
src/
  pipeline.py          Review 2: data quality, cleaning, validation, leakage-safe split,
                       feature engineering, benchmark-table integration, EDA (figs 1-10)
  eda_extended.py      Extended EDA across every column (figs 8, 11-20, out/eda_*.csv)
  train_models.py      Review 3: 7 regressors + 7 classifiers, baseline vs tuned
                       (GridSearchCV / RandomizedSearchCV), full metric suites,
                       significance test, figs 21-31, model export
  review3_cells.py     Cell definitions of the Review 3 notebook
  build_notebooks.py   Builds and executes both notebooks
  build_report.py      Generates Review3_Report.docx (+ PDF via MS Word) from the results
data/raw/              food_wastage_data.csv
out/                   All CSV / JSON result tables
figures/               fig01-fig31 PNG figures
models/                regressor.pkl, classifier.pkl, model_zoo.pkl (all tuned models),
                       scaler.pkl, benchmark_table.pkl, metadata.json
notebooks/             Executed Review 2 and Review 3 notebooks
app/streamlit_app.py   Interactive dashboard
```

## Setup

Use Python 3.11. A `venv` is already set up in this folder:

```
py -3.11 -m venv venv
./venv/Scripts/pip install -r requirements.txt
```

## Reproduce everything

```
./venv/Scripts/python src/pipeline.py          # cleaning, features, EDA (figs 1-20)          ~1 min
./venv/Scripts/python src/train_models.py      # training + tuning + comparison (figs 21-31) ~8 min
./venv/Scripts/python src/build_notebooks.py   # re-executes both notebooks                  ~6 min
./venv/Scripts/python src/build_report.py --pdf  # Review3_Report.docx and .pdf
```

`build_notebooks.py` runs on a Jupyter kernel named `pods-venv`. Register it once with
`./venv/Scripts/python -m ipykernel install --user --name pods-venv`.

## Running the dashboard

```
./venv/Scripts/python -m streamlit run app/streamlit_app.py
```

| Tab | What it does |
|---|---|
| Forecast | Booking form -> predicted waste, risk tier + probabilities, recommended action, suggested reduced quantity, donation alert, segment benchmark, what-if curve over quantity, per-feature explanation (XGBoost TreeSHAP), every tuned model's forecast |
| Data Explorer | Interactive EDA over every column with filters, CI bar charts, risk-tier mix, heatmaps, scatter, statistical tables, figure gallery |
| Model Performance | Final KPIs, tuned and baseline tables, before/after tuning, significance test, per-tier F1, best hyperparameters, figure gallery |
| Batch Forecast | Upload a CSV of bookings, then download forecasts, actions and alerts |
| About | Methodology and limitations |

The sidebar switches between any of the tuned models. The defaults are the selected final models.

## Results (held-out test set, 324 bookings)

| Task | Final model | CV score (5-fold) | Test |
|---|---|---|---|
| Regression | XGBoost (`tree_method="hist"`), tuned | R2 0.918 | R2 0.932, RMSE 2.71, MAE 1.89 |
| Classification | Random Forest, tuned | macro-F1 0.891 | macro-F1 0.879, accuracy 0.910, ROC-AUC 0.975 |

The tuned XGBoost classifier is statistically tied with the Random Forest (corrected resampled t-test,
p = 0.69). It reaches test macro-F1 0.889 and can be selected in the dashboard. Every model, metric and
hyperparameter is listed in `out/` and in the report.

## Methodology summary

- **Leakage-safe split.** The 80/20 split is made before anything is computed from the target, such as the
  risk-tier cut points or segment benchmarks.
- **Pre-event features only.** The model inputs are `food_per_guest`, `surplus_units`, `pricing_ordinal`,
  `is_high_price`, `is_sit_down` and `guest_scale`, plus one-hot encodings of the categorical columns,
  33 design columns in total. Two automated leakage guards are applied.
- **Models.** Seven per task: baseline, Linear/Logistic, KNN, SVM, Random Forest, HistGradientBoosting
  and XGBoost (hist). Scale-sensitive models scale inside the CV pipeline.
- **Tuning.** GridSearchCV for small discrete spaces and RandomizedSearchCV (25-50 configurations) for the
  ensembles, all on the same 5 folds. The before and after results are both reported.
- **Selection.** The model with the highest tuned CV score is selected, with a Nadeau-Bengio corrected t-test
  against the runner-up. The test set is used only for the final check.
- **Prescriptive engine.** LOW -> normal preparation; MEDIUM -> reduce quantity by 10-15%;
  HIGH -> reduce by 15-20% and raise a donation-partner alert.

## Limitations

- Limited value granularity: 61 guest/quantity combinations, and most waste values are multiples of 5.
  Re-validate on the caterer's own operational records before deployment.
- There is no date field, so weekday, seasonal-calendar and holiday effects are out of scope.
- Pricing is a 3-level category, which limits price analysis to tier comparisons.
- Forecasts inform preparation decisions. They never override guest-contract minimums, dietary
  requirements, service buffers or food-safety rules.
