"""
Cell definitions for notebooks/Review3_Model_Building_and_Comparison.ipynb.
Built and executed by src/build_notebooks.py.
"""

REVIEW3_CELLS = [
("md", """# Review 3 -- Model Building and Comparison Analysis
**Food Waste Prediction and Meal-Planning Optimization for Event Catering**

This notebook executes the full Review 3 workflow end to end on the leakage-safe split from Review 2:

| Deliverable | Section |
|---|---|
| Extended EDA across every column (carried into model design) | A |
| 1. Model selection and justification | 1 |
| 2. Model development (train/test split + 5-fold CV) | 2 |
| 3. Model evaluation (MAE, MSE, RMSE, R2 / Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix) | 3 |
| 4. Model performance comparison (tables, charts, strengths and limitations) | 4 |
| 5. Hyperparameter tuning (Grid Search, Randomized Search) and before/after comparison | 5 |
| 6. Final model selection with statistical justification | 6 |
| End-to-end product: prescriptive engine and dashboard inference path | 7 |

Two tasks are solved: **regression** of `wastage_food_amount` and **classification** of the LOW / MEDIUM / HIGH risk tier."""),

("code", """%matplotlib inline
import sys, pathlib, json, warnings
ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Image, display
import pipeline as P
import eda_extended as E
import train_models as T
get_ipython().run_line_magic('matplotlib', 'inline')   # the modules force the Agg backend on import
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 30); pd.set_option('display.width', 200)

D = T.prepare_data()          # runs the Review 2 pipeline: clean -> split -> tiers -> features -> encode
train_df, test_df = D['artifacts']['train_df'], D['artifacts']['test_df']
print('Design matrix:', D['X_train'].shape, 'train /', D['X_test'].shape, 'test')
print('Risk-tier cut points (training quantiles): Q1 =', D['q1'], ' Q3 =', D['q3'])"""),

("md", """## A. Extended exploratory analysis across every column
Review 2 focused on the two dominant drivers. Here every categorical column is examined against waste, the
waste *rate*, the risk tier, and against each other. All statistics use the **training partition only**."""),

("code", """E.numeric_summary(train_df)"""),
("code", """cat = E.category_summary(train_df)
cat"""),
("code", """fig = E.fig_waste_by_every_category(train_df); display(fig); plt.close(fig)"""),
("code", """fig = E.fig_waste_rate_by_category(train_df); display(fig); plt.close(fig)"""),
("md", """**Reading:** waste *rate* (waste / quantity prepared) separates the same two columns as raw waste --
pricing tier and preparation method -- so the effect is not just a by-product of larger events preparing more food."""),
("code", """assoc = E.risk_tier_association(train_df)
assoc"""),
("code", """fig = E.fig_risk_tier_composition(train_df); display(fig); plt.close(fig)"""),
("code", """anova = pd.DataFrame([E.two_way_anova(train_df, a, b) for a, b in E.TWO_WAY_PAIRS + [('pricing','preparation_method')]])
anova"""),
("code", """fig = E.fig_two_way_heatmaps(train_df); display(fig); plt.close(fig)"""),
("code", """cv = E.cramers_v_matrix(train_df)
fig = E.fig_cramers_v(cv); display(fig); plt.close(fig)"""),
("code", """for fn in [E.fig_guests_quantity_map, E.fig_event_type_profile, E.fig_ecdf_by_segment,
           E.fig_waste_per_guest, E.fig_season_location_storage]:
    fig = fn(train_df); display(fig); plt.close(fig)"""),
("code", """fig = E.fig_pairplot(train_df); display(fig); plt.close(fig)"""),
("md", """**What the extended EDA changes in the modelling plan**

* Pricing (Cramer's V with tier ~0.55), guest-scale band (~0.38) and preparation method (~0.32) are the only strong
  associations with the risk tier. Food type, event type, storage, purchase history, seasonality and location are
  weak (V < 0.1). Location has a small main effect: Rural bookings have fewer HIGH tiers, partly because they prepare
  less food on average.
* Preparation method matters beyond the sit-down flag (Finger Food and Buffet differ), so it is now one-hot encoded
  in full: **33 design columns** instead of 30, which adds about 0.008 to cross-validated R2.
* The secondary columns show no meaningful two-way interactions (all interaction p > 0.1), apart from
  pricing x preparation method. A model that captures interactions automatically (tree ensembles) is therefore
  preferred over hand-built interaction terms.
* Waste depends on guests and quantity non-linearly *within* each pricing tier (Figure 14, ECDFs), so purely linear
  models are expected to under-fit."""),

("md", """## 1. Model selection and justification
Algorithms span four families, so the comparison says something about the *shape* of the problem and not
only about which library wins:"""),
("code", """reg_specs = T.regression_candidates(D['scale_cols'])
clf_specs = T.classification_candidates(D['scale_cols'])
pd.DataFrame([{'task':'regression','model':n,'search':s['search'],'rationale':s['why']} for n,s in reg_specs.items()] +
             [{'task':'classification','model':n,'search':s['search'],'rationale':s['why']} for n,s in clf_specs.items()])"""),
("md", """* **Baselines** (mean / most-frequent) give the floor every model must beat.
* **Linear / Logistic Regression**: interpretable and additive. They test whether the relationship is essentially linear.
* **KNN, SVM (RBF)**: non-parametric and kernel methods. They need feature scaling, which is done *inside* the CV
  pipeline so that every fold fits its own scaler.
* **Random Forest**: bagging of deep trees. It is robust and captures the pricing x service interaction with no manual terms.
* **HistGradientBoosting** and **XGBoost (`tree_method="hist"`)**: two histogram-based gradient-boosting
  implementations. Histogram binning suits this dataset well because guests, quantity and the target take only a
  small number of distinct values."""),

("md", """## 2-3. Model development and evaluation -- regression
Every model is trained twice: once with **library-default hyperparameters (baseline)**, and once after
**Grid / Randomized search** over 5-fold CV on the training partition. The 324-booking test partition is touched
only for the final scoring."""),
("code", """reg_table, reg_folds, reg_preds, reg_fitted, reg_params, reg_info, reg_searches, _ = T.run_regression(D)
reg_table[reg_table.stage=='baseline'].round(4)"""),
("code", """reg_table[reg_table.stage=='tuned'].sort_values('cv_R2_mean', ascending=False).round(4)"""),

("md", """## 2-3. Model development and evaluation -- classification
Macro-F1 is the primary metric. Accuracy would be misleading here, because predicting MEDIUM every time already
scores about 50%. Accuracy, balanced accuracy, macro precision and recall, weighted F1, one-vs-rest ROC-AUC,
log-loss and the recall of the HIGH tier (the tier that triggers donation alerts) are all reported too."""),
("code", """best_reg, runner_reg, _ = T.select_best(reg_table, 'cv_R2_mean', ['Naive (mean)'])
clf_table, clf_folds, clf_preds, clf_probas, clf_fitted, clf_params, clf_info, clf_searches, _ = \\
    T.run_classification(D, best_regressor=(best_reg, reg_fitted[best_reg]))
clf_table[clf_table.stage=='baseline'].round(4)"""),
("code", """clf_table[clf_table.stage=='tuned'].sort_values('cv_macro_F1_mean', ascending=False).round(4)"""),

("md", """## 4. Model performance comparison
Figures are regenerated from the objects above, saved to `figures/`, and shown inline."""),
("code", """from sklearn.base import clone
T.FIG_DIR.mkdir(exist_ok=True)
T.fig_metric_comparison(reg_table[reg_table.model!='Naive (mean)'], ['cv_R2_mean','test_R2','test_RMSE','test_MAE'],
                        'Figure 21 -- Regression model comparison, baseline vs tuned', 'fig21_regression_comparison.png',
                        {'test_RMSE': False, 'test_MAE': False})
display(Image(T.FIG_DIR / 'fig21_regression_comparison.png'))"""),
("code", """T.fig_pred_vs_actual(reg_preds, D['y_test_reg'], 'fig22_pred_vs_actual.png')
display(Image(T.FIG_DIR / 'fig22_pred_vs_actual.png'))"""),
("code", """clf_plot = clf_table[~clf_table.model.str.contains('Dummy|cut points')]
T.fig_metric_comparison(clf_plot, ['cv_macro_F1_mean','test_macro_F1','test_accuracy','test_ROC_AUC_ovr'],
                        'Figure 24 -- Classification model comparison, baseline vs tuned', 'fig24_classification_comparison.png', {})
display(Image(T.FIG_DIR / 'fig24_classification_comparison.png'))"""),
("code", """T.fig_confusion_grid(clf_preds, D['y_test_clf'], 'fig25_confusion_matrices.png')
display(Image(T.FIG_DIR / 'fig25_confusion_matrices.png'))"""),
("code", """best_clf, runner_clf, _ = T.select_best(clf_table, 'cv_macro_F1_mean', ['Dummy (most frequent)'])
T.fig_roc(clf_probas, D['y_test_clf'], best_clf, 'fig26_roc_curves.png')
display(Image(T.FIG_DIR / 'fig26_roc_curves.png'))"""),
("code", """folds = pd.DataFrame(reg_folds + clf_folds)
T.fig_cv_distribution(folds[folds.model!='Naive (mean)'], 'regression', 'Figure 31a -- 5-fold CV R2 per model',
                      'fig31a_cv_scores_regression.png', 'R2')
T.fig_cv_distribution(folds[folds.model!='Dummy (most frequent)'], 'classification', 'Figure 31b -- 5-fold CV macro-F1 per model',
                      'fig31b_cv_scores_classification.png', 'macro-F1')
display(Image(T.FIG_DIR / 'fig31a_cv_scores_regression.png')); display(Image(T.FIG_DIR / 'fig31b_cv_scores_classification.png'))"""),
("md", """### Strengths and limitations of each model

| Model | Strengths observed | Limitations observed |
|---|---|---|
| Linear / Logistic Regression | Fast, transparent coefficients, stable across folds | Additive form cannot represent the pricing x service interaction; lowest R2 / F1 of the real models |
| KNN | No training phase; captures local structure | Sensitive to scaling and to the one-hot dimensions; degrades on sparse guest/quantity combinations |
| SVR / SVM (RBF) | Largest gain from tuning (C, gamma, epsilon); smooth decision surface | Slower to tune; probability outputs need internal CV; still behind the tree ensembles |
| Random Forest | Strong out of the box; best HIGH-tier F1 (the donation-alert tier); little tuning needed | Large model on disk; averaging limits how sharply it can fit the round-number targets |
| HistGradientBoosting | Fast histogram boosting, competitive with no feature work | Most variable across random-search draws; slightly below XGBoost |
| XGBoost (hist) | Best regression on CV and test; best LOW-tier F1 and best test macro-F1; regularisation (lambda, alpha, gamma) controls over-fitting | Most hyperparameters; the default settings under-perform until tuned |

**Regressor + cut points** (classifying by thresholding the predicted amount) is clearly worse than a dedicated
classifier. 29% of bookings sit *exactly* on a cut point (20 or 35 units), so a small regression error flips the tier.
This justifies keeping two separate models."""),

("md", """## 5. Hyperparameter tuning and optimisation
Grid Search is used for small, discrete spaces (Linear/Ridge, KNN, SVM). Randomized Search with continuous
distributions is used for the tree ensembles (25-50 sampled configurations x 5 folds)."""),
("code", """pd.DataFrame([{'task':'regression','model':n, **reg_info[n], 'best_params': reg_params[n]} for n in reg_params] +
             [{'task':'classification','model':n, **clf_info[n], 'best_params': clf_params[n]} for n in clf_params])"""),
("code", """reg_ba = T.before_after(reg_table, 'cv_R2_mean', 'test_R2').round(4)
clf_ba = T.before_after(clf_table, 'cv_macro_F1_mean', 'test_macro_F1').round(4)
display(reg_ba); display(clf_ba)
T.fig_before_after(reg_ba, clf_ba, 'fig27_tuning_before_after.png')
display(Image(T.FIG_DIR / 'fig27_tuning_before_after.png'))"""),
("code", """T.fig_xgb_search_landscape(reg_searches['XGBoost'], clf_searches['XGBoost'], 'fig30_xgboost_search_landscape.png')
display(Image(T.FIG_DIR / 'fig30_xgboost_search_landscape.png'))"""),
("md", """**Reading:** tuning helps most where the defaults were poorly matched to the data. SVR gains the most, because
the default C is too small for a target in units. XGBoost is next: its default depth 6 and learning rate 0.3 over-fit,
and the tuned model uses shallower trees, a lower learning rate and more rounds. Random Forest and
HistGradientBoosting change only marginally, because their defaults were already close to optimal."""),

("md", """## 6. Final model selection
The rule fixed *before* looking at the test set: choose the model with the highest **tuned 5-fold CV score**. The
winner is then tested against the runner-up with the **Nadeau-Bengio corrected resampled t-test** on 5x3 repeated
CV. A naive paired t-test is over-confident here because the CV training sets overlap."""),
("code", """from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
sig = pd.DataFrame([
    {'task':'regression','best':best_reg,'runner_up':runner_reg,
     **T.corrected_ttest(clone(reg_fitted[best_reg]), clone(reg_fitted[runner_reg]), D['X_train'], D['y_train_reg'],
                         RepeatedKFold(n_splits=5, n_repeats=3, random_state=P.SEED), 'r2')},
    {'task':'classification','best':best_clf,'runner_up':runner_clf,
     **T.corrected_ttest(clone(clf_fitted[best_clf]), clone(clf_fitted[runner_clf]), D['X_train'], D['y_train_clf'],
                         RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=P.SEED), 'f1_macro')},
])
print('Selected regressor :', best_reg, '| runner-up:', runner_reg)
print('Selected classifier:', best_clf, '| runner-up:', runner_clf)
sig.round(4)"""),
("code", """from sklearn.metrics import classification_report, r2_score, f1_score
print(classification_report(D['y_test_clf'], clf_preds[best_clf], target_names=T.LABELS, digits=3))
T.fig_residuals(best_reg, D['y_test_reg'], reg_preds[best_reg], test_df, 'fig23_residuals_final_regressor.png')
display(Image(T.FIG_DIR / 'fig23_residuals_final_regressor.png'))"""),
("code", """reg_perm = T.grouped_permutation_importance(reg_fitted[best_reg], D['X_test'], D['y_test_reg'], r2_score)
clf_perm = T.grouped_permutation_importance(clf_fitted[best_clf], D['X_test'], D['y_test_clf'],
                                            lambda y, p: f1_score(y, np.asarray(p).astype(int), average='macro'))
gain = reg_fitted['XGBoost'].get_booster().get_score(importance_type='gain')
xgb_gain = pd.DataFrame({'feature': list(gain), 'gain': list(gain.values())}).sort_values('gain', ascending=False)
T.fig_feature_importance(reg_perm, clf_perm, xgb_gain, best_reg, best_clf, 'fig28_feature_importance_final.png')
display(Image(T.FIG_DIR / 'fig28_feature_importance_final.png'))
reg_perm.head(8)"""),
("code", """T.fig_learning_curves(clone(reg_fitted[best_reg]), clone(clf_fitted[best_clf]), D, best_reg, best_clf, 'fig29_learning_curves.png')
display(Image(T.FIG_DIR / 'fig29_learning_curves.png'))"""),
("code", """ceiling = json.load(open(ROOT / 'out' / 'pipeline_manifest.json'))['attainable_ceiling']['implied_r2_ceiling']
best_row = reg_table[(reg_table.stage=='tuned') & (reg_table.model==best_reg)].iloc[0]
print(f"Final regressor test R2 = {best_row.test_R2:.4f}, irreducible-noise ceiling estimated in Review 2 ~ {ceiling}")
print(f"-> the model captures {100*best_row.test_R2/ceiling:.1f}% of the attainable explained variance")"""),
("md", """**Justification.**
* The **XGBoost (hist)** regressor has the highest tuned CV R2 and the best test R2, RMSE and MAE. It comes close to
  the irreducible-noise ceiling estimated in Review 2, which leaves little room for any other model.
* For the risk tier, the CV-selected classifier and XGBoost are statistically indistinguishable (corrected t-test
  p > 0.05). Both reach a ROC-AUC of about 0.97, so the choice between them is not decisive. The dashboard exposes
  every tuned model so that this robustness can be checked interactively.
* The learning curves flatten, so the remaining error is noise in the labels rather than a lack of data or of model
  capacity."""),

("md", """## 7. End-to-end product: booking -> forecast -> tier -> action -> alert
The dashboard (`app/streamlit_app.py`) uses exactly this inference path: `pipeline.prepare_inputs`, followed by the
saved models in `models/`. Those models are written by `python src/train_models.py`, which runs the same code shown above."""),
("code", """import joblib
meta = json.load(open(ROOT / 'models' / 'metadata.json'))
regressor = joblib.load(ROOT / 'models' / 'regressor.pkl')
classifier = joblib.load(ROOT / 'models' / 'classifier.pkl')
bookings = pd.DataFrame([
    dict(number_of_guests=460, quantity_of_food=495, pricing='High', event_type='Wedding', preparation_method='Sit-down Dinner',
         type_of_food='Meat', storage_conditions='Refrigerated', purchase_history='Regular', seasonality='Winter', geographical_location='Urban'),
    dict(number_of_guests=300, quantity_of_food=380, pricing='Moderate', event_type='Corporate', preparation_method='Buffet',
         type_of_food='Vegetables', storage_conditions='Room Temperature', purchase_history='Occasional', seasonality='Summer', geographical_location='Suburban'),
    dict(number_of_guests=220, quantity_of_food=300, pricing='Low', event_type='Birthday', preparation_method='Finger Food',
         type_of_food='Baked Goods', storage_conditions='Refrigerated', purchase_history='Regular', seasonality='All Seasons', geographical_location='Rural'),
])
X = P.prepare_inputs(bookings, meta['design_matrix_columns'], meta['feature_columns_raw']).astype(float)
out = bookings[['number_of_guests','quantity_of_food','pricing','preparation_method']].copy()
out['predicted_waste'] = regressor.predict(X).round(1)
out['risk_tier'] = [T.LABELS[i] for i in classifier.predict(X).astype(int)]
out['action'] = out['risk_tier'].map(P.ACTION_MAP)
out['donation_alert'] = out['risk_tier'].eq('HIGH')
out"""),
("md", """Launch the interactive dashboard with:

```
./venv/Scripts/python -m streamlit run app/streamlit_app.py
```"""),
]
