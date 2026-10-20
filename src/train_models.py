"""
Review 3: model building, hyperparameter tuning and comparison analysis.

Two learning tasks are solved on the leakage-safe split produced by
src/pipeline.py:

  Regression      wastage_food_amount (units of food wasted per booking)
  Classification  LOW / MEDIUM / HIGH waste-risk tier (training-quantile cut points)

Stages
  1. Baseline   every candidate algorithm is trained with library-default
                hyperparameters and scored with 5-fold CV on the training
                partition plus the held-out test partition.
  2. Tuning     GridSearchCV (small, discrete spaces) or RandomizedSearchCV
                (large / continuous spaces) on the same 5 folds.
  3. Compare    before-vs-after tables, full metric suites, fold-level score
                distributions, a corrected resampled t-test between the two
                best tuned models, learning curves and feature importance.
  4. Select     the best model of each task is chosen by tuned CV score (never
                by test score) and exported with the whole tuned model zoo for
                the Streamlit dashboard.

XGBoost runs with the histogram tree method (tree_method="hist"), and
scikit-learn's HistGradientBoosting is included as a second histogram-based
gradient-boosting implementation.

Run: python src/train_models.py
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score, roc_curve, auc,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold,
    StratifiedKFold, cross_validate, learning_curve,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from pipeline import run_pipeline, TARGET, SEED, OUT_DIR, FIG_DIR, ROOT

warnings.filterwarnings("ignore", category=UserWarning)
MODELS_DIR = ROOT / "models"
LABELS = ["LOW", "MEDIUM", "HIGH"]
LABEL_MAP = {lbl: i for i, lbl in enumerate(LABELS)}
TIER_COLORS = {"LOW": "#2E9E6B", "MEDIUM": "#E3A21A", "HIGH": "#D1495B"}
STAGE_COLORS = {"baseline": "#B8B3E6", "tuned": "#5B4FCF"}
N_CV = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def prepare_data(artifacts: dict | None = None) -> dict:
    a = artifacts or run_pipeline()
    train_df, test_df = a["train_df"], a["test_df"]
    return {
        "artifacts": a,
        "X_train": a["train_encoded"].astype(float), "X_test": a["test_encoded"].astype(float),
        "y_train_reg": train_df[TARGET].to_numpy(dtype=float),
        "y_test_reg": test_df[TARGET].to_numpy(dtype=float),
        "y_train_clf": train_df["risk_tier"].map(LABEL_MAP).to_numpy(dtype=int),
        "y_test_clf": test_df["risk_tier"].map(LABEL_MAP).to_numpy(dtype=int),
        "scale_cols": a["scale_cols"], "q1": a["q1"], "q3": a["q3"],
        "kf": KFold(n_splits=N_CV, shuffle=True, random_state=SEED),
        "skf": StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED),
    }


def scaled(model, scale_cols):
    """Scale-sensitive models get a scaler *inside* the pipeline, so every CV fold
    fits its own scaler and no validation-fold statistic leaks into training."""
    pre = ColumnTransformer([("num", StandardScaler(), scale_cols)], remainder="passthrough",
                            verbose_feature_names_out=False)
    return Pipeline([("scale", pre), ("model", model)])


# ---------------------------------------------------------------------------
# Candidate models and their search spaces
# ---------------------------------------------------------------------------
def regression_candidates(scale_cols):
    m = "model__"
    return {
        "Naive (mean)": {"estimator": DummyRegressor(strategy="mean"), "search": None,
                          "why": "Floor: any useful model must beat predicting the training mean."},
        "Linear Regression": {
            "estimator": scaled(LinearRegression(), scale_cols),
            "tune_estimator": scaled(Ridge(), scale_cols), "search": "grid",
            "params": {f"{m}alpha": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]},
            "why": "Interpretable additive baseline; tests how far a linear fit gets (tuned as Ridge)."},
        "KNN": {
            "estimator": scaled(KNeighborsRegressor(), scale_cols), "search": "grid",
            "params": {f"{m}n_neighbors": [3, 5, 7, 10, 15, 20, 30], f"{m}weights": ["uniform", "distance"],
                       f"{m}p": [1, 2]},
            "why": "Non-parametric; bookings with similar guests/quantity/tier should waste alike."},
        "SVR": {
            "estimator": scaled(SVR(), scale_cols), "search": "grid",
            "params": {f"{m}C": [1, 10, 30, 100], f"{m}epsilon": [0.1, 0.5, 1.0, 2.0],
                       f"{m}gamma": ["scale", 0.01, 0.05]},
            "why": "RBF kernel captures smooth non-linearity; epsilon-insensitive loss resists label noise."},
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=SEED, n_jobs=1), "search": "random", "n_iter": 25,
            "params": {"n_estimators": [200, 300, 500, 700], "max_depth": [None, 6, 10, 16, 24],
                       "min_samples_leaf": [1, 2, 4, 8], "max_features": [1.0, 0.7, 0.5, "sqrt"]},
            "why": "Bagged trees model the pricing x service interaction without manual terms."},
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingRegressor(random_state=SEED), "search": "random", "n_iter": 40,
            "params": {"learning_rate": loguniform(0.01, 0.3), "max_iter": randint(100, 800),
                       "max_leaf_nodes": [7, 15, 31, 63], "min_samples_leaf": randint(5, 60),
                       "l2_regularization": loguniform(1e-4, 10), "max_depth": [None, 3, 4, 6, 8]},
            "why": "Histogram-binned gradient boosting; fast, strong on tabular data with few distinct values."},
        "XGBoost": {
            "estimator": XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=1),
            "search": "random", "n_iter": 50,
            "params": {"n_estimators": randint(150, 900), "max_depth": randint(2, 8),
                       "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
                       "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 12),
                       "gamma": uniform(0, 2), "reg_lambda": loguniform(0.1, 20),
                       "reg_alpha": loguniform(1e-3, 5), "max_bin": [64, 128, 256]},
            "why": "Regularised gradient boosting with histogram splits; usually the strongest tabular learner."},
    }


def classification_candidates(scale_cols):
    m = "model__"
    return {
        "Dummy (most frequent)": {"estimator": DummyClassifier(strategy="most_frequent"), "search": None,
                                   "why": "Floor: always predicts the majority tier (MEDIUM)."},
        "Logistic Regression": {
            "estimator": scaled(LogisticRegression(max_iter=3000), scale_cols), "search": "grid",
            "params": {f"{m}C": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100], f"{m}class_weight": [None, "balanced"]},
            "why": "Linear, calibrated probabilities; interpretable multinomial baseline."},
        "KNN": {
            "estimator": scaled(KNeighborsClassifier(), scale_cols), "search": "grid",
            "params": {f"{m}n_neighbors": [3, 5, 7, 10, 15, 20, 30], f"{m}weights": ["uniform", "distance"],
                       f"{m}p": [1, 2]},
            "why": "Instance-based; similar bookings should share a tier."},
        "SVM (RBF)": {
            "estimator": scaled(SVC(probability=True, random_state=SEED), scale_cols), "search": "grid",
            "params": {f"{m}C": [0.3, 1, 3, 10, 30, 100], f"{m}gamma": ["scale", 0.01, 0.05, 0.1],
                       f"{m}class_weight": [None, "balanced"]},
            "why": "Maximum-margin non-linear boundaries between adjacent tiers."},
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=SEED, n_jobs=1), "search": "random", "n_iter": 25,
            "params": {"n_estimators": [200, 300, 500, 700], "max_depth": [None, 6, 10, 16, 24],
                       "min_samples_leaf": [1, 2, 4, 8], "max_features": ["sqrt", 0.5, 0.7, 1.0],
                       "class_weight": [None, "balanced", "balanced_subsample"]},
            "why": "Robust ensemble; class weighting can lift the minority LOW tier."},
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingClassifier(random_state=SEED), "search": "random", "n_iter": 40,
            "params": {"learning_rate": loguniform(0.01, 0.3), "max_iter": randint(100, 800),
                       "max_leaf_nodes": [7, 15, 31, 63], "min_samples_leaf": randint(5, 60),
                       "l2_regularization": loguniform(1e-4, 10), "max_depth": [None, 3, 4, 6, 8],
                       "class_weight": [None, "balanced"]},
            "why": "Histogram-binned gradient boosting with native multi-class loss."},
        "XGBoost": {
            "estimator": XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=1, eval_metric="mlogloss"),
            "search": "random", "n_iter": 50,
            "params": {"n_estimators": randint(150, 900), "max_depth": randint(2, 8),
                       "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
                       "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 12),
                       "gamma": uniform(0, 2), "reg_lambda": loguniform(0.1, 20),
                       "reg_alpha": loguniform(1e-3, 5), "max_bin": [64, 128, 256]},
            "why": "Regularised multi-class boosting (softprob) with histogram splits."},
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_regressor(name, model, D, stage):
    X_tr, X_te, y_tr, y_te = D["X_train"], D["X_test"], D["y_train_reg"], D["y_test_reg"]
    cv = cross_validate(model, X_tr, y_tr, cv=D["kf"], n_jobs=-1,
                        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error",
                                 "mae": "neg_mean_absolute_error"})
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    pred = model.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    row = {
        "model": name, "stage": stage,
        "cv_R2_mean": cv["test_r2"].mean(), "cv_R2_std": cv["test_r2"].std(),
        "cv_RMSE_mean": -cv["test_rmse"].mean(), "cv_MAE_mean": -cv["test_mae"].mean(),
        "test_MAE": mean_absolute_error(y_te, pred), "test_MSE": mse, "test_RMSE": mse ** 0.5,
        "test_R2": r2_score(y_te, pred),
        "test_within_5_units_pct": 100 * np.mean(np.abs(y_te - pred) <= 5),
        "fit_time_s": fit_s,
    }
    folds = [{"task": "regression", "model": name, "stage": stage, "fold": i, "score": s}
             for i, s in enumerate(cv["test_r2"])]
    return row, model, pred, folds


def evaluate_classifier(name, model, D, stage):
    X_tr, X_te, y_tr, y_te = D["X_train"], D["X_test"], D["y_train_clf"], D["y_test_clf"]
    cv = cross_validate(model, X_tr, y_tr, cv=D["skf"], n_jobs=-1,
                        scoring={"f1": "f1_macro", "acc": "accuracy", "bacc": "balanced_accuracy"})
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    pred = np.asarray(model.predict(X_te)).astype(int)
    proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
    row = {
        "model": name, "stage": stage,
        "cv_macro_F1_mean": cv["test_f1"].mean(), "cv_macro_F1_std": cv["test_f1"].std(),
        "cv_accuracy_mean": cv["test_acc"].mean(), "cv_balanced_acc_mean": cv["test_bacc"].mean(),
        "test_accuracy": accuracy_score(y_te, pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "test_precision_macro": precision_score(y_te, pred, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_te, pred, average="macro", zero_division=0),
        "test_macro_F1": f1_score(y_te, pred, average="macro"),
        "test_weighted_F1": f1_score(y_te, pred, average="weighted"),
        "test_ROC_AUC_ovr": roc_auc_score(y_te, proba, multi_class="ovr", average="macro") if proba is not None else np.nan,
        "test_log_loss": log_loss(y_te, proba, labels=[0, 1, 2]) if proba is not None else np.nan,
        "test_HIGH_recall": recall_score(y_te, pred, labels=[2], average="macro", zero_division=0),
        "fit_time_s": fit_s,
    }
    folds = [{"task": "classification", "model": name, "stage": stage, "fold": i, "score": s}
             for i, s in enumerate(cv["test_f1"])]
    return row, model, pred, proba, folds


def tune(name, spec, X, y, cv, scoring):
    est = spec.get("tune_estimator", spec["estimator"])
    if spec["search"] == "grid":
        search = GridSearchCV(est, spec["params"], cv=cv, scoring=scoring, n_jobs=-1)
    else:
        search = RandomizedSearchCV(est, spec["params"], n_iter=spec["n_iter"], cv=cv, scoring=scoring,
                                    n_jobs=-1, random_state=SEED)
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0
    n_cand = len(search.cv_results_["params"])
    print(f"  tuned {name:22s} {spec['search']:6s} {n_cand:3d} candidates x {N_CV} folds "
          f"in {elapsed:5.1f}s -> CV {scoring}={search.best_score_:.4f}")
    params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
              for k, v in search.best_params_.items()}
    return search, params, {"search": spec["search"], "candidates": n_cand, "seconds": round(elapsed, 1)}


def unfitted_best(search):
    from sklearn.base import clone
    return clone(search.best_estimator_)


def corrected_ttest(model_a, model_b, X, y, cv, scoring):
    """Nadeau & Bengio (2003) corrected resampled t-test on repeated k-fold scores.
    The naive paired t-test is over-confident because CV training sets overlap."""
    sa = cross_validate(model_a, X, y, cv=cv, scoring=scoring, n_jobs=-1)["test_score"]
    sb = cross_validate(model_b, X, y, cv=cv, scoring=scoring, n_jobs=-1)["test_score"]
    d = sa - sb
    k = len(d)
    n_test_over_train = 1 / (N_CV - 1)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test_over_train) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_a": sa.mean(), "mean_b": sb.mean(), "mean_diff": d.mean(), "t": t, "p_value": p,
            "folds": k, "a_wins": int((d > 0).sum())}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_metric_comparison(table, metrics, title, fname, higher_better):
    models = table[table["stage"] == "tuned"]["model"].tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        piv = table.pivot(index="model", columns="stage", values=metric).reindex(models)
        y = np.arange(len(models))
        h = 0.38
        for j, stage in enumerate(["baseline", "tuned"]):
            if stage in piv:
                ax.barh(y + (j - 0.5) * h, piv[stage], height=h, color=STAGE_COLORS[stage], label=stage)
                for yi, v in zip(y + (j - 0.5) * h, piv[stage]):
                    if pd.notna(v):
                        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(metric + ("  (higher = better)" if higher_better.get(metric, True) else "  (lower = better)"),
                     fontsize=9)
        ax.margins(x=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, fname)


def fig_cv_distribution(folds_df, task, title, fname, ylabel):
    d = folds_df[(folds_df["task"] == task)]
    order = (d[d["stage"] == "tuned"].groupby("model")["score"].mean().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=d, x="model", y="score", hue="stage", order=order, palette=STAGE_COLORS, ax=ax)
    sns.stripplot(data=d, x="model", y="score", hue="stage", order=order, dodge=True, color="k",
                  size=3, ax=ax, legend=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    lo = d[~d["model"].str.contains("Naive|Dummy")]["score"].min()
    ax.set_ylim(max(lo - 0.05, -0.05), 1.0)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, fname)


def fig_pred_vs_actual(preds, y_test, fname):
    names = [n for n in preds if not n.startswith("Naive")]
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    lim = [y_test.min() - 3, y_test.max() + 3]
    for ax, n in zip(axes.flat, names):
        p = preds[n]
        ax.scatter(y_test, p, alpha=0.45, s=14, color="#5B4FCF")
        ax.plot(lim, lim, "r--", linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{n}\nR2={r2_score(y_test, p):.3f}  RMSE={mean_squared_error(y_test, p) ** 0.5:.2f}", fontsize=9)
        ax.set_xlabel("actual waste"); ax.set_ylabel("predicted waste")
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Figure 22 -- Predicted vs actual waste on the held-out test set (tuned regressors)")
    fig.tight_layout()
    _save(fig, fname)


def fig_residuals(name, y_test, pred, test_df, fname):
    res = y_test - pred
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    axes[0].scatter(pred, res, alpha=0.45, s=14)
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("residual"); axes[0].set_title("Residuals vs fitted")
    sns.histplot(res, bins=30, kde=True, ax=axes[1], color="#5B4FCF")
    axes[1].set_title(f"Residual distribution (mean {res.mean():.2f}, sd {res.std():.2f})")
    stats.probplot(res, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q")
    rd = pd.DataFrame({"pricing": test_df["pricing"].values, "preparation": test_df["preparation_method"].values,
                       "abs_error": np.abs(res)})
    sns.barplot(data=rd, x="pricing", y="abs_error", hue="preparation", order=["Low", "Moderate", "High"],
                ax=axes[3], errorbar=("ci", 95))
    axes[3].set_title("Mean absolute error by segment")
    axes[3].legend(fontsize=7)
    fig.suptitle(f"Figure 23 -- Residual diagnostics for the final regressor ({name}), held-out test set")
    fig.tight_layout()
    _save(fig, fname)


def fig_confusion_grid(preds, y_test, fname):
    names = [n for n in preds]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows))
    for ax, n in zip(axes.flat, names):
        cm = confusion_matrix(y_test, preds[n], labels=[0, 1, 2])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{c}\n{p:.0f}%" for c, p in zip(r, rp)] for r, rp in zip(cm, cm_pct)])
        sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Purples", vmin=0, vmax=100, cbar=False,
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(f"{n}\nmacro-F1={f1_score(y_test, preds[n], average='macro'):.3f}", fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Figure 25 -- Confusion matrices on the held-out test set (counts and row %)")
    fig.tight_layout()
    _save(fig, fname)


def fig_roc(probas, y_test, best_name, fname):
    yb = label_binarize(y_test, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for n, p in probas.items():
        if p is None or n.startswith("Dummy"):
            continue
        fpr_grid = np.linspace(0, 1, 200)
        tprs = []
        for k in range(3):
            fpr, tpr, _ = roc_curve(yb[:, k], p[:, k])
            tprs.append(np.interp(fpr_grid, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        axes[0].plot(fpr_grid, mean_tpr, label=f"{n} (AUC={auc(fpr_grid, mean_tpr):.3f})",
                     linewidth=2.5 if n == best_name else 1.2)
    axes[0].plot([0, 1], [0, 1], "k:", linewidth=1)
    axes[0].set_title("Macro-averaged one-vs-rest ROC, all tuned classifiers")
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].legend(fontsize=8, loc="lower right")
    p = probas[best_name]
    for k, lbl in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(yb[:, k], p[:, k])
        axes[1].plot(fpr, tpr, color=TIER_COLORS[lbl], linewidth=2, label=f"{lbl} vs rest (AUC={auc(fpr, tpr):.3f})")
    axes[1].plot([0, 1], [0, 1], "k:", linewidth=1)
    axes[1].set_title(f"Per-class ROC, final classifier ({best_name})")
    axes[1].set_xlabel("false positive rate")
    axes[1].legend(fontsize=9, loc="lower right")
    fig.suptitle("Figure 26 -- ROC curves on the held-out test set")
    fig.tight_layout()
    _save(fig, fname)


def fig_before_after(reg_ba, clf_ba, fname):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, ba, metric, title in [(axes[0], reg_ba, "cv_R2", "Regression: 5-fold CV R2"),
                                  (axes[1], clf_ba, "cv_macro_F1", "Classification: 5-fold CV macro-F1")]:
        b = ba[~ba["model"].str.contains("Naive|Dummy")].sort_values(f"{metric}_tuned", ascending=True)
        y = np.arange(len(b))
        ax.hlines(y, b[f"{metric}_baseline"], b[f"{metric}_tuned"], color="grey", linewidth=2)
        ax.scatter(b[f"{metric}_baseline"], y, color=STAGE_COLORS["baseline"], s=70, label="baseline (defaults)", zorder=3)
        ax.scatter(b[f"{metric}_tuned"], y, color=STAGE_COLORS["tuned"], s=70, label="tuned", zorder=3)
        for yi, (_, r) in zip(y, b.iterrows()):
            delta = r[f"{metric}_tuned"] - r[f"{metric}_baseline"]
            ax.text(max(r[f"{metric}_tuned"], r[f"{metric}_baseline"]) + 0.004, yi, f"{delta:+.3f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(b["model"])
        ax.set_title(title)
        ax.margins(x=0.12)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Figure 27 -- Effect of hyperparameter tuning (before -> after)")
    fig.tight_layout()
    _save(fig, fname)


def grouped_permutation_importance(model, X, y, scorer, n_repeats=20, seed=SEED):
    """Permutes all one-hot columns of an original feature together, so importance is
    reported per business feature (e.g. event_type) instead of per dummy column."""
    groups = {}
    for c in X.columns:
        base = next((g for g in ["type_of_food", "event_type", "storage_conditions", "purchase_history",
                                  "seasonality", "preparation_method", "geographical_location", "guest_scale"]
                     if c.startswith(g + "_")), c)
        groups.setdefault(base, []).append(c)
    rng = np.random.default_rng(seed)
    base_score = scorer(y, model.predict(X))
    rows = []
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[idx]
            drops.append(base_score - scorer(y, model.predict(Xp)))
        rows.append({"feature": g, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def fig_feature_importance(reg_perm, clf_perm, xgb_gain, reg_name, clf_name, fname):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, d, title, color in [
        (axes[0], reg_perm, f"Permutation importance -- {reg_name} (drop in test R2)", "#5B4FCF"),
        (axes[1], clf_perm, f"Permutation importance -- {clf_name} (drop in test macro-F1)", "#D1495B"),
    ]:
        d = d.sort_values("importance_mean")
        ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color=color, alpha=0.85)
        ax.set_title(title, fontsize=10)
    g = xgb_gain.head(15).sort_values("gain")
    axes[2].barh(g["feature"], g["gain"], color="#2E9E6B")
    axes[2].set_title("XGBoost regressor -- gain importance (top 15 design columns)", fontsize=10)
    fig.suptitle("Figure 28 -- What drives the predictions")
    fig.tight_layout()
    _save(fig, fname)


def fig_learning_curves(reg_model, clf_model, D, reg_name, clf_name, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sizes = np.linspace(0.1, 1.0, 8)
    for ax, model, X, y, cv, scoring, title in [
        (axes[0], reg_model, D["X_train"], D["y_train_reg"], D["kf"], "r2", f"{reg_name} (R2)"),
        (axes[1], clf_model, D["X_train"], D["y_train_clf"], D["skf"], "f1_macro", f"{clf_name} (macro-F1)"),
    ]:
        n, tr, va = learning_curve(model, X, y, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=-1)
        for scores, lbl, c in [(tr, "training", "#D1495B"), (va, "cross-validation", "#5B4FCF")]:
            ax.plot(n, scores.mean(1), marker="o", color=c, label=lbl)
            ax.fill_between(n, scores.mean(1) - scores.std(1), scores.mean(1) + scores.std(1), color=c, alpha=0.15)
        ax.set_xlabel("training bookings")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Figure 29 -- Learning curves of the final models")
    fig.tight_layout()
    _save(fig, fname)


def fig_xgb_search_landscape(reg_search, clf_search, fname):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for row, (search, label) in enumerate([(reg_search, "CV R2"), (clf_search, "CV macro-F1")]):
        r = pd.DataFrame(search.cv_results_)
        for ax, p in zip(axes[row], ["param_learning_rate", "param_n_estimators", "param_max_depth"]):
            ax.scatter(r[p].astype(float), r["mean_test_score"], c=r["mean_test_score"], cmap="viridis", s=30)
            best = r.loc[r["rank_test_score"] == 1].iloc[0]
            ax.scatter([float(best[p])], [best["mean_test_score"]], marker="*", s=250, color="#D1495B", label="best")
            if p == "param_learning_rate":
                ax.set_xscale("log")
            ax.set_xlabel(p.replace("param_", ""))
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
        axes[row][0].set_title(f"XGBoost {'regressor' if row == 0 else 'classifier'} random-search landscape", loc="left")
    fig.suptitle("Figure 30 -- XGBoost (hist) hyperparameter search: every sampled configuration")
    fig.tight_layout()
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_regression(D):
    specs = regression_candidates(D["scale_cols"])
    rows, folds, preds, fitted, best_params, search_info, searches = [], [], {}, {}, {}, {}, {}
    print("\n=== Regression: baseline (default hyperparameters) ===")
    for name, spec in specs.items():
        r, _, _, f = evaluate_regressor(name, spec["estimator"], D, "baseline")
        rows.append(r); folds += f
    print("=== Regression: tuning ===")
    for name, spec in specs.items():
        if spec["search"] is None:
            model = spec["estimator"]
        else:
            search, params, info = tune(name, spec, D["X_train"], D["y_train_reg"], D["kf"], "r2")
            model, best_params[name], search_info[name], searches[name] = unfitted_best(search), params, info, search
        r, m, p, f = evaluate_regressor(name, model, D, "tuned")
        rows.append(r); folds += f
        preds[name], fitted[name] = p, m
    table = pd.DataFrame(rows)
    return table, folds, preds, fitted, best_params, search_info, searches, specs


def run_classification(D, best_regressor=None):
    specs = classification_candidates(D["scale_cols"])
    rows, folds, preds, probas, fitted, best_params, search_info, searches = [], [], {}, {}, {}, {}, {}, {}
    print("\n=== Classification: baseline (default hyperparameters) ===")
    for name, spec in specs.items():
        r, _, _, _, f = evaluate_classifier(name, spec["estimator"], D, "baseline")
        rows.append(r); folds += f
    print("=== Classification: tuning ===")
    for name, spec in specs.items():
        if spec["search"] is None:
            model = spec["estimator"]
        else:
            search, params, info = tune(name, spec, D["X_train"], D["y_train_clf"], D["skf"], "f1_macro")
            model, best_params[name], search_info[name], searches[name] = unfitted_best(search), params, info, search
        r, m, p, pr, f = evaluate_classifier(name, model, D, "tuned")
        rows.append(r); folds += f
        preds[name], probas[name], fitted[name] = p, pr, m
    table = pd.DataFrame(rows)

    # Alternative route: classify by thresholding the best regressor's prediction
    # at the same training-quantile cut points used to define the tiers.
    if best_regressor is not None:
        name_r, reg = best_regressor
        pw = reg.predict(D["X_test"])
        tier_pred = np.where(pw < D["q1"], 0, np.where(pw >= D["q3"], 2, 1))
        y_te = D["y_test_clf"]
        table = pd.concat([table, pd.DataFrame([{
            "model": f"{name_r} regressor + cut points", "stage": "tuned",
            "test_accuracy": accuracy_score(y_te, tier_pred),
            "test_balanced_accuracy": balanced_accuracy_score(y_te, tier_pred),
            "test_precision_macro": precision_score(y_te, tier_pred, average="macro", zero_division=0),
            "test_recall_macro": recall_score(y_te, tier_pred, average="macro", zero_division=0),
            "test_macro_F1": f1_score(y_te, tier_pred, average="macro"),
            "test_weighted_F1": f1_score(y_te, tier_pred, average="weighted"),
            "test_HIGH_recall": recall_score(y_te, tier_pred, labels=[2], average="macro", zero_division=0),
        }])], ignore_index=True)
        preds[f"{name_r} regressor + cut points"] = tier_pred
    return table, folds, preds, probas, fitted, best_params, search_info, searches, specs


def before_after(table, metric_cv, metric_test):
    piv = table.pivot_table(index="model", columns="stage", values=[metric_cv, metric_test], aggfunc="first")
    out = pd.DataFrame({
        "model": piv.index,
        f"{metric_cv.replace('_mean', '')}_baseline": piv[(metric_cv, "baseline")].values,
        f"{metric_cv.replace('_mean', '')}_tuned": piv[(metric_cv, "tuned")].values,
        f"{metric_test}_baseline": piv[(metric_test, "baseline")].values,
        f"{metric_test}_tuned": piv[(metric_test, "tuned")].values,
    })
    out = out.dropna(subset=[f"{metric_cv.replace('_mean', '')}_baseline"])
    out[f"{metric_cv.replace('_mean', '')}_gain"] = out.iloc[:, 2] - out.iloc[:, 1]
    out[f"{metric_test}_gain"] = out.iloc[:, 4] - out.iloc[:, 3]
    return out.reset_index(drop=True)


def select_best(table, cv_metric, exclude):
    t = table[(table["stage"] == "tuned") & ~table["model"].isin(exclude) & table[cv_metric].notna()]
    t = t.sort_values(cv_metric, ascending=False)
    return t.iloc[0]["model"], t.iloc[1]["model"], t


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    D = prepare_data()
    test_df = D["artifacts"]["test_df"]

    # ---------------- Regression ----------------
    reg_table, reg_folds, reg_preds, reg_fitted, reg_params, reg_info, reg_searches, reg_specs = run_regression(D)
    best_reg, runner_reg, reg_rank = select_best(reg_table, "cv_R2_mean", ["Naive (mean)"])
    print(f"\nSelected regressor: {best_reg} (runner-up {runner_reg})")

    # ---------------- Classification ----------------
    clf_table, clf_folds, clf_preds, clf_probas, clf_fitted, clf_params, clf_info, clf_searches, clf_specs = \
        run_classification(D, best_regressor=(best_reg, reg_fitted[best_reg]))
    best_clf, runner_clf, clf_rank = select_best(clf_table, "cv_macro_F1_mean", ["Dummy (most frequent)"])
    print(f"Selected classifier: {best_clf} (runner-up {runner_clf})")

    # ---------------- Significance of the winner over the runner-up ----------------
    from sklearn.base import clone
    sig_rows = []
    for task, best, runner, fitted, y, cv, scoring in [
        ("regression", best_reg, runner_reg, reg_fitted, D["y_train_reg"],
         RepeatedKFold(n_splits=N_CV, n_repeats=3, random_state=SEED), "r2"),
        ("classification", best_clf, runner_clf, clf_fitted, D["y_train_clf"],
         RepeatedStratifiedKFold(n_splits=N_CV, n_repeats=3, random_state=SEED), "f1_macro"),
    ]:
        res = corrected_ttest(clone(fitted[best]), clone(fitted[runner]), D["X_train"], y, cv, scoring)
        sig_rows.append({"task": task, "metric": scoring, "best_model": best, "runner_up": runner, **res})
    sig_df = pd.DataFrame(sig_rows)
    print("\n=== Corrected resampled t-test (5x3 repeated CV) ===")
    print(sig_df.round(4).to_string(index=False))

    # ---------------- Tables ----------------
    reg_table = reg_table.round(4)
    clf_table = clf_table.round(4)
    reg_table[reg_table["stage"] == "baseline"].to_csv(OUT_DIR / "regression_baseline_metrics.csv", index=False)
    reg_table[reg_table["stage"] == "tuned"].to_csv(OUT_DIR / "regression_model_comparison.csv", index=False)
    clf_table[clf_table["stage"] == "baseline"].to_csv(OUT_DIR / "classification_baseline_metrics.csv", index=False)
    clf_table[clf_table["stage"] == "tuned"].to_csv(OUT_DIR / "classification_model_comparison.csv", index=False)
    reg_ba = before_after(reg_table, "cv_R2_mean", "test_R2").round(4)
    clf_ba = before_after(clf_table, "cv_macro_F1_mean", "test_macro_F1").round(4)
    reg_ba.to_csv(OUT_DIR / "regression_tuning_before_after.csv", index=False)
    clf_ba.to_csv(OUT_DIR / "classification_tuning_before_after.csv", index=False)
    pd.DataFrame(reg_folds + clf_folds).to_csv(OUT_DIR / "cv_fold_scores.csv", index=False)
    sig_df.round(6).to_csv(OUT_DIR / "final_model_significance.csv", index=False)

    tuning_log = {
        "regression": {n: {**reg_info[n], "best_params": reg_params[n], "why": reg_specs[n]["why"]} for n in reg_params},
        "classification": {n: {**clf_info[n], "best_params": clf_params[n], "why": clf_specs[n]["why"]} for n in clf_params},
    }
    with open(OUT_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(tuning_log, f, indent=2, default=str)
    pd.DataFrame(
        [{"task": "regression", "model": n, "rationale": s["why"]} for n, s in reg_specs.items()] +
        [{"task": "classification", "model": n, "rationale": s["why"]} for n, s in clf_specs.items()]
    ).to_csv(OUT_DIR / "model_selection_rationale.csv", index=False)

    best_pred = clf_preds[best_clf]
    report = classification_report(D["y_test_clf"], best_pred, target_names=LABELS, output_dict=True)
    pd.DataFrame(report).transpose().round(4).to_csv(OUT_DIR / "classification_report_best_model.csv")
    cm = confusion_matrix(D["y_test_clf"], best_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT_DIR / "confusion_matrix_best_model.csv")

    per_class = []
    for n, p in clf_preds.items():
        pr = precision_score(D["y_test_clf"], p, labels=[0, 1, 2], average=None, zero_division=0)
        rc = recall_score(D["y_test_clf"], p, labels=[0, 1, 2], average=None, zero_division=0)
        f1 = f1_score(D["y_test_clf"], p, labels=[0, 1, 2], average=None, zero_division=0)
        for k, lbl in enumerate(LABELS):
            per_class.append({"model": n, "tier": lbl, "precision": pr[k], "recall": rc[k], "f1": f1[k]})
    pd.DataFrame(per_class).round(4).to_csv(OUT_DIR / "per_class_metrics_all_models.csv", index=False)

    # Feature importance of the final models
    reg_perm = grouped_permutation_importance(reg_fitted[best_reg], D["X_test"], D["y_test_reg"], r2_score)
    clf_perm = grouped_permutation_importance(
        clf_fitted[best_clf], D["X_test"], D["y_test_clf"],
        lambda y, p: f1_score(y, np.asarray(p).astype(int), average="macro"))
    xgb_reg = reg_fitted["XGBoost"]
    gain = xgb_reg.get_booster().get_score(importance_type="gain")
    xgb_gain = pd.DataFrame({"feature": list(gain), "gain": list(gain.values())}).sort_values("gain", ascending=False)
    reg_perm.assign(task="regression", model=best_reg).round(5).to_csv(
        OUT_DIR / "feature_importance_final_regressor.csv", index=False)
    clf_perm.assign(task="classification", model=best_clf).round(5).to_csv(
        OUT_DIR / "feature_importance_final_classifier.csv", index=False)
    xgb_gain.round(3).to_csv(OUT_DIR / "xgboost_gain_importance.csv", index=False)

    # ---------------- Figures ----------------
    sns.set_theme(style="whitegrid")
    fig_metric_comparison(reg_table[reg_table["model"] != "Naive (mean)"],
                          ["cv_R2_mean", "test_R2", "test_RMSE", "test_MAE"],
                          "Figure 21 -- Regression model comparison, baseline vs tuned",
                          "fig21_regression_comparison.png",
                          {"test_RMSE": False, "test_MAE": False})
    fig_pred_vs_actual(reg_preds, D["y_test_reg"], "fig22_pred_vs_actual.png")
    fig_residuals(best_reg, D["y_test_reg"], reg_preds[best_reg], test_df, "fig23_residuals_final_regressor.png")
    clf_plot = clf_table[~clf_table["model"].str.contains("Dummy|cut points")]
    fig_metric_comparison(clf_plot, ["cv_macro_F1_mean", "test_macro_F1", "test_accuracy", "test_ROC_AUC_ovr"],
                          "Figure 24 -- Classification model comparison, baseline vs tuned",
                          "fig24_classification_comparison.png", {})
    fig_confusion_grid(clf_preds, D["y_test_clf"], "fig25_confusion_matrices.png")
    fig_roc(clf_probas, D["y_test_clf"], best_clf, "fig26_roc_curves.png")
    fig_before_after(reg_ba, clf_ba, "fig27_tuning_before_after.png")
    fig_feature_importance(reg_perm, clf_perm, xgb_gain, best_reg, best_clf, "fig28_feature_importance_final.png")
    fig_learning_curves(clone(reg_fitted[best_reg]), clone(clf_fitted[best_clf]), D, best_reg, best_clf,
                        "fig29_learning_curves.png")
    fig_xgb_search_landscape(reg_searches["XGBoost"], clf_searches["XGBoost"], "fig30_xgboost_search_landscape.png")
    folds_df = pd.DataFrame(reg_folds + clf_folds)
    fig_cv_distribution(folds_df[~folds_df["model"].isin(["Naive (mean)"])], "regression",
                        "Figure 31a -- 5-fold CV R2 per model (each dot = one fold)",
                        "fig31a_cv_scores_regression.png", "R2")
    fig_cv_distribution(folds_df[~folds_df["model"].isin(["Dummy (most frequent)"])], "classification",
                        "Figure 31b -- 5-fold CV macro-F1 per model (each dot = one fold)",
                        "fig31b_cv_scores_classification.png", "macro-F1")

    # ---------------- Persist artefacts for the dashboard ----------------
    a = D["artifacts"]
    joblib.dump(reg_fitted[best_reg], MODELS_DIR / "regressor.pkl")
    joblib.dump(clf_fitted[best_clf], MODELS_DIR / "classifier.pkl", compress=3)
    joblib.dump(a["scaler"], MODELS_DIR / "scaler.pkl")
    joblib.dump(a["bench"], MODELS_DIR / "benchmark_table.pkl")
    joblib.dump({"regressors": {k: v for k, v in reg_fitted.items() if k != "Naive (mean)"},
                 "classifiers": {k: v for k, v in clf_fitted.items() if k != "Dummy (most frequent)"}},
                MODELS_DIR / "model_zoo.pkl", compress=3)

    def row_dict(table, name):
        r = table[(table["stage"] == "tuned") & (table["model"] == name)].iloc[0]
        return {k: (None if pd.isna(v) else v) for k, v in r.items()}

    train_raw = a["train_df"]
    metadata = {
        "feature_columns_raw": a["feature_cols"],
        "design_matrix_columns": D["X_train"].columns.tolist(),
        "scale_columns": a["scale_cols"],
        "risk_tier_cutpoints": {"Q1": a["q1"], "Q3": a["q3"]},
        "regressor_name": best_reg,
        "regressor_runner_up": runner_reg,
        "regressor_uses_scaling": False,
        "regressor_best_params": reg_params.get(best_reg, {}),
        "classifier_name": best_clf,
        "classifier_runner_up": runner_clf,
        "classifier_uses_scaling": False,
        "classifier_label_map": LABEL_MAP,
        "classifier_best_params": clf_params.get(best_clf, {}),
        "regression_metrics": row_dict(reg_table, best_reg),
        "classification_metrics": row_dict(clf_table, best_clf),
        "training_ranges": {
            "number_of_guests": [int(train_raw["number_of_guests"].min()), int(train_raw["number_of_guests"].max())],
            "quantity_of_food": [int(train_raw["quantity_of_food"].min()), int(train_raw["quantity_of_food"].max())],
        },
        "train_rows": int(len(D["X_train"])), "test_rows": int(len(D["X_test"])),
        "xgboost_tree_method": "hist",
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n=== Regression (tuned) ===")
    print(reg_table[reg_table.stage == "tuned"][["model", "cv_R2_mean", "test_R2", "test_RMSE", "test_MAE"]].to_string(index=False))
    print("\n=== Classification (tuned) ===")
    print(clf_table[clf_table.stage == "tuned"][["model", "cv_macro_F1_mean", "test_macro_F1", "test_accuracy",
                                                  "test_ROC_AUC_ovr"]].to_string(index=False))
    print(f"\nSaved model artefacts to {MODELS_DIR}")
    return {"reg_table": reg_table, "clf_table": clf_table, "sig": sig_df, "metadata": metadata}


if __name__ == "__main__":
    main()
