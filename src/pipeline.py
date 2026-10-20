"""
Review 2 pipeline: data quality assessment, cleaning, leakage-safe feature
engineering, data integration, and exploratory data analysis for the
Food Waste Prediction and Meal-Planning Optimization for Event Catering project.

Executes end to end from the raw CSV and writes every artefact referenced in
Review2_Report.pdf Section 8.2 (out/*.csv, figures/fig01-fig10.png).

Run: python src/pipeline.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "food_wastage_data.csv"
OUT_DIR = ROOT / "out"
FIG_DIR = ROOT / "figures"
SEED = 42

sns.set_theme(style="whitegrid")

CATEGORICAL_COLS = [
    "type_of_food", "event_type", "storage_conditions", "purchase_history",
    "seasonality", "preparation_method", "geographical_location", "pricing",
]
NUMERIC_COLS = ["number_of_guests", "quantity_of_food"]
TARGET = "wastage_food_amount"


# ---------------------------------------------------------------------------
# Section 1: load + standardise column names (DQ-06)
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Section 2: data quality assessment
# ---------------------------------------------------------------------------
def data_quality_assessment(df: pd.DataFrame) -> dict:
    report = {}
    report["shape"] = df.shape
    report["missing_values"] = int(df.isna().sum().sum())
    report["exact_duplicates"] = int(df.duplicated().sum())

    feature_cols = [c for c in df.columns if c != TARGET]
    dup_features = df.duplicated(subset=feature_cols, keep=False)
    conflict_mask = dup_features & ~df.duplicated(keep=False)
    report["feature_duplicate_conflict_rows"] = int(conflict_mask.sum())

    # Outliers on numeric columns (1.5x and 3.0x IQR, z-score)
    outlier_rows = []
    for col in NUMERIC_COLS + [TARGET]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        outlier_rows.append({
            "column": col,
            "outliers_1.5_iqr": int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()),
            "outliers_3.0_iqr": int(((df[col] < q1 - 3.0 * iqr) | (df[col] > q3 + 3.0 * iqr)).sum()),
            "z_gt_3": int((z.abs() > 3).sum()),
            "skew": round(float(df[col].skew()), 3),
            "kurtosis": round(float(df[col].kurtosis()), 3),
        })
    report["outlier_table"] = pd.DataFrame(outlier_rows)

    # Cross-field validity
    report["quantity_lt_guests"] = int((df["quantity_of_food"] < df["number_of_guests"]).sum())
    report["negative_waste"] = int((df[TARGET] < 0).sum())
    report["waste_exceeds_quantity"] = int((df[TARGET] > df["quantity_of_food"]).sum())

    # Value granularity
    combo = df.groupby(["number_of_guests", "quantity_of_food"]).size().sort_values(ascending=False)
    report["distinct_guest_qty_combos"] = int(combo.shape[0])
    report["top7_combo_coverage_pct"] = round(100 * combo.head(7).sum() / len(df), 1)
    report["distinct_target_values"] = int(df[TARGET].nunique())
    report["target_divisible_by_5_pct"] = round(100 * (df[TARGET] % 5 == 0).mean(), 1)

    # Class balance
    imbalance = {}
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts()
        imbalance[col] = round(float(counts.max() / counts.min()), 2)
    report["categorical_imbalance"] = imbalance

    return report


def build_dq_register(report: dict) -> pd.DataFrame:
    rows = [
        ("DQ-01", "Missing values", f"{report['missing_values']} across all columns", "No imputation required"),
        ("DQ-02", "Duplicate records", f"{report['exact_duplicates']} exact duplicates", "Remove all; impact quantified"),
        ("DQ-03", "Feature-duplicate conflicts", f"{report['feature_duplicate_conflict_rows']} rows, identical features, differing target", "Retain; used to bound attainable R2"),
        ("DQ-04", "Outliers", f"{int(report['outlier_table'].loc[report['outlier_table'].column==TARGET,'z_gt_3'].iloc[0])} target values with |z|>3", "Retain; influence tested"),
        ("DQ-05", "Cross-field logic", f"{report['quantity_lt_guests']} records with quantity < guests", "Flag as unit ambiguity; retain"),
        ("DQ-06", "Encoding", "UTF-8 BOM on header", "Read with utf-8-sig; snake_case rename"),
        ("DQ-07", "Value granularity", f"{report['distinct_guest_qty_combos']} combinations; {report['target_divisible_by_5_pct']}% round targets", "Recorded as a limitation"),
        ("DQ-08", "Class balance", f"max imbalance {max(report['categorical_imbalance'].values())}x", "Note for effect-size interpretation"),
    ]
    return pd.DataFrame(rows, columns=["id", "issue", "finding", "action"])


# ---------------------------------------------------------------------------
# Section 3: cleaning + validation contract
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned


def validate(df: pd.DataFrame, quantity_col: str = "quantity_of_food") -> None:
    """Six invariants; raises AssertionError (not a warning) on failure."""
    assert df.isna().sum().sum() == 0, "missing values present"
    assert df.duplicated().sum() == 0, "duplicate rows present"
    assert (df["number_of_guests"] > 0).all(), "non-positive guest count"
    assert (df["quantity_of_food"] > 0).all(), "non-positive quantity"
    if TARGET in df.columns:
        assert (df[TARGET] >= 0).all(), "negative target"
        assert (df[TARGET] <= df[quantity_col]).all(), "target exceeds quantity prepared"
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            assert df[col].notna().all(), f"unseen/null category in {col}"


# ---------------------------------------------------------------------------
# Section 4: partitioning, risk tiers, feature engineering (leakage-safe)
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def derive_risk_tiers(train_df: pd.DataFrame, test_df: pd.DataFrame):
    q1, q3 = train_df[TARGET].quantile([0.25, 0.75])

    def tier(y):
        return np.where(y < q1, "LOW", np.where(y >= q3, "HIGH", "MEDIUM"))

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["risk_tier"] = tier(train_df[TARGET])
    test_df["risk_tier"] = tier(test_df[TARGET])
    return train_df, test_df, float(q1), float(q3)


PRICING_ORDER = {"Low": 0, "Moderate": 1, "High": 2}
ACTION_MAP = {
    "LOW": "Normal preparation",
    "MEDIUM": "Reduce quantity of food by 10-15%",
    "HIGH": "Reduce quantity of food by 15-20% and flag for donation-partner alert",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["food_per_guest"] = df["quantity_of_food"] / (df["number_of_guests"] + 1)
    df["surplus_units"] = df["quantity_of_food"] - df["number_of_guests"]
    df["pricing_ordinal"] = df["pricing"].map(PRICING_ORDER)
    df["is_high_price"] = (df["pricing"] == "High").astype(int)
    df["is_sit_down"] = (df["preparation_method"] == "Sit-down Dinner").astype(int)
    df["guest_scale"] = pd.cut(
        df["number_of_guests"], bins=[0, 275, 350, 425, 10_000],
        labels=["small", "medium", "large", "very_large"],
    )
    if "risk_tier" in df.columns:
        df["prescriptive_action"] = df["risk_tier"].map(ACTION_MAP)
    return df


ONEHOT_COLS = ["type_of_food", "event_type", "storage_conditions", "purchase_history",
               "seasonality", "preparation_method", "geographical_location", "guest_scale"]


def prepare_inputs(raw_df: pd.DataFrame, design_columns: list[str], feature_cols: list[str]) -> pd.DataFrame:
    """Turns raw booking rows (snake_case columns, no target) into the exact design
    matrix the models were trained on. Used by the dashboard and batch scoring, so
    inference goes through the same feature engineering as training."""
    fe = engineer_features(raw_df)
    onehot = [c for c in ONEHOT_COLS if c in feature_cols]
    encoded = pd.get_dummies(fe[feature_cols], columns=onehot, dtype=int)
    return encoded.reindex(columns=design_columns, fill_value=0)


def leakage_checks(feature_cols: list[str], design_df: pd.DataFrame, target: pd.Series) -> None:
    forbidden_substrings = ["waste", "target", "wastage"]
    for col in feature_cols:
        assert not any(s in col.lower() for s in forbidden_substrings), f"target-derived name leaked: {col}"
    numeric = design_df[feature_cols].select_dtypes(include=[np.number])
    corr = numeric.apply(lambda s: s.corr(target)).abs()
    offenders = corr[corr > 0.95]
    assert offenders.empty, f"feature too correlated with target (possible leakage): {offenders.to_dict()}"


def encode(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    onehot_cols = [c for c in CATEGORICAL_COLS if c != "pricing" and c in feature_cols] + (
        ["guest_scale"] if "guest_scale" in feature_cols else []
    )
    passthrough_cols = [c for c in feature_cols if c not in onehot_cols]

    train_encoded = pd.get_dummies(train_df[feature_cols], columns=onehot_cols, dtype=int)
    test_encoded = pd.get_dummies(test_df[feature_cols], columns=onehot_cols, dtype=int)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    scaler = StandardScaler()
    scale_cols = [c for c in passthrough_cols if c not in ("pricing_ordinal", "is_high_price", "is_sit_down")]
    scale_cols = [c for c in scale_cols if c in train_encoded.columns]
    scaler.fit(train_encoded[scale_cols])

    return train_encoded, test_encoded, scaler, scale_cols


# ---------------------------------------------------------------------------
# Section 5: data integration -- segment benchmark table
# ---------------------------------------------------------------------------
def build_benchmark_table(train_df: pd.DataFrame) -> pd.DataFrame:
    grp = train_df.groupby(["pricing", "preparation_method"])[TARGET]
    bench = grp.agg(mean_waste="mean", median_waste="median", std_waste="std", n="size").reset_index()
    bench["mean_waste_per_guest"] = train_df.groupby(["pricing", "preparation_method"]).apply(
        lambda g: (g[TARGET] / (g["number_of_guests"] + 1)).mean(), include_groups=False
    ).values
    bench["reliable"] = bench["n"] >= 50
    return bench


def merge_benchmark(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    merged = df.merge(
        bench[["pricing", "preparation_method", "mean_waste", "median_waste", "std_waste", "mean_waste_per_guest"]],
        on=["pricing", "preparation_method"], how="left", validate="many_to_one",
    )
    assert len(merged) == before, "row-count changed during benchmark merge (fan-out)"
    assert merged["mean_waste"].notna().all(), "unmatched segment key in benchmark merge"
    return merged


def analytical_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    views = {
        "high_risk_bookings": df[df["risk_tier"] == "HIGH"],
        "premium_sit_down": df[(df["pricing"] == "High") & (df["preparation_method"] == "Sit-down Dinner")],
        "large_events_300plus": df[df["number_of_guests"] >= 300],
        "budget_finger_food": df[(df["pricing"] == "Low") & (df["preparation_method"] == "Finger Food")],
    }
    return views


# ---------------------------------------------------------------------------
# Section 6: exploratory data analysis
# ---------------------------------------------------------------------------
def run_eda(train_df: pd.DataFrame) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    insights = {}

    # Figure 1: univariate distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS + [TARGET]):
        sns.histplot(train_df[col], ax=ax, bins=20, color="#5B4FCF")
        ax.axvline(train_df[col].mean(), color="orange", linestyle="--", label="mean")
        ax.axvline(train_df[col].median(), color="black", linestyle=":", label="median")
        ax.set_title(f"{col}\nskew {train_df[col].skew():.2f} kurtosis {train_df[col].kurtosis():.2f}")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 1 -- Univariate distributions (training split)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_univariate.png", dpi=110)
    plt.close(fig)

    # Figure 2: categorical level balance
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        order = train_df[col].value_counts().index
        sns.countplot(y=train_df[col], order=order, ax=ax, color="#5B4FCF")
        counts = train_df[col].value_counts()
        imbalance = counts.max() / counts.min()
        ax.set_title(f"{col} (imbalance {imbalance:.1f}x)")
    fig.suptitle("Figure 2 -- Categorical level balance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_categorical_balance.png", dpi=110)
    plt.close(fig)

    # Figure 3: numeric drivers vs target
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.regplot(x=train_df[col], y=train_df[TARGET], ax=ax,
                    scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "red"})
        r = train_df[col].corr(train_df[TARGET])
        rho = train_df[col].corr(train_df[TARGET], method="spearman")
        ax.set_title(f"r={r:.3f}  rho={rho:.3f}")
    fig.suptitle("Figure 3 -- Numeric drivers against the target")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_numeric_drivers.png", dpi=110)
    plt.close(fig)

    # Categorical effect sizes: ANOVA + Kruskal-Wallis + eta^2
    effect_rows = []
    for col in CATEGORICAL_COLS:
        groups = [g[TARGET].values for _, g in train_df.groupby(col)]
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        grand_mean = train_df[TARGET].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((train_df[TARGET] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_total
        effect_rows.append({
            "feature": col, "k": len(groups), "F": round(f_stat, 2),
            "anova_p": p_anova, "kruskal_p": p_kw, "eta2": round(eta2, 4),
            "effect": "Large" if eta2 > 0.14 else ("Medium" if eta2 > 0.06 else "Negligible"),
        })
    effect_df = pd.DataFrame(effect_rows).sort_values("eta2", ascending=False)

    # Figure 4: the strongest categorical effects
    top2 = effect_df.head(2)["feature"].tolist()
    fig, axes = plt.subplots(1, len(top2), figsize=(6 * len(top2), 5))
    if len(top2) == 1:
        axes = [axes]
    for ax, col in zip(axes, top2):
        order = train_df.groupby(col)[TARGET].mean().sort_values().index
        sns.boxplot(x=col, y=TARGET, data=train_df, order=order, ax=ax, color="#A9C6FF")
        means = train_df.groupby(col)[TARGET].mean().reindex(order)
        ax.scatter(range(len(order)), means.values, color="orange", zorder=5, label="mean")
        ax.legend()
        eta2 = effect_df.set_index("feature").loc[col, "eta2"]
        ax.set_title(f"{col} (eta^2={eta2:.4f})")
    fig.suptitle("Figure 4 -- Strongest categorical effects")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_top_categorical_effects.png", dpi=110)
    plt.close(fig)

    # Figure 5: correlation matrix on engineered numeric features
    engineered = engineer_features(train_df)
    corr_cols = ["number_of_guests", "quantity_of_food", "food_per_guest", "surplus_units",
                 "pricing_ordinal", "is_high_price", "is_sit_down", TARGET]
    corr = engineered[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Figure 5 -- Pearson correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_correlation_matrix.png", dpi=110)
    plt.close(fig)

    # VIF (approximate, via 1/(1-R^2) of pairwise regression for the flagged pairs)
    def pair_vif(a, b):
        r = engineered[a].corr(engineered[b])
        return round(1 / (1 - r ** 2), 2)
    high_corr_pairs = []
    cols_no_target = [c for c in corr_cols if c != TARGET]
    for i, a in enumerate(cols_no_target):
        for b in cols_no_target[i + 1:]:
            r = engineered[a].corr(engineered[b])
            if abs(r) > 0.7:
                high_corr_pairs.append({"pair": f"{a} + {b}", "r": round(r, 3), "vif_approx": pair_vif(a, b)})
    vif_df = pd.DataFrame(high_corr_pairs)

    # Figure 6: pricing x preparation-method interaction
    pivot = train_df.pivot_table(index="pricing", columns="preparation_method", values=TARGET, aggfunc="mean")
    pivot = pivot.reindex(["Low", "Moderate", "High"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", ax=axes[0])
    axes[0].set_title("Mean waste by pricing x preparation method")
    for method in pivot.columns:
        axes[1].plot(pivot.index, pivot[method], marker="o", label=method)
    axes[1].legend()
    axes[1].set_title("Interaction plot -- non-parallel lines = non-additive")
    fig.suptitle("Figure 6 -- Pricing x preparation-method interaction")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_interaction.png", dpi=110)
    plt.close(fig)

    # Two-way interaction F-test (sequential least-squares ANOVA; valid for the
    # unbalanced pricing x preparation-method design)
    from eda_extended import two_way_anova
    anova = two_way_anova(train_df, "pricing", "preparation_method")
    f_interaction, p_interaction = anova["F_ab"], anova["p_ab"]
    df_interaction, df_within = (3 - 1) * (3 - 1), len(train_df) - 9
    insights["interaction_F"] = round(float(f_interaction), 2)
    insights["interaction_p"] = float(p_interaction)
    insights["interaction_df"] = (df_interaction, df_within)

    # Figure 7: guest count vs waste within each pricing tier
    fig, ax = plt.subplots(figsize=(8, 6))
    slopes = {}
    for tier, color in zip(["Low", "Moderate", "High"], ["#4C72B0", "#DD8452", "#55A868"]):
        sub = train_df[train_df["pricing"] == tier]
        slope, intercept, r, p, se = stats.linregress(sub["number_of_guests"], sub[TARGET])
        slopes[tier] = round(float(slope), 3)
        ax.scatter(sub["number_of_guests"], sub[TARGET], alpha=0.3, s=12, color=color)
        xs = np.linspace(sub["number_of_guests"].min(), sub["number_of_guests"].max(), 50)
        ax.plot(xs, intercept + slope * xs, color=color,
                label=f"{tier}: slope {slope:.3f}, R2 {r ** 2:.2f}")
    ax.legend()
    ax.set_title("Figure 7 -- Guest count vs waste, fitted separately within each pricing tier")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_slope_by_tier.png", dpi=110)
    plt.close(fig)
    insights["guest_slope_by_tier"] = slopes

    # Feature importance: impurity + permutation + mutual information
    feature_cols = ["number_of_guests", "quantity_of_food", "food_per_guest", "surplus_units",
                     "pricing_ordinal", "is_high_price", "is_sit_down"]
    from sklearn.inspection import permutation_importance
    from sklearn.feature_selection import mutual_info_regression

    X = engineered[feature_cols]
    y = engineered[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    impurity = pd.Series(rf.feature_importances_, index=feature_cols)
    perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=SEED, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=feature_cols)
    mi = pd.Series(mutual_info_regression(X, y, random_state=SEED), index=feature_cols)

    imp_df = pd.DataFrame({
        "impurity_train": impurity / impurity.sum(),
        "permutation_test": perm_imp / perm_imp.abs().sum() if perm_imp.abs().sum() else perm_imp,
        "mutual_information": mi / mi.sum() if mi.sum() else mi,
    }).sort_values("impurity_train", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    imp_df.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Figure 9 -- Feature importance under three methods")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_feature_importance.png", dpi=110)
    plt.close(fig)

    # Figure 10: residual diagnostics
    y_pred = rf.predict(X_te)
    residuals = y_te.values - y_pred
    pricing_te = train_df.loc[y_te.index, "pricing"] if set(y_te.index).issubset(train_df.index) else None
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("residual"); axes[0].set_title("residuals vs fitted")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("normal Q-Q of residuals")
    if pricing_te is not None:
        resid_df = pd.DataFrame({"pricing": pricing_te.values, "residual": residuals})
        sns.boxplot(x="pricing", y="residual", data=resid_df, order=["Low", "Moderate", "High"], ax=axes[2])
        axes[2].set_title("residual spread by pricing tier")
        resid_sd_by_tier = resid_df.groupby("pricing")["residual"].std().to_dict()
    else:
        resid_sd_by_tier = {}
    fig.suptitle("Figure 10 -- Residual diagnostics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_residual_diagnostics.png", dpi=110)
    plt.close(fig)
    insights["residual_sd_by_tier"] = {k: round(v, 2) for k, v in resid_sd_by_tier.items()}

    insights["pearson_guests"] = round(float(train_df["number_of_guests"].corr(train_df[TARGET])), 3)
    insights["pearson_quantity"] = round(float(train_df["quantity_of_food"].corr(train_df[TARGET])), 3)
    insights["effect_sizes"] = effect_df
    insights["vif_table"] = vif_df
    insights["feature_importance"] = imp_df.reset_index().rename(columns={"index": "feature"})

    return insights


def attainable_ceiling(cleaned_df: pd.DataFrame) -> dict:
    feature_cols = [c for c in cleaned_df.columns if c != TARGET]
    dup_mask = cleaned_df.duplicated(subset=feature_cols, keep=False)
    conflict_groups = cleaned_df[dup_mask].groupby(feature_cols)[TARGET]
    dof_all = int(conflict_groups.ngroups if conflict_groups.ngroups == 0 else sum(g.size - 1 for _, g in conflict_groups))
    var_all = float(pd.concat([g - g.mean() for _, g in conflict_groups]).pow(2).sum() / dof_all) if dof_all > 0 else None

    strong_cols = ["pricing", "preparation_method", "number_of_guests", "quantity_of_food"]
    strong_groups = cleaned_df.groupby(strong_cols)[TARGET]
    dof_strong = sum(max(g.size - 1, 0) for _, g in strong_groups)
    resid_var_strong = pd.concat([g - g.mean() for _, g in strong_groups]).pow(2).sum() / dof_strong
    ceiling = 1 - resid_var_strong / cleaned_df[TARGET].var()

    return {
        "dof_all_features": dof_all,
        "residual_variance_all_features": round(var_all, 3) if var_all else None,
        "dof_strong_features": int(dof_strong),
        "residual_variance_strong_features": round(float(resid_var_strong), 3),
        "implied_r2_ceiling": round(float(ceiling), 4),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    dq = data_quality_assessment(raw)
    register = build_dq_register(dq)
    register.to_csv(OUT_DIR / "data_quality_register.csv", index=False)

    cleaned = clean(raw)
    validate(cleaned)
    cleaned.to_csv(OUT_DIR / "cleaned_event_catering.csv", index=False)

    ceiling = attainable_ceiling(cleaned)

    train_df, test_df = split(cleaned)
    train_df, test_df, q1, q3 = derive_risk_tiers(train_df, test_df)

    ks_stat, ks_p = stats.ks_2samp(train_df[TARGET], test_df[TARGET])

    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)

    feature_cols = [
        "number_of_guests", "quantity_of_food", "food_per_guest", "surplus_units",
        "pricing_ordinal", "is_high_price", "is_sit_down", "guest_scale",
        "type_of_food", "event_type", "storage_conditions", "purchase_history",
        "seasonality", "geographical_location", "preparation_method",
    ]
    leakage_checks(feature_cols, train_fe, train_fe[TARGET])
    train_encoded, test_encoded, scaler, scale_cols = encode(train_fe, test_fe, feature_cols)

    bench = build_benchmark_table(train_fe)
    bench.to_csv(OUT_DIR / "segment_benchmark_table.csv", index=False)
    train_merged = merge_benchmark(train_fe, bench)
    test_merged = merge_benchmark(test_fe, bench)

    views = analytical_views(pd.concat([train_merged, test_merged], ignore_index=True))
    for name, view in views.items():
        view.to_csv(OUT_DIR / f"view_{name}.csv", index=False)

    insights = run_eda(train_df)
    import eda_extended
    eda_extended.run_extended_eda(train_df)
    insights["effect_sizes"].to_csv(OUT_DIR / "categorical_effect_sizes.csv", index=False)
    insights["feature_importance"].to_csv(OUT_DIR / "feature_importance.csv", index=False)
    if not insights["vif_table"].empty:
        insights["vif_table"].to_csv(OUT_DIR / "vif_table.csv", index=False)

    risk_dist_train = train_df["risk_tier"].value_counts(normalize=True).round(3).to_dict()
    risk_dist_test = test_df["risk_tier"].value_counts(normalize=True).round(3).to_dict()

    key_insights = pd.DataFrame([
        ("I-1", "Pricing tier is the dominant driver of waste",
         f"eta2={insights['effect_sizes'].set_index('feature').loc['pricing','eta2']}"),
        ("I-2", "Guest count and quantity of food correlate strongly with waste",
         f"r_guests={insights['pearson_guests']}, r_quantity={insights['pearson_quantity']}"),
        ("I-3", "Pricing and preparation method interact (rank reversal)",
         f"F={insights['interaction_F']}, p={insights['interaction_p']:.2e}"),
        ("I-4", "The guest-count effect is tier-dependent", f"slopes={insights['guest_slope_by_tier']}"),
        ("I-5", "Error variance is not constant across tiers", f"residual_sd={insights['residual_sd_by_tier']}"),
        ("I-6", "Attainable accuracy is bounded by irreducible label noise",
         f"implied_R2_ceiling={ceiling['implied_r2_ceiling']}"),
        ("I-7", "Duplicates were a source of potential train/test leakage",
         f"{dq['exact_duplicates']} exact duplicates removed before splitting"),
        ("I-8", "The dataset has limited value granularity (records logged in rounded units)",
         f"{dq['distinct_guest_qty_combos']} guest/quantity combinations; {dq['target_divisible_by_5_pct']}% of targets divisible by 5"),
        ("I-9", "Risk tiers are not evenly distributed", f"train={risk_dist_train}, test={risk_dist_test}"),
        ("I-10", "Train/test target distributions are statistically indistinguishable",
         f"KS p={ks_p:.3f}"),
    ], columns=["id", "insight", "evidence"])
    key_insights.to_csv(OUT_DIR / "key_insights.csv", index=False)

    manifest = {
        "raw_shape": dq["shape"],
        "cleaned_shape": list(cleaned.shape),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "risk_tier_cutpoints": {"Q1": q1, "Q3": q3},
        "risk_tier_distribution": {"train": risk_dist_train, "test": risk_dist_test},
        "ks_test_train_vs_test_target": {"statistic": round(float(ks_stat), 4), "p_value": round(float(ks_p), 4)},
        "attainable_ceiling": ceiling,
        "feature_columns": feature_cols,
        "scaled_columns": scale_cols,
        "design_matrix_columns": train_encoded.shape[1],
    }
    with open(OUT_DIR / "pipeline_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(json.dumps(manifest, indent=2, default=str))
    return {
        "train_df": train_fe, "test_df": test_fe,
        "train_merged": train_merged, "test_merged": test_merged,
        "train_encoded": train_encoded, "test_encoded": test_encoded,
        "scaler": scaler, "scale_cols": scale_cols,
        "feature_cols": feature_cols, "bench": bench,
        "q1": q1, "q3": q3,
    }


if __name__ == "__main__":
    run_pipeline()
