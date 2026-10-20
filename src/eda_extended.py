"""
Extended exploratory data analysis across every column of the dataset.

Review 2 concentrated on the two dominant drivers (pricing tier and
preparation method). This module widens the lens to all eight categorical
columns and both numeric columns, and to how they interact with each other
and with the LOW/MEDIUM/HIGH risk tier.

Every statistic is computed on the training partition only, so nothing here
is informed by the held-out test set.

Figures written to figures/:
  fig08_waste_by_every_category.png    waste distribution for every level of every categorical column
  fig11_waste_rate_by_category.png     waste as % of food prepared, per level, with 95% CI
  fig12_two_way_heatmaps.png           mean waste for four pairs of categorical columns
  fig13_risk_tier_composition.png      LOW/MEDIUM/HIGH share within every categorical level
  fig14_guests_quantity_map.png        guests x quantity booking map coloured by waste
  fig15_event_type_profile.png         event type broken down by service style and food type
  fig16_cramers_v_matrix.png           association strength between all categorical columns
  fig17_ecdf_by_segment.png            cumulative distribution of waste by pricing tier and service style
  fig18_waste_per_guest.png            waste per guest by guest-scale band and event type
  fig19_pairplot_by_risk_tier.png      numeric and engineered features coloured by risk tier
  fig20_season_location_storage.png    seasonality x location x storage conditions

Tables written to out/:
  eda_numeric_summary.csv, eda_category_summary.csv, eda_cramers_v_matrix.csv,
  eda_risk_tier_association.csv, eda_two_way_anova.csv
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
FIG_DIR = ROOT / "figures"

TARGET = "wastage_food_amount"
CATEGORICAL_COLS = [
    "type_of_food", "event_type", "storage_conditions", "purchase_history",
    "seasonality", "preparation_method", "geographical_location", "pricing",
]
NUMERIC_COLS = ["number_of_guests", "quantity_of_food"]
TIER_ORDER = ["LOW", "MEDIUM", "HIGH"]
TIER_COLORS = {"LOW": "#2E9E6B", "MEDIUM": "#E3A21A", "HIGH": "#D1495B"}
LEVEL_ORDER = {
    "pricing": ["Low", "Moderate", "High"],
    "preparation_method": ["Finger Food", "Buffet", "Sit-down Dinner"],
    "seasonality": ["Summer", "Winter", "All Seasons"],
    "geographical_location": ["Rural", "Suburban", "Urban"],
}
ACCENT = "#5B4FCF"


def _order(df: pd.DataFrame, col: str) -> list:
    if col in LEVEL_ORDER:
        return [lvl for lvl in LEVEL_ORDER[col] if lvl in set(df[col])]
    return df.groupby(col)[TARGET].mean().sort_values().index.tolist()


def _with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """EDA-only derived columns. waste_pct / waste_per_guest use the target, so they
    are never model inputs -- they exist only to describe the training data."""
    d = df.copy()
    d["waste_pct"] = 100 * d[TARGET] / d["quantity_of_food"]
    d["waste_per_guest"] = d[TARGET] / d["number_of_guests"]
    d["food_per_guest"] = d["quantity_of_food"] / (d["number_of_guests"] + 1)
    d["guest_scale"] = pd.cut(d["number_of_guests"], bins=[0, 275, 350, 425, 10_000],
                              labels=["small", "medium", "large", "very_large"])
    return d


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = _with_derived(df)
    cols = NUMERIC_COLS + ["food_per_guest", TARGET, "waste_pct", "waste_per_guest"]
    rows = []
    for c in cols:
        s = d[c]
        rows.append({
            "column": c, "mean": s.mean(), "std": s.std(), "min": s.min(),
            "q1": s.quantile(0.25), "median": s.median(), "q3": s.quantile(0.75),
            "max": s.max(), "skew": s.skew(), "kurtosis": s.kurtosis(),
            "distinct_values": s.nunique(),
        })
    return pd.DataFrame(rows).round(3)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = _with_derived(df)
    rows = []
    for col in CATEGORICAL_COLS:
        for lvl, g in d.groupby(col):
            rows.append({
                "column": col, "level": lvl, "n": len(g), "share_pct": 100 * len(g) / len(d),
                "mean_waste": g[TARGET].mean(), "median_waste": g[TARGET].median(),
                "std_waste": g[TARGET].std(), "mean_waste_pct": g["waste_pct"].mean(),
                "mean_guests": g["number_of_guests"].mean(),
                "mean_quantity": g["quantity_of_food"].mean(),
                "high_risk_share_pct": 100 * (g["risk_tier"] == "HIGH").mean() if "risk_tier" in g else np.nan,
            })
    return pd.DataFrame(rows).round(3)


def cramers_v_matrix(df: pd.DataFrame) -> pd.DataFrame:
    d = _with_derived(df)
    cols = CATEGORICAL_COLS + ["guest_scale"]
    m = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            m.loc[a, b] = 1.0 if a == b else cramers_v(d[a], d[b])
    return m.round(3)


def risk_tier_association(df: pd.DataFrame) -> pd.DataFrame:
    d = _with_derived(df)
    rows = []
    for col in CATEGORICAL_COLS + ["guest_scale"]:
        table = pd.crosstab(d[col], d["risk_tier"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        v = cramers_v(d[col], d["risk_tier"])
        rows.append({"feature": col, "chi2": round(chi2, 2), "dof": dof, "p_value": p,
                     "cramers_v": round(v, 3),
                     "association": "Strong" if v > 0.3 else ("Moderate" if v > 0.1 else "Weak/none")})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def _rss(df: pd.DataFrame, terms: list[str]) -> tuple[float, int]:
    """Residual sum of squares and model rank of an OLS fit on one-hot encoded terms."""
    X = pd.get_dummies(df[terms].astype(str), drop_first=True, dtype=float) if terms else pd.DataFrame(index=df.index)
    X.insert(0, "const", 1.0)
    y = df[TARGET].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    resid = y - X.to_numpy() @ beta
    return float(resid @ resid), int(np.linalg.matrix_rank(X.to_numpy()))


def two_way_anova(df: pd.DataFrame, a: str, b: str) -> dict:
    """Sequential (Type-I) two-way ANOVA a, b|a, a:b|a,b fitted by least squares,
    so it stays valid when the two factors are correlated (unbalanced design)."""
    d = df[[a, b, TARGET]].copy()
    d["_cell"] = d[a].astype(str) + " | " + d[b].astype(str)
    rss0, p0 = _rss(d, [])
    rss_a, p_a = _rss(d, [a])
    rss_ab_add, p_add = _rss(d, [a, b])
    rss_full, p_full = _rss(d, ["_cell"])
    df_w = len(d) - p_full
    ms_w = rss_full / df_w
    out = {"factor_a": a, "factor_b": b}
    for name, ss, dof in [("a", rss0 - rss_a, p_a - p0), ("b", rss_a - rss_ab_add, p_add - p_a),
                          ("ab", rss_ab_add - rss_full, p_full - p_add)]:
        dof = max(dof, 1)
        f = (ss / dof) / ms_w
        out[f"F_{name}"] = round(float(f), 2)
        out[f"p_{name}"] = float(stats.f.sf(f, dof, df_w))
        out[f"partial_eta2_{name}"] = round(float(ss / (ss + rss_full)), 4)
    return out


# ---------------------------------------------------------------------------
# Figures (each returns the Figure so notebooks can show it inline)
# ---------------------------------------------------------------------------
def fig_waste_by_every_category(df):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    gm = df[TARGET].mean()
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        order = _order(df, col)
        sns.boxplot(x=col, y=TARGET, data=df, order=order, ax=ax, color="#C9C3F5", fliersize=2)
        means = df.groupby(col)[TARGET].mean().reindex(order)
        ax.scatter(range(len(order)), means.values, color="#D1495B", zorder=5, s=30, label="mean")
        ax.axhline(gm, color="grey", linestyle=":", linewidth=1, label="overall mean")
        groups = [g[TARGET].values for _, g in df.groupby(col)]
        p = stats.kruskal(*groups).pvalue
        ax.set_title(f"{col}\nKruskal-Wallis p={p:.2g}", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Figure 8 -- Waste distribution for every level of every categorical column (training split)")
    fig.tight_layout()
    return fig


def fig_waste_rate_by_category(df):
    d = _with_derived(df)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    overall = d["waste_pct"].mean()
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        order = _order(d, col)
        g = d.groupby(col)["waste_pct"].agg(["mean", "std", "size"]).reindex(order)
        ci = 1.96 * g["std"] / np.sqrt(g["size"])
        ax.bar(range(len(order)), g["mean"], yerr=ci, color=ACCENT, alpha=0.85, capsize=4)
        ax.axhline(overall, color="#D1495B", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, fontsize=8)
        ax.set_title(col, fontsize=10)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, m + 0.15, f"{m:.2f}%\nn={n}", ha="center", fontsize=7)
    axes[0, 0].set_ylabel("waste as % of food prepared")
    axes[1, 0].set_ylabel("waste as % of food prepared")
    fig.suptitle("Figure 11 -- Waste rate (waste / quantity prepared) by category, with 95% CI; dashed = overall mean")
    fig.tight_layout()
    return fig


TWO_WAY_PAIRS = [
    ("event_type", "type_of_food"),
    ("seasonality", "storage_conditions"),
    ("geographical_location", "pricing"),
    ("purchase_history", "event_type"),
]


def fig_two_way_heatmaps(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (a, b) in zip(axes.flat, TWO_WAY_PAIRS):
        pivot = df.pivot_table(index=a, columns=b, values=TARGET, aggfunc="mean")
        pivot = pivot.reindex(index=_order(df, a), columns=_order(df, b))
        counts = df.pivot_table(index=a, columns=b, values=TARGET, aggfunc="size").reindex_like(pivot)
        annot = pivot.round(1).astype(str) + "\n(n=" + counts.astype(int).astype(str) + ")"
        sns.heatmap(pivot, annot=annot, fmt="", cmap="YlOrRd", ax=ax, cbar_kws={"label": "mean waste"})
        res = two_way_anova(df, a, b)
        ax.set_title(f"{a} x {b}\ninteraction F={res['F_ab']}, p={res['p_ab']:.2g}", fontsize=10)
    fig.suptitle("Figure 12 -- Mean waste for pairs of secondary categorical columns")
    fig.tight_layout()
    return fig


def fig_risk_tier_composition(df):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        order = _order(df, col)
        share = pd.crosstab(df[col], df["risk_tier"], normalize="index").reindex(index=order, columns=TIER_ORDER).fillna(0)
        bottom = np.zeros(len(order))
        for tier in TIER_ORDER:
            ax.bar(range(len(order)), share[tier] * 100, bottom=bottom, color=TIER_COLORS[tier], label=tier)
            for i, v in enumerate(share[tier] * 100):
                if v > 7:
                    ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white")
            bottom += share[tier].to_numpy() * 100
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f"{col}  (Cramer's V={cramers_v(df[col], df['risk_tier']):.2f})", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Figure 13 -- Risk-tier composition within each categorical level")
    fig.tight_layout()
    return fig


def fig_guests_quantity_map(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    agg = df.groupby(["number_of_guests", "quantity_of_food"])[TARGET].agg(["mean", "size"]).reset_index()
    sc = axes[0].scatter(agg["number_of_guests"], agg["quantity_of_food"], c=agg["mean"],
                         s=20 + agg["size"] * 3, cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.3)
    lim = [min(df["number_of_guests"].min(), df["quantity_of_food"].min()) - 10,
           max(df["number_of_guests"].max(), df["quantity_of_food"].max()) + 10]
    axes[0].plot(lim, lim, color="grey", linestyle="--", linewidth=1, label="quantity = guests")
    axes[0].set_xlabel("number_of_guests"); axes[0].set_ylabel("quantity_of_food")
    axes[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axes[0], label="mean waste")
    axes[0].set_title(f"Booking map: {len(agg)} distinct guest/quantity combinations (size = bookings)")
    d = _with_derived(df)
    for tier in ["Low", "Moderate", "High"]:
        sub = d[d["pricing"] == tier]
        sns.kdeplot(sub["food_per_guest"], ax=axes[1], label=tier, fill=True, alpha=0.3, warn_singular=False)
    axes[1].set_xlabel("food_per_guest (quantity / guests)")
    axes[1].set_title("Preparation intensity by pricing tier")
    axes[1].legend(title="pricing")
    fig.suptitle("Figure 14 -- How guest count and quantity prepared combine")
    fig.tight_layout()
    return fig


def fig_event_type_profile(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ev_order = _order(df, "event_type")
    sns.violinplot(x="event_type", y=TARGET, hue="preparation_method", data=df, order=ev_order,
                   hue_order=LEVEL_ORDER["preparation_method"], ax=axes[0], cut=0, inner="quartile",
                   density_norm="width")
    axes[0].set_title("Waste by event type, split by service style")
    axes[0].legend(fontsize=8, title="preparation")
    mix = pd.crosstab(df["event_type"], df["preparation_method"], normalize="index").reindex(
        index=ev_order, columns=LEVEL_ORDER["preparation_method"]) * 100
    mix.plot(kind="barh", stacked=True, ax=axes[1], colormap="Set2")
    axes[1].set_xlabel("% of bookings"); axes[1].set_title("Service-style mix per event type")
    axes[1].legend(fontsize=8)
    sns.pointplot(x="type_of_food", y=TARGET, hue="event_type", data=df, ax=axes[2],
                  order=_order(df, "type_of_food"), dodge=0.4, errorbar=("ci", 95), markersize=4)
    axes[2].set_title("Mean waste (95% CI) by food type and event type")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].legend(fontsize=8)
    fig.suptitle("Figure 15 -- Event-type profile")
    fig.tight_layout()
    return fig


def fig_cramers_v(matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(matrix.astype(float), mask=mask, annot=True, fmt=".2f", cmap="mako_r", vmin=0, vmax=1, ax=ax)
    ax.set_title("Figure 16 -- Cramer's V association between categorical columns (bias-corrected)")
    fig.tight_layout()
    return fig


def fig_ecdf_by_segment(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.ecdfplot(data=df, x=TARGET, hue="pricing", hue_order=LEVEL_ORDER["pricing"], ax=axes[0])
    axes[0].set_title("ECDF of waste by pricing tier")
    sns.ecdfplot(data=df, x=TARGET, hue="preparation_method", hue_order=LEVEL_ORDER["preparation_method"], ax=axes[1])
    axes[1].set_title("ECDF of waste by preparation method")
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    for ax in axes:
        ax.axvline(q1, color="grey", linestyle=":", linewidth=1)
        ax.axvline(q3, color="grey", linestyle=":", linewidth=1)
        ax.text(q1, 0.02, " Q1", fontsize=8); ax.text(q3, 0.02, " Q3", fontsize=8)
    fig.suptitle("Figure 17 -- Cumulative distribution of waste; dotted lines = risk-tier cut points")
    fig.tight_layout()
    return fig


def fig_waste_per_guest(df):
    d = _with_derived(df)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x="guest_scale", y="waste_per_guest", data=d, ax=axes[0], color="#A9C6FF")
    axes[0].set_title("Waste per guest by guest-scale band")
    sns.barplot(x="event_type", y="waste_per_guest", hue="pricing", hue_order=LEVEL_ORDER["pricing"],
                data=d, ax=axes[1], errorbar=("ci", 95), order=_order(d, "event_type"))
    axes[1].set_title("Mean waste per guest by event type and pricing tier")
    fig.suptitle("Figure 18 -- Waste per guest (descriptive only; target-derived, never a model input)")
    fig.tight_layout()
    return fig


def fig_pairplot(df):
    d = _with_derived(df)
    d["surplus_units"] = d["quantity_of_food"] - d["number_of_guests"]
    cols = ["number_of_guests", "quantity_of_food", "food_per_guest", "surplus_units", TARGET]
    g = sns.pairplot(d[cols + ["risk_tier"]], hue="risk_tier", hue_order=TIER_ORDER, palette=TIER_COLORS,
                     corner=True, plot_kws={"alpha": 0.45, "s": 14}, diag_kind="kde")
    g.figure.suptitle("Figure 19 -- Numeric and engineered features coloured by risk tier", y=1.01)
    return g.figure


def fig_season_location_storage(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    sns.barplot(x="seasonality", y=TARGET, hue="geographical_location", data=df,
                order=LEVEL_ORDER["seasonality"], hue_order=LEVEL_ORDER["geographical_location"],
                ax=axes[0], errorbar=("ci", 95), palette="crest")
    axes[0].set_title("Mean waste by season and location (95% CI)")
    sns.barplot(x="storage_conditions", y=TARGET, hue="type_of_food", data=df, ax=axes[1],
                errorbar=("ci", 95), palette="flare")
    axes[1].set_title("Mean waste by storage condition and food type (95% CI)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Figure 20 -- Operational context columns: season, location, storage")
    fig.tight_layout()
    return fig


FIGURES = [
    ("fig08_waste_by_every_category.png", fig_waste_by_every_category),
    ("fig11_waste_rate_by_category.png", fig_waste_rate_by_category),
    ("fig12_two_way_heatmaps.png", fig_two_way_heatmaps),
    ("fig13_risk_tier_composition.png", fig_risk_tier_composition),
    ("fig14_guests_quantity_map.png", fig_guests_quantity_map),
    ("fig15_event_type_profile.png", fig_event_type_profile),
    ("fig17_ecdf_by_segment.png", fig_ecdf_by_segment),
    ("fig18_waste_per_guest.png", fig_waste_per_guest),
    ("fig19_pairplot_by_risk_tier.png", fig_pairplot),
    ("fig20_season_location_storage.png", fig_season_location_storage),
]


def run_extended_eda(train_df: pd.DataFrame, save: bool = True) -> dict:
    """Builds every extended-EDA table and figure from the training partition."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    tables = {
        "numeric_summary": numeric_summary(train_df),
        "category_summary": category_summary(train_df),
        "cramers_v_matrix": cramers_v_matrix(train_df),
        "risk_tier_association": risk_tier_association(train_df),
        "two_way_anova": pd.DataFrame([two_way_anova(train_df, a, b) for a, b in
                                       TWO_WAY_PAIRS + [("pricing", "preparation_method")]]),
    }
    if save:
        tables["numeric_summary"].to_csv(OUT_DIR / "eda_numeric_summary.csv", index=False)
        tables["category_summary"].to_csv(OUT_DIR / "eda_category_summary.csv", index=False)
        tables["cramers_v_matrix"].to_csv(OUT_DIR / "eda_cramers_v_matrix.csv")
        tables["risk_tier_association"].to_csv(OUT_DIR / "eda_risk_tier_association.csv", index=False)
        tables["two_way_anova"].to_csv(OUT_DIR / "eda_two_way_anova.csv", index=False)

        for fname, fn in FIGURES:
            fig = fn(train_df)
            fig.savefig(FIG_DIR / fname, dpi=110, bbox_inches="tight")
            plt.close(fig)
        fig = fig_cramers_v(tables["cramers_v_matrix"])
        fig.savefig(FIG_DIR / "fig16_cramers_v_matrix.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    return tables
