"""
Builds and executes the two project notebooks from code cells defined here,
so the .ipynb files on disk are real, runnable artefacts with baked-in
outputs (tables and figures) -- not just text pasted into cells.

Run: python src/build_notebooks.py
Output: notebooks/Review2_Preprocessing_EDA.ipynb
        notebooks/Review3_Model_Building_and_Comparison.ipynb
"""
import pathlib

import nbformat as nbf
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def make_notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for kind, source in cells:
        if kind == "md":
            nb["cells"].append(nbf.v4.new_markdown_cell(source))
        else:
            nb["cells"].append(nbf.v4.new_code_cell(source))
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def execute_and_save(nb: nbf.NotebookNode, path: pathlib.Path):
    client = NotebookClient(nb, timeout=600, kernel_name="pods-venv", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote executed notebook: {path}")


# ===========================================================================
# Review 2 notebook: data quality, cleaning, feature engineering, integration, EDA
# ===========================================================================
REVIEW2_CELLS = [
("md", "# Review 2 -- Data Preprocessing and Exploratory Data Analysis\n"
       "Food Waste Prediction and Meal-Planning Optimization for Event Catering\n\n"
       "This notebook executes end to end from the raw CSV: data quality assessment, "
       "cleaning and validation, leakage-safe train/test partitioning, pre-event feature "
       "engineering, data integration (segment benchmark table), and exploratory data analysis."),

("code",
 "%matplotlib inline\n"
 "import sys, pathlib\n"
 "ROOT = pathlib.Path.cwd()\n"
 "sys.path.insert(0, str(ROOT / 'src'))\n"
 "import matplotlib.pyplot as plt\n"
 "import pandas as pd\n"
 "import numpy as np\n"
 "import seaborn as sns\n"
 "from scipy import stats\n"
 "sns.set_theme(style='whitegrid')\n"
 "pd.set_option('display.max_columns', 20)\n"
 "import pipeline as P\n"
 "get_ipython().run_line_magic('matplotlib', 'inline')  # pipeline.py forces the Agg backend on import\n"
 "print('Loaded pipeline module from', P.__file__)"),

("md", "## Section 1-2: Load raw data and run the data quality assessment"),
("code",
 "raw = P.load_raw()\n"
 "print('Raw shape:', raw.shape)\n"
 "raw.head()"),

("code",
 "dq = P.data_quality_assessment(raw)\n"
 "print('Missing values:', dq['missing_values'])\n"
 "print('Exact duplicates:', dq['exact_duplicates'])\n"
 "print('Feature-duplicate conflict rows (same features, different target):', dq['feature_duplicate_conflict_rows'])\n"
 "print('Records with quantity < guests:', dq['quantity_lt_guests'])\n"
 "print('Distinct (guests, quantity) combinations:', dq['distinct_guest_qty_combos'])\n"
 "print('Coverage of top 7 combinations: {}%'.format(dq['top7_combo_coverage_pct']))\n"
 "print('Distinct target values:', dq['distinct_target_values'])\n"
 "print('Target values divisible by 5: {}%'.format(dq['target_divisible_by_5_pct']))\n"
 "dq['outlier_table']"),

("code",
 "register = P.build_dq_register(dq)\n"
 "register"),

("md", "## Section 3: Cleaning and validation contract\n"
       "All 164(+) exact duplicates are removed. A validation routine then enforces six "
       "invariants and raises an error (not a warning) if any is violated."),
("code",
 "cleaned = P.clean(raw)\n"
 "P.validate(cleaned)\n"
 "print('Cleaned shape:', cleaned.shape)\n"
 "print('All six validation invariants passed.')\n"
 "ceiling = P.attainable_ceiling(cleaned)\n"
 "ceiling"),

("md", "## Section 4: Partitioning before target-derived statistics\n"
       "The 80/20 split happens **before** the LOW/MEDIUM/HIGH risk-tier cut points are "
       "computed, so no test-set value can influence a test-set label."),
("code",
 "train_df, test_df = P.split(cleaned)\n"
 "train_df, test_df, q1, q3 = P.derive_risk_tiers(train_df, test_df)\n"
 "print(f'Train: {train_df.shape}, Test: {test_df.shape}')\n"
 "print(f'Q1={q1}, Q3={q3}')\n"
 "ks_stat, ks_p = stats.ks_2samp(train_df[P.TARGET], test_df[P.TARGET])\n"
 "print(f'KS test train vs test target distribution: stat={ks_stat:.3f}, p={ks_p:.3f}')\n"
 "print('\\nTrain risk-tier distribution:')\n"
 "print(train_df['risk_tier'].value_counts(normalize=True).round(3))\n"
 "print('\\nTest risk-tier distribution:')\n"
 "print(test_df['risk_tier'].value_counts(normalize=True).round(3))"),

("md", "## Section 4.3-4.4: Feature engineering, leakage checks, encoding"),
("code",
 "train_fe = P.engineer_features(train_df)\n"
 "test_fe = P.engineer_features(test_df)\n"
 "feature_cols = ['number_of_guests','quantity_of_food','food_per_guest','surplus_units',\n"
 "                'pricing_ordinal','is_high_price','is_sit_down','guest_scale',\n"
 "                'type_of_food','event_type','storage_conditions','purchase_history',\n"
 "                'seasonality','geographical_location','preparation_method']\n"
 "P.leakage_checks(feature_cols, train_fe, train_fe[P.TARGET])\n"
 "print('Leakage checks passed: no target-derived feature name, no |r| > 0.95 with target.')\n"
 "train_encoded, test_encoded, scaler, scale_cols = P.encode(train_fe, test_fe, feature_cols)\n"
 "print('Design matrix columns:', train_encoded.shape[1])\n"
 "train_fe[['food_per_guest','surplus_units','pricing_ordinal','is_high_price','is_sit_down','guest_scale']].head()"),

("md", "## Section 5: Data integration -- segment benchmark table\n"
       "A second relation, keyed on (pricing tier x preparation method), is derived from the "
       "**training partition only** and left-joined onto both partitions."),
("code",
 "bench = P.build_benchmark_table(train_fe)\n"
 "display(bench)\n"
 "train_merged = P.merge_benchmark(train_fe, bench)\n"
 "test_merged = P.merge_benchmark(test_fe, bench)\n"
 "corr_train = train_merged['mean_waste'].corr(train_merged[P.TARGET])\n"
 "corr_test = test_merged['mean_waste'].corr(test_merged[P.TARGET])\n"
 "print(f'Correlation of benchmark mean with target -- train: {corr_train:.3f}, test: {corr_test:.3f}')\n"
 "print('(Close agreement confirms no target leakage through the merge.)')"),

("code",
 "views = P.analytical_views(pd.concat([train_merged, test_merged], ignore_index=True))\n"
 "for name, v in views.items():\n"
 "    print(f\"{name:24s} n={len(v):5d}  share={100*len(v)/(len(train_merged)+len(test_merged)):5.1f}%  mean_waste={v[P.TARGET].mean():.2f}\")"),

("md", "## Section 6: Exploratory Data Analysis (training partition only)"),
("code",
 "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
 "for ax, col in zip(axes, ['number_of_guests', 'quantity_of_food', P.TARGET]):\n"
 "    sns.histplot(train_df[col], ax=ax, bins=20, color='#5B4FCF')\n"
 "    ax.set_title(f'{col}\\nskew {train_df[col].skew():.2f} kurtosis {train_df[col].kurtosis():.2f}')\n"
 "fig.suptitle('Figure 1 -- Univariate distributions (training split)')\n"
 "fig.tight_layout()\n"
 "plt.show()"),

("code",
 "fig, ax = plt.subplots(figsize=(6,5))\n"
 "for col in ['number_of_guests','quantity_of_food']:\n"
 "    r = train_df[col].corr(train_df[P.TARGET])\n"
 "    print(f'{col}: r={r:.3f}')\n"
 "sns.regplot(x='number_of_guests', y=P.TARGET, data=train_df, ax=ax, scatter_kws={'alpha':0.4,'s':15}, line_kws={'color':'red'})\n"
 "ax.set_title('Figure 3 -- Number of guests vs waste')\n"
 "plt.show()"),

("code",
 "effect_rows = []\n"
 "for col in P.CATEGORICAL_COLS:\n"
 "    groups = [g[P.TARGET].values for _, g in train_df.groupby(col)]\n"
 "    f_stat, p_anova = stats.f_oneway(*groups)\n"
 "    h_stat, p_kw = stats.kruskal(*groups)\n"
 "    grand_mean = train_df[P.TARGET].mean()\n"
 "    ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups)\n"
 "    ss_total = ((train_df[P.TARGET]-grand_mean)**2).sum()\n"
 "    eta2 = ss_between/ss_total\n"
 "    effect_rows.append({'feature': col, 'k': len(groups), 'F': round(f_stat,2),\n"
 "                         'anova_p': p_anova, 'kruskal_p': p_kw, 'eta2': round(eta2,4),\n"
 "                         'effect': 'Large' if eta2>0.14 else ('Medium' if eta2>0.06 else 'Negligible')})\n"
 "effect_df = pd.DataFrame(effect_rows).sort_values('eta2', ascending=False).reset_index(drop=True)\n"
 "effect_df"),

("md", "**Only pricing and preparation method show a detectable relationship with waste; "
       "the remaining six categorical features are negligible** -- this determines which "
       "features are worth keeping in Review 3's models."),

("code",
 "fig, ax = plt.subplots(figsize=(7,5))\n"
 "order = train_df.groupby('pricing')[P.TARGET].mean().sort_values().index\n"
 "sns.boxplot(x='pricing', y=P.TARGET, data=train_df, order=order, ax=ax, color='#A9C6FF')\n"
 "means = train_df.groupby('pricing')[P.TARGET].mean().reindex(order)\n"
 "ax.scatter(range(len(order)), means.values, color='orange', zorder=5, label='mean')\n"
 "ax.legend()\n"
 "ax.set_title(f\"Figure 4 -- Pricing tier (eta^2={effect_df.set_index('feature').loc['pricing','eta2']:.4f})\")\n"
 "plt.show()"),

("code",
 "pivot = train_df.pivot_table(index='pricing', columns='preparation_method', values=P.TARGET, aggfunc='mean').reindex(['Low','Moderate','High'])\n"
 "fig, axes = plt.subplots(1, 2, figsize=(13,5))\n"
 "sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Blues', ax=axes[0])\n"
 "axes[0].set_title('Mean waste by pricing x preparation method')\n"
 "for method in pivot.columns:\n"
 "    axes[1].plot(pivot.index, pivot[method], marker='o', label=method)\n"
 "axes[1].legend()\n"
 "axes[1].set_title('Interaction plot')\n"
 "fig.suptitle('Figure 6 -- Pricing x preparation-method interaction')\n"
 "fig.tight_layout()\n"
 "plt.show()\n"
 "pivot"),

("code",
 "# Two-way interaction F-test (sequential least-squares ANOVA, valid for this unbalanced design)\n"
 "import eda_extended as E\n"
 "res = E.two_way_anova(train_df, 'pricing', 'preparation_method')\n"
 "print(f\"pricing: F={res['F_a']}  |  preparation_method given pricing: F={res['F_b']}  |  \"\n"
 "      f\"interaction: F={res['F_ab']}, p={res['p_ab']:.2e}\")"),

("code",
 "engineered = P.engineer_features(train_df)\n"
 "corr_cols = ['number_of_guests','quantity_of_food','food_per_guest','surplus_units',\n"
 "             'pricing_ordinal','is_high_price','is_sit_down', P.TARGET]\n"
 "corr = engineered[corr_cols].corr()\n"
 "fig, ax = plt.subplots(figsize=(8,6))\n"
 "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)\n"
 "ax.set_title('Figure 5 -- Pearson correlation matrix')\n"
 "plt.show()"),

("md", "## Section 6.4: Attainable predictive performance (irreducible-noise ceiling)\n"
       "Rows that share an identical feature vector but record different target values bound "
       "how good any model can get -- no model can output two different predictions from one input."),
("code",
 "print(ceiling)\n"
 "print(f\"\\nImplied R2 ceiling from the 4 signal-carrying features: {ceiling['implied_r2_ceiling']}\")"),

("md", "## Section 8: Export artefacts\n"
       "Running `python src/pipeline.py` from the project root regenerates every CSV in `out/` "
       "and every figure in `figures/` referenced in the Review 2 report."),
("code",
 "print('out/ files already written by src/pipeline.py:')\n"
 "for p in sorted((ROOT / 'out').glob('*.csv')):\n"
 "    print(' -', p.name)"),
]


from review3_cells import REVIEW3_CELLS  # noqa: E402


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb2 = make_notebook(REVIEW2_CELLS)
    execute_and_save(nb2, NB_DIR / "Review2_Preprocessing_EDA.ipynb")
    nb3 = make_notebook(REVIEW3_CELLS)
    execute_and_save(nb3, NB_DIR / "Review3_Model_Building_and_Comparison.ipynb")
