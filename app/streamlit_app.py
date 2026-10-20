"""
Streamlit dashboard for the Food Waste Prediction and Meal-Planning
Optimization for Event Catering project.

Booking Information -> Waste Prediction -> Risk Classification ->
Preparation Recommendation -> Donation Alert.

Tabs
  Forecast           single-booking forecast, what-if quantity sweep, model explanation
  Data Explorer      interactive EDA over every column of the cleaned dataset
  Model Performance  baseline vs tuned comparison, significance test, figure gallery
  Batch Forecast     score a whole CSV of upcoming bookings and download the result
  About              pipeline, methodology and limitations

Run: streamlit run app/streamlit_app.py
"""
import json
import pathlib
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "out"
FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))

from pipeline import prepare_inputs, ACTION_MAP  # noqa: E402

st.set_page_config(page_title="Event Catering Waste Forecast", page_icon="🍽️", layout="wide")

TARGET = "wastage_food_amount"
LABELS = ["LOW", "MEDIUM", "HIGH"]
TIER_COLORS = {"LOW": "#2E9E6B", "MEDIUM": "#E3A21A", "HIGH": "#D1495B"}
REDUCTION = {"LOW": 0.0, "MEDIUM": 0.125, "HIGH": 0.175}  # midpoints of the 10-15% / 15-20% bands
CATEGORY_OPTIONS = {
    "type_of_food": ["Meat", "Baked Goods", "Dairy Products", "Fruits", "Vegetables"],
    "event_type": ["Corporate", "Wedding", "Birthday", "Social Gathering"],
    "storage_conditions": ["Refrigerated", "Room Temperature"],
    "purchase_history": ["Regular", "Occasional"],
    "seasonality": ["Winter", "Summer", "All Seasons"],
    "preparation_method": ["Buffet", "Sit-down Dinner", "Finger Food"],
    "geographical_location": ["Urban", "Suburban", "Rural"],
    "pricing": ["Low", "Moderate", "High"],
}
RAW_COLUMNS = ["number_of_guests", "quantity_of_food"] + list(CATEGORY_OPTIONS)
PRETTY = {c: c.replace("_", " ").capitalize() for c in RAW_COLUMNS + [TARGET]}
FEATURE_GROUPS = ["type_of_food", "event_type", "storage_conditions", "purchase_history", "seasonality",
                  "preparation_method", "geographical_location", "guest_scale"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    zoo = joblib.load(MODELS_DIR / "model_zoo.pkl")
    bench = joblib.load(MODELS_DIR / "benchmark_table.pkl")
    return zoo, bench, metadata


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = OUT_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    path = OUT_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("﻿", "").strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def design_matrix(raw: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    return prepare_inputs(raw, metadata["design_matrix_columns"], metadata["feature_columns_raw"]).astype(float)


def predict_frame(raw: pd.DataFrame, regressor, classifier, metadata) -> pd.DataFrame:
    X = design_matrix(raw, metadata)
    waste = np.clip(regressor.predict(X), 0, None)
    tier_idx = np.asarray(classifier.predict(X)).astype(int)
    out = pd.DataFrame({"predicted_waste": waste, "risk_tier": [LABELS[i] for i in tier_idx]}, index=raw.index)
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X)
        for k, lbl in enumerate(LABELS):
            out[f"p_{lbl}"] = proba[:, k]
    return out


def xgb_contributions(regressor, X: pd.DataFrame) -> pd.DataFrame | None:
    """Per-feature additive contributions (TreeSHAP values computed natively by XGBoost),
    summed back over the one-hot columns of each business feature."""
    try:
        import xgboost as xgb
        booster = regressor.get_booster()
    except Exception:
        return None
    contrib = booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)[0]
    s = pd.Series(contrib[:-1], index=X.columns)
    grouped = {}
    for col, v in s.items():
        base = next((g for g in FEATURE_GROUPS if col.startswith(g + "_")), col)
        grouped[base] = grouped.get(base, 0.0) + float(v)
    df = pd.DataFrame({"feature": list(grouped), "contribution": list(grouped.values())})
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs"), float(contrib[-1])


def segment_benchmark(bench, pricing, preparation_method):
    seg = bench[(bench["pricing"] == pricing) & (bench["preparation_method"] == preparation_method)]
    return None if seg.empty else seg.iloc[0]


# ---------------------------------------------------------------------------
# Tab 1: Forecast
# ---------------------------------------------------------------------------
def tab_forecast(zoo, bench, metadata, reg_name, clf_name):
    regressor, classifier = zoo["regressors"][reg_name], zoo["classifiers"][clf_name]
    g_lo, g_hi = metadata["training_ranges"]["number_of_guests"]
    q_lo, q_hi = metadata["training_ranges"]["quantity_of_food"]

    with st.form("booking_form"):
        st.subheader("Booking details")
        c1, c2, c3 = st.columns(3)
        with c1:
            guests = st.number_input("Number of guests", 50, 1000, 350, step=5)
            quantity = st.number_input("Quantity of food prepared", 50, 1200, 420, step=5)
            pricing = st.selectbox("Pricing tier", CATEGORY_OPTIONS["pricing"], index=2)
        with c2:
            event_type = st.selectbox("Event type", CATEGORY_OPTIONS["event_type"], index=1)
            preparation_method = st.selectbox("Preparation method", CATEGORY_OPTIONS["preparation_method"], index=1)
            type_of_food = st.selectbox("Type of food", CATEGORY_OPTIONS["type_of_food"])
        with c3:
            storage_conditions = st.selectbox("Storage conditions", CATEGORY_OPTIONS["storage_conditions"])
            purchase_history = st.selectbox("Client purchase history", CATEGORY_OPTIONS["purchase_history"])
            seasonality = st.selectbox("Seasonality", CATEGORY_OPTIONS["seasonality"])
            geographical_location = st.selectbox("Geographical location", CATEGORY_OPTIONS["geographical_location"])
        submitted = st.form_submit_button("Forecast waste", type="primary", width="stretch")

    if submitted:
        st.session_state["booking"] = dict(
            number_of_guests=int(guests), quantity_of_food=int(quantity), pricing=pricing,
            event_type=event_type, preparation_method=preparation_method, type_of_food=type_of_food,
            storage_conditions=storage_conditions, purchase_history=purchase_history,
            seasonality=seasonality, geographical_location=geographical_location)
    booking = st.session_state.get("booking")
    if booking is None:
        st.info("Enter the booking parameters and press **Forecast waste**.")
        return

    raw = pd.DataFrame([booking])
    res = predict_frame(raw, regressor, classifier, metadata).iloc[0]
    waste, tier = float(res["predicted_waste"]), res["risk_tier"]
    guests, quantity = booking["number_of_guests"], booking["quantity_of_food"]

    if not (g_lo <= guests <= g_hi) or not (q_lo <= quantity <= q_hi):
        st.warning(f"This booking lies outside the range the models were trained on "
                   f"(guests {g_lo}-{g_hi}, quantity {q_lo}-{q_hi}). Tree models do not extrapolate, "
                   f"so treat the forecast as indicative.")

    st.subheader("Forecast")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted waste (units)", f"{waste:.1f}")
    m2.metric("Risk tier", tier)
    m3.metric("Waste as % of food prepared", f"{100 * waste / quantity:.1f}%")
    m4.metric("Expected waste per guest", f"{waste / guests:.3f}")

    q1, q3 = metadata["risk_tier_cutpoints"]["Q1"], metadata["risk_tier_cutpoints"]["Q3"]
    reg_tier = "LOW" if waste < q1 else ("HIGH" if waste >= q3 else "MEDIUM")
    if reg_tier != tier:
        st.caption(f"The regressor's amount ({waste:.1f}) falls in the {reg_tier} band "
                   f"(cut points {q1:g} / {q3:g}); the classifier says {tier}. Bookings near a cut "
                   f"point are genuinely borderline. Check the tier probabilities below.")

    color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[tier]
    st.markdown(f"### Recommended action: :{color}[{ACTION_MAP[tier]}]")
    if REDUCTION[tier] > 0:
        new_qty = int(round(quantity * (1 - REDUCTION[tier])))
        new_raw = raw.assign(quantity_of_food=new_qty)
        new_waste = float(predict_frame(new_raw, regressor, classifier, metadata).iloc[0]["predicted_waste"])
        a1, a2, a3 = st.columns(3)
        a1.metric("Suggested quantity", f"{new_qty}", delta=f"-{quantity - new_qty} units", delta_color="inverse")
        a2.metric("Projected waste at suggested quantity", f"{new_waste:.1f}", delta=f"{new_waste - waste:+.1f}",
                  delta_color="inverse")
        a3.metric("Food per guest after reduction", f"{new_qty / guests:.2f}")
    if tier == "HIGH":
        st.error("🔔 **Donation-partner alert**: the predicted surplus for this booking is high. "
                 "Contact a local food-recovery partner ahead of the event.")

    left, right = st.columns(2)
    with left:
        if "p_LOW" in res:
            st.markdown("**Risk-tier probabilities**")
            pdf = pd.DataFrame({"tier": LABELS, "probability": [res[f"p_{l}"] for l in LABELS]})
            chart = alt.Chart(pdf).mark_bar().encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("tier:N", sort=LABELS, title=None),
                color=alt.Color("tier:N", scale=alt.Scale(domain=LABELS, range=[TIER_COLORS[l] for l in LABELS]),
                                legend=None),
                tooltip=["tier", alt.Tooltip("probability:Q", format=".1%")]).properties(height=160)
            st.altair_chart(chart, width="stretch")
        seg = segment_benchmark(bench, booking["pricing"], booking["preparation_method"])
        if seg is not None:
            st.markdown("**Segment benchmark**")
            st.write(f"Past **{booking['pricing']}**-price **{booking['preparation_method']}** bookings "
                     f"(n={int(seg['n'])}) averaged **{seg['mean_waste']:.1f}** units of waste "
                     f"(median {seg['median_waste']:.1f}, sd {seg['std_waste']:.1f}). "
                     f"This booking is forecast at **{waste - seg['mean_waste']:+.1f}** units vs the segment mean.")
    with right:
        st.markdown("**What-if: predicted waste vs quantity prepared**")
        qty_grid = np.arange(max(q_lo - 40, 50), q_hi + 41, 5)
        sweep_raw = pd.concat([raw] * len(qty_grid), ignore_index=True).assign(quantity_of_food=qty_grid)
        sweep = predict_frame(sweep_raw, regressor, classifier, metadata)
        sweep["quantity_of_food"] = qty_grid
        line = alt.Chart(sweep).mark_line(color="#5B4FCF").encode(
            x=alt.X("quantity_of_food:Q", title="quantity of food prepared"),
            y=alt.Y("predicted_waste:Q", title="predicted waste"))
        pts = alt.Chart(sweep).mark_circle(size=40).encode(
            x="quantity_of_food:Q", y="predicted_waste:Q",
            color=alt.Color("risk_tier:N", scale=alt.Scale(domain=LABELS, range=[TIER_COLORS[l] for l in LABELS])),
            tooltip=["quantity_of_food", alt.Tooltip("predicted_waste:Q", format=".1f"), "risk_tier"])
        rule = alt.Chart(pd.DataFrame({"q": [quantity]})).mark_rule(strokeDash=[4, 4]).encode(x="q:Q")
        st.altair_chart((line + pts + rule).properties(height=260), width="stretch")
        st.caption("Dashed line = this booking. Shaded region outside the training range is extrapolation.")

    with st.expander("Why this prediction? (feature contributions)", expanded=False):
        out = xgb_contributions(regressor, design_matrix(raw, metadata))
        if out is None:
            st.write("Per-booking contributions are available when the regression model is XGBoost.")
        else:
            contrib, base = out
            st.write(f"Average training booking: **{base:.1f}** units. Each bar shows how much a feature "
                     f"moved this booking's forecast up (red) or down (green) from that baseline.")
            ch = alt.Chart(contrib).mark_bar().encode(
                x=alt.X("contribution:Q", title="contribution to predicted waste (units)"),
                y=alt.Y("feature:N", sort=None, title=None),
                color=alt.condition("datum.contribution > 0", alt.value("#D1495B"), alt.value("#2E9E6B")),
                tooltip=["feature", alt.Tooltip("contribution:Q", format="+.2f")]).properties(height=330)
            st.altair_chart(ch, width="stretch")

    with st.expander("Every tuned model's forecast for this booking"):
        rows = []
        for n, m in zoo["regressors"].items():
            rows.append({"task": "regression", "model": n,
                         "forecast": f"{float(np.clip(m.predict(design_matrix(raw, metadata)), 0, None)[0]):.1f} units"})
        for n, m in zoo["classifiers"].items():
            rows.append({"task": "classification", "model": n,
                         "forecast": LABELS[int(np.asarray(m.predict(design_matrix(raw, metadata)))[0])]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 2: Data Explorer
# ---------------------------------------------------------------------------
EDA_FIGURES = {
    "Fig 1 - Univariate distributions": "fig01_univariate.png",
    "Fig 2 - Categorical level balance": "fig02_categorical_balance.png",
    "Fig 3 - Numeric drivers": "fig03_numeric_drivers.png",
    "Fig 4 - Strongest categorical effects": "fig04_top_categorical_effects.png",
    "Fig 5 - Correlation matrix": "fig05_correlation_matrix.png",
    "Fig 6 - Pricing x preparation interaction": "fig06_interaction.png",
    "Fig 7 - Guest slope by pricing tier": "fig07_slope_by_tier.png",
    "Fig 8 - Waste by every categorical level": "fig08_waste_by_every_category.png",
    "Fig 9 - Feature importance (EDA)": "fig09_feature_importance.png",
    "Fig 10 - Residual diagnostics (EDA)": "fig10_residual_diagnostics.png",
    "Fig 11 - Waste rate by category": "fig11_waste_rate_by_category.png",
    "Fig 12 - Two-way heatmaps": "fig12_two_way_heatmaps.png",
    "Fig 13 - Risk-tier composition": "fig13_risk_tier_composition.png",
    "Fig 14 - Guests x quantity map": "fig14_guests_quantity_map.png",
    "Fig 15 - Event-type profile": "fig15_event_type_profile.png",
    "Fig 16 - Cramer's V associations": "fig16_cramers_v_matrix.png",
    "Fig 17 - ECDF by segment": "fig17_ecdf_by_segment.png",
    "Fig 18 - Waste per guest": "fig18_waste_per_guest.png",
    "Fig 19 - Pair plot by risk tier": "fig19_pairplot_by_risk_tier.png",
    "Fig 20 - Season, location, storage": "fig20_season_location_storage.png",
}


def tab_explorer(metadata):
    data = load_csv("cleaned_event_catering.csv")
    if data is None:
        st.error("out/cleaned_event_catering.csv not found. Run `python src/pipeline.py` first.")
        return
    q1, q3 = metadata["risk_tier_cutpoints"]["Q1"], metadata["risk_tier_cutpoints"]["Q3"]
    data = data.assign(
        risk_tier=np.where(data[TARGET] < q1, "LOW", np.where(data[TARGET] >= q3, "HIGH", "MEDIUM")),
        waste_pct=100 * data[TARGET] / data["quantity_of_food"],
        food_per_guest=data["quantity_of_food"] / data["number_of_guests"])

    with st.expander("Filters", expanded=False):
        cols = st.columns(4)
        filters = {}
        for i, c in enumerate(CATEGORY_OPTIONS):
            levels = sorted(data[c].unique())
            filters[c] = cols[i % 4].multiselect(PRETTY[c], levels, default=levels, key=f"f_{c}")
        g_range = st.slider("Number of guests", int(data.number_of_guests.min()), int(data.number_of_guests.max()),
                            (int(data.number_of_guests.min()), int(data.number_of_guests.max())))
    mask = data["number_of_guests"].between(*g_range)
    for c, keep in filters.items():
        mask &= data[c].isin(keep)
    d = data[mask]
    if d.empty:
        st.warning("No bookings match the filters.")
        return

    k = st.columns(5)
    k[0].metric("Bookings", f"{len(d):,}")
    k[1].metric("Mean waste", f"{d[TARGET].mean():.1f}")
    k[2].metric("Median waste", f"{d[TARGET].median():.0f}")
    k[3].metric("Mean waste rate", f"{d['waste_pct'].mean():.2f}%")
    k[4].metric("HIGH-risk share", f"{100 * (d['risk_tier'] == 'HIGH').mean():.1f}%")

    st.markdown("#### Waste by any column")
    c1, c2 = st.columns([1, 2])
    col = c1.selectbox("Column", list(CATEGORY_OPTIONS), index=7, format_func=PRETTY.get)
    metric = c1.radio("Measure", ["Mean waste", "Waste rate (% of quantity)", "HIGH-risk share"])
    field = {"Mean waste": TARGET, "Waste rate (% of quantity)": "waste_pct", "HIGH-risk share": "is_high"}[metric]
    dd = d.assign(is_high=(d["risk_tier"] == "HIGH").astype(float) * 100)
    agg = dd.groupby(col).agg(value=(field, "mean"), n=(field, "size"), sd=(field, "std")).reset_index()
    agg["ci"] = 1.96 * agg["sd"].fillna(0) / np.sqrt(agg["n"])
    agg["lo"], agg["hi"] = agg["value"] - agg["ci"], agg["value"] + agg["ci"]
    bars = alt.Chart(agg).mark_bar(color="#5B4FCF").encode(
        x=alt.X(f"{col}:N", sort="-y", title=PRETTY[col]), y=alt.Y("value:Q", title=metric),
        tooltip=[col, alt.Tooltip("value:Q", format=".2f"), "n"])
    err = alt.Chart(agg).mark_errorbar().encode(x=alt.X(f"{col}:N", sort="-y"), y="lo:Q", y2="hi:Q")
    c2.altair_chart((bars + err).properties(height=300), width="stretch")
    with c1:
        groups = [g[TARGET].values for _, g in d.groupby(col)]
        if len(groups) > 1:
            from scipy import stats
            st.caption(f"Kruskal-Wallis on waste across {PRETTY[col].lower()} levels: "
                       f"p = {stats.kruskal(*groups).pvalue:.3g}")

    st.markdown("#### Risk-tier mix and waste distribution")
    c3, c4 = st.columns(2)
    mix = d.groupby([col, "risk_tier"]).size().reset_index(name="n")
    c3.altair_chart(alt.Chart(mix).mark_bar().encode(
        x=alt.X(f"{col}:N", title=PRETTY[col]), y=alt.Y("n:Q", stack="normalize", title="share", axis=alt.Axis(format="%")),
        color=alt.Color("risk_tier:N", scale=alt.Scale(domain=LABELS, range=[TIER_COLORS[l] for l in LABELS]),
                        sort=LABELS), order=alt.Order("risk_tier:N"),
        tooltip=[col, "risk_tier", "n"]).properties(height=300), width="stretch")
    c4.altair_chart(alt.Chart(d).transform_density(TARGET, groupby=[col], as_=[TARGET, "density"]).mark_line().encode(
        x=alt.X(f"{TARGET}:Q", title="waste"), y="density:Q", color=alt.Color(f"{col}:N", title=PRETTY[col])
    ).properties(height=300), width="stretch")

    st.markdown("#### Two-column heatmap")
    h1, h2, h3 = st.columns([1, 1, 3])
    a = h1.selectbox("Rows", list(CATEGORY_OPTIONS), index=1, format_func=PRETTY.get)
    b = h2.selectbox("Columns", [c for c in CATEGORY_OPTIONS if c != a], index=5, format_func=PRETTY.get)
    hm = d.groupby([a, b])[TARGET].agg(["mean", "size"]).reset_index()
    base = alt.Chart(hm).encode(x=alt.X(f"{b}:N", title=PRETTY[b]), y=alt.Y(f"{a}:N", title=PRETTY[a]))
    h3.altair_chart((base.mark_rect().encode(color=alt.Color("mean:Q", scale=alt.Scale(scheme="orangered"),
                                                                title="mean waste"),
                                             tooltip=[a, b, alt.Tooltip("mean:Q", format=".1f"), "size"]) +
                     base.mark_text(fontSize=11).encode(text=alt.Text("mean:Q", format=".1f"))).properties(height=260),
                    width="stretch")

    st.markdown("#### Guests, quantity and waste")
    s1, s2 = st.columns([1, 3])
    xcol = s1.selectbox("X axis", ["number_of_guests", "quantity_of_food", "food_per_guest"], format_func=lambda c: c.replace("_", " "))
    hue = s1.selectbox("Colour by", list(CATEGORY_OPTIONS) + ["risk_tier"], index=7, format_func=lambda c: c.replace("_", " "))
    s2.altair_chart(alt.Chart(d).mark_circle(opacity=0.5, size=35).encode(
        x=alt.X(f"{xcol}:Q", scale=alt.Scale(zero=False)), y=alt.Y(f"{TARGET}:Q", title="waste"),
        color=f"{hue}:N", tooltip=RAW_COLUMNS + [TARGET]).interactive().properties(height=340), width="stretch")

    st.markdown("#### Statistical summaries (training partition)")
    t1, t2 = st.columns(2)
    assoc = load_csv("eda_risk_tier_association.csv")
    effects = load_csv("categorical_effect_sizes.csv")
    if assoc is not None:
        t1.caption("Association of each column with the risk tier (chi-square, Cramer's V)")
        t1.dataframe(assoc, hide_index=True, width="stretch")
    if effects is not None:
        t2.caption("Effect of each column on waste amount (ANOVA, Kruskal-Wallis, eta squared)")
        t2.dataframe(effects, hide_index=True, width="stretch")
    anova = load_csv("eda_two_way_anova.csv")
    if anova is not None:
        st.caption("Two-way ANOVA (sequential sums of squares) for pairs of columns")
        st.dataframe(anova, hide_index=True, width="stretch")

    st.markdown("#### EDA figure gallery")
    pick = st.selectbox("Figure", list(EDA_FIGURES))
    path = FIG_DIR / EDA_FIGURES[pick]
    if path.exists():
        st.image(str(path), width="stretch")
    with st.expander("Show filtered data"):
        st.dataframe(d, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 3: Model performance
# ---------------------------------------------------------------------------
MODEL_FIGURES = {
    "Fig 21 - Regression comparison": "fig21_regression_comparison.png",
    "Fig 22 - Predicted vs actual": "fig22_pred_vs_actual.png",
    "Fig 23 - Final regressor residuals": "fig23_residuals_final_regressor.png",
    "Fig 24 - Classification comparison": "fig24_classification_comparison.png",
    "Fig 25 - Confusion matrices": "fig25_confusion_matrices.png",
    "Fig 26 - ROC curves": "fig26_roc_curves.png",
    "Fig 27 - Tuning before vs after": "fig27_tuning_before_after.png",
    "Fig 28 - Feature importance": "fig28_feature_importance_final.png",
    "Fig 29 - Learning curves": "fig29_learning_curves.png",
    "Fig 30 - XGBoost search landscape": "fig30_xgboost_search_landscape.png",
    "Fig 31a - CV fold scores (regression)": "fig31a_cv_scores_regression.png",
    "Fig 31b - CV fold scores (classification)": "fig31b_cv_scores_classification.png",
}


def tab_models(metadata):
    rm, cm = metadata["regression_metrics"], metadata["classification_metrics"]
    st.markdown(f"#### Final models: **{metadata['regressor_name']}** regressor and "
                f"**{metadata['classifier_name']}** classifier")
    k = st.columns(6)
    k[0].metric("Test R²", f"{rm['test_R2']:.3f}")
    k[1].metric("Test RMSE", f"{rm['test_RMSE']:.2f}")
    k[2].metric("Test MAE", f"{rm['test_MAE']:.2f}")
    k[3].metric("Test macro-F1", f"{cm['test_macro_F1']:.3f}")
    k[4].metric("Test accuracy", f"{cm['test_accuracy']:.3f}")
    k[5].metric("Test ROC-AUC", f"{cm['test_ROC_AUC_ovr']:.3f}")

    reg, clf = load_csv("regression_model_comparison.csv"), load_csv("classification_model_comparison.csv")
    reg_b, clf_b = load_csv("regression_baseline_metrics.csv"), load_csv("classification_baseline_metrics.csv")
    st.markdown("#### Regression: tuned models")
    if reg is not None:
        st.dataframe(reg.drop(columns="stage").sort_values("cv_R2_mean", ascending=False), hide_index=True, width="stretch")
        long = pd.concat([reg_b, reg])[lambda x: x["model"] != "Naive (mean)"]
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("cv_R2_mean:Q", title="5-fold CV R²", scale=alt.Scale(domain=[0.6, 1])),
            y=alt.Y("model:N", sort="-x", title=None), yOffset="stage:N",
            color=alt.Color("stage:N", scale=alt.Scale(domain=["baseline", "tuned"], range=["#B8B3E6", "#5B4FCF"])),
            tooltip=["model", "stage", alt.Tooltip("cv_R2_mean:Q", format=".4f")]).properties(height=280),
            width="stretch")
    st.markdown("#### Classification: tuned models")
    if clf is not None:
        st.dataframe(clf.drop(columns="stage").sort_values("cv_macro_F1_mean", ascending=False), hide_index=True,
                     width="stretch")
        long = pd.concat([clf_b, clf])[lambda x: ~x["model"].str.contains("Dummy|cut points")]
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("cv_macro_F1_mean:Q", title="5-fold CV macro-F1", scale=alt.Scale(domain=[0.6, 1])),
            y=alt.Y("model:N", sort="-x", title=None), yOffset="stage:N",
            color=alt.Color("stage:N", scale=alt.Scale(domain=["baseline", "tuned"], range=["#B8B3E6", "#5B4FCF"])),
            tooltip=["model", "stage", alt.Tooltip("cv_macro_F1_mean:Q", format=".4f")]).properties(height=280),
            width="stretch")

    c1, c2 = st.columns(2)
    for col, name, title in [(c1, "regression_tuning_before_after.csv", "Tuning gain: regression"),
                             (c2, "classification_tuning_before_after.csv", "Tuning gain: classification")]:
        t = load_csv(name)
        if t is not None:
            col.markdown(f"**{title}**")
            col.dataframe(t, hide_index=True, width="stretch")

    sig = load_csv("final_model_significance.csv")
    if sig is not None:
        st.markdown("**Is the winner significantly better than the runner-up?** "
                    "(Nadeau-Bengio corrected resampled t-test, 5x3 repeated CV)")
        st.dataframe(sig, hide_index=True, width="stretch")

    per_class = load_csv("per_class_metrics_all_models.csv")
    if per_class is not None:
        st.markdown("**Per-tier precision / recall / F1 on the test set**")
        st.altair_chart(alt.Chart(per_class[~per_class["model"].str.contains("Dummy")]).mark_rect().encode(
            x=alt.X("tier:N", sort=LABELS), y=alt.Y("model:N", title=None), color=alt.Color("f1:Q", scale=alt.Scale(scheme="purples")),
            tooltip=["model", "tier", alt.Tooltip("precision:Q", format=".3f"), alt.Tooltip("recall:Q", format=".3f"),
                     alt.Tooltip("f1:Q", format=".3f")]).properties(height=260) +
            alt.Chart(per_class[~per_class["model"].str.contains("Dummy")]).mark_text().encode(
                x=alt.X("tier:N", sort=LABELS), y="model:N", text=alt.Text("f1:Q", format=".2f")), width="stretch")

    params = load_json("best_hyperparameters.json")
    if params:
        with st.expander("Best hyperparameters found by Grid / Randomized search"):
            st.json(params, expanded=False)

    st.markdown("#### Model figure gallery")
    pick = st.selectbox("Figure", list(MODEL_FIGURES), key="model_fig")
    path = FIG_DIR / MODEL_FIGURES[pick]
    if path.exists():
        st.image(str(path), width="stretch")


# ---------------------------------------------------------------------------
# Tab 4: Batch forecast
# ---------------------------------------------------------------------------
def tab_batch(zoo, metadata, reg_name, clf_name):
    st.markdown("Upload a CSV of upcoming bookings with the dataset's columns (original names such as "
                "`Number of Guests` or snake_case both work). Any `Wastage Food Amount` column is ignored.")
    up = st.file_uploader("Bookings CSV", type="csv")
    use_sample = st.button("Or score 25 sample bookings from the dataset")
    if up is not None:
        raw = normalise_columns(pd.read_csv(up, encoding="utf-8-sig"))
    elif use_sample or st.session_state.get("batch_sample"):
        st.session_state["batch_sample"] = True
        test = load_csv("cleaned_event_catering.csv")
        raw = test[RAW_COLUMNS + [TARGET]].sample(25, random_state=7).reset_index(drop=True) if test is not None else None
    else:
        return
    if raw is None:
        return
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return
    bad = {c: sorted(set(raw[c]) - set(CATEGORY_OPTIONS[c])) for c in CATEGORY_OPTIONS}
    bad = {c: v for c, v in bad.items() if v}
    if bad:
        st.error(f"Unknown category values: {bad}")
        return
    preds = predict_frame(raw[RAW_COLUMNS], zoo["regressors"][reg_name], zoo["classifiers"][clf_name], metadata)
    result = pd.concat([raw.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    result["recommended_action"] = result["risk_tier"].map(ACTION_MAP)
    result["suggested_quantity"] = (result["quantity_of_food"] * (1 - result["risk_tier"].map(REDUCTION))).round().astype(int)
    result["donation_alert"] = result["risk_tier"] == "HIGH"

    k = st.columns(4)
    k[0].metric("Bookings scored", len(result))
    k[1].metric("Total predicted waste", f"{result['predicted_waste'].sum():.0f} units")
    k[2].metric("Donation alerts", int(result["donation_alert"].sum()))
    k[3].metric("Food saved by following actions",
                f"{int((result['quantity_of_food'] - result['suggested_quantity']).sum())} units")
    if TARGET in result.columns:
        err = (result[TARGET] - result["predicted_waste"]).abs()
        st.caption(f"Actual waste is known for these bookings: mean absolute error {err.mean():.2f} units.")
    st.dataframe(result.round(3), hide_index=True, width="stretch")
    st.download_button("Download forecasts as CSV", result.to_csv(index=False).encode(), "waste_forecasts.csv",
                       "text/csv")


# ---------------------------------------------------------------------------
# Tab 5: About
# ---------------------------------------------------------------------------
def tab_about(metadata):
    st.markdown(f"""
**Pipeline:** Booking Information → Waste Prediction → Risk Classification → Preparation Recommendation → Donation Alert

**Data:** *Food Wastage Data in Restaurant* (Kaggle), 1,782 records × 11 columns; 164 exact duplicates removed →
1,618 bookings, split 80/20 ({metadata['train_rows']} train / {metadata['test_rows']} test, seed 42) **before** any
target-derived statistic was computed.

**Features (pre-event only):** guests, quantity, food per guest, surplus units, pricing (ordinal + high-price flag),
sit-down flag, guest-scale band, and one-hot encodings of food type, event type, storage, purchase history,
seasonality, location and preparation method ({len(metadata['design_matrix_columns'])} design columns).

**Risk tiers:** LOW < {metadata['risk_tier_cutpoints']['Q1']:g} ≤ MEDIUM < {metadata['risk_tier_cutpoints']['Q3']:g} ≤ HIGH
(training-set quartiles).

**Models compared:** Linear/Logistic Regression, KNN, SVM, Random Forest, HistGradientBoosting and XGBoost
(`tree_method="hist"`), each trained with default settings and then tuned with Grid/Randomized search on 5-fold CV.
The final model of each task is the one with the best **tuned CV score**; the test set is used only for the final check.

**Prescriptive engine:** LOW → normal preparation · MEDIUM → reduce quantity 10–15% · HIGH → reduce 15–20% and alert a
donation partner.

**Limitations:** the dataset has limited value granularity (61 guest/quantity combinations, targets mostly multiples
of 5), no date field (no weekday or holiday effects), and pricing is a 3-level tier. Forecasts support staff
decisions; they never override contract minimums, dietary requirements, service buffers or food-safety rules.
""")


# ---------------------------------------------------------------------------
def main():
    st.title("🍽️ Event Catering Waste Forecast")
    st.caption("Booking Information → Waste Prediction → Risk Classification → Preparation Recommendation → Donation Alert")
    try:
        zoo, bench, metadata = load_artifacts()
    except FileNotFoundError:
        st.error("Model artefacts not found. Run `python src/train_models.py` from the project root first.")
        return

    with st.sidebar:
        st.header("Models")
        reg_names = list(zoo["regressors"])
        clf_names = list(zoo["classifiers"])
        reg_name = st.selectbox("Regression model", reg_names, index=reg_names.index(metadata["regressor_name"]))
        clf_name = st.selectbox("Classification model", clf_names, index=clf_names.index(metadata["classifier_name"]))
        st.caption(f"Defaults are the selected final models: **{metadata['regressor_name']}** "
                   f"(CV R² {metadata['regression_metrics']['cv_R2_mean']:.3f}) and "
                   f"**{metadata['classifier_name']}** (CV macro-F1 "
                   f"{metadata['classification_metrics']['cv_macro_F1_mean']:.3f}).")
        st.divider()
        st.caption(f"Trained on {metadata['train_rows']} bookings, guests "
                   f"{metadata['training_ranges']['number_of_guests'][0]}-{metadata['training_ranges']['number_of_guests'][1]}, "
                   f"quantity {metadata['training_ranges']['quantity_of_food'][0]}-{metadata['training_ranges']['quantity_of_food'][1]}.")

    t1, t2, t3, t4, t5 = st.tabs(["🔮 Forecast", "📊 Data Explorer", "🏆 Model Performance", "📁 Batch Forecast", "ℹ️ About"])
    with t1:
        tab_forecast(zoo, bench, metadata, reg_name, clf_name)
    with t2:
        tab_explorer(metadata)
    with t3:
        tab_models(metadata)
    with t4:
        tab_batch(zoo, metadata, reg_name, clf_name)
    with t5:
        tab_about(metadata)


if __name__ == "__main__":
    main()
