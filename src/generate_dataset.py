"""
Regenerates a raw event-catering food-wastage dataset that reproduces the schema,
value ranges, category proportions, and statistical relationships documented in
Review 1 and Review 2 of this project (pricing-tier dominance, guest/quantity
correlation with waste, a pricing x preparation-method interaction with a rank
reversal in the Moderate tier, heteroscedastic noise by pricing tier, a limited
pool of (guests, quantity) combinations, ~77% of target values landing on
multiples of five, 164 exact duplicate rows, and 58 rows that share a feature
vector with another row but disagree on the target).

The project now runs on the actual Kaggle dataset ("Food Wastage Data in
Restaurant"), stored at data/raw/food_wastage_data.csv. This script is kept
only as a historical/dev fallback for regenerating a statistically similar
synthetic dataset -- it is NOT the raw-data source and does not write to
data/raw/food_wastage_data.csv, so running it will never overwrite the real
Kaggle file.

Run: python src/generate_dataset.py
Output: data/raw/event_catering_raw_synthetic.csv (1782 rows x 11 columns,
includes the BOM-prefixed header and the 164 exact duplicates described in
Review 2).
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
N_BASE = 1560          # rows generated independently
N_CONFLICT = 58         # rows that clone another row's features but not its target
N_EXACT_DUP = 164        # rows that are then cloned in full (features + target)
N_GUEST_QTY_PAIRS = 61   # distinct (guests, quantity) combinations allowed

CATEGORIES = {
    "Type of Food": {
        "values": ["Meat", "Baked Goods", "Dairy Products", "Fruits", "Vegetables"],
        "p": [317, 304, 273, 236, 164],
    },
    "Event Type": {
        "values": ["Corporate", "Social Gathering", "Wedding", "Birthday"],
        "p": [365, 331, 326, 272],
    },
    "Storage Conditions": {
        "values": ["Room Temperature", "Refrigerated"],
        "p": [842, 452],
    },
    "Purchase History": {
        "values": ["Regular", "Occasional"],
        "p": [1133, 161],
    },
    "Seasonality": {
        "values": ["Winter", "All Seasons", "Summer"],
        "p": [468, 430, 396],
    },
    "Geographical Location": {
        "values": ["Suburban", "Urban", "Rural"],
        "p": [590, 412, 292],
    },
}
# Normalise counts (taken from Review 2 Fig. 2, computed on a 1294-row split) to probabilities.
for spec in CATEGORIES.values():
    total = sum(spec["p"])
    spec["p"] = [c / total for c in spec["p"]]

PRICING_P = {"Low": 332 / 1294, "Moderate": 487 / 1294, "High": 475 / 1294}
PREP_P = {"Sit-down Dinner": 574 / 1294, "Finger Food": 503 / 1294, "Buffet": 217 / 1294}

# Cell means from Review 2 Figure 6 (pricing tier x preparation method), used as the
# deterministic base of the target so the documented rank-reversal interaction survives.
INTERACTION_BASE = {
    "Low": {"Sit-down Dinner": 25.1, "Buffet": 24.1, "Finger Food": 18.6},
    "Moderate": {"Buffet": 27.0, "Sit-down Dinner": 24.6, "Finger Food": 20.1},
    "High": {"Sit-down Dinner": 41.5, "Buffet": 36.0, "Finger Food": 35.1},
}
GUEST_SLOPE = {"Low": 0.087, "Moderate": 0.061, "High": 0.042}
NOISE_SD = {"Low": 2.2, "Moderate": 2.8, "High": 4.2}
GUEST_CENTER = 318
QUANTITY_SLOPE = 0.03
QUANTITY_CENTER = 400


def _sample_categorical(rng, spec, n):
    return rng.choice(spec["values"], size=n, p=spec["p"])


def _build_guest_quantity_pool(rng):
    """A fixed pool of (guests, quantity) pairs, sampled with skewed frequency
    so a handful of combinations dominate (Review 2 Sec. 2.4: 7 combinations
    cover 67.4% of records)."""
    guests = rng.integers(207, 492, size=N_GUEST_QTY_PAIRS)
    quantity = np.clip(
        (guests * 0.85 + rng.normal(60, 40, size=N_GUEST_QTY_PAIRS)).round().astype(int),
        280, 500,
    )
    pool = list(zip(guests.tolist(), quantity.tolist()))
    # Zipf-like weights so a few pairs are heavily reused.
    weights = 1.0 / np.arange(1, N_GUEST_QTY_PAIRS + 1)
    weights = weights / weights.sum()
    return pool, weights


def _draw_target(rng, pricing, prep, guests, quantity):
    base = INTERACTION_BASE[pricing][prep]
    guest_effect = GUEST_SLOPE[pricing] * (guests - GUEST_CENTER)
    quantity_effect = QUANTITY_SLOPE * (quantity - QUANTITY_CENTER)
    noise = rng.normal(0, NOISE_SD[pricing])
    value = base + guest_effect + quantity_effect + noise
    if rng.random() < 0.77:
        value = round(value / 5) * 5
    else:
        value = round(value)
    return int(np.clip(value, 10, 63))


def _generate_rows(rng, n, pool, weights):
    idx = rng.choice(len(pool), size=n, p=weights)
    guests = np.array([pool[i][0] for i in idx])
    quantity = np.array([pool[i][1] for i in idx])
    pricing = rng.choice(list(PRICING_P), size=n, p=list(PRICING_P.values()))
    prep = rng.choice(list(PREP_P), size=n, p=list(PREP_P.values()))

    rows = {
        "Type of Food": _sample_categorical(rng, CATEGORIES["Type of Food"], n),
        "Number of Guests": guests,
        "Event Type": _sample_categorical(rng, CATEGORIES["Event Type"], n),
        "Quantity of Food": quantity,
        "Storage Conditions": _sample_categorical(rng, CATEGORIES["Storage Conditions"], n),
        "Purchase History": _sample_categorical(rng, CATEGORIES["Purchase History"], n),
        "Seasonality": _sample_categorical(rng, CATEGORIES["Seasonality"], n),
        "Preparation Method": prep,
        "Geographical Location": _sample_categorical(rng, CATEGORIES["Geographical Location"], n),
        "Pricing": pricing,
    }
    target = [
        _draw_target(rng, rows["Pricing"][i], rows["Preparation Method"][i], guests[i], quantity[i])
        for i in range(n)
    ]
    rows["Wastage Food Amount"] = target
    return pd.DataFrame(rows)


def generate(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool, weights = _build_guest_quantity_pool(rng)

    base = _generate_rows(rng, N_BASE, pool, weights)

    # 58 feature-duplicate conflicts: clone another row's 10 feature columns,
    # then redraw the target from fresh noise so it very likely differs.
    feature_cols = [c for c in base.columns if c != "Wastage Food Amount"]
    clone_src = base.sample(n=N_CONFLICT, random_state=seed, replace=True).reset_index(drop=True)
    conflicts = clone_src[feature_cols].copy()
    conflicts["Wastage Food Amount"] = [
        _draw_target(
            rng,
            row["Pricing"], row["Preparation Method"],
            row["Number of Guests"], row["Quantity of Food"],
        )
        for _, row in conflicts.iterrows()
    ]
    # Guarantee an actual disagreement with the source row.
    same = conflicts["Wastage Food Amount"].to_numpy() == clone_src["Wastage Food Amount"].to_numpy()
    conflicts.loc[same, "Wastage Food Amount"] = np.clip(
        conflicts.loc[same, "Wastage Food Amount"] + rng.choice([-5, 5], size=int(same.sum())), 10, 63
    )

    unique_df = pd.concat([base, conflicts], ignore_index=True)  # 1618 rows, matches Review 2 Sec. 3.2

    exact_dups = unique_df.sample(n=N_EXACT_DUP, random_state=seed + 1, replace=False).copy()
    raw_df = pd.concat([unique_df, exact_dups], ignore_index=True)  # 1782 rows, matches Review 1/2 raw size
    raw_df = raw_df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return raw_df


if __name__ == "__main__":
    import pathlib

    out_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "event_catering_raw_synthetic.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    # Write with a UTF-8 BOM to reproduce the encoding defect documented as DQ-06.
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns to {out_path}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Unique rows after dedup: {df.drop_duplicates().shape[0]}")
