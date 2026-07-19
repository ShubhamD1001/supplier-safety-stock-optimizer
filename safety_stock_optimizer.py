"""
=============================================================================
 INVENTORY & SUPPLIER RELIABILITY OPTIMIZATION
 Safety Stock Modeling with Dual Variability (Demand + Lead Time)
=============================================================================
 Author  : <Your Name>
 Role    : Supply Chain Data Science Portfolio Project
 Stack   : Python (pandas, numpy, scipy) -> CSV -> Tableau

 BUSINESS PROBLEM
 ----------------
 Stockouts are driven by TWO independent sources of uncertainty:
   1. Demand variability  - customers don't order the same quantity every day.
   2. Supply variability  - suppliers don't deliver in the same number of days
                            every time (lead time is a random variable).

 A naive safety stock model (Z * sigma_D * sqrt(L)) ignores supplier
 unreliability and systematically UNDER-stocks when lead times are volatile.

 This script implements the industry-standard DUAL-VARIABILITY formula:

        SS = Z * sqrt( L_bar * sigma_D^2  +  D_bar^2 * sigma_L^2 )

 Where:
   Z        = service factor (z-score of the target cycle service level,
              e.g. 95% service level -> Z = 1.645, from the inverse normal CDF)
   L_bar    = average lead time in days (supplier performance)
   sigma_L  = standard deviation of lead time in days (supplier RELIABILITY)
   D_bar    = average daily demand in units
   sigma_D  = standard deviation of daily demand in units

   NOTE: sigma_D^2 and sigma_L^2 in the formula are the VARIANCES
   (i.e., the standard deviations squared).

 The first term  (L_bar * sigma_D^2)  captures demand risk during lead time.
 The second term (D_bar^2 * sigma_L^2) captures supplier delivery risk.
 Because the two risks are assumed independent, their variances add, and the
 square root converts the combined variance back into units of inventory.

 PIPELINE
 --------
   Step 1: Load & clean the transactional order dataset.
   Step 2: Aggregate lead time statistics per supplier segment (L_bar, sigma_L).
   Step 3: Aggregate daily demand statistics per product segment (D_bar, sigma_D).
   Step 4: Compute Safety Stock + Reorder Point at a target service level.
   Step 5: Score each supplier segment into a 2x2 Risk Matrix
           (lead time volatility x late-delivery rate) for Tableau.
   Step 6: Export a tidy, Tableau-ready CSV.

 DATASET
 -------
 Built for the Kaggle "DataCo Smart Supply Chain" dataset
 (DataCoSupplyChainDataset.csv), but every column name is configurable in
 CONFIG below, so any dataset with (order date, scheduled days, actual days,
 quantity, category/supplier key) will work. If no file is found, the script
 generates a realistic synthetic dataset so the pipeline is fully reproducible.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG - edit this block only; the rest of the pipeline is dataset-agnostic
# ---------------------------------------------------------------------------
CONFIG = {
    # Path to the raw Kaggle file. DataCo ships as latin-1 encoded CSV.
    "input_path": "DataCoSupplyChainDataset.csv",
    "encoding": "latin-1",

    # Column mapping (defaults = DataCo Smart Supply Chain column names).
    "col_order_date": "order date (DateOrders)",   # order placement timestamp
    "col_actual_days": "Days for shipping (real)", # realized lead time (days)
    "col_sched_days": "Days for shipment (scheduled)",  # promised lead time
    "col_quantity": "Order Item Quantity",         # units ordered per line
    "col_segment": "Category Name",                # supplier / product segment
    "col_late_flag": "Late_delivery_risk",         # 1 if delivered late

    # Modeling parameters
    "service_level": 0.95,     # target cycle service level (95% -> Z ~ 1.645)
    "min_orders": 30,          # drop segments with too few orders to model
    "output_path": "supplier_risk_metrics.csv",
}


# ---------------------------------------------------------------------------
# STEP 0 (fallback): synthetic data generator for reproducibility
# ---------------------------------------------------------------------------
def generate_synthetic_orders(n_orders: int = 40_000, seed: int = 42) -> pd.DataFrame:
    """
    Create a realistic synthetic order log if the Kaggle file is absent.

    Each 'segment' behaves like a distinct supplier with its own true mean
    lead time, lead time volatility, and daily demand level - so downstream
    risk scoring produces a meaningful spread on the Tableau risk matrix.
    """
    rng = np.random.default_rng(seed)
    segments = {
        # name:              (mean_lead, sd_lead, mean_daily_units)
        "Electronics":        (4.0, 0.8, 220),
        "Sporting Goods":     (3.5, 1.6, 180),   # volatile supplier
        "Furniture":          (6.5, 2.4, 90),    # slow AND volatile
        "Apparel":            (3.0, 0.5, 300),   # fast and reliable
        "Auto Parts":         (5.0, 1.2, 140),
        "Garden & Outdoor":   (5.5, 2.0, 110),
    }
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    rows = []
    for seg, (mu_l, sd_l, mu_d) in segments.items():
        n = n_orders // len(segments)
        order_dates = rng.choice(dates, size=n)
        lead = np.clip(rng.normal(mu_l, sd_l, size=n), 1, None).round()
        sched = np.full(n, round(mu_l))               # promised lead time
        qty = rng.poisson(lam=max(mu_d / 40, 1), size=n) + 1
        rows.append(pd.DataFrame({
            CONFIG["col_order_date"]: order_dates,
            CONFIG["col_actual_days"]: lead,
            CONFIG["col_sched_days"]: sched,
            CONFIG["col_quantity"]: qty,
            CONFIG["col_segment"]: seg,
            CONFIG["col_late_flag"]: (lead > sched).astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# STEP 1: Load & clean
# ---------------------------------------------------------------------------
def load_data(cfg: dict) -> pd.DataFrame:
    """
    Load the raw order log and enforce a clean modeling schema.

    Cleaning rules (documented so reviewers can audit assumptions):
      - Parse order dates; drop rows where the date cannot be parsed.
      - Coerce lead time & quantity to numeric; drop nulls.
      - Remove impossible records (lead time <= 0 or quantity <= 0).
    """
    if os.path.exists(cfg["input_path"]):
        df = pd.read_csv(cfg["input_path"], encoding=cfg["encoding"])
        source = f"Kaggle file: {cfg['input_path']}"
    else:
        df = generate_synthetic_orders(n_orders=12_000)
        df.to_csv("raw_orders_data.csv", index=False)   # save raw input
        source = "synthetic generator -> saved as raw_orders_data.csv"
    print(f"[Step 1] Loaded {len(df):,} order lines from {source}")

    df = df.copy()
    df[cfg["col_order_date"]] = pd.to_datetime(df[cfg["col_order_date"]],
                                               errors="coerce")
    for col in (cfg["col_actual_days"], cfg["col_quantity"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[cfg["col_order_date"], cfg["col_actual_days"],
                           cfg["col_quantity"], cfg["col_segment"]])
    df = df[(df[cfg["col_actual_days"]] > 0) & (df[cfg["col_quantity"]] > 0)]
    print(f"[Step 1] Cleaned: removed {before - len(df):,} invalid rows "
          f"-> {len(df):,} rows retained")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Lead time statistics per segment  (L_bar, sigma_L)
# ---------------------------------------------------------------------------
def lead_time_stats(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Measure supplier PERFORMANCE (mean lead time) and RELIABILITY
    (lead time standard deviation + late-delivery rate) per segment.
    """
    g = df.groupby(cfg["col_segment"])
    out = pd.DataFrame({
        "n_orders": g.size(),
        "avg_lead_time_days": g[cfg["col_actual_days"]].mean(),
        # ddof=1 -> sample standard deviation (we observe a sample of orders,
        # not the full population of all possible deliveries)
        "std_lead_time_days": g[cfg["col_actual_days"]].std(ddof=1),
        "late_delivery_rate": g[cfg["col_late_flag"]].mean(),
    })
    # Coefficient of variation: volatility normalized by the mean, so a
    # 2-day swing on a 10-day lead time isn't scored like one on a 3-day lead.
    out["lead_time_cv"] = out["std_lead_time_days"] / out["avg_lead_time_days"]
    print(f"[Step 2] Lead time stats computed for {len(out)} segments")
    return out


# ---------------------------------------------------------------------------
# STEP 3: Daily demand statistics per segment  (D_bar, sigma_D)
# ---------------------------------------------------------------------------
def demand_stats(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Convert the order log into a DAILY demand series per segment, then
    measure its mean and standard deviation.

    Key detail: days with zero orders are real observations of zero demand.
    We reindex each segment onto the full calendar and fill missing days
    with 0 - otherwise sigma_D is biased downward and safety stock is
    understated.
    """
    daily = (df.groupby([cfg["col_segment"],
                         df[cfg["col_order_date"]].dt.normalize()])
               [cfg["col_quantity"]].sum())

    full_range = pd.date_range(df[cfg["col_order_date"]].min().normalize(),
                               df[cfg["col_order_date"]].max().normalize(),
                               freq="D")
    records = {}
    for seg, series in daily.groupby(level=0):
        s = series.droplevel(0).reindex(full_range, fill_value=0)
        records[seg] = {"avg_daily_demand": s.mean(),
                        "std_daily_demand": s.std(ddof=1)}
    out = pd.DataFrame.from_dict(records, orient="index")
    print(f"[Step 3] Daily demand stats computed over "
          f"{len(full_range)} calendar days")
    return out


# ---------------------------------------------------------------------------
# STEP 4: Safety Stock + Reorder Point (the core model)
# ---------------------------------------------------------------------------
def compute_safety_stock(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply the dual-variability formula per segment:

        SS = Z * sqrt( L_bar * sigma_D^2  +  D_bar^2 * sigma_L^2 )

    and derive the Reorder Point:

        ROP = D_bar * L_bar + SS
        (expected demand during lead time, plus the safety buffer)

    Z is derived from the target service level via the inverse normal CDF
    (scipy.stats.norm.ppf), NOT hard-coded - so the business can re-run the
    model at 90%, 97.5%, 99% etc. by changing one config value.
    """
    z = stats.norm.ppf(cfg["service_level"])
    print(f"[Step 4] Service level {cfg['service_level']:.1%} -> Z = {z:.3f}")

    m = metrics.copy()
    demand_risk = m["avg_lead_time_days"] * m["std_daily_demand"] ** 2
    supply_risk = m["avg_daily_demand"] ** 2 * m["std_lead_time_days"] ** 2

    m["service_factor_z"] = z
    m["safety_stock_units"] = z * np.sqrt(demand_risk + supply_risk)
    m["reorder_point_units"] = (m["avg_daily_demand"] * m["avg_lead_time_days"]
                                + m["safety_stock_units"])

    # Decompose WHERE the risk comes from - a great talking point in Tableau:
    # "62% of Furniture's buffer exists because the supplier is unreliable,
    #  not because customers are unpredictable."
    total = demand_risk + supply_risk
    m["pct_risk_from_supplier"] = supply_risk / total
    m["pct_risk_from_demand"] = demand_risk / total
    return m


# ---------------------------------------------------------------------------
# STEP 5: Supplier Risk Matrix classification
# ---------------------------------------------------------------------------
def classify_risk(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket each segment into a 2x2 risk matrix for the Tableau dashboard:

        X-axis: lead time CV (delivery volatility)
        Y-axis: late delivery rate (promise-keeping)

    Thresholds use the cross-segment MEDIAN so the matrix always splits the
    portfolio into relative quadrants regardless of dataset scale.
    """
    m = metrics.copy()
    cv_med = m["lead_time_cv"].median()
    late_med = m["late_delivery_rate"].median()

    def quadrant(row) -> str:
        volatile = row["lead_time_cv"] > cv_med
        late = row["late_delivery_rate"] > late_med
        if volatile and late:
            return "Critical Risk"        # volatile AND frequently late
        if volatile:
            return "Volatility Risk"      # on-time on average, but erratic
        if late:
            return "Chronic Lateness"     # consistent, but consistently late
        return "Reliable"

    m["risk_quadrant"] = m.apply(quadrant, axis=1)
    print("[Step 5] Risk quadrants assigned:",
          m["risk_quadrant"].value_counts().to_dict())
    return m


# ---------------------------------------------------------------------------
# STEP 6: Orchestrate + export for Tableau
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    cfg = CONFIG
    df = load_data(cfg)

    metrics = lead_time_stats(df, cfg).join(demand_stats(df, cfg))
    metrics = metrics[metrics["n_orders"] >= cfg["min_orders"]]

    metrics = compute_safety_stock(metrics, cfg)
    metrics = classify_risk(metrics)

    # Tidy for Tableau: one row per segment, rounded, index as a column.
    export = (metrics.round(3)
                     .reset_index()
                     .rename(columns={"index": "segment",
                                      cfg["col_segment"]: "segment"}))
    export.to_csv(cfg["output_path"], index=False)
    print(f"[Step 6] Exported {len(export)} rows -> {cfg['output_path']}")

    cols = ["segment", "avg_lead_time_days", "std_lead_time_days",
            "avg_daily_demand", "std_daily_demand",
            "safety_stock_units", "reorder_point_units", "risk_quadrant"]
    print("\n=== SAFETY STOCK RECOMMENDATIONS ===")
    print(export[cols].to_string(index=False))
    return export


if __name__ == "__main__":
    main()
