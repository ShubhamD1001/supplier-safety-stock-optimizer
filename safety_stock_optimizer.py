"""
=============================================================================
 INVENTORY & SUPPLIER RELIABILITY OPTIMIZATION
 Safety Stock Modeling with Dual Variability (Demand + Lead Time)
=============================================================================
 Author  : Shubham Dhyani
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
    # Generate a date range for synthetic orders
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    rows = []
    for seg, (mu_l, sd_l, mu_d) in segments.items():
        n = n_orders // len(segments)
        # Randomly choose order dates from the generated range
        order_dates = rng.choice(dates, size=n)
        # Simulate lead time with some randomness, ensuring it's at least 1 day
        lead = np.clip(rng.normal(mu_l, sd_l, size=n), 1, None).round()
        # Promised lead time is a rounded mean lead time
        sched = np.full(n, round(mu_l))
        # Simulate quantity using Poisson distribution, ensuring it's at least 1
        qty = rng.poisson(lam=max(mu_d / 40, 1), size=n) + 1
        rows.append(pd.DataFrame({
            CONFIG["col_order_date"]: order_dates,
            CONFIG["col_actual_days"]: lead,
            CONFIG["col_sched_days"]: sched,
            CONFIG["col_quantity"]: qty,
            CONFIG["col_segment"]: seg,
            # Determine if delivery was late based on actual vs scheduled lead time
            CONFIG["col_late_flag"]: (lead > sched).astype(int),
        }))
    # Concatenate all generated dataframes into a single dataframe
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
    # Check if the input CSV file exists
    if os.path.exists(cfg["input_path"]):
        # Load data from the specified CSV file
        df = pd.read_csv(cfg["input_path"], encoding=cfg["encoding"])
        source = f"Kaggle file: {cfg['input_path']}"
    else:
        # Generate synthetic orders if the file is not found
        df = generate_synthetic_orders(n_orders=12_000)
        # Save the synthetic data to a CSV file
        df.to_csv("raw_orders_data.csv", index=False)
        source = "synthetic generator -> saved as raw_orders_data.csv"
    print(f"[Step 1] Loaded {len(df):,} order lines from {source}")

    df = df.copy()
    # Convert the order date column to datetime objects, coercing errors to NaT
    df[cfg["col_order_date"]] = pd.to_datetime(df[cfg["col_order_date"]],
                                               errors="coerce")
    # Convert actual days and quantity columns to numeric, coercing errors to NaN
    for col in (cfg["col_actual_days"], cfg["col_quantity"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    # Drop rows with any NaN values in critical columns
    df = df.dropna(subset=[cfg["col_order_date"], cfg["col_actual_days"],
                           cfg["col_quantity"], cfg["col_segment"]])
    # Filter out records where actual days or quantity are non-positive
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
    # Group data by the segment column
    g = df.groupby(cfg["col_segment"])
    out = pd.DataFrame({
        "n_orders": g.size(),  # Number of orders per segment
        "avg_lead_time_days": g[cfg["col_actual_days"]].mean(),  # Average lead time
        # Standard deviation of lead time (sample standard deviation)
        "std_lead_time_days": g[cfg["col_actual_days"]].std(ddof=1),
        "late_delivery_rate": g[cfg["col_late_flag"]].mean(),  # Mean late delivery rate
    })
    # Coefficient of variation: volatility normalized by the mean
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
    # Group by segment and normalized order date to sum quantities for daily demand
    daily = (df.groupby([cfg["col_segment"],
                         df[cfg["col_order_date"]].dt.normalize()])
               [cfg["col_quantity"]].sum())

    # Create a full date range from the earliest to the latest order date
    full_range = pd.date_range(df[cfg["col_order_date"]].min().normalize(),
                               df[cfg["col_order_date"]].max().normalize(),
                               freq="D")
    records = {}
    # Iterate through each segment's daily demand series
    for seg, series in daily.groupby(level=0):
        # Drop the segment level, reindex to the full date range, filling missing days with 0
        s = series.droplevel(0).reindex(full_range, fill_value=0)
        records[seg] = {"avg_daily_demand": s.mean(),  # Average daily demand
                        "std_daily_demand": s.std(ddof=1)}  # Standard deviation of daily demand
    # Create a DataFrame from the collected demand statistics
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
    # Calculate Z-score for the target service level using inverse normal CDF
    z = stats.norm.ppf(cfg["service_level"])
    print(f"[Step 4] Service level {cfg['service_level']:.1%} -> Z = {z:.3f}")

    m = metrics.copy()
    # Calculate demand risk component: average lead time * demand variance
    demand_risk = m["avg_lead_time_days"] * m["std_daily_demand"] ** 2
    # Calculate supply risk component: average daily demand squared * lead time variance
    supply_risk = m["avg_daily_demand"] ** 2 * m["std_lead_time_days"] ** 2

    m["service_factor_z"] = z
    # Calculate safety stock using the dual-variability formula
    m["safety_stock_units"] = z * np.sqrt(demand_risk + supply_risk)
    # Calculate reorder point: expected demand during lead time + safety stock
    m["reorder_point_units"] = (m["avg_daily_demand"] * m["avg_lead_time_days"]
                                + m["safety_stock_units"])

    # Decompose WHERE the risk comes from - a great talking point in Tableau:
    # "62% of Furniture's buffer exists because the supplier is unreliable,
    #  not because customers are unpredictable."
    total = demand_risk + supply_risk
    # Percentage of risk attributed to supplier variability
    m["pct_risk_from_supplier"] = supply_risk / total
    # Percentage of risk attributed to demand variability
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
    # Calculate median lead time coefficient of variation across all segments
    cv_med = m["lead_time_cv"].median()
    # Calculate median late delivery rate across all segments
    late_med = m["late_delivery_rate"].median()

    # Define a function to classify each segment into a risk quadrant
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

    # Apply the quadrant classification function to each row
    m["risk_quadrant"] = m.apply(quadrant, axis=1)
    print("[Step 5] Risk quadrants assigned:",
          m["risk_quadrant"].value_counts().to_dict())
    return m


# ---------------------------------------------------------------------------
# STEP 6: Orchestrate + export for Tableau
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    # Get the configuration settings
    cfg = CONFIG
    # Load and clean the data
    df = load_data(cfg)

    # Compute lead time and demand statistics, then join them
    metrics = lead_time_stats(df, cfg).join(demand_stats(df, cfg))
    # Filter out segments with fewer orders than the minimum threshold
    metrics = metrics[metrics["n_orders"] >= cfg["min_orders"]]

    # Compute safety stock and reorder point
    metrics = compute_safety_stock(metrics, cfg)
    # Classify segments into risk quadrants
    metrics = classify_risk(metrics)

    # Prepare data for Tableau: round numerical values, reset index, rename columns
    export = (metrics.round(3)
                     .reset_index()
                     .rename(columns={
                         "index": "segment",
                         cfg["col_segment"]: "segment" # In case index name is the same as col_segment
                         }))
    # Export the final metrics to a CSV file
    export.to_csv(cfg["output_path"], index=False)
    print(f"[Step 6] Exported {len(export)} rows -> {cfg['output_path']}")

    # Define columns to display in the console output
    cols = ["segment", "avg_lead_time_days", "std_lead_time_days",
            "avg_daily_demand", "std_daily_demand",
            "safety_stock_units", "reorder_point_units", "risk_quadrant"]
    print("\n=== SAFETY STOCK RECOMMENDATIONS ===")
    # Print the selected columns to the console
    print(export[cols].to_string(index=False))
    return export


# Execute the main function when the script is run
if __name__ == "__main__":
    main()




