# 📦 Inventory & Supplier Reliability Optimization

**Statistical safety stock modeling in Python to mitigate stockout risk driven by demand volatility and supplier unreliability.**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![pandas](https://img.shields.io/badge/pandas-✓-green) ![numpy](https://img.shields.io/badge/numpy-✓-green) ![scipy](https://img.shields.io/badge/scipy-✓-green) ![Tableau](https://img.shields.io/badge/Tableau-Dashboard-orange)

---

## 🎯 Business Problem

Stockouts are caused by **two independent sources of uncertainty**:

1. **Demand variability** — customers don't order the same quantity every day.
2. **Supply variability** — suppliers don't deliver in the same number of days every time.

A naive safety stock model (`Z · σ_D · √L`) ignores supplier unreliability and **systematically under-stocks** when lead times are volatile. This project implements the industry-standard **dual-variability formula**:

```
SS = Z · √( L̄ · σ_D²  +  D̄² · σ_L² )
```

| Symbol | Meaning |
|--------|---------|
| `Z` | Service factor — z-score of the target cycle service level (95% → 1.645) |
| `L̄` | Average lead time (days) |
| `σ_L` | Standard deviation of lead time (supplier **reliability**) |
| `D̄` | Average daily demand (units) |
| `σ_D` | Standard deviation of daily demand (demand **volatility**) |

The first term captures **demand risk during lead time**; the second captures **supplier delivery risk**. Since the risks are independent, their variances add.

---

## 📂 Repository Structure

```
├── safety_stock_optimizer.py    # Full pipeline (load → model → export)
├── raw_orders_data.csv          # Input: 12,000 order lines (2024–2025)
├── supplier_risk_metrics.csv    # Output: Tableau/Power BI-ready metrics
└── README.md
```

---

## 🔄 Pipeline

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `load_data()` | Loads & cleans orders (parses dates, drops invalid lead times / quantities) |
| 2 | `lead_time_stats()` | Per segment: `L̄`, `σ_L`, late-delivery rate, lead-time CV |
| 3 | `demand_stats()` | Builds a **daily** demand series — zero-demand days reinserted so `σ_D` isn't biased downward |
| 4 | `compute_safety_stock()` | Applies the dual-variability formula; derives Reorder Point `ROP = D̄·L̄ + SS`; decomposes risk into supplier % vs demand % |
| 5 | `classify_risk()` | 2×2 Supplier Risk Matrix: lead-time volatility × late-delivery rate |
| 6 | `main()` | Exports a tidy CSV for BI dashboards |

**Key modeling decisions:**
- `Z` derived via `scipy.stats.norm.ppf(service_level)` — rerun at 90/97.5/99% by changing one config value.
- Sample standard deviation (`ddof=1`) since observed orders are a sample.
- Risk-quadrant thresholds use cross-segment medians, so the matrix adapts to any dataset scale.

---

## 📊 Sample Output (95% service level)

| Segment | L̄ (days) | σ_L | D̄ (units/day) | Safety Stock | Reorder Point | Risk Quadrant |
|---------|-----------|------|----------------|--------------|----------------|----------------|
| Apparel | 2.98 | 0.56 | 23.2 | 46 | 115 | ✅ Reliable |
| Sporting Goods | 3.60 | 1.54 | 15.1 | 49 | 104 | ⚠️ Volatility Risk |
| Furniture | 6.49 | 2.42 | 8.9 | 44 | 101 | 🔴 Critical Risk |

> Insight the dashboard surfaces: for Furniture, **most of the buffer exists because the supplier is unreliable, not because customers are unpredictable** (`pct_risk_from_supplier`).

---

## 🚀 How to Run

```bash
pip install pandas numpy scipy
python safety_stock_optimizer.py
```

Works out of the box on the included dataset. To use the **Kaggle DataCo Smart Supply Chain** dataset instead, download `DataCoSupplyChainDataset.csv` into the repo root — the script auto-detects it (column mapping already matches; edit `CONFIG` for any other dataset).

---

## 📈 Dashboard

Connect `supplier_risk_metrics.csv` to Tableau or Power BI:
- **Supplier Risk Matrix** — scatter: lead-time CV (x) × late-delivery rate (y), sized by safety stock, colored by risk quadrant, median reference lines forming the quadrants.
- **Safety Stock vs Reorder Point** — bar chart per segment.
- **Risk Decomposition** — stacked bars: % of buffer driven by supplier vs demand.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `scipy` · `Tableau / Power BI`

## 📄 License

MIT
