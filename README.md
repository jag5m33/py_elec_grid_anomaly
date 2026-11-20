# GB Electricity Demand: Anomaly Detection

**Question:** which half-hours of Great Britain's electricity demand were genuinely unusual, beyond the normal daily, weekly and seasonal pattern?

**Data:** [NESO Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data), 2018–2023: 105,168 half-hours.
**Tools:** Databricks, PySpark, Spark SQL, Python (pandas, statsmodels, scikit-learn), MLflow.

> Status: in progress. The pipeline runs end to end in Databricks. I'm now improving the model after finding a bias (below).

## The pipeline (Databricks)

| Step | Notebook | What it does |
|---|---|---|
| 1. Load | `databricks/01_load.ipynb` | Reads six yearly CSVs from a Databricks volume with PySpark and saves one table |
| 2. Clean | `databricks/02_clean.ipynb` | Fixes mixed date formats, builds UTC timestamps from the 48 half-hour periods, checks for gaps and missing values, adds date features |
| 3. Model | `databricks/03_models.ipynb` | MSTL decomposition, then isolation forest, with runs tracked in MLflow |
| 4. Analyse | `databricks/04_analysis.ipynb` | Spark SQL queries on the flagged half-hours |

The original pandas version is in `notebooks/`, and the reusable model code is in `src/`.

## Method

1. **MSTL decomposition** (`src/mstl.py`) splits national demand into trend + daily pattern + weekly pattern + **residual**. The residual is what demand did beyond its normal pattern, so that's where anomalies should show.
2. **Isolation forest** (`src/isolation_forest.py`) is unsupervised. It flags points that are easy to separate from the rest.
3. **PCA** (`src/pca.py`) is a second detector that flags rows it can't rebuild well from 8 components (93% of variance explained).

## What I found so far

- **The first model was biased.** Trained on all 11 columns, isolation forest mostly flagged **winter Sundays**. Columns like weekday and month have edges (Sunday, January, December) that are easy to isolate. Christmas Day and the Covid lockdown got **no flags**.
- **The fix:** train on the residual only, so the model sees only the "surprise" in demand. The Databricks version now does this, and I'm checking whether it catches known events.

## Data dictionary (main columns)

| Column | Meaning |
|---|---|
| `SETTLEMENT_DATE`, `SETTLEMENT_PERIOD` | Date and half-hour of the day (1–48) |
| `ND` | National demand (MW): the series searched for anomalies |
| `EMBEDDED_SOLAR_GENERATION`, `EMBEDDED_WIND_GENERATION` | Estimated small solar and wind output (MW) |
| `*_FLOW` | Interconnector flows to other countries (positive = import) |

## Next steps

- Check the residual-only model against known events (Christmas, Covid lockdown, heatwaves).
- Compare isolation forest and PCA flags, and keep the days both agree on.
- Move the cleaning checks into SQL.
