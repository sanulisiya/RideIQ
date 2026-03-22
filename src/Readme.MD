# 🛰️ Demand-Sentinel
### Spatio-Temporal Multi-Modal Engine for Urban Mobility Optimization

---

## Project Structure

```
demand_sentinel/
├── data/
│   └── uber_nyc_enriched.csv      ← PUT YOUR DATASET HERE
├── src/
│   ├── 01_preprocess.py           ← Feature engineering
│   ├── 02_train.py                ← XGBoost model training
│   ├── 03_explain.py              ← SHAP explainability
│   └── 04_dashboard.py            ← Streamlit dashboard
├── models/                        ← Auto-created on first run
├── outputs/                       ← Auto-created on first run
├── run_all.py                     ← Runs steps 1–3 in sequence
└── requirements.txt
```

---

## Setup

```bash
# 1. Create a virtual environment (in PyCharm or terminal)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
#    Copy uber_nyc_enriched.csv into the /data folder
```

---

## Running the Pipeline

**Option A — Run everything at once:**
```bash
python run_all.py
```

**Option B — Run step by step (recommended for learning):**
```bash
python src/01_preprocess.py   # ~10 seconds
python src/02_train.py        # ~2–5 minutes
python src/03_explain.py      # ~1–2 minutes
```

**Launch the dashboard:**
```bash
streamlit run src/04_dashboard.py
```
Then open `http://localhost:8501` in your browser.

---

## What Each Script Does

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `01_preprocess.py` | Raw CSV | `data/processed_features.csv` | Cleans data, creates lag & cyclical features |
| `02_train.py` | Processed CSV | `models/xgb_demand_model.json` | Trains XGBoost with Poisson loss |
| `03_explain.py` | Model + data | `outputs/shap_*.csv/png` | Computes SHAP feature attributions |
| `04_dashboard.py` | All outputs | Streamlit app | Interactive heatmap + SHAP panel |

---

## Key Technical Decisions

### Why Poisson Loss?
Ride demand is **count data** (0, 1, 2, … rides). Poisson regression:
- Guarantees predictions ≥ 0
- Handles the skewed distribution of ride counts correctly
- Outperforms MSE on this type of target

### Why Cyclical Encoding?
`hour = 23` and `hour = 0` are numerically 23 apart but only 1 hour apart.
Sine/cosine transforms map time onto a unit circle so the model "sees" continuity.

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

### Why Lag Features?
The single strongest predictor of demand *now* is demand *1 hour ago*.
`pickups_lag1h` and `pickups_lag24h` encode recent history directly.

### SHAP Explainability
TreeExplainer computes exact Shapley values for XGBoost trees.
Each prediction decomposes as:
```
prediction = base_value + Σ(SHAP values for all features)
```
This powers the "Why?" panel in the dashboard.

---

## Expected Evaluation Metrics

| Metric | Expected Range |
|--------|---------------|
| RMSE   | 80–150 rides  |
| MAPE   | 8–15%         |
| 95% CI | ± 100–200 rides |

---

## Dataset Columns

| Column | Description |
|--------|-------------|
| `pickup_dt` | Datetime of pickup (hourly) |
| `borough` | NYC borough |
| `pickups` | **Target** — ride count in that hour |
| `spd` | Wind speed |
| `vsb` | Visibility |
| `temp` | Temperature (°F) |
| `dewp` | Dew point |
| `slp` | Sea-level pressure |
| `pcp01` | 1-hour precipitation (mm) |
| `pcp06` | 6-hour precipitation |
| `pcp24` | 24-hour precipitation |
| `sd` | Snow depth |
| `hday` | Holiday flag (Y/N) |