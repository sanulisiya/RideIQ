
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED    = "data/processed_features.csv"
MODEL_PATH   = "models/xgb_demand_model.json"
FEATURE_FILE = "models/feature_names.txt"

FEATURE_LABELS = {
    "pickups_lag1h"     : "Demand 1h ago",
    "pickups_lag24h"    : "Demand 24h ago",
    "pickups_roll3h"    : "3h rolling average",
    "precip_mm"         : "Precipitation (mm)",
    "is_raining"        : "Rain flag",
    "is_heavy_rain"     : "Heavy rain flag",
    "temperature_C"     : "Temperature (C)",
    "humidity_%"        : "Humidity (%)",
    "hour_sin"          : "Hour of day (sin)",
    "hour_cos"          : "Hour of day (cos)",
    "dow_sin"           : "Day of week (sin)",
    "dow_cos"           : "Day of week (cos)",
    "month_sin"         : "Month (sin)",
    "month_cos"         : "Month (cos)",
    "is_weekend"        : "Weekend flag",
    "is_rush_hour"      : "Rush hour flag",
    "is_night"          : "Night flag",
    "is_holiday"        : "Public holiday",
    "city_id"           : "City / Zone",
    "dist_from_colombo" : "Distance from Colombo",
}

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("[1/4] Loading model and data...")
df = pd.read_csv(PROCESSED)

with open(FEATURE_FILE) as f:
    FEATURES = [l.strip() for l in f.readlines()]

X = df[FEATURES]

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

with open("outputs/evaluation_report.json") as f:
    metrics = json.load(f)

# Use test set (last 20%)
split_idx = int(len(X) * 0.8)
X_explain  = X.iloc[split_idx:].reset_index(drop=True)
df_explain = df.iloc[split_idx:].reset_index(drop=True)

# Subsample for speed
SAMPLE = min(3000, len(X_explain))
np.random.seed(42)
idx      = np.random.choice(len(X_explain), SAMPLE, replace=False)
X_sample = X_explain.iloc[idx]

print(f"      Explaining {SAMPLE} rows...")

# ── 2. SHAP values ─────────────────────────────────────────────────────────────
print("[2/4] Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
base_value  = float(explainer.expected_value)
print(f"      Base value (model average): {base_value:.2f} rides")

# ── 3. Summary plot ────────────────────────────────────────────────────────────
print("[3/4] Saving SHAP summary plot...")
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_sample,
    feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES],
    max_display=15,
    show=False,
    plot_type="bar",
)
plt.title("SHAP Feature Importance — RideIQ Sri Lanka", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: outputs/shap_summary_plot.png")

# ── 4. Sample human-readable explanation ──────────────────────────────────────
print("[4/4] Generating sample explanation...")

def generate_explanation(pos, df_ref, X_ref, shap_vals, base_val, model, features):
    row    = df_ref.iloc[pos]
    x_row  = X_ref.iloc[pos]
    sv     = shap_vals[pos]
    pred   = float(model.predict(x_row.to_frame().T)[0])
    actual = float(row["pickup_count"])
    ci     = metrics["ci_95_half"]

    sv_df = pd.DataFrame({
        "feature": features,
        "shap":    sv,
    }).sort_values("shap", key=abs, ascending=False).head(5)

    city    = row.get("city", "Unknown")
    pickup  = pd.to_datetime(row["pickup_datetime"])
    hour    = pickup.strftime("%H:%M")

    lines = [
        "=" * 54,
        "  DEMAND-SENTINEL  |  Prediction Explanation",
        "=" * 54,
        f"  Zone     : {city}",
        f"  Time     : {hour}  ({pickup.strftime('%A')})",
        f"  Actual   : {int(actual):>5} rides",
        f"  Predicted: {int(pred):>5} rides   (95% CI: +/-{ci:.0f})",
        f"  Base val : {base_val:>5.1f} rides  (model average)",
        "-" * 54,
        "  TOP FEATURE CONTRIBUTIONS (SHAP)",
        "-" * 54,
    ]
    for _, r in sv_df.iterrows():
        label = FEATURE_LABELS.get(r["feature"], r["feature"])
        sign  = "+" if r["shap"] > 0 else ""
        arrow = "increases demand" if r["shap"] > 0 else "decreases demand"
        lines.append(f"  {label:<30} {sign}{r['shap']:.1f}  ({arrow})")

    lines += ["-" * 54, "  NARRATIVE", "-" * 54]
    for _, r in sv_df.iterrows():
        label = FEATURE_LABELS.get(r["feature"], r["feature"])
        verb  = "Added" if r["shap"] > 0 else "Reduced"
        lines.append(f"  * {verb} ~{abs(r['shap']):.0f} rides because of [{label}]")
    lines.append("=" * 54)
    return "\n".join(lines)

# Pick the highest-demand row from the sample
best_pos = int(X_sample.reset_index(drop=True)["pickups_lag1h"].idxmax())
expl = generate_explanation(
    pos       = best_pos,
    df_ref    = df_explain.iloc[idx].reset_index(drop=True),
    X_ref     = X_sample.reset_index(drop=True),
    shap_vals = shap_values,
    base_val  = base_value,
    model     = model,
    features  = FEATURES,
)
print("\n" + expl)

with open("outputs/sample_explanation.txt", "w") as f:
    f.write(expl)

# Save raw SHAP values
pd.DataFrame(shap_values, columns=FEATURES).to_csv("outputs/shap_values.csv", index=False)

print("\nDone!")
print("  outputs/shap_summary_plot.png")
print("  outputs/shap_values.csv")
print("  outputs/sample_explanation.txt")