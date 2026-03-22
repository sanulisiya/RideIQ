"""
 train.py

"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import json, os, warnings
warnings.filterwarnings("ignore")
np.random.seed(99)

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PART 1 — GENERATE DATASET
# ══════════════════════════════════════════════════════════════
print("=" * 58)
print("  PART 1: Generating realistic Sri Lanka fare dataset")
print("=" * 58)

CITIES = {
    "Colombo":     (6.9271, 79.8612),
    "Kandy":       (7.2906, 80.6337),
    "Galle":       (6.0535, 80.2210),
    "Negombo":     (7.2008, 79.8380),
    "Jaffna":      (9.6615, 80.0255),
    "Matara":      (5.9549, 80.5550),
}
city_names = list(CITIES.keys())

# Each vehicle reacts differently to surge factors
# Wide random ranges force the model to learn patterns not memorize formulas
VEHICLES = {
    "bike": {
        "rate_km":    (40,  75),     # LKR/km — random per trip
        "flag":       (50,  150),    # base flag fall
        "rain_sens":  (0.0, 0.3),    # low rain sensitivity
        "rush_sens":  (0.8, 1.2),    # navigates traffic better
        "night_sens": (0.9, 1.5),    # riskier at night
        "driver_neg": (0.70, 1.40),  # widest negotiation range
        "ac_premium": 1.0,
    },
    "tuk_tuk": {
        "rate_km":    (65,  105),
        "flag":       (80,  200),
        "rain_sens":  (0.1, 0.6),    # moderate rain effect
        "rush_sens":  (1.1, 1.5),    # gets stuck in traffic
        "night_sens": (1.0, 1.3),
        "driver_neg": (0.75, 1.30),
        "ac_premium": 1.0,
    },
    "car": {
        "rate_km":    (100, 160),
        "flag":       (150, 350),
        "rain_sens":  (0.05, 0.25),  # least rain sensitive
        "rush_sens":  (1.2, 1.7),    # worst in traffic
        "night_sens": (0.95, 1.2),   # safer at night
        "driver_neg": (0.85, 1.20),  # least negotiation
        "ac_premium": 1.08,          # AC premium
    },
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2-p1)/2)**2 +
         np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

rows = []
N_PER_VEHICLE = 10000

for vtype, cfg in VEHICLES.items():
    print(f"  Generating {N_PER_VEHICLE:,} {vtype} trips...")
    for _ in range(N_PER_VEHICLE):
        o = np.random.choice(city_names)
        d = np.random.choice([c for c in city_names if c != o])
        olat, olon = CITIES[o]
        dlat, dlon = CITIES[d]

        olat += np.random.normal(0, 0.03)
        olon += np.random.normal(0, 0.03)
        dlat += np.random.normal(0, 0.03)
        dlon += np.random.normal(0, 0.03)

        dist = max(3.0, haversine(olat, olon, dlat, dlon))

        dt    = pd.Timestamp("2024-01-01") + pd.Timedelta(
                    seconds=int(np.random.uniform(0, 1.55e7)))
        hour  = dt.hour
        dow   = dt.dayofweek
        month = dt.month

        rain = float(np.random.exponential(4) if np.random.random() < 0.32 else 0)
        temp = float(np.clip(np.random.normal(29, 4), 17, 42))
        hum  = float(np.clip(np.random.normal(75, 13), 30, 100))
        hol  = int(np.random.random() < 0.07)

        rate     = np.random.uniform(*cfg["rate_km"])
        flag     = np.random.uniform(*cfg["flag"])
        base     = flag + dist * rate
        is_rush  = (7 <= hour <= 9) or (17 <= hour <= 19)
        is_night = (hour >= 22) or (hour <= 5)

        rain_m   = 1 + np.random.uniform(*cfg["rain_sens"]) * (min(rain, 40) / 40)
        rush_m   = np.random.uniform(*cfg["rush_sens"])  if is_rush  else np.random.uniform(0.85, 1.05)
        night_m  = np.random.uniform(*cfg["night_sens"]) if is_night else np.random.uniform(0.92, 1.05)
        wknd_m   = np.random.uniform(1.0,  1.2)  if dow >= 5 else np.random.uniform(0.88, 1.08)
        hol_m    = np.random.uniform(1.05, 1.35) if hol      else np.random.uniform(0.92, 1.05)
        driver_m = np.random.uniform(*cfg["driver_neg"])
        city_m   = np.random.uniform(1.05, 1.35) if o == "Colombo" else np.random.uniform(0.90, 1.15)
        fuel_m   = np.random.uniform(0.90, 1.15)

        fare = base * rain_m * rush_m * night_m * wknd_m * hol_m * driver_m * city_m * fuel_m * cfg["ac_premium"]
        fare = max(250, round(fare, 2))

        rows.append({
            "datetime":      dt.strftime("%Y-%m-%d %H:%M:%S"),
            "origin_city":   o,               "dest_city":    d,
            "origin_lat":    round(olat, 6),  "origin_lon":   round(olon, 6),
            "dest_lat":      round(dlat, 6),  "dest_lon":     round(dlon, 6),
            "distance_km":   round(dist, 2),
            "vehicle_type":  vtype,
            "fare_LKR":      fare,
            "precip_mm":     round(rain, 2),
            "temperature_C": round(temp, 1),
            "humidity_pct":  round(hum,  1),
            "is_holiday":    hol,
        })

df = pd.DataFrame(rows)
df.to_csv("data/sl_ride_fares.csv", index=False)
print(f"\n  Saved {len(df):,} rows -> data/sl_ride_fares.csv")
print("\n  Distance-fare correlation (healthy range: 0.75-0.85):")
for v in ["bike", "tuk_tuk", "car"]:
    sub  = df[df["vehicle_type"] == v]
    corr = sub["distance_km"].corr(sub["fare_LKR"])
    print(f"    {v:<10}  corr={corr:.3f}  mean=LKR {sub['fare_LKR'].mean():.0f}  std=LKR {sub['fare_LKR'].std():.0f}")

# ══════════════════════════════════════════════════════════════
# PART 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print()
print("=" * 58)
print("  PART 2: Feature engineering")
print("=" * 58)

df["datetime"]     = pd.to_datetime(df["datetime"])
df["hour"]         = df["datetime"].dt.hour
df["day_of_week"]  = df["datetime"].dt.dayofweek
df["month"]        = df["datetime"].dt.month
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

# Cyclical encoding — 11 PM and midnight are neighbours, not 23 apart
df["hour_sin"]     = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]     = np.cos(2 * np.pi * df["hour"]  / 24)
df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)

df["is_rush_hour"] = (((df["hour"] >= 7)  & (df["hour"] <= 9)) |
                      ((df["hour"] >= 17) & (df["hour"] <= 19))).astype(int)
df["is_night"]     = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
df["is_raining"]   = (df["precip_mm"] > 0).astype(int)
df["is_heavy_rain"]= (df["precip_mm"] >= 10).astype(int)
df["heat_index"]   = (df["temperature_C"] + 0.33 *
                      (df["humidity_pct"] / 100 * 6.105 *
                      np.exp(17.27 * df["temperature_C"] / (237.7 + df["temperature_C"]))) - 4)

# City encoding
all_cities = sorted(set(df["origin_city"]) | set(df["dest_city"]))
city_map   = {c: i for i, c in enumerate(all_cities)}
df["origin_id"]    = df["origin_city"].map(city_map)
df["dest_id"]      = df["dest_city"].map(city_map)
df["route_id"]     = df.apply(
    lambda r: city_map[min(r["origin_city"], r["dest_city"])] * 100 +
              city_map[max(r["origin_city"], r["dest_city"])], axis=1)
df["dist_bucket"]  = pd.cut(df["distance_km"], bins=[0, 50, 150, 300, 9999],
                             labels=[0, 1, 2, 3]).astype(int)

# Non-linear and interaction features
df["dist_sq"]      = df["distance_km"] ** 2
df["rain_x_rush"]  = df["precip_mm"] * df["is_rush_hour"]   # rain + rush combined
df["night_x_rain"] = df["is_night"]   * df["precip_mm"]     # night + rain combined

vehicle_map = {"bike": 0, "tuk_tuk": 1, "car": 2}
df["vehicle_id"]   = df["vehicle_type"].map(vehicle_map)

# Save metadata for app.py
with open("models/city_map.json",    "w") as f: json.dump(city_map,    f)
with open("models/vehicle_map.json", "w") as f: json.dump(vehicle_map, f)
city_coords = df.groupby("origin_city")[["origin_lat", "origin_lon"]].mean().round(4)
city_coords.columns = ["lat", "lon"]
city_coords.to_json("models/city_coords.json")
df.to_csv("data/processed_fares.csv", index=False)
print(f"  Saved {len(df):,} processed rows -> data/processed_fares.csv")

# ══════════════════════════════════════════════════════════════
# PART 3 — TRAIN ONE MODEL PER VEHICLE
# ══════════════════════════════════════════════════════════════
print()
print("=" * 58)
print("  PART 3: Training separate XGBoost model per vehicle")
print("=" * 58)

# Car gets heat_index as an extra feature (AC comfort pricing)
# Bike and tuk_tuk share the same feature set
BASE_FEATURES = [
    "distance_km", "dist_sq", "dist_bucket",
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "month", "month_sin", "month_cos",
    "is_weekend", "is_rush_hour", "is_night", "is_holiday",
    "precip_mm", "is_raining", "is_heavy_rain",
    "temperature_C", "humidity_pct",
    "rain_x_rush", "night_x_rain",
    "origin_id", "dest_id", "route_id",
]

VEHICLE_CONFIGS = {
    "bike": {
        "features": BASE_FEATURES,
        "params":   dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                         min_child_weight=8,  subsample=0.75, colsample_bytree=0.75,
                         reg_alpha=1.5, reg_lambda=3.0, gamma=0.5),
    },
    "tuk_tuk": {
        "features": BASE_FEATURES,
        "params":   dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                         min_child_weight=8,  subsample=0.75, colsample_bytree=0.75,
                         reg_alpha=1.5, reg_lambda=3.0, gamma=0.5),
    },
    "car": {
        "features": BASE_FEATURES + ["heat_index"],   # extra feature for car
        "params":   dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                         min_child_weight=8,  subsample=0.75, colsample_bytree=0.75,
                         reg_alpha=1.0, reg_lambda=2.5, gamma=0.3),
    },
}

# Save per-vehicle feature lists — app.py reads this to build the right feature vector
vehicle_features = {v: cfg["features"] for v, cfg in VEHICLE_CONFIGS.items()}
with open("models/vehicle_features.json", "w") as f:
    json.dump(vehicle_features, f, indent=2)

kf          = KFold(n_splits=5, shuffle=True, random_state=42)
metrics_all = {}

print(f"\n  {'Vehicle':<10} {'CV MAE':>12} {'CV MAPE':>10} {'CV R2':>8}  Verdict")
print(f"  {'-'*58}")

for vtype, cfg in VEHICLE_CONFIGS.items():
    sub      = df[df["vehicle_type"] == vtype].reset_index(drop=True)
    features = cfg["features"]
    X        = sub[features].values
    y        = sub["fare_LKR"].values

    fold_maes, fold_mapes, fold_r2s = [], [], []

    for fold, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(
            **cfg["params"],
            objective    = "reg:squarederror",
            random_state = 42 + fold,
            n_jobs       = -1,
            verbosity    = 0,
        )
        m.fit(X[tr], y[tr])
        preds = np.maximum(m.predict(X[va]), 0)
        fold_maes.append(mean_absolute_error(y[va],  preds))
        fold_mapes.append(mean_absolute_percentage_error(y[va], preds) * 100)
        fold_r2s.append(r2_score(y[va], preds))

    cv_mae  = float(np.mean(fold_maes))
    cv_mape = float(np.mean(fold_mapes))
    cv_r2   = float(np.mean(fold_r2s))
    std_mae = float(np.std(fold_maes))
    verdict = ("Excellent" if cv_mape < 8 else
               "Good"      if cv_mape < 15 else
               "Acceptable" if cv_mape < 22 else "Needs review")

    print(f"  {vtype:<10} LKR {cv_mae:>7.0f} +/-{std_mae:<5.0f}  {cv_mape:>7.1f}%  {cv_r2:>6.3f}  {verdict}")

    # Train final model on ALL data and save
    final = xgb.XGBRegressor(
        **cfg["params"],
        objective    = "reg:squarederror",
        random_state = 42,
        n_jobs       = -1,
        verbosity    = 0,
    )
    final.fit(X, y)
    final.save_model(f"models/fare_model_{vtype}.json")

    metrics_all[vtype] = {
        "mae":      round(cv_mae,  1),
        "mape":     round(cv_mape, 2),
        "r2":       round(cv_r2,   4),
        "accuracy": round(100 - cv_mape, 1),
    }

with open("models/fare_metrics.json", "w") as f:
    json.dump(metrics_all, f, indent=2)

# Unified feature list for backwards compatibility
all_feats = list(dict.fromkeys(f for cfg in VEHICLE_CONFIGS.values() for f in cfg["features"]))
with open("models/fare_features.txt", "w") as f:
    f.write("\n".join(all_feats))

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print()
print("=" * 58)
print("  TRAINING COMPLETE")
print("=" * 58)
for vtype, m in metrics_all.items():
    stars = ("★★★★★" if m["mape"] < 10 else
             "★★★★☆" if m["mape"] < 15 else
             "★★★☆☆" if m["mape"] < 20 else "★★☆☆☆")
    print(f"  {vtype:<10}  {stars}  Accuracy: {m['accuracy']}%   MAE: LKR {m['mae']:,.0f}   R2: {m['r2']}")
print()
print("  Files saved:")
print("    models/fare_model_bike.json")
print("    models/fare_model_tuk_tuk.json")
print("    models/fare_model_car.json")
print("    models/vehicle_features.json")
print("    models/city_map.json")
print("    models/city_coords.json")
print("    models/fare_metrics.json")
print()
print("    python -m streamlit run src/app.py")
