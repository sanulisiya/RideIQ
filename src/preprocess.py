# Data Pre Processing and feature enginerring sectio

import pandas as pd
import numpy as np
import os

#Loading Dataset
RAW  = "data/sl_ride_fares.csv"
OUT  = "data/processed_fares.csv"

print("[1/5] Loading data...")
df = pd.read_csv(RAW, parse_dates=["datetime"])
print(f"      {len(df):,} rows | vehicles: {df['vehicle_type'].unique()}")

print("[2/5] Extracting time features...")
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"]       = df["datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Cyclical time encoding
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

# Demand flag
df["is_rush_hour"] = (
    ((df["hour"] >= 7)  & (df["hour"] <= 9)) |
    ((df["hour"] >= 17) & (df["hour"] <= 19))
).astype(int)
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

print("[3/5] Engineering weather & route features...")
df["is_raining"]    = (df["precip_mm"] > 0).astype(int)
df["is_heavy_rain"] = (df["precip_mm"] >= 10).astype(int)
df["heat_index"]    = df["temperature_C"] + 0.33 * (df["humidity_pct"] / 100 * 6.105 *
                      np.exp(17.27 * df["temperature_C"] / (237.7 + df["temperature_C"]))) - 4

# Distance buckets — short/medium/
df["dist_bucket"] = pd.cut(df["distance_km"],
                            bins=[0, 50, 150, 300, 9999],
                            labels=[0, 1, 2, 3]).astype(int)

print("[4/5] Encoding cities...")
all_cities = sorted(set(df["origin_city"]) | set(df["dest_city"]))
city_map   = {c: i for i, c in enumerate(all_cities)}
df["origin_id"] = df["origin_city"].map(city_map)
df["dest_id"]   = df["dest_city"].map(city_map)

# Route ID — symmetric pair encoding
df["route_id"] = df.apply(
    lambda r: city_map[min(r["origin_city"], r["dest_city"])] * 100 +
              city_map[max(r["origin_city"], r["dest_city"])], axis=1
)

print("[5/5] Encoding vehicle type and saving...")
vehicle_map = {"bike": 0, "tuk_tuk": 1, "car": 2}
df["vehicle_id"] = df["vehicle_type"].map(vehicle_map)

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

import json
with open("models/city_map.json",    "w") as f: json.dump(city_map,    f)
with open("models/vehicle_map.json", "w") as f: json.dump(vehicle_map, f)

city_coords = df.groupby("origin_city")[["origin_lat","origin_lon"]].mean().round(4)
city_coords.columns = ["lat","lon"]
city_coords.to_json("models/city_coords.json")

df.to_csv(OUT, index=False)
print(f"\nSaved {len(df):,} rows -> {OUT}")
print(f"Cities encoded: {city_map}")
print("Done")