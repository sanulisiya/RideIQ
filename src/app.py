
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json, os
from datetime import datetime, date, time as dtime

st.set_page_config(
    page_title="RideIQ | Sri Lanka",
    page_icon="🛺",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 760px; }
.fare-card {
    border-radius: 16px; padding: 1.4rem 1.6rem;
    margin: 0.4rem 0; display: flex;
    align-items: center; justify-content: space-between;
}
.fare-card.bike    { background: #0f2d1a; border: 1px solid #1d5c30; }
.fare-card.tuk_tuk { background: #2d1f00; border: 1px solid #7a4f00; }
.fare-card.car     { background: #0a1f35; border: 1px solid #1a4a7a; }
.fare-left  { display: flex; align-items: center; gap: 14px; }
.fare-icon  { font-size: 2.2rem; line-height: 1; }
.fare-name  { font-size: 1.05rem; font-weight: 600; }
.fare-desc  { font-size: 0.78rem; opacity: 0.55; margin-top: 2px; }
.fare-right { text-align: right; }
.fare-price { font-size: 2rem; font-weight: 700; }
.fare-range { font-size: 0.78rem; opacity: 0.5; margin-top: 2px; }
.fare-card.bike    .fare-price { color: #4ade80; }
.fare-card.tuk_tuk .fare-price { color: #fbbf24; }
.fare-card.car     .fare-price { color: #60a5fa; }
.tag { display: inline-block; padding: 3px 10px; border-radius: 20px;
       font-size: 0.72rem; font-weight: 500; margin: 3px 3px 3px 0; }
.tag-rain  { background:#0a2a4a; color:#60c8ff; border:1px solid #1a5a8a; }
.tag-rush  { background:#3a1a00; color:#ffb347; border:1px solid #8a4400; }
.tag-night { background:#1a0a3a; color:#c0a0ff; border:1px solid #5a3a9a; }
.tag-hol   { background:#0a2a0a; color:#80ff80; border:1px solid #1a6a1a; }
.tag-wknd  { background:#2a1a0a; color:#ffd080; border:1px solid #7a4a00; }
.summary-box { background:#111827; border:1px solid #1f2937;
               border-radius:12px; padding:1rem 1.2rem; margin-bottom:1.2rem; }
.summary-row { display:flex; justify-content:space-between;
               font-size:0.88rem; padding:3px 0; }
.summary-label { opacity:0.5; }
.summary-value { font-weight:500; }
.stButton > button {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    font-size:1rem !important; padding:0.65rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Path helper — works regardless of where streamlit is launched from ─────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

def mp(fname):
    return os.path.join(BASE, fname)

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for v in ["bike", "tuk_tuk", "car"]:
        m = xgb.XGBRegressor()
        m.load_model(mp(f"fare_model_{v}.json"))
        models[v] = m
    return models

@st.cache_data
def load_meta():
    with open(mp("city_map.json"))         as f: city_map   = json.load(f)
    with open(mp("fare_metrics.json"))     as f: metrics    = json.load(f)
    with open(mp("city_coords.json"))      as f: coords_raw = json.load(f)
    with open(mp("vehicle_features.json")) as f: vfeatures  = json.load(f)
    coords = {city: (coords_raw["lat"][city], coords_raw["lon"][city])
              for city in coords_raw["lat"]}
    return city_map, metrics, coords, vfeatures

try:
    models = load_models()
    city_map, metrics, city_coords, vehicle_features = load_meta()
except FileNotFoundError as e:
    st.error(f"**Models not found:** {e}\n\nRun training first:\n```\npython src/train.py\n```")
    st.stop()

CITIES = sorted(city_map.keys())

VEHICLE_META = {
    "bike":    {"icon": "🏍️", "label": "Motorbike", "desc": "Fastest · Budget pick"},
    "tuk_tuk": {"icon": "🛺",  "label": "Tuk-Tuk",   "desc": "Classic Sri Lanka ride"},
    "car":     {"icon": "🚗",  "label": "Car",        "desc": "Comfortable · Air-conditioned"},
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2-p1)/2)**2 +
         np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

def dist_bucket(d):
    if d < 50:  return 0
    if d < 150: return 1
    if d < 300: return 2
    return 3

def build_features(dist, hour, dow, month, is_weekend, is_holiday,
                   precip, temp, humidity, origin, destination, vtype):
    """Build exact feature vector for each vehicle model."""
    is_rush  = int((7<=hour<=9) or (17<=hour<=19))
    is_night = int(hour>=22 or hour<=5)
    is_rain  = int(precip > 0)
    is_heavy = int(precip >= 10)
    heat_idx = temp + 0.33*(humidity/100*6.105*np.exp(17.27*temp/(237.7+temp)))-4
    min_city = min(origin, destination)
    max_city = max(origin, destination)
    route_id = city_map.get(min_city,0)*100 + city_map.get(max_city,0)

    all_feat = {
        "distance_km":   dist,
        "dist_sq":       dist ** 2,
        "dist_bucket":   dist_bucket(dist),
        "hour":          hour,
        "day_of_week":   dow,
        "month":         month,
        "hour_sin":      np.sin(2*np.pi*hour/24),
        "hour_cos":      np.cos(2*np.pi*hour/24),
        "month_sin":     np.sin(2*np.pi*month/12),
        "month_cos":     np.cos(2*np.pi*month/12),
        "is_weekend":    is_weekend,
        "is_rush_hour":  is_rush,
        "is_night":      is_night,
        "is_holiday":    int(is_holiday),
        "precip_mm":     precip,
        "temperature_C": temp,
        "humidity_pct":  humidity,
        "heat_index":    heat_idx,
        "is_raining":    is_rain,
        "is_heavy_rain": is_heavy,
        "rain_x_rush":   precip * is_rush,
        "night_x_rain":  is_night * precip,
        "origin_id":     city_map.get(origin, 0),
        "dest_id":       city_map.get(destination, 0),
        "route_id":      route_id,
    }
    feats = vehicle_features[vtype]
    return pd.DataFrame([[all_feat[f] for f in feats]], columns=feats)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🛺 RideIQ — Sri Lanka Fare Predictor")
st.markdown("Instant fare estimates for **Motorbike**, **Tuk-Tuk**, and **Car** — powered by machine learning.")

a1, a2, a3 = st.columns(3)
a1.metric("Bike accuracy",    f"{metrics['bike']['accuracy']:.1f}%")
a2.metric("Tuk-Tuk accuracy", f"{metrics['tuk_tuk']['accuracy']:.1f}%")
a3.metric("Car accuracy",     f"{metrics['car']['accuracy']:.1f}%")

st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("**Trip Details**")
c1, c2 = st.columns(2)
with c1:
    origin = st.selectbox("📍 From", CITIES,
                           index=CITIES.index("Colombo") if "Colombo" in CITIES else 0)
with c2:
    dest_opts   = [c for c in CITIES if c != origin]
    destination = st.selectbox("📍 To", dest_opts,
                                index=dest_opts.index("Kandy") if "Kandy" in dest_opts else 0)

c3, c4 = st.columns(2)
with c3:
    trip_date = st.date_input("📅 Date", value=date.today())
with c4:
    trip_hour = st.slider("🕐 Departure hour", 0, 23,
                           datetime.now().hour, format="%d:00")

st.markdown("**Weather Conditions**")
w1, w2, w3 = st.columns(3)
with w1: precip   = st.number_input("🌧 Rainfall (mm)", 0.0, 200.0, 0.0, step=0.5)
with w2: temp     = st.number_input("🌡 Temperature (°C)", 15.0, 42.0, 29.0, step=0.5)
with w3: humidity = st.number_input("💧 Humidity (%)", 30.0, 100.0, 75.0, step=1.0)

is_holiday = st.checkbox("🎉 Public holiday")
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Fare", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict:
    if origin == destination:
        st.error("Origin and destination must be different.")
        st.stop()

    olat, olon = city_coords[origin]
    dlat, dlon = city_coords[destination]
    dist       = haversine_km(olat, olon, dlat, dlon)

    dt         = datetime.combine(trip_date, dtime(trip_hour, 0))
    dow        = dt.weekday()
    month      = dt.month
    is_weekend = int(dow >= 5)
    is_rush    = (7<=trip_hour<=9) or (17<=trip_hour<=19)
    is_night   = (trip_hour>=22) or (trip_hour<=5)

    tags = ""
    if precip > 0:  tags += '<span class="tag tag-rain">🌧 Rain surge</span>'
    if is_rush:     tags += '<span class="tag tag-rush">⚡ Rush hour</span>'
    if is_night:    tags += '<span class="tag tag-night">🌙 Night rate</span>'
    if is_holiday:  tags += '<span class="tag tag-hol">🎉 Holiday</span>'
    if is_weekend:  tags += '<span class="tag tag-wknd">📅 Weekend</span>'
    if not tags:    tags  = '<span style="opacity:0.4;font-size:0.82rem;">No surcharges active</span>'

    st.divider()
    st.markdown(f"""
    <div class="summary-box">
      <div class="summary-row">
        <span class="summary-label">Route</span>
        <span class="summary-value">{origin} → {destination}</span>
      </div>
      <div class="summary-row">
        <span class="summary-label">Distance</span>
        <span class="summary-value">{dist:.1f} km</span>
      </div>
      <div class="summary-row">
        <span class="summary-label">Departure</span>
        <span class="summary-value">{dt.strftime('%a %d %b %Y, %H:00')}</span>
      </div>
      <div class="summary-row">
        <span class="summary-label">Weather</span>
        <span class="summary-value">{temp:.1f}°C · {humidity:.0f}% humidity · {precip:.1f}mm rain</span>
      </div>
      <div class="summary-row" style="margin-top:6px;">
        <span class="summary-label">Active factors</span>
        <span>{tags}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Fare Estimates")

    results = {}
    for vtype in ["bike", "tuk_tuk", "car"]:
        feat = build_features(
            dist=dist, hour=trip_hour, dow=dow, month=month,
            is_weekend=is_weekend, is_holiday=is_holiday,
            precip=precip, temp=temp, humidity=humidity,
            origin=origin, destination=destination, vtype=vtype,
        )
        pred = float(models[vtype].predict(feat)[0])
        pred = max(pred, 250)
        results[vtype] = (pred, metrics[vtype]["mae"])

    for vtype, (pred, mae) in results.items():
        meta = VEHICLE_META[vtype]
        lo   = max(250, pred - mae)
        hi   = pred + mae
        st.markdown(f"""
        <div class="fare-card {vtype}">
          <div class="fare-left">
            <div class="fare-icon">{meta['icon']}</div>
            <div>
              <div class="fare-name">{meta['label']}</div>
              <div class="fare-desc">{meta['desc']}</div>
            </div>
          </div>
          <div class="fare-right">
            <div class="fare-price">LKR {pred:,.0f}</div>
            <div class="fare-range">Range: {lo:,.0f} – {hi:,.0f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    cheapest       = min(results, key=lambda v: results[v][0])
    most_expensive = max(results, key=lambda v: results[v][0])
    saving         = results[most_expensive][0] - results[cheapest][0]
    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.9rem 1.1rem;background:#0a1f0a;
                border:1px solid #1a4a1a;border-radius:12px;font-size:0.9rem;">
        💡 <strong>Best value:</strong> {VEHICLE_META[cheapest]['label']} saves you
        <strong>LKR {saving:,.0f}</strong> vs {VEHICLE_META[most_expensive]['label']}.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Cost per km**")
    p1, p2, p3 = st.columns(3)
    for col, vtype in zip([p1, p2, p3], ["bike", "tuk_tuk", "car"]):
        pred, _ = results[vtype]
        col.metric(VEHICLE_META[vtype]["label"],
                   f"LKR {pred/dist:.0f}/km",
                   f"Total: {pred:,.0f}")

st.divider()
st.markdown(
    "<div style='text-align:center;opacity:0.35;font-size:0.78rem;'>"
    "RideIQ · XGBoost ML · Sri Lanka · Estimates only</div>",
    unsafe_allow_html=True,
)