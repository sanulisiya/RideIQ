

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json, os, sys
from datetime import datetime, date, time as dtime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlp_parser import parse_trip_query
from best_time import render_best_time_tab

st.set_page_config(page_title="RideIQ | Sri Lanka", page_icon="🛺", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.block-container{padding-top:1.8rem;padding-bottom:2rem;max-width:780px;}
.fare-card{border-radius:16px;padding:1.4rem 1.6rem;margin:0.4rem 0;display:flex;align-items:center;justify-content:space-between;}
.fare-card.bike   {background:#0f2d1a;border:1px solid #1d5c30;}
.fare-card.tuk_tuk{background:#2d1f00;border:1px solid #7a4f00;}
.fare-card.car    {background:#0a1f35;border:1px solid #1a4a7a;}
.fare-left{display:flex;align-items:center;gap:14px;}
.fare-icon{font-size:2.2rem;line-height:1;}
.fare-name{font-size:1.05rem;font-weight:600;}
.fare-desc{font-size:0.78rem;opacity:0.55;margin-top:2px;}
.fare-right{text-align:right;}
.fare-price{font-size:2rem;font-weight:700;}
.fare-range{font-size:0.78rem;opacity:0.5;margin-top:2px;}
.fare-card.bike    .fare-price{color:#4ade80;}
.fare-card.tuk_tuk .fare-price{color:#fbbf24;}
.fare-card.car     .fare-price{color:#60a5fa;}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:500;margin:3px 3px 3px 0;}
.tag-rain {background:#0a2a4a;color:#60c8ff;border:1px solid #1a5a8a;}
.tag-rush {background:#3a1a00;color:#ffb347;border:1px solid #8a4400;}
.tag-night{background:#1a0a3a;color:#c0a0ff;border:1px solid #5a3a9a;}
.tag-hol  {background:#0a2a0a;color:#80ff80;border:1px solid #1a6a1a;}
.tag-wknd {background:#2a1a0a;color:#ffd080;border:1px solid #7a4a00;}
.summary-box{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem;}
.summary-row{display:flex;justify-content:space-between;font-size:0.88rem;padding:3px 0;}
.summary-label{opacity:0.5;}
.summary-value{font-weight:500;}
.nlp-box{background:#0a1a2a;border:1px solid #1a4a7a;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;}
.nlp-row{display:flex;justify-content:space-between;font-size:12px;padding:3px 0;}
.nlp-key{color:#60a5fa;font-family:monospace;min-width:130px;}
.nlp-val{color:var(--color-text-primary);font-weight:500;}
.stButton>button{background:linear-gradient(135deg,#1d4ed8,#2563eb)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;font-size:1rem!important;padding:0.65rem!important;}
</style>
""", unsafe_allow_html=True)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
def mp(f): return os.path.join(BASE, f)

@st.cache_resource
def load_models():
    m = {}
    for v in ["bike","tuk_tuk","car"]:
        x = xgb.XGBRegressor()
        x.load_model(mp(f"fare_model_{v}.json"))
        m[v] = x
    return m

@st.cache_data
def load_meta():
    with open(mp("city_map.json"))         as f: city_map  = json.load(f)
    with open(mp("fare_metrics.json"))     as f: metrics   = json.load(f)
    with open(mp("city_coords.json"))      as f: cr        = json.load(f)
    with open(mp("vehicle_features.json")) as f: vfeat     = json.load(f)
    coords = {c:(cr["lat"][c],cr["lon"][c]) for c in cr["lat"]}
    return city_map, metrics, coords, vfeat

try:
    models = load_models()
    city_map, metrics, city_coords, vehicle_features = load_meta()
except FileNotFoundError as e:
    st.error(f"**Models not found:** {e}\n\nRun:\n```\npython src/train.py\n```")
    st.stop()

CITIES = sorted(city_map.keys())
VLABELS = {"bike":"Motorbike","tuk_tuk":"Tuk-Tuk","car":"Car"}
VMETA   = {
    "bike":    {"icon":"🏍️","label":"Motorbike","desc":"Fastest · Budget pick"},
    "tuk_tuk": {"icon":"🛺", "label":"Tuk-Tuk",  "desc":"Classic Sri Lanka ride"},
    "car":     {"icon":"🚗", "label":"Car",       "desc":"Comfortable · Air-conditioned"},
}

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371; p1,p2=np.radians(lat1),np.radians(lat2)
    a=(np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2)
    return 2*R*np.arcsin(np.sqrt(a))

def dist_bucket(d): return 0 if d<50 else(1 if d<150 else(2 if d<300 else 3))

def build_feat(dist,hour,dow,month,wknd,hol,precip,temp,hum,orig,dest,vtype):
    rush=int((7<=hour<=9)or(17<=hour<=19)); night=int(hour>=22 or hour<=5)
    heat=temp+0.33*(hum/100*6.105*np.exp(17.27*temp/(237.7+temp)))-4
    mc,xc=min(orig,dest),max(orig,dest)
    af={"distance_km":dist,"dist_sq":dist**2,"dist_bucket":dist_bucket(dist),
        "hour":hour,"day_of_week":dow,"month":month,
        "hour_sin":np.sin(2*np.pi*hour/24),"hour_cos":np.cos(2*np.pi*hour/24),
        "month_sin":np.sin(2*np.pi*month/12),"month_cos":np.cos(2*np.pi*month/12),
        "is_weekend":wknd,"is_rush_hour":rush,"is_night":night,"is_holiday":int(hol),
        "precip_mm":precip,"temperature_C":temp,"humidity_pct":hum,"heat_index":heat,
        "is_raining":int(precip>0),"is_heavy_rain":int(precip>=10),
        "rain_x_rush":precip*rush,"night_x_rain":night*precip,
        "origin_id":city_map.get(orig,0),"dest_id":city_map.get(dest,0),
        "route_id":city_map.get(mc,0)*100+city_map.get(xc,0)}
    feats=vehicle_features[vtype]
    return pd.DataFrame([[af[f] for f in feats]],columns=feats)

def show_fares(orig,dest,dist,hour,dt,precip,temp,hum,hol,vfilter=None):
    dow=dt.weekday(); month=dt.month; wknd=int(dow>=5)
    rush=(7<=hour<=9)or(17<=hour<=19); night=(hour>=22)or(hour<=5)
    tags=""
    if precip>0: tags+='<span class="tag tag-rain">🌧 Rain surge</span>'
    if rush:     tags+='<span class="tag tag-rush">⚡ Rush hour</span>'
    if night:    tags+='<span class="tag tag-night">🌙 Night rate</span>'
    if hol:      tags+='<span class="tag tag-hol">🎉 Holiday</span>'
    if wknd:     tags+='<span class="tag tag-wknd">📅 Weekend</span>'
    if not tags: tags='<span style="opacity:0.4;font-size:0.82rem;">No surcharges active</span>'

    st.markdown(f"""<div class="summary-box">
      <div class="summary-row"><span class="summary-label">Route</span><span class="summary-value">{orig} → {dest}</span></div>
      <div class="summary-row"><span class="summary-label">Distance</span><span class="summary-value">{dist:.1f} km</span></div>
      <div class="summary-row"><span class="summary-label">Departure</span><span class="summary-value">{dt.strftime('%a %d %b %Y, %H:00')}</span></div>
      <div class="summary-row"><span class="summary-label">Weather</span><span class="summary-value">{temp:.1f}°C · {hum:.0f}% humidity · {precip:.1f}mm rain</span></div>
      <div class="summary-row" style="margin-top:6px;"><span class="summary-label">Active factors</span><span>{tags}</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Fare Estimates")
    vtypes = [vfilter] if vfilter else ["bike","tuk_tuk","car"]
    results={}
    for vt in vtypes:
        feat=build_feat(dist,hour,dow,month,wknd,hol,precip,temp,hum,orig,dest,vt)
        pred=max(float(models[vt].predict(feat)[0]),250)
        results[vt]=(pred,metrics[vt]["mae"])

    for vt,(pred,mae) in results.items():
        m=VMETA[vt]; lo=max(250,pred-mae); hi=pred+mae
        st.markdown(f"""<div class="fare-card {vt}">
          <div class="fare-left"><div class="fare-icon">{m['icon']}</div>
            <div><div class="fare-name">{m['label']}</div><div class="fare-desc">{m['desc']}</div></div>
          </div>
          <div class="fare-right"><div class="fare-price">LKR {pred:,.0f}</div>
            <div class="fare-range">Range: {lo:,.0f} – {hi:,.0f}</div></div>
        </div>""", unsafe_allow_html=True)

    if len(results)>1:
        cheap=min(results,key=lambda v:results[v][0])
        pricey=max(results,key=lambda v:results[v][0])
        saving=results[pricey][0]-results[cheap][0]
        st.markdown(f"""<div style="margin-top:1rem;padding:0.9rem 1.1rem;background:#0a1f0a;
            border:1px solid #1a4a1a;border-radius:12px;font-size:0.9rem;">
            💡 <strong>Best value:</strong> {VMETA[cheap]['label']} saves <strong>LKR {saving:,.0f}</strong>
            vs {VMETA[pricey]['label']}.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Cost per km**")
    cols=st.columns(len(results))
    for col,vt in zip(cols,results):
        pred,_=results[vt]
        col.metric(VMETA[vt]["label"],f"LKR {pred/dist:.0f}/km",f"Total: {pred:,.0f}")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🛺 RideIQ — Sri Lanka Fare Predictor")
st.markdown("Instant fare estimates for **Motorbike**, **Tuk-Tuk**, and **Car** — powered by ML + NLP.")

a1,a2,a3=st.columns(3)
a1.metric("Bike accuracy",    f"{metrics['bike']['accuracy']:.1f}%")
a2.metric("Tuk-Tuk accuracy", f"{metrics['tuk_tuk']['accuracy']:.1f}%")
a3.metric("Car accuracy",     f"{metrics['car']['accuracy']:.1f}%")

st.divider()

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_predict, tab_besttime = st.tabs(["🔍 Fare Predictor", "📊 Best Time to Book"])

with tab_besttime:
    render_best_time_tab(city_map, city_coords, vehicle_features, models, metrics)

with tab_predict:
    mode = st.radio("Input mode",
        ["🧠 NLP — just type your trip", "📋 Form — fill fields manually"],
        horizontal=True, label_visibility="collapsed")
    use_nlp = mode.startswith("🧠")

# ══════════════════════════════════════════════════════════════════════════════
# NLP MODE
# ══════════════════════════════════════════════════════════════════════════════
if use_nlp:
    st.markdown("**Describe your trip in plain English**")
    st.caption("Mention: cities · vehicle type · time · weather")

    examples = [
        "tuk tuk from Colombo to Kandy 8am heavy rain",
        "car from Negombo to Galle tonight",
        "bike colombo to matara morning",
        "cab jaffna to colombo 6pm holiday",
    ]
    ecols = st.columns(2)
    for i,ex in enumerate(examples):
        if ecols[i%2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["nlp_q"] = ex

    nlp_q = st.text_area("Query", value=st.session_state.get("nlp_q",""),
                          height=80, placeholder="e.g. tuk tuk from Colombo to Kandy 8am heavy rain",
                          label_visibility="collapsed")

    with st.expander("Optional: set temperature and humidity"):
        nc1,nc2=st.columns(2)
        with nc1: nlp_temp=st.number_input("Temperature (°C)",15.0,42.0,29.0,0.5,key="nt")
        with nc2: nlp_hum =st.number_input("Humidity (%)",30.0,100.0,75.0,1.0,key="nh")

    if st.button("🧠 Parse & Predict", use_container_width=True) and nlp_q.strip():
        p = parse_trip_query(nlp_q)

        conf_color=("#4ade80" if p["confidence"]>=80 else
                    "#fbbf24" if p["confidence"]>=50 else "#f87171")

        st.markdown(f"""<div class="nlp-box">
          <div style="font-size:12px;font-weight:500;color:#60a5fa;margin-bottom:8px;">
            NLP extraction · Confidence: <span style="color:{conf_color}">{p['confidence']}%</span>
          </div>
          <div style="height:6px;background:#1a2a3a;border-radius:3px;margin-bottom:10px;overflow:hidden;">
            <div style="height:100%;width:{p['confidence']}%;background:{conf_color};border-radius:3px;"></div>
          </div>
          <div class="nlp-row"><span class="nlp-key">origin</span><span class="nlp-val">{p['origin'] or '— not detected'}</span></div>
          <div class="nlp-row"><span class="nlp-key">destination</span><span class="nlp-val">{p['destination'] or '— not detected'}</span></div>
          <div class="nlp-row"><span class="nlp-key">vehicle</span><span class="nlp-val">{VLABELS.get(p['vehicle_type'],'—')}</span></div>
          <div class="nlp-row"><span class="nlp-key">departure hour</span><span class="nlp-val">{p['hour']:02d}:00</span></div>
          <div class="nlp-row"><span class="nlp-key">rainfall</span><span class="nlp-val">{p['precip_mm']} mm</span></div>
          <div class="nlp-row"><span class="nlp-key">holiday</span><span class="nlp-val">{'Yes' if p['is_holiday'] else 'No'}</span></div>
        </div>""", unsafe_allow_html=True)

        for w in p["warnings"]: st.warning(w)

        if p["origin"] and p["destination"]:
            if p["origin"]==p["destination"]:
                st.error("Origin and destination are the same city.")
            else:
                olat,olon=city_coords[p["origin"]]
                dlat,dlon=city_coords[p["destination"]]
                dist=haversine_km(olat,olon,dlat,dlon)
                dt  =datetime.combine(date.today(),dtime(p["hour"],0))
                st.divider()
                show_fares(p["origin"],p["destination"],dist,p["hour"],dt,
                           p["precip_mm"],nlp_temp,nlp_hum,p["is_holiday"],
                           p["vehicle_type"])
        else:
            st.error("Could not detect both cities. Try: 'car from Colombo to Kandy'")

# ══════════════════════════════════════════════════════════════════════════════
# FORM MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("**Trip Details**")
    c1,c2=st.columns(2)
    with c1: origin=st.selectbox("📍 From",CITIES,index=CITIES.index("Colombo") if "Colombo" in CITIES else 0)
    with c2:
        do=[c for c in CITIES if c!=origin]
        destination=st.selectbox("📍 To",do,index=do.index("Kandy") if "Kandy" in do else 0)

    c3,c4=st.columns(2)
    with c3: trip_date=st.date_input("📅 Date",value=date.today())
    with c4: trip_hour=st.slider("🕐 Departure hour",0,23,datetime.now().hour,format="%d:00")

    st.markdown("**Weather Conditions**")
    w1,w2,w3=st.columns(3)
    with w1: precip  =st.number_input("🌧 Rainfall (mm)",0.0,200.0,0.0,step=0.5)
    with w2: temp    =st.number_input("🌡 Temperature (°C)",15.0,42.0,29.0,step=0.5)
    with w3: humidity=st.number_input("💧 Humidity (%)",30.0,100.0,75.0,step=1.0)

    is_holiday=st.checkbox("🎉 Public holiday")
    st.markdown("<br>",unsafe_allow_html=True)

    if st.button("🔍 Predict Fare",use_container_width=True):
        if origin==destination:
            st.error("Origin and destination must be different.")
            st.stop()
        olat,olon=city_coords[origin]; dlat,dlon=city_coords[destination]
        dist=haversine_km(olat,olon,dlat,dlon)
        dt=datetime.combine(trip_date,dtime(trip_hour,0))
        st.divider()
        show_fares(origin,destination,dist,trip_hour,dt,precip,temp,humidity,is_holiday)

st.divider()
st.markdown("<div style='text-align:center;opacity:0.35;font-size:0.78rem;'>RideIQ · XGBoost ML + NLP · Sri Lanka · Estimates only</div>",
            unsafe_allow_html=True)