
import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import shap
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Demand-Sentinel",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="metric-container"] {
    background: #0f1923;
    border: 1px solid #1e3040;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# Loaders
@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model("models/xgb_demand_model.json")
    return m

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_features.csv",
                       parse_dates=["pickup_datetime"])

@st.cache_data
def load_metrics():
    with open("outputs/evaluation_report.json") as f:
        return json.load(f)

@st.cache_data
def load_fi():
    return pd.read_csv("outputs/feature_importance.csv")

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

with open("models/feature_names.txt") as f:
    FEATURES = [l.strip() for l in f.readlines()]

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

# Header
st.markdown("## 🛰️ Demand-Sentinel — Sri Lanka")
st.markdown("**Spatio-Temporal Ride Demand Forecasting** · XGBoost + SHAP")

try:
    df      = load_data()
    model   = load_model()
    metrics = load_metrics()
    fi      = load_fi()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}\n\nRun the pipeline first:\n```\npython run_all.py\n```")
    st.stop()

#  Sidebar
with st.sidebar:
    st.header("Controls")

    cities = sorted(df["city"].dropna().unique())
    selected_cities = st.multiselect("Cities", cities, default=cities)

    date_min = df["pickup_datetime"].min().date()
    date_max = df["pickup_datetime"].max().date()
    date_range = st.date_input("Date range", [date_min, date_max],
                               min_value=date_min, max_value=date_max)

    hour_range = st.slider("Hour of day", 0, 23, (0, 23))

    st.divider()
    st.subheader("Model Metrics")
    st.metric("RMSE",   f"{metrics['rmse']:.2f} rides")
    st.metric("MAE",    f"{metrics['mae']:.2f} rides")
    st.metric("MAPE",   f"{metrics['mape']:.1f}%")
    st.metric("95% CI", f"+/- {metrics['ci_95_half']:.0f} rides")

# Filter
mask = (
    df["city"].isin(selected_cities) &
    (df["pickup_datetime"].dt.date >= date_range[0]) &
    (df["pickup_datetime"].dt.date <= date_range[1]) &
    (df["pickup_datetime"].dt.hour >= hour_range[0]) &
    (df["pickup_datetime"].dt.hour <= hour_range[1])
)
filtered = df[mask].copy()

# KPI Cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Rides",      f"{int(filtered['pickup_count'].sum()):,}")
k2.metric("Peak Hour Demand", f"{int(filtered['pickup_count'].max()):,}")
k3.metric("Avg Hourly",       f"{filtered['pickup_count'].mean():.1f}")
k4.metric("Records in view",  f"{len(filtered):,}")

st.divider()

#  Heatmap + Map
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Zone x Hour Demand Heatmap")
    hm = (
        filtered.groupby(["city", filtered["pickup_datetime"].dt.hour])["pickup_count"]
        .mean().reset_index()
    )
    hm.columns = ["City", "Hour", "Avg Demand"]
    pivot = hm.pivot(index="City", columns="Hour", values="Avg Demand").fillna(0)

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale="RdYlBu_r",
        aspect="auto",
        labels=dict(x="Hour", y="City", color="Avg Rides"),
    )
    fig_hm.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_hm, use_container_width=True)

with col2:
    st.subheader("Pickup Location Map")
    map_data = filtered.sample(min(2000, len(filtered)), random_state=42)
    fig_map = px.scatter_mapbox(
        map_data,
        lat="latitude", lon="longitude",
        color="pickup_count",
        size="pickup_count",
        color_continuous_scale="RdYlBu_r",
        mapbox_style="carto-darkmatter",
        zoom=6,
        center={"lat": 7.8731, "lon": 80.7718},
        hover_data={"city": True, "pickup_count": True,
                    "latitude": False, "longitude": False},
        labels={"pickup_count": "Rides"},
    )
    fig_map.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

st.divider()

#  SHAP Panel
st.subheader("SHAP Explainability — Why did the model predict this?")

col_l, col_r = st.columns([1, 1.5])

with col_l:
    st.markdown("**Select a zone + time to explain**")
    explain_city = st.selectbox("City", cities,
                                index=cities.index("Colombo") if "Colombo" in cities else 0)
    explain_date = st.date_input("Date",
                                 value=df["pickup_datetime"].iloc[len(df)//2].date(),
                                 min_value=date_min, max_value=date_max,
                                 key="xdate")
    explain_hour = st.slider("Hour", 0, 23, 17)
    run_shap = st.button("Explain this prediction", use_container_width=True)

with col_r:
    if run_shap:
        match = df[
            (df["city"] == explain_city) &
            (df["pickup_datetime"].dt.date == explain_date) &
            (df["pickup_datetime"].dt.hour == explain_hour)
        ]

        if match.empty:
            st.warning("No data found for that combination. Try a different date or hour.")
        else:
            row    = match.iloc[0]
            X_row  = match[FEATURES].iloc[[0]]
            pred   = float(model.predict(X_row)[0])
            actual = float(row["pickup_count"])

            explainer  = get_explainer(model)
            sv         = explainer.shap_values(X_row)[0]
            base_val   = float(explainer.expected_value)

            c1, c2, c3 = st.columns(3)
            c1.metric("Actual",    f"{int(actual)}")
            c2.metric("Predicted", f"{int(pred)}", f"+/- {metrics['ci_95_half']:.0f}")
            c3.metric("Base avg",  f"{base_val:.0f}")

            sv_df = pd.DataFrame({
                "Feature":    [FEATURE_LABELS.get(f, f) for f in FEATURES],
                "SHAP Value": sv,
            }).sort_values("SHAP Value", key=abs, ascending=True).tail(10)

            sv_df["Direction"] = sv_df["SHAP Value"].apply(
                lambda x: "Increases demand" if x > 0 else "Decreases demand"
            )

            fig_shap = px.bar(
                sv_df, x="SHAP Value", y="Feature",
                color="Direction",
                color_discrete_map={
                    "Increases demand": "#E74C3C",
                    "Decreases demand": "#2E86C1",
                },
                orientation="h",
                title=f"{explain_city} at {explain_hour:02d}:00",
            )
            fig_shap.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis_title="",
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("**Narrative explanation**")
            top5 = pd.DataFrame({"feature": FEATURES, "shap": sv}) \
                     .sort_values("shap", key=abs, ascending=False).head(5)
            for _, r in top5.iterrows():
                lbl  = FEATURE_LABELS.get(r["feature"], r["feature"])
                verb = "Added" if r["shap"] > 0 else "Reduced"
                st.markdown(f"- {verb} **{abs(r['shap']):.0f} rides** due to *{lbl}*")
    else:
        st.info("Select a city and time, then click **Explain this prediction**.")

st.divider()

#  Feature Importance + Distribution
col_fi, col_dist = st.columns(2)

with col_fi:
    st.subheader("Feature Importance (XGBoost)")
    top_fi = fi.head(12).copy()
    top_fi["Feature"] = top_fi["feature"].apply(lambda x: FEATURE_LABELS.get(x, x))
    fig_fi = px.bar(
        top_fi.sort_values("importance"),
        x="importance", y="Feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Reds",
    )
    fig_fi.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with col_dist:
    st.subheader("Demand Distribution by City")
    cap = filtered["pickup_count"].quantile(0.98)
    fig_box = px.box(
        filtered[filtered["pickup_count"] <= cap],
        x="city", y="pickup_count",
        color="city",
        labels={"pickup_count": "Rides/hour", "city": ""},
    )
    fig_box.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

#  Row 4: Rain impact
st.divider()
st.subheader("Rain Impact on Demand")

rain_df = filtered.copy()
rain_df["Weather"] = rain_df["is_raining"].map({1: "Raining", 0: "Dry"})
fig_rain = px.box(
    rain_df[rain_df["pickup_count"] <= rain_df["pickup_count"].quantile(0.98)],
    x="city", y="pickup_count",
    color="Weather",
    color_discrete_map={"Raining": "#2E86C1", "Dry": "#E67E22"},
    labels={"pickup_count": "Rides/hour", "city": ""},
    title="Demand: Raining vs Dry by City",
)
fig_rain.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_rain, use_container_width=True)

st.markdown("---")
st.markdown(
    "<small>Demand-Sentinel · XGBoost + SHAP · Sri Lanka Synthetic Dataset · Streamlit + Plotly</small>",
    unsafe_allow_html=True,
)