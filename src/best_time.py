"""
best_time.py — Best Time to Book tab for RideIQ
================================================
Renders a Streamlit tab showing:
  • 24-hour fare chart for all 3 vehicles
  • Cheapest hour recommendation per vehicle
  • Side-by-side vehicle comparison at any hour
  • Full savings summary table

Called from app.py:
    render_best_time_tab(city_map, city_coords, vehicle_features, models, metrics)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date


# ─────────────────────────────────────────────
# Internal helpers  (mirror app.py logic)
# ─────────────────────────────────────────────

def _dist_bucket(d):
    return 0 if d < 50 else (1 if d < 150 else (2 if d < 300 else 3))


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2 +
         np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def _build_feat(hour, dist, dow, month, precip, temp, hum,
                is_holiday, origin, destination,
                city_map, vehicle_features, vtype):
    rush  = int((7 <= hour <= 9) or (17 <= hour <= 19))
    night = int(hour >= 22 or hour <= 5)
    heat  = temp + 0.33 * (hum / 100 * 6.105 * np.exp(17.27 * temp / (237.7 + temp))) - 4
    mc, xc = min(origin, destination), max(origin, destination)

    af = {
        "distance_km":   dist,
        "dist_sq":       dist ** 2,
        "dist_bucket":   _dist_bucket(dist),
        "hour":          hour,
        "day_of_week":   dow,
        "month":         month,
        "hour_sin":      np.sin(2 * np.pi * hour / 24),
        "hour_cos":      np.cos(2 * np.pi * hour / 24),
        "month_sin":     np.sin(2 * np.pi * month / 12),
        "month_cos":     np.cos(2 * np.pi * month / 12),
        "is_weekend":    int(dow >= 5),
        "is_rush_hour":  rush,
        "is_night":      night,
        "is_holiday":    int(is_holiday),
        "precip_mm":     precip,
        "temperature_C": temp,
        "humidity_pct":  hum,
        "heat_index":    heat,
        "is_raining":    int(precip > 0),
        "is_heavy_rain": int(precip >= 10),
        "rain_x_rush":   precip * rush,
        "night_x_rain":  night * precip,
        "origin_id":     city_map.get(origin, 0),
        "dest_id":       city_map.get(destination, 0),
        "route_id":      city_map.get(mc, 0) * 100 + city_map.get(xc, 0),
    }
    feats = vehicle_features[vtype]
    return pd.DataFrame([[af[f] for f in feats]], columns=feats)


def get_hourly_fares(models, vehicle_features, city_map,
                     origin, destination, dist,
                     precip, temp, hum, is_holiday, trip_date):
    """
    Returns 24-hour fare predictions for all 3 vehicles + cheapest window.
    """
    dow   = trip_date.weekday()
    month = trip_date.month
    out   = {"hours": list(range(24)), "bike": [], "tuk_tuk": [], "car": []}

    for hour in range(24):
        for vtype in ["bike", "tuk_tuk", "car"]:
            feat = _build_feat(hour, dist, dow, month, precip, temp, hum,
                               is_holiday, origin, destination,
                               city_map, vehicle_features, vtype)
            pred = max(float(models[vtype].predict(feat)[0]), 250)
            out[vtype].append(round(pred, 0))

    out["cheapest"] = {}
    for vtype in ["bike", "tuk_tuk", "car"]:
        fares = out[vtype]
        lo, hi = min(fares), max(fares)
        out["cheapest"][vtype] = {
            "hour":       fares.index(lo),
            "fare":       lo,
            "worst_hour": fares.index(hi),
            "worst_fare": hi,
            "saving":     round(hi - lo, 0),
            "saving_pct": round((hi - lo) / hi * 100, 1),
        }
    return out


# ─────────────────────────────────────────────
# Vehicle display metadata
# ─────────────────────────────────────────────

VMETA = {
    "bike":    {"icon": "🏍️", "label": "Motorbike", "color": "#4ade80", "bg": "#0f2d1a", "border": "#1d5c30"},
    "tuk_tuk": {"icon": "🛺",  "label": "Tuk-Tuk",   "color": "#fbbf24", "bg": "#2d1f00", "border": "#7a4f00"},
    "car":     {"icon": "🚗",  "label": "Car",        "color": "#60a5fa", "bg": "#0a1f35", "border": "#1a4a7a"},
}


# ─────────────────────────────────────────────
# Main tab renderer
# ─────────────────────────────────────────────

def render_best_time_tab(city_map, city_coords, vehicle_features, models, metrics):
    """
    Full Streamlit render for the 'Best Time to Book' tab.

    Parameters
    ----------
    city_map         : dict  — city name → integer id
    city_coords      : dict  — city name → (lat, lon)
    vehicle_features : dict  — vehicle type → list of feature names
    models           : dict  — vehicle type → loaded XGBRegressor
    metrics          : dict  — vehicle type → {"mae": ..., "accuracy": ...}
    """

    st.markdown("### 📊 Best Time to Book")
    st.markdown(
        "See how fares change **hour by hour** across all vehicles. "
        "Find the cheapest window before you book."
    )

    # ── Input form ────────────────────────────────────────────
    cities = sorted(city_map.keys())

    with st.form("best_time_form"):
        c1, c2 = st.columns(2)
        with c1:
            origin = st.selectbox(
                "📍 From", cities,
                index=cities.index("Colombo") if "Colombo" in cities else 0,
                key="bt_origin",
            )
        with c2:
            dest_options = [c for c in cities if c != origin]
            destination  = st.selectbox(
                "📍 To", dest_options,
                index=dest_options.index("Kandy") if "Kandy" in dest_options else 0,
                key="bt_dest",
            )

        c3, c4 = st.columns(2)
        with c3:
            trip_date = st.date_input("📅 Date", value=date.today(), key="bt_date")
        with c4:
            precip = st.number_input("🌧 Rainfall (mm)", 0.0, 200.0, 0.0, step=0.5, key="bt_rain")

        c5, c6 = st.columns(2)
        with c5:
            temp = st.number_input("🌡 Temperature (°C)", 15.0, 42.0, 29.0, step=0.5, key="bt_temp")
        with c6:
            hum  = st.number_input("💧 Humidity (%)", 30.0, 100.0, 75.0, step=1.0, key="bt_hum")

        is_holiday = st.checkbox("🎉 Public holiday", key="bt_hol")
        submitted  = st.form_submit_button("📈 Analyse Hourly Fares", use_container_width=True)

    if not submitted:
        st.info("Select your route and conditions above, then click **Analyse Hourly Fares**.")
        return

    if origin == destination:
        st.error("Origin and destination must be different cities.")
        return

    # Distance
    olat, olon = city_coords[origin]
    dlat, dlon = city_coords[destination]
    dist = _haversine_km(olat, olon, dlat, dlon)

    # Run predictions
    with st.spinner("Running 72 predictions (24h × 3 vehicles)…"):
        data = get_hourly_fares(
            models, vehicle_features, city_map,
            origin, destination, dist,
            precip, temp, hum, is_holiday, trip_date,
        )

    hours = data["hours"]
    st.divider()

    # ══════════════════════════════════════════
    # Section 1 — 24-hour fare chart
    # ══════════════════════════════════════════
    st.markdown("#### 📈 Hourly Fare Chart")
    st.caption(
        f"{origin} → {destination} · {dist:.1f} km · "
        f"{trip_date.strftime('%a %d %b %Y')}"
    )

    fig = go.Figure()

    for vtype in ["bike", "tuk_tuk", "car"]:
        m     = VMETA[vtype]
        fares = data[vtype]
        best  = data["cheapest"][vtype]

        # Line
        fig.add_trace(go.Scatter(
            x=hours, y=fares,
            mode="lines+markers",
            name=f"{m['icon']} {m['label']}",
            line=dict(color=m["color"], width=2.5),
            marker=dict(size=5),
            hovertemplate=(
                f"<b>{m['label']}</b><br>"
                "Hour: %{x:02d}:00<br>"
                "Fare: LKR %{y:,.0f}<extra></extra>"
            ),
        ))

        # Best-hour star marker
        fig.add_trace(go.Scatter(
            x=[best["hour"]], y=[best["fare"]],
            mode="markers",
            name=f"Best {m['label']}",
            marker=dict(color=m["color"], size=13, symbol="star",
                        line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertemplate=(
                f"<b>⭐ Cheapest {m['label']}</b><br>"
                f"{best['hour']:02d}:00 → LKR {best['fare']:,.0f}<extra></extra>"
            ),
        ))

    # Rush-hour shading
    for rh_start, rh_end, label in [(7, 9, "Rush AM"), (17, 19, "Rush PM")]:
        fig.add_vrect(
            x0=rh_start, x1=rh_end,
            fillcolor="rgba(255,180,0,0.08)",
            layer="below", line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(color="#fbbf24", size=10),
        )

    # Night shading
    for nx0, nx1 in [(0, 6), (22, 23)]:
        fig.add_vrect(
            x0=nx0, x1=nx1,
            fillcolor="rgba(100,80,200,0.07)",
            layer="below", line_width=0,
        )

    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="Hour of day",
            tickmode="array",
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            gridcolor="#1f2937",
            showline=False,
        ),
        yaxis=dict(title="Fare (LKR)", gridcolor="#1f2937", showline=False),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ = cheapest hour · 🟡 shading = rush hours · 🟣 shading = night hours")

    st.divider()

    # ══════════════════════════════════════════
    # Section 2 — Cheapest hour cards
    # ══════════════════════════════════════════
    st.markdown("#### 💡 Cheapest Hour Recommendations")

    cols = st.columns(3)
    for col, vtype in zip(cols, ["bike", "tuk_tuk", "car"]):
        m    = VMETA[vtype]
        best = data["cheapest"][vtype]
        mae  = metrics[vtype]["mae"]

        with col:
            st.markdown(f"""
            <div style="background:{m['bg']};border:1px solid {m['border']};
                        border-radius:14px;padding:1.1rem 1.2rem;text-align:center;">
              <div style="font-size:2rem;">{m['icon']}</div>
              <div style="font-size:0.95rem;font-weight:600;margin:4px 0;">{m['label']}</div>
              <div style="font-size:1.7rem;font-weight:700;color:{m['color']};">
                LKR {best['fare']:,.0f}
              </div>
              <div style="font-size:0.75rem;opacity:0.5;margin-bottom:8px;">
                Range: {max(250, best['fare'] - mae):,.0f} – {best['fare'] + mae:,.0f}
              </div>
              <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:0.5rem;">
                <div style="font-size:0.8rem;opacity:0.6;">Best hour</div>
                <div style="font-size:1.1rem;font-weight:600;">{best['hour']:02d}:00</div>
              </div>
              <div style="margin-top:8px;font-size:0.78rem;opacity:0.55;">
                Avoid {best['worst_hour']:02d}:00 (LKR {best['worst_fare']:,.0f})
              </div>
              <div style="margin-top:6px;font-size:0.82rem;color:{m['color']};">
                Save LKR {best['saving']:,.0f} ({best['saving_pct']}%)
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════
    # Section 3 — Compare vehicles at any hour
    # ══════════════════════════════════════════
    st.markdown("#### 🔍 Compare All Vehicles at a Specific Hour")

    compare_hour = st.slider(
        "Select hour to compare", 0, 23,
        value=data["cheapest"]["tuk_tuk"]["hour"],
        format="%d:00",
        key="bt_compare_hour",
    )

    fares_at_hour = {vt: data[vt][compare_hour] for vt in ["bike", "tuk_tuk", "car"]}
    cheapest_vt   = min(fares_at_hour, key=fares_at_hour.get)
    priciest_vt   = max(fares_at_hour, key=fares_at_hour.get)
    saving_across = fares_at_hour[priciest_vt] - fares_at_hour[cheapest_vt]

    # Contextual badges
    is_rush_h  = (7 <= compare_hour <= 9) or (17 <= compare_hour <= 19)
    is_night_h = compare_hour >= 22 or compare_hour <= 5
    badges = ""
    if is_rush_h:
        badges += ('<span style="background:#3a1a00;color:#fbbf24;border:1px solid #8a4400;'
                   'border-radius:20px;padding:2px 10px;font-size:0.72rem;margin-right:6px;">'
                   '⚡ Rush hour</span>')
    if is_night_h:
        badges += ('<span style="background:#1a0a3a;color:#c0a0ff;border:1px solid #5a3a9a;'
                   'border-radius:20px;padding:2px 10px;font-size:0.72rem;">'
                   '🌙 Night rate</span>')
    if badges:
        st.markdown(badges, unsafe_allow_html=True)

    c_cols = st.columns(3)
    for col, vtype in zip(c_cols, ["bike", "tuk_tuk", "car"]):
        m     = VMETA[vtype]
        fare  = fares_at_hour[vtype]
        mae   = metrics[vtype]["mae"]
        delta = fare - fares_at_hour[cheapest_vt]

        col.metric(
            label      = f"{m['icon']} {m['label']}" + (" ⭐" if vtype == cheapest_vt else ""),
            value      = f"LKR {fare:,.0f}",
            delta      = (f"+{delta:,.0f} vs cheapest" if delta > 0 else "Cheapest option"),
            delta_color= "inverse" if delta > 0 else "normal",
            help       = f"MAE ±LKR {mae:.0f} · Range: {max(250, fare - mae):,.0f}–{fare + mae:,.0f}",
        )

    if saving_across > 0:
        st.markdown(f"""
        <div style="margin-top:0.8rem;padding:0.85rem 1.1rem;background:#0a1f0a;
                    border:1px solid #1a4a1a;border-radius:12px;font-size:0.88rem;">
          💡 At <strong>{compare_hour:02d}:00</strong>,
          choosing <strong>{VMETA[cheapest_vt]['icon']} {VMETA[cheapest_vt]['label']}</strong>
          saves <strong>LKR {saving_across:,.0f}</strong>
          over {VMETA[priciest_vt]['icon']} {VMETA[priciest_vt]['label']}.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════
    # Section 4 — Savings summary table
    # ══════════════════════════════════════════
    st.markdown("#### 📋 Full Savings Summary")

    rows = []
    for vtype in ["bike", "tuk_tuk", "car"]:
        best = data["cheapest"][vtype]
        rows.append({
            "Vehicle":       f"{VMETA[vtype]['icon']} {VMETA[vtype]['label']}",
            "Best Hour":     f"{best['hour']:02d}:00",
            "Cheapest Fare": f"LKR {best['fare']:,.0f}",
            "Worst Hour":    f"{best['worst_hour']:02d}:00",
            "Peak Fare":     f"LKR {best['worst_fare']:,.0f}",
            "You Save":      f"LKR {best['saving']:,.0f}",
            "Saving %":      f"{best['saving_pct']}%",
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Vehicle"),
        use_container_width=True,
    )

    st.caption(
        f"Predictions for {origin} → {destination} · {dist:.1f} km · "
        f"{trip_date.strftime('%A %d %b %Y')} · "
        f"{precip:.1f}mm rain · {temp:.1f}°C · {hum:.0f}% humidity"
        + (" · 🎉 Holiday" if is_holiday else "")
    )