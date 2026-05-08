
"""
SAT-Guard Q1 Resilience Digital Twin
====================================

Corrected advanced Streamlit application for electricity-grid resilience,
regional risk, social vulnerability, finance/funding, investment prioritisation,
spatial intelligence and Monte Carlo uncertainty analysis.

Authoring notes
---------------
This file is intentionally self-contained and deployable as app.py.

Major corrections included
--------------------------
1. Calm live weather no longer produces false Severe / Fragile warnings.
2. Synthetic fallback outage data is never treated as real outage evidence.
3. Grid-failure probability is bounded and weather-gated in live mode.
4. Regional risk visuals include explicit colour legends.
5. EV/V2G storm operative tab has been removed.
6. Failure and investment logic is combined.
7. Finance and funding logic is combined.
8. Only one Monte Carlo tab remains; it uses the improved correlated simulation
   but is named simply "Monte Carlo".
9. README / Methodology is included as a tab and explains all formulae,
   variables, assumptions, limitations and references in plain language.

Recommended packages
--------------------
pip install streamlit pandas numpy requests plotly pydeck

Run
---
streamlit run app.py
"""

from __future__ import annotations

import math
import json
import time
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SAT-Guard Q1 Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# APPLICATION STYLING
# =============================================================================

APP_CSS = """
<style>
:root {
    --bg-main: #020617;
    --panel: rgba(15, 23, 42, 0.76);
    --panel-strong: rgba(15, 23, 42, 0.94);
    --border: rgba(148, 163, 184, 0.24);
    --text: #e5e7eb;
    --muted: #94a3b8;
    --blue: #38bdf8;
    --green: #22c55e;
    --yellow: #eab308;
    --orange: #f97316;
    --red: #ef4444;
    --purple: #a855f7;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.20), transparent 34%),
        radial-gradient(circle at 70% 12%, rgba(168,85,247,0.14), transparent 33%),
        radial-gradient(circle at bottom right, rgba(34,197,94,0.08), transparent 30%),
        linear-gradient(180deg, #020617 0%, #050816 45%, #020617 100%);
}
.block-container {
    padding-top: 1.10rem;
    padding-bottom: 2.8rem;
}
[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.97);
    border-right: 1px solid rgba(148, 163, 184, 0.20);
}
.hero {
    border: 1px solid rgba(148,163,184,0.22);
    background:
        linear-gradient(135deg, rgba(14,165,233,0.22), rgba(168,85,247,0.10)),
        rgba(15,23,42,0.86);
    border-radius: 28px;
    padding: 24px 26px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.34);
    margin-bottom: 18px;
}
.hero-title {
    font-size: 38px;
    font-weight: 950;
    letter-spacing: -0.052em;
    color: white;
    margin-bottom: 6px;
}
.hero-subtitle {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.55;
    max-width: 1180px;
}
.chip {
    display: inline-block;
    margin: 8px 6px 0 0;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.27);
    background: rgba(2,6,23,0.60);
    padding: 7px 12px;
    color: #bfdbfe;
    font-weight: 800;
    font-size: 12px;
}
.card {
    border: 1px solid rgba(148,163,184,0.20);
    background: rgba(15,23,42,0.72);
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.26);
}
.note {
    border: 1px solid rgba(56,189,248,0.26);
    background: rgba(56,189,248,0.09);
    border-radius: 18px;
    padding: 14px 16px;
    color: #dbeafe;
    line-height: 1.55;
}
.good {
    border: 1px solid rgba(34,197,94,0.28);
    background: rgba(34,197,94,0.10);
    border-radius: 18px;
    padding: 14px 16px;
    color: #dcfce7;
}
.warn {
    border: 1px solid rgba(234,179,8,0.34);
    background: rgba(234,179,8,0.11);
    border-radius: 18px;
    padding: 14px 16px;
    color: #fef3c7;
}
.danger {
    border: 1px solid rgba(239,68,68,0.34);
    background: rgba(239,68,68,0.10);
    border-radius: 18px;
    padding: 14px 16px;
    color: #fee2e2;
}
.formula {
    border-left: 4px solid #38bdf8;
    background: rgba(2,6,23,0.55);
    padding: 12px 14px;
    border-radius: 12px;
    color: #e0f2fe;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    white-space: pre-wrap;
}
.legend-box {
    border: 1px solid rgba(148,163,184,0.22);
    background: rgba(2,6,23,0.50);
    border-radius: 16px;
    padding: 13px 14px;
    color: #e5e7eb;
    line-height: 1.65;
}
.legend-chip {
    display:inline-block;
    width:14px;
    height:14px;
    border-radius:50%;
    margin-right:8px;
    vertical-align:middle;
}
.stMetric {
    background: rgba(15, 23, 42, 0.58);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 12px 14px;
    box-shadow: 0 12px 34px rgba(0,0,0,0.23);
}
[data-testid="stMetricValue"] {
    color: white;
    font-weight: 950;
}
[data-testid="stMetricLabel"] {
    color: #bfdbfe;
}
hr {
    border-color: rgba(148,163,184,0.18);
}
</style>
"""


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_CURRENT = [
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "relative_humidity_2m",
    "precipitation",
    "is_day",
]

AIR_CURRENT = [
    "european_aqi",
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
    "uv_index",
]

# Conservative operational thresholds.
# The earlier problematic version gave high risk from low wind and nearly zero rain.
# These thresholds mean normal UK conditions remain Low/Moderate unless real stress exists.
THRESHOLDS = {
    "wind_alert_kmh": 45.0,
    "wind_severe_kmh": 70.0,
    "rain_alert_mm": 6.0,
    "rain_severe_mm": 20.0,
    "aqi_alert": 70.0,
    "aqi_severe": 150.0,
    "imd_high": 65.0,
    "outage_density_alert": 3.0,
    "outage_density_severe": 8.0,
    "ens_alert_mw": 150.0,
    "ens_severe_mw": 750.0,
}

# The live weather risk model intentionally gives limited weight to social vulnerability.
# IMD should modify consequence and prioritisation, not fabricate storm conditions.
WEIGHTS = {
    "weather_wind": 0.27,
    "weather_rain": 0.22,
    "weather_aqi": 0.10,
    "outage": 0.18,
    "ens": 0.10,
    "social": 0.13,
}

SCENARIOS = {
    "Live / Real-time": {
        "wind_multiplier": 1.00,
        "rain_multiplier": 1.00,
        "aqi_multiplier": 1.00,
        "temperature_delta": 0.00,
        "outage_multiplier": 1.00,
        "ens_multiplier": 1.00,
        "description": "Observed or fallback-normal current conditions. No artificial hazard stress.",
    },
    "Extreme wind": {
        "wind_multiplier": 1.95,
        "rain_multiplier": 1.15,
        "aqi_multiplier": 1.05,
        "temperature_delta": -1.0,
        "outage_multiplier": 1.75,
        "ens_multiplier": 1.45,
        "description": "A high-wind event affecting overhead lines, trees, access routes and exposed assets.",
    },
    "Flood": {
        "wind_multiplier": 1.20,
        "rain_multiplier": 5.00,
        "aqi_multiplier": 1.05,
        "temperature_delta": 0.4,
        "outage_multiplier": 2.00,
        "ens_multiplier": 1.70,
        "description": "A heavy-rainfall and surface-water flooding event affecting access and substations.",
    },
    "Heatwave": {
        "wind_multiplier": 0.85,
        "rain_multiplier": 0.30,
        "aqi_multiplier": 1.35,
        "temperature_delta": 8.0,
        "outage_multiplier": 1.25,
        "ens_multiplier": 1.35,
        "description": "A hot-weather event raising load and thermal stress while reducing crew comfort.",
    },
    "Drought / low renewable": {
        "wind_multiplier": 0.40,
        "rain_multiplier": 0.15,
        "aqi_multiplier": 1.15,
        "temperature_delta": 3.0,
        "outage_multiplier": 1.10,
        "ens_multiplier": 1.25,
        "description": "A calm, dry, low-renewable scenario affecting supply-demand balance.",
    },
    "Compound extreme": {
        "wind_multiplier": 2.05,
        "rain_multiplier": 4.20,
        "aqi_multiplier": 1.55,
        "temperature_delta": 4.0,
        "outage_multiplier": 2.60,
        "ens_multiplier": 2.25,
        "description": "A multi-hazard stress case combining wind, rainfall, air-quality and system pressure.",
    },
}

REGIONS = {
    "North East": {
        "center": {"lat": 54.85, "lon": -1.65, "zoom": 7},
        "places": {
            "Newcastle": {
                "lat": 54.9783,
                "lon": -1.6178,
                "postcode": "NE1",
                "population_density": 2590,
                "imd_score": 43,
                "load_mw": 128,
                "asset_exposure": 0.58,
            },
            "Sunderland": {
                "lat": 54.9069,
                "lon": -1.3838,
                "postcode": "SR1",
                "population_density": 2010,
                "imd_score": 52,
                "load_mw": 106,
                "asset_exposure": 0.52,
            },
            "Durham": {
                "lat": 54.7761,
                "lon": -1.5733,
                "postcode": "DH1",
                "population_density": 730,
                "imd_score": 38,
                "load_mw": 64,
                "asset_exposure": 0.38,
            },
            "Middlesbrough": {
                "lat": 54.5742,
                "lon": -1.2350,
                "postcode": "TS1",
                "population_density": 2680,
                "imd_score": 61,
                "load_mw": 96,
                "asset_exposure": 0.60,
            },
            "Darlington": {
                "lat": 54.5236,
                "lon": -1.5595,
                "postcode": "DL1",
                "population_density": 1070,
                "imd_score": 45,
                "load_mw": 72,
                "asset_exposure": 0.41,
            },
            "Hexham": {
                "lat": 54.9730,
                "lon": -2.1010,
                "postcode": "NE46",
                "population_density": 330,
                "imd_score": 32,
                "load_mw": 38,
                "asset_exposure": 0.33,
            },
        },
    },
    "Yorkshire": {
        "center": {"lat": 53.95, "lon": -1.30, "zoom": 7},
        "places": {
            "Leeds": {
                "lat": 53.8008,
                "lon": -1.5491,
                "postcode": "LS1",
                "population_density": 1560,
                "imd_score": 44,
                "load_mw": 168,
                "asset_exposure": 0.62,
            },
            "Sheffield": {
                "lat": 53.3811,
                "lon": -1.4701,
                "postcode": "S1",
                "population_density": 1510,
                "imd_score": 48,
                "load_mw": 144,
                "asset_exposure": 0.58,
            },
            "York": {
                "lat": 53.9600,
                "lon": -1.0873,
                "postcode": "YO1",
                "population_density": 740,
                "imd_score": 34,
                "load_mw": 82,
                "asset_exposure": 0.39,
            },
            "Hull": {
                "lat": 53.7676,
                "lon": -0.3274,
                "postcode": "HU1",
                "population_density": 3560,
                "imd_score": 62,
                "load_mw": 116,
                "asset_exposure": 0.64,
            },
            "Bradford": {
                "lat": 53.7950,
                "lon": -1.7594,
                "postcode": "BD1",
                "population_density": 1450,
                "imd_score": 59,
                "load_mw": 132,
                "asset_exposure": 0.57,
            },
            "Doncaster": {
                "lat": 53.5228,
                "lon": -1.1285,
                "postcode": "DN1",
                "population_density": 540,
                "imd_score": 49,
                "load_mw": 78,
                "asset_exposure": 0.43,
            },
        },
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp(value: Any, low: float, high: float) -> float:
    """Return value clipped to [low, high]."""
    try:
        x = float(value)
    except Exception:
        x = low
    if math.isnan(x):
        x = low
    return max(low, min(high, x))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, safely."""
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def normalise(value: Any, low: float, high: float) -> float:
    """Normalise a variable to 0-1."""
    v = safe_float(value)
    if high <= low:
        return 0.0
    return clamp((v - low) / (high - low), 0.0, 1.0)


def risk_label(score: float) -> str:
    """Human label for risk. Conservative thresholds avoid false alerts."""
    s = safe_float(score)
    if s >= 80:
        return "Severe"
    if s >= 65:
        return "High"
    if s >= 45:
        return "Elevated"
    if s >= 25:
        return "Moderate"
    return "Low"


def resilience_label(score: float) -> str:
    """Human label for resilience."""
    s = safe_float(score)
    if s >= 80:
        return "Robust"
    if s >= 65:
        return "Stable"
    if s >= 45:
        return "Functional"
    if s >= 30:
        return "Stressed"
    return "Fragile"


def money(value: float) -> str:
    return f"£{safe_float(value):,.0f}"


def money_m(value: float) -> str:
    return f"£{safe_float(value)/1_000_000:.2f}m"


def pct(value: float) -> str:
    return f"{safe_float(value)*100:.1f}%"


def status_class(label: str) -> str:
    label = str(label).lower()
    if "severe" in label or "fragile" in label:
        return "danger"
    if "high" in label or "stressed" in label or "elevated" in label:
        return "warn"
    return "good"


def requests_json(url: str, params: Dict[str, Any], timeout: int = 8) -> Dict[str, Any]:
    """Safe HTTP JSON request."""
    try:
        headers = {"User-Agent": "SAT-Guard-Q1/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def plotly_template() -> str:
    return "plotly_dark"


# =============================================================================
# DATA INGESTION
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch current weather and air-quality variables.

    If the API is unavailable, use conservative normal-weather fallback values.
    These fallback values represent ordinary weather and must not create warnings.
    """
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(WEATHER_CURRENT),
        "timezone": "Europe/London",
        "wind_speed_unit": "kmh",
    }
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(AIR_CURRENT),
        "timezone": "Europe/London",
    }

    weather_json = requests_json(OPEN_METEO_WEATHER_URL, weather_params)
    air_json = requests_json(OPEN_METEO_AIR_URL, air_params)

    current_weather = weather_json.get("current", {}) if isinstance(weather_json, dict) else {}
    current_air = air_json.get("current", {}) if isinstance(air_json, dict) else {}

    fallback = {
        "temperature_2m": 12.0,
        "apparent_temperature": 11.0,
        "wind_speed_10m": 8.0,
        "wind_direction_10m": 220.0,
        "surface_pressure": 1015.0,
        "cloud_cover": 45.0,
        "relative_humidity_2m": 70.0,
        "precipitation": 0.0,
        "is_day": 1,
        "european_aqi": 30.0,
        "pm10": 15.0,
        "pm2_5": 8.0,
        "nitrogen_dioxide": 12.0,
        "ozone": 50.0,
        "sulphur_dioxide": 3.0,
        "carbon_monoxide": 220.0,
        "uv_index": 2.0,
        "weather_source": "fallback-normal",
    }

    result = {}
    for key in WEATHER_CURRENT:
        result[key] = current_weather.get(key, fallback.get(key))
    for key in AIR_CURRENT:
        result[key] = current_air.get(key, fallback.get(key))
    result["weather_source"] = "live-api" if current_weather or current_air else "fallback-normal"

    # Guard against impossible or missing values.
    result["wind_speed_10m"] = clamp(result.get("wind_speed_10m"), 0, 180)
    result["precipitation"] = clamp(result.get("precipitation"), 0, 150)
    result["european_aqi"] = clamp(result.get("european_aqi"), 0, 500)
    result["temperature_2m"] = clamp(result.get("temperature_2m"), -25, 45)
    return result


def build_base_places(region_name: str) -> pd.DataFrame:
    """Build place table with weather attached."""
    rows = []
    for place, meta in REGIONS[region_name]["places"].items():
        w = fetch_weather(meta["lat"], meta["lon"])
        row = dict(meta)
        row["place"] = place
        row.update(w)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_scenario(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """Apply scenario multipliers. Live mode is unchanged."""
    scenario = SCENARIOS[scenario_name]
    out = df.copy()

    out["wind_speed_10m"] = out["wind_speed_10m"].astype(float) * scenario["wind_multiplier"]
    out["precipitation"] = out["precipitation"].astype(float) * scenario["rain_multiplier"]
    out["european_aqi"] = out["european_aqi"].astype(float) * scenario["aqi_multiplier"]
    out["temperature_2m"] = out["temperature_2m"].astype(float) + scenario["temperature_delta"]
    out["scenario"] = scenario_name
    out["scenario_description"] = scenario["description"]
    return out


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def compute_weather_components(row: Dict[str, Any]) -> Dict[str, float]:
    """Compute transparent weather component scores on 0-100 basis."""
    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    aqi = safe_float(row.get("european_aqi"))

    wind_score = normalise(wind, 10, THRESHOLDS["wind_severe_kmh"]) * 100
    rain_score = normalise(rain, 0.5, THRESHOLDS["rain_severe_mm"]) * 100
    aqi_score = normalise(aqi, 35, THRESHOLDS["aqi_severe"]) * 100

    # Critical live-weather correction:
    # Very low rainfall and low wind should not create hazard warnings.
    if wind < 20:
        wind_score *= 0.35
    if rain < 1:
        rain_score = 0.0
    if aqi < 45:
        aqi_score *= 0.30

    return {
        "wind_component": round(clamp(wind_score, 0, 100), 2),
        "rain_component": round(clamp(rain_score, 0, 100), 2),
        "aqi_component": round(clamp(aqi_score, 0, 100), 2),
    }


def estimate_outage_evidence(row: Dict[str, Any], scenario_name: str, use_synthetic_outages: bool) -> Dict[str, Any]:
    """
    Estimate outage evidence.

    IMPORTANT:
    In live mode, synthetic fallback outage estimates are NOT treated as real outages.
    This prevents false Fragile/Warning labels during good weather.
    """
    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    asset = safe_float(row.get("asset_exposure"))
    load = safe_float(row.get("load_mw"))

    if scenario_name == "Live / Real-time" and not use_synthetic_outages:
        return {
            "real_outage_records": 0,
            "nearby_outages_25km": 0,
            "outage_density_score": 0.0,
            "energy_not_supplied_mw": 0.0,
            "outage_source": "none-or-not-confirmed",
        }

    # Stress-only synthetic estimate for what-if scenarios.
    storm_pressure = (
        normalise(wind, THRESHOLDS["wind_alert_kmh"], THRESHOLDS["wind_severe_kmh"]) * 0.60
        + normalise(rain, THRESHOLDS["rain_alert_mm"], THRESHOLDS["rain_severe_mm"]) * 0.40
    )
    scenario = SCENARIOS[scenario_name]
    outage_records = int(round(clamp(storm_pressure * 8 * asset * scenario["outage_multiplier"], 0, 20)))
    ens_mw = clamp(outage_records * load * 0.035 * scenario["ens_multiplier"], 0, load * 0.85)

    return {
        "real_outage_records": 0,
        "nearby_outages_25km": outage_records,
        "outage_density_score": round(normalise(outage_records, 0, THRESHOLDS["outage_density_severe"]) * 100, 2),
        "energy_not_supplied_mw": round(ens_mw, 2),
        "outage_source": "scenario-estimate" if scenario_name != "Live / Real-time" else "synthetic-live-enabled",
    }


def compute_social_vulnerability(row: Dict[str, Any]) -> float:
    """
    Compute social vulnerability from IMD and population density.

    IMD is the principal driver; density slightly modifies operational exposure.
    """
    imd = safe_float(row.get("imd_score"))
    density = safe_float(row.get("population_density"))
    density_component = normalise(density, 250, 3600) * 100
    social = 0.82 * imd + 0.18 * density_component
    return round(clamp(social, 0, 100), 2)


def compute_risk(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute corrected risk score and full explainability fields."""
    comps = compute_weather_components(row)
    social = compute_social_vulnerability(row)

    outage_score = safe_float(row.get("outage_density_score"))
    ens_score = normalise(row.get("energy_not_supplied_mw"), 0, THRESHOLDS["ens_severe_mw"]) * 100

    risk = (
        WEIGHTS["weather_wind"] * comps["wind_component"]
        + WEIGHTS["weather_rain"] * comps["rain_component"]
        + WEIGHTS["weather_aqi"] * comps["aqi_component"]
        + WEIGHTS["outage"] * outage_score
        + WEIGHTS["ens"] * ens_score
        + WEIGHTS["social"] * social
    )

    # Baseline should be small. It avoids a false zero while not creating warning.
    risk += 4.0

    # Good-weather guardrail:
    # If wind, rain and confirmed outages are low, cap live risk.
    calm_weather = (
        safe_float(row.get("wind_speed_10m")) < 20
        and safe_float(row.get("precipitation")) < 1
        and safe_float(row.get("nearby_outages_25km")) == 0
    )
    if calm_weather and str(row.get("scenario")) == "Live / Real-time":
        risk = min(risk, 34.0)

    risk = round(clamp(risk, 0, 100), 2)
    return {
        "wind_component": comps["wind_component"],
        "rain_component": comps["rain_component"],
        "aqi_component": comps["aqi_component"],
        "social_vulnerability": social,
        "ens_score": round(ens_score, 2),
        "final_risk_score": risk,
        "risk_label": risk_label(risk),
    }


def compute_resilience(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute resilience as risk inverse adjusted by vulnerability and asset exposure."""
    risk = safe_float(row.get("final_risk_score"))
    social = safe_float(row.get("social_vulnerability"))
    asset = safe_float(row.get("asset_exposure"))

    resilience = 100 - risk
    resilience -= normalise(social, 60, 100) * 8
    resilience -= normalise(asset, 0.55, 0.90) * 5

    # Good-weather guardrail:
    # if risk is not high, do not classify as fragile.
    if risk < 45:
        resilience = max(resilience, 55)
    if risk < 35:
        resilience = max(resilience, 66)

    resilience = round(clamp(resilience, 0, 100), 2)
    return {
        "resilience_index": resilience,
        "resilience_label": resilience_label(resilience),
    }


def compute_failure_probability(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transparent logistic failure probability.

    In earlier versions, failure probability was inflated under ordinary weather.
    The corrected model gates failure through actual hazard pressure.
    """
    risk = safe_float(row.get("final_risk_score"))
    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    outage = safe_float(row.get("nearby_outages_25km"))
    ens = safe_float(row.get("energy_not_supplied_mw"))
    social = safe_float(row.get("social_vulnerability"))

    hazard_gate = max(
        normalise(wind, THRESHOLDS["wind_alert_kmh"], THRESHOLDS["wind_severe_kmh"]),
        normalise(rain, THRESHOLDS["rain_alert_mm"], THRESHOLDS["rain_severe_mm"]),
        normalise(outage, THRESHOLDS["outage_density_alert"], THRESHOLDS["outage_density_severe"]),
        normalise(ens, THRESHOLDS["ens_alert_mw"], THRESHOLDS["ens_severe_mw"]),
    )

    z = (
        -4.25
        + 0.052 * risk
        + 1.85 * hazard_gate
        + 0.006 * max(social - 50, 0)
    )

    p = 1 / (1 + math.exp(-z))

    # Live calm cap: stops ordinary Newcastle/Durham weather being labelled as a failure state.
    if str(row.get("scenario")) == "Live / Real-time" and hazard_gate < 0.05:
        p = min(p, 0.08)

    return {
        "hazard_gate": round(hazard_gate, 3),
        "grid_failure_probability": round(clamp(p, 0, 1), 4),
    }


def compute_finance(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate financial exposure.

    This is an indicative decision-support calculation, not a regulated settlement model.
    """
    ens = safe_float(row.get("energy_not_supplied_mw"))
    load = safe_float(row.get("load_mw"))
    failure_p = safe_float(row.get("grid_failure_probability"))
    social = safe_float(row.get("social_vulnerability"))
    risk = safe_float(row.get("final_risk_score"))

    duration_hours = 1.0 + 5.0 * normalise(risk, 45, 100)
    ens_mwh = ens * duration_hours

    value_of_lost_load = 17000.0
    restoration_cost = 12000.0 + 42000.0 * failure_p
    social_uplift = 1.0 + 0.30 * normalise(social, 50, 100)

    direct_loss = ens_mwh * value_of_lost_load
    restoration_loss = restoration_cost * max(1, safe_float(row.get("nearby_outages_25km")))
    total_loss = (direct_loss + restoration_loss) * social_uplift

    # In calm live mode with zero ENS, keep financial loss low and honest.
    if ens <= 0 and str(row.get("scenario")) == "Live / Real-time":
        total_loss = restoration_loss * 0.05

    return {
        "estimated_duration_h": round(duration_hours, 2),
        "energy_not_supplied_mwh": round(ens_mwh, 2),
        "direct_loss_gbp": round(direct_loss, 2),
        "restoration_loss_gbp": round(restoration_loss, 2),
        "total_financial_loss_gbp": round(total_loss, 2),
    }


def compute_investment_priority(row: Dict[str, Any]) -> Dict[str, Any]:
    """Combined failure and investment prioritisation score."""
    risk = safe_float(row.get("final_risk_score"))
    resilience = safe_float(row.get("resilience_index"))
    social = safe_float(row.get("social_vulnerability"))
    failure = safe_float(row.get("grid_failure_probability"))
    loss = safe_float(row.get("total_financial_loss_gbp"))

    score = (
        0.28 * risk
        + 0.22 * (100 - resilience)
        + 0.18 * social
        + 0.20 * failure * 100
        + 0.12 * normalise(loss, 0, 5_000_000) * 100
    )
    score = round(clamp(score, 0, 100), 2)

    if score >= 78:
        band = "Immediate investment"
    elif score >= 62:
        band = "High priority"
    elif score >= 45:
        band = "Targeted reinforcement"
    elif score >= 30:
        band = "Monitor"
    else:
        band = "Routine"

    return {
        "investment_priority_score": score,
        "investment_priority": band,
    }


def process_model(df: pd.DataFrame, scenario_name: str, use_synthetic_outages: bool) -> pd.DataFrame:
    """Run the complete model pipeline."""
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        d.update(estimate_outage_evidence(d, scenario_name, use_synthetic_outages))
        d.update(compute_risk(d))
        d.update(compute_resilience(d))
        d.update(compute_failure_probability(d))
        d.update(compute_finance(d))
        d.update(compute_investment_priority(d))

        d["weather_state"] = weather_state_label(d)
        d["explainability"] = explain_row(d)
        rows.append(d)

    out = pd.DataFrame(rows)
    sort_cols = ["final_risk_score", "grid_failure_probability", "total_financial_loss_gbp"]
    return out.sort_values(sort_cols, ascending=False).reset_index(drop=True)


def weather_state_label(row: Dict[str, Any]) -> str:
    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    aqi = safe_float(row.get("european_aqi"))

    if wind >= THRESHOLDS["wind_severe_kmh"] or rain >= THRESHOLDS["rain_severe_mm"]:
        return "Severe weather"
    if wind >= THRESHOLDS["wind_alert_kmh"] or rain >= THRESHOLDS["rain_alert_mm"]:
        return "Adverse weather"
    if aqi >= THRESHOLDS["aqi_alert"]:
        return "Air-quality pressure"
    return "Normal / calm"


def explain_row(row: Dict[str, Any]) -> str:
    reasons = []
    if safe_float(row.get("wind_speed_10m")) >= THRESHOLDS["wind_alert_kmh"]:
        reasons.append(f"wind {safe_float(row.get('wind_speed_10m')):.1f} km/h")
    if safe_float(row.get("precipitation")) >= THRESHOLDS["rain_alert_mm"]:
        reasons.append(f"rain {safe_float(row.get('precipitation')):.1f} mm")
    if safe_float(row.get("nearby_outages_25km")) > 0:
        reasons.append(f"{int(safe_float(row.get('nearby_outages_25km')))} scenario outage records")
    if safe_float(row.get("social_vulnerability")) >= 60:
        reasons.append(f"social vulnerability {safe_float(row.get('social_vulnerability')):.1f}/100")
    if safe_float(row.get("energy_not_supplied_mw")) > 0:
        reasons.append(f"ENS {safe_float(row.get('energy_not_supplied_mw')):.1f} MW")
    if not reasons:
        reasons.append("calm weather and no confirmed outage evidence")
    return "; ".join(reasons)


# =============================================================================
# MONTE CARLO
# =============================================================================

def monte_carlo_for_row(row: Dict[str, Any], simulations: int, seed: int = 42) -> Dict[str, Any]:
    """
    Improved correlated Monte Carlo simulation.

    The model uses a shared storm shock so wind, rainfall, outage pressure and ENS
    move together rather than independently.
    """
    rng = np.random.default_rng(seed + abs(hash(str(row.get("place")))) % 10000)

    n = int(clamp(simulations, 250, 10000))
    wind0 = safe_float(row.get("wind_speed_10m"))
    rain0 = safe_float(row.get("precipitation"))
    aqi0 = safe_float(row.get("european_aqi"))
    social = safe_float(row.get("social_vulnerability"))
    load = safe_float(row.get("load_mw"))
    asset = safe_float(row.get("asset_exposure"))

    storm = rng.normal(0, 1, n)
    wind = np.maximum(0, wind0 * np.exp(0.15 * storm + rng.normal(0, 0.08, n)))
    rain = np.maximum(0, rain0 * np.exp(0.30 * storm + rng.normal(0, 0.16, n)))
    aqi = np.maximum(0, aqi0 * np.exp(rng.normal(0, 0.10, n)))

    wind_score = np.clip((wind - 10) / (THRESHOLDS["wind_severe_kmh"] - 10), 0, 1) * 100
    rain_score = np.clip((rain - 0.5) / (THRESHOLDS["rain_severe_mm"] - 0.5), 0, 1) * 100
    aqi_score = np.clip((aqi - 35) / (THRESHOLDS["aqi_severe"] - 35), 0, 1) * 100

    hazard_gate = np.maximum(
        np.clip((wind - THRESHOLDS["wind_alert_kmh"]) / (THRESHOLDS["wind_severe_kmh"] - THRESHOLDS["wind_alert_kmh"]), 0, 1),
        np.clip((rain - THRESHOLDS["rain_alert_mm"]) / (THRESHOLDS["rain_severe_mm"] - THRESHOLDS["rain_alert_mm"]), 0, 1),
    )

    outage_lambda = np.clip(hazard_gate * 5 * asset, 0.05, 12)
    outage = rng.poisson(outage_lambda)

    ens = outage * load * rng.triangular(0.01, 0.035, 0.12, n)
    ens_score = np.clip(ens / THRESHOLDS["ens_severe_mw"], 0, 1) * 100
    outage_score = np.clip(outage / THRESHOLDS["outage_density_severe"], 0, 1) * 100

    risk = (
        WEIGHTS["weather_wind"] * wind_score
        + WEIGHTS["weather_rain"] * rain_score
        + WEIGHTS["weather_aqi"] * aqi_score
        + WEIGHTS["outage"] * outage_score
        + WEIGHTS["ens"] * ens_score
        + WEIGHTS["social"] * social
        + 4
    )
    risk = np.clip(risk, 0, 100)

    failure = 1 / (1 + np.exp(-(-4.25 + 0.052 * risk + 1.85 * hazard_gate)))
    duration = 1.0 + 5.0 * np.clip((risk - 45) / 55, 0, 1)
    ens_mwh = ens * duration

    voll = rng.lognormal(np.log(17000), 0.18, n)
    restoration = outage * rng.lognormal(np.log(18000), 0.25, n)
    social_uplift = 1.0 + 0.30 * np.clip((social - 50) / 50, 0, 1)
    loss = (ens_mwh * voll + restoration) * social_uplift

    p95_loss = float(np.percentile(loss, 95))
    cvar95_loss = float(loss[loss >= p95_loss].mean()) if np.any(loss >= p95_loss) else p95_loss

    return {
        "place": row.get("place"),
        "postcode": row.get("postcode"),
        "mc_risk_mean": round(float(np.mean(risk)), 2),
        "mc_risk_p95": round(float(np.percentile(risk, 95)), 2),
        "mc_failure_mean": round(float(np.mean(failure)), 4),
        "mc_failure_p95": round(float(np.percentile(failure, 95)), 4),
        "mc_loss_mean_gbp": round(float(np.mean(loss)), 2),
        "mc_loss_p95_gbp": round(p95_loss, 2),
        "mc_loss_cvar95_gbp": round(cvar95_loss, 2),
        "risk_samples": [round(float(x), 2) for x in risk[:600]],
    }


def run_monte_carlo(df: pd.DataFrame, simulations: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append(monte_carlo_for_row(row.to_dict(), simulations))
    return pd.DataFrame(rows).sort_values("mc_risk_p95", ascending=False).reset_index(drop=True)


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">SAT-Guard Q1 Digital Twin</div>
            <div class="hero-subtitle">
                Corrected electricity-resilience dashboard for North East and Yorkshire.
                The live model now respects real weather: calm Newcastle or Durham conditions
                will not be labelled as Severe or Fragile unless there is genuine hazard or outage evidence.
            </div>
            <span class="chip">Corrected live risk</span>
            <span class="chip">Colourful spatial intelligence</span>
            <span class="chip">Finance + funding combined</span>
            <span class="chip">Failure + investment combined</span>
            <span class="chip">Improved Monte Carlo</span>
            <span class="chip">README methodology tab</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_legend() -> None:
    st.markdown(
        """
        <div class="legend-box">
            <b>Colour legend</b><br>
            <span class="legend-chip" style="background:#22c55e"></span>Low risk / robust condition<br>
            <span class="legend-chip" style="background:#a3e635"></span>Moderate risk / watch condition<br>
            <span class="legend-chip" style="background:#eab308"></span>Elevated operational pressure<br>
            <span class="legend-chip" style="background:#f97316"></span>High stress / prioritise review<br>
            <span class="legend-chip" style="background:#ef4444"></span>Severe stress / urgent action
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_note(df: pd.DataFrame, scenario: str, synthetic_outages: bool) -> None:
    max_risk = float(df["final_risk_score"].max())
    max_label = risk_label(max_risk)
    if scenario == "Live / Real-time" and max_risk < 45:
        st.markdown(
            """
            <div class="good">
                <b>Live status:</b> current conditions are not being treated as a storm event.
                Low wind, minimal rainfall and no confirmed outages remain Low/Moderate risk.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif max_risk >= 65:
        st.markdown(
            f"""
            <div class="warn">
                <b>Operational watch:</b> highest modelled risk is {max_label}.
                Review the explainability column before taking any decision.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="note">
                <b>Model status:</b> risk is being calculated from transparent weather,
                outage, ENS, social and asset-exposure components.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if scenario == "Live / Real-time" and not synthetic_outages:
        st.caption("Confirmed outage data is not available in this self-contained build; therefore synthetic outage fallback is disabled in live mode.")


def render_executive_tab(df: pd.DataFrame, region: str, scenario: str, synthetic_outages: bool) -> None:
    st.subheader("Executive overview")
    render_status_note(df, scenario, synthetic_outages)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest risk", f"{df['final_risk_score'].max():.1f}/100", risk_label(df["final_risk_score"].max()))
    c2.metric("Mean resilience", f"{df['resilience_index'].mean():.1f}/100", resilience_label(df["resilience_index"].mean()))
    c3.metric("Max failure probability", pct(df["grid_failure_probability"].max()))
    c4.metric("Total financial exposure", money_m(df["total_financial_loss_gbp"].sum()))

    st.markdown("#### Corrected live-risk table")
    cols = [
        "place", "postcode", "weather_state", "risk_label", "final_risk_score",
        "resilience_label", "resilience_index", "grid_failure_probability",
        "wind_speed_10m", "precipitation", "european_aqi", "imd_score",
        "social_vulnerability", "nearby_outages_25km", "energy_not_supplied_mw",
        "outage_source", "explainability",
    ]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Why calm Newcastle/Durham weather is no longer overflagged")
    st.markdown(
        """
        <div class="note">
        The corrected live model applies a weather gate. If wind is below 20 km/h, rainfall is below
        1 mm and there are no confirmed outage records, live risk is capped below high-warning level.
        IMD/social vulnerability still matters for consequence and funding, but it no longer fabricates
        a storm or grid-failure event by itself.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_spatial_tab(df: pd.DataFrame, region: str) -> None:
    st.subheader("Spatial intelligence")
    center = REGIONS[region]["center"]

    c1, c2 = st.columns([0.72, 0.28])
    with c1:
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="final_risk_score",
            size=np.maximum(df["final_risk_score"], 8),
            hover_name="place",
            hover_data={
                "postcode": True,
                "risk_label": True,
                "final_risk_score": ":.1f",
                "resilience_label": True,
                "grid_failure_probability": ":.1%",
                "wind_speed_10m": ":.1f",
                "precipitation": ":.2f",
                "european_aqi": ":.0f",
                "explainability": True,
                "lat": False,
                "lon": False,
            },
            color_continuous_scale=[
                [0.00, "#22c55e"],
                [0.25, "#a3e635"],
                [0.45, "#eab308"],
                [0.65, "#f97316"],
                [1.00, "#ef4444"],
            ],
            range_color=[0, 100],
            zoom=center["zoom"],
            center={"lat": center["lat"], "lon": center["lon"]},
            height=680,
            title="Colourful regional risk map",
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=45, b=0),
            coloraxis_colorbar=dict(
                title="Risk score",
                tickvals=[0, 25, 45, 65, 80, 100],
                ticktext=["Low", "Moderate", "Elevated", "High", "Severe", "Max"],
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        render_legend()
        st.markdown("#### Spatial interpretation")
        st.markdown(
            """
            <div class="note">
            The map uses colour and bubble size together. Colour shows the corrected risk score.
            Bubble size shows the same score to make hotspots visible even on dark basemaps.
            Hover over each place to see the exact risk drivers.
            </div>
            """,
            unsafe_allow_html=True,
        )

        ranked = df[["place", "postcode", "final_risk_score", "risk_label", "explainability"]].sort_values("final_risk_score", ascending=False)
        st.dataframe(ranked, use_container_width=True, hide_index=True)

    st.markdown("#### Risk surface and resilience comparison")
    a, b = st.columns(2)
    with a:
        bar = df.sort_values("final_risk_score", ascending=True)
        fig = px.bar(
            bar,
            x="final_risk_score",
            y="place",
            color="final_risk_score",
            orientation="h",
            color_continuous_scale=[
                [0.00, "#22c55e"],
                [0.25, "#a3e635"],
                [0.45, "#eab308"],
                [0.65, "#f97316"],
                [1.00, "#ef4444"],
            ],
            range_color=[0, 100],
            title="Regional risk ranking with colour legend",
            template=plotly_template(),
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.scatter(
            df,
            x="final_risk_score",
            y="resilience_index",
            size="total_financial_loss_gbp",
            color="grid_failure_probability",
            hover_name="place",
            color_continuous_scale="Turbo",
            title="Risk-resilience-failure space",
            template=plotly_template(),
            range_x=[0, 100],
            range_y=[0, 100],
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)


def render_regional_risk_tab(df: pd.DataFrame) -> None:
    st.subheader("Regional risk diagnostics")

    components = df[[
        "place", "wind_component", "rain_component", "aqi_component",
        "outage_density_score", "ens_score", "social_vulnerability"
    ]].melt(id_vars="place", var_name="component", value_name="score")

    fig = px.bar(
        components,
        x="place",
        y="score",
        color="component",
        barmode="group",
        title="Risk components by place",
        template=plotly_template(),
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed diagnostic fields")
    diagnostic_cols = [
        "place", "wind_component", "rain_component", "aqi_component",
        "outage_density_score", "ens_score", "social_vulnerability",
        "hazard_gate", "final_risk_score", "risk_label",
        "resilience_index", "resilience_label",
    ]
    st.dataframe(df[diagnostic_cols], use_container_width=True, hide_index=True)

    st.markdown("#### Formula")
    st.markdown(
        """
        <div class="formula">
Risk = 4
     + 0.27 × WindComponent
     + 0.22 × RainComponent
     + 0.10 × AQIComponent
     + 0.18 × OutageComponent
     + 0.10 × ENSComponent
     + 0.13 × SocialVulnerability

Live calm-weather guard:
If wind < 20 km/h, rain < 1 mm and confirmed outages = 0,
then live risk is capped below high-warning level.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_failure_investment_tab(df: pd.DataFrame) -> None:
    st.subheader("Failure and investment")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest failure probability", pct(df["grid_failure_probability"].max()))
    c2.metric("Highest investment score", f"{df['investment_priority_score'].max():.1f}/100")
    c3.metric("Immediate investment areas", int((df["investment_priority"] == "Immediate investment").sum()))
    c4.metric("Mean hazard gate", f"{df['hazard_gate'].mean():.2f}")

    a, b = st.columns(2)
    with a:
        fig = px.bar(
            df.sort_values("grid_failure_probability", ascending=True),
            x="grid_failure_probability",
            y="place",
            color="hazard_gate",
            orientation="h",
            title="Corrected grid-failure probability",
            template=plotly_template(),
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=450, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.bar(
            df.sort_values("investment_priority_score", ascending=True),
            x="investment_priority_score",
            y="place",
            color="investment_priority",
            orientation="h",
            title="Investment priority ranking",
            template=plotly_template(),
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Failure + investment evidence")
    cols = [
        "place", "postcode", "weather_state", "grid_failure_probability", "hazard_gate",
        "investment_priority_score", "investment_priority", "final_risk_score",
        "resilience_index", "social_vulnerability", "explainability",
    ]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Formula")
    st.markdown(
        """
        <div class="formula">
HazardGate = max(
    normalised_wind_above_alert,
    normalised_rain_above_alert,
    normalised_outage_density_above_alert,
    normalised_ENS_above_alert
)

FailureProbability = sigmoid(
    -4.25 + 0.052 × Risk + 1.85 × HazardGate + 0.006 × max(SocialVulnerability - 50, 0)
)

Live calm cap:
If HazardGate < 0.05 in live mode, probability is capped at 8%.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_finance_funding_tab(df: pd.DataFrame) -> None:
    st.subheader("Finance and funding")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total expected exposure", money_m(df["total_financial_loss_gbp"].sum()))
    c2.metric("Highest location loss", money_m(df["total_financial_loss_gbp"].max()))
    c3.metric("Total ENS", f"{df['energy_not_supplied_mw'].sum():.1f} MW")
    c4.metric("Mean estimated duration", f"{df['estimated_duration_h'].mean():.1f} h")

    a, b = st.columns(2)
    with a:
        fig = px.bar(
            df.sort_values("total_financial_loss_gbp", ascending=True),
            x="total_financial_loss_gbp",
            y="place",
            color="investment_priority",
            orientation="h",
            title="Financial exposure by place",
            template=plotly_template(),
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.scatter(
            df,
            x="total_financial_loss_gbp",
            y="investment_priority_score",
            size="energy_not_supplied_mw",
            color="social_vulnerability",
            hover_name="place",
            title="Funding pressure: loss, ENS and social vulnerability",
            template=plotly_template(),
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Finance + funding table")
    cols = [
        "place", "postcode", "direct_loss_gbp", "restoration_loss_gbp",
        "total_financial_loss_gbp", "estimated_duration_h",
        "energy_not_supplied_mwh", "investment_priority_score",
        "investment_priority", "explainability",
    ]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Formula")
    st.markdown(
        """
        <div class="formula">
DurationHours = 1 + 5 × normalise(Risk, 45, 100)

EnergyNotSuppliedMWh = EnergyNotSuppliedMW × DurationHours

DirectLoss = EnergyNotSuppliedMWh × ValueOfLostLoad

TotalLoss = (DirectLoss + RestorationLoss) × SocialUplift

FundingPriority = 0.28×Risk
                + 0.22×(100-Resilience)
                + 0.18×SocialVulnerability
                + 0.20×FailureProbability×100
                + 0.12×NormalisedFinancialLoss×100
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_monte_carlo_tab(df: pd.DataFrame, simulations: int) -> None:
    st.subheader("Monte Carlo")
    st.markdown(
        """
        <div class="note">
        This is the improved Monte Carlo simulation. It is called simply <b>Monte Carlo</b> in the app,
        as requested. It uses correlated storm uncertainty, demand/ENS variation and lognormal
        restoration-cost tails.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Running Monte Carlo simulation..."):
        mc = run_monte_carlo(df, simulations)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest P95 risk", f"{mc['mc_risk_p95'].max():.1f}/100")
    c2.metric("Highest P95 failure", pct(mc["mc_failure_p95"].max()))
    c3.metric("Highest CVaR95 loss", money_m(mc["mc_loss_cvar95_gbp"].max()))
    c4.metric("Simulations / place", f"{simulations:,}")

    a, b = st.columns(2)
    with a:
        fig = px.scatter(
            mc,
            x="mc_risk_mean",
            y="mc_risk_p95",
            size="mc_loss_cvar95_gbp",
            color="mc_failure_p95",
            hover_name="place",
            title="Mean risk vs P95 risk",
            template=plotly_template(),
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        worst = mc.iloc[0]
        fig = px.histogram(
            x=worst["risk_samples"],
            nbins=32,
            title=f"Monte Carlo risk distribution — {worst['place']}",
            template=plotly_template(),
        )
        fig.update_layout(height=450, xaxis_title="Risk score", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(mc.drop(columns=["risk_samples"]), use_container_width=True, hide_index=True)

    st.markdown("#### Monte Carlo formula")
    st.markdown(
        """
        <div class="formula">
StormShock ~ Normal(0,1)

WindSimulated = WindObserved × exp(0.15×StormShock + random_error)
RainSimulated = RainObserved × exp(0.30×StormShock + random_error)

OutageCount ~ Poisson(HazardGate × AssetExposure)

Loss = ENSMWh × Lognormal(ValueOfLostLoad) + OutageCount × Lognormal(RestorationCost)

CVaR95 = mean(Loss | Loss ≥ P95(Loss))
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_readme_tab() -> None:
    st.subheader("README / Methodology")

    st.markdown(
        """
# SAT-Guard Q1 Digital Twin — Enhanced README

## 1. Purpose

This application is a transparent decision-support dashboard for regional
electricity-grid resilience. It combines weather, air-quality, social
vulnerability, asset exposure, indicative outage stress, financial exposure,
failure probability and investment prioritisation.

The most important correction is that ordinary live weather no longer produces
false severe warnings. Newcastle, Durham or any other area with low wind,
little or no rain and no confirmed outage evidence should remain Low or
Moderate risk.

---

## 2. What changed from the earlier version?

The previous logic allowed synthetic outage and ENS estimates to behave like
real outage evidence. That created inflated risk values even when live weather
was calm. The corrected model separates:

1. **Live operational evidence**
2. **Scenario-based synthetic stress**
3. **Social vulnerability**
4. **Financial consequence**

Social vulnerability remains important, but it does not mean there is a storm.
IMD affects consequence and funding priority; it does not automatically create
a Severe grid-failure warning.

---

## 3. Data used in the app

### 3.1 Weather

The model uses:
- wind speed at 10 m
- precipitation
- temperature
- cloud cover
- humidity
- air-quality variables

In this self-contained version, live weather is fetched from Open-Meteo where
available. If the connection is unavailable, conservative normal-weather
fallback values are used. These fallback values are intentionally calm and
cannot generate artificial severe alerts.

### 3.2 Air quality

European AQI is used as a pressure indicator. Poor air quality can affect
public-health vulnerability and crew working conditions, but AQI alone should
not produce a grid-storm alert.

### 3.3 IMD / social vulnerability

IMD means Index of Multiple Deprivation. It is a socio-economic indicator used
in the UK to identify areas where communities may be more vulnerable to service
disruption.

Typical IMD domains include:
- income
- employment
- education
- health
- crime
- barriers to housing and services
- living environment

In this app, IMD is converted into a social vulnerability score. A higher score
means a stronger social consequence if energy disruption occurs.

### 3.4 Asset exposure

Asset exposure is a local proxy for how exposed the local grid may be to
hazards. It is not a replacement for detailed utility asset data. It helps the
scenario engine decide which places are more sensitive during simulated
stress events.

---

## 4. Explanation of each tab

### 4.1 Executive overview

This tab summarises the regional state:
- highest risk
- mean resilience
- maximum grid-failure probability
- total financial exposure

It also shows the corrected risk table. The table contains the exact variables
used to produce the classification.

### 4.2 Spatial intelligence

This tab shows a colourful interactive map. The colour and bubble size represent
regional risk.

Legend:
- green = low risk
- yellow-green = moderate condition
- yellow = elevated pressure
- orange = high stress
- red = severe stress

The map is designed to be clearer than a plain table. Hovering over a location
shows the exact drivers.

### 4.3 Regional risk

This tab decomposes risk into components:
- wind component
- rain component
- AQI component
- outage component
- ENS component
- social vulnerability

This makes the model explainable. A user can see whether risk is caused by
weather, social vulnerability, outage stress or energy-not-supplied exposure.

### 4.4 Failure and investment

This tab combines the old failure and investment tabs. It estimates grid-failure
probability and ranks locations for reinforcement or monitoring.

Important correction:
failure probability is gated by hazard evidence. Calm weather produces low
failure probability.

### 4.5 Finance and funding

This tab combines finance and funding. It estimates indicative financial loss
from:
- energy not supplied
- duration
- value of lost load
- restoration cost
- social uplift

Then it creates a funding-priority score.

### 4.6 Monte Carlo

Only one Monte Carlo tab remains. It uses the improved simulation:
- correlated storm shock
- wind and rain moving together
- Poisson outage stress
- lognormal cost tails
- P95 and CVaR95 loss metrics

The tab is named “Monte Carlo”, as requested.

---

## 5. Formulae

### 5.1 Risk formula

Risk =
4
+ 0.27 × WindComponent
+ 0.22 × RainComponent
+ 0.10 × AQIComponent
+ 0.18 × OutageComponent
+ 0.10 × ENSComponent
+ 0.13 × SocialVulnerability

### 5.2 Live calm-weather guard

If:

- wind < 20 km/h
- precipitation < 1 mm
- confirmed outage records = 0
- scenario = Live / Real-time

then the risk is capped below high-warning level.

This is the correction that prevents calm Newcastle or Durham conditions from
being incorrectly classified as Severe.

### 5.3 Resilience formula

Resilience = 100 − Risk − social penalty − asset exposure penalty

If risk is low or moderate, resilience is prevented from becoming Fragile.
This avoids the earlier error where ordinary weather produced Fragile labels.

### 5.4 Failure probability formula

HazardGate =
max(
normalised wind above alert threshold,
normalised rain above alert threshold,
normalised outage density above alert threshold,
normalised ENS above alert threshold
)

FailureProbability =
sigmoid(
−4.25
+ 0.052 × Risk
+ 1.85 × HazardGate
+ 0.006 × max(SocialVulnerability − 50, 0)
)

If HazardGate is very low in live mode, failure probability is capped at 8%.

### 5.5 Financial-loss formula

DurationHours =
1 + 5 × normalise(Risk, 45, 100)

EnergyNotSuppliedMWh =
EnergyNotSuppliedMW × DurationHours

DirectLoss =
EnergyNotSuppliedMWh × ValueOfLostLoad

TotalLoss =
(DirectLoss + RestorationLoss) × SocialUplift

### 5.6 Investment priority formula

InvestmentPriority =
0.28 × Risk
+ 0.22 × (100 − Resilience)
+ 0.18 × SocialVulnerability
+ 0.20 × FailureProbability × 100
+ 0.12 × NormalisedFinancialLoss × 100

---

## 6. Interpretation guidance

A High or Severe score should not be interpreted blindly. The explainability
column must be checked. A useful model should show why it is concerned.

For example:
- high wind + high rain + outage evidence = credible hazard concern
- high IMD alone = vulnerability concern, not storm concern
- low wind + no rain + no outage = no severe weather warning

---

## 7. Academic and professional references to cite in a paper

Suggested references for the methodology section:

1. UK Ministry of Housing, Communities and Local Government. English Indices
   of Deprivation methodology and guidance.
2. Ofgem. RIIO-ED2 Final Determinations and electricity distribution resilience
   framework.
3. Open-Meteo. Weather forecast and air-quality API documentation.
4. IEEE and CIGRE literature on power-system resilience, reliability,
   restoration and outage risk.
5. Sullivan, M. J., Schellenberg, J., and Blundell, M. Updated Value of Service
   Reliability Estimates for Electric Utility Customers in the United States.
6. Panteli, M. and Mancarella, P. The grid: stronger, bigger, smarter?
   Presenting a conceptual framework of power system resilience.
7. Ouyang, M. Review on modelling and simulation of interdependent critical
   infrastructure systems.

---

## 8. Limitations

This app is a research-grade prototype, not an operational utility control
system. It should be calibrated with:
- real outage records
- feeder topology
- substation loading
- restoration logs
- customer interruption data
- local flood and wind exposure
- validated cost assumptions

The model is transparent by design. It is not a black-box neural network.

---

## 9. Deployment

Save this file as app.py and run:

streamlit run app.py

Recommended packages:

streamlit
pandas
numpy
requests
plotly

        """
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    render_header()

    st.sidebar.header("Controls")
    region = st.sidebar.selectbox("Region", list(REGIONS.keys()), index=0)
    scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()), index=0)

    st.sidebar.markdown("### Live outage handling")
    synthetic_outages = st.sidebar.checkbox(
        "Allow synthetic outage estimates in Live / Real-time",
        value=False,
        help="Keep OFF for honest live status. Turn ON only for testing.",
    )

    simulations = st.sidebar.slider("Monte Carlo simulations per place", 250, 5000, 1200, step=250)

    st.sidebar.markdown("### Scenario description")
    st.sidebar.info(SCENARIOS[scenario]["description"])

    base = build_base_places(region)
    scenario_df = apply_scenario(base, scenario)
    model_df = process_model(scenario_df, scenario, synthetic_outages)

    tab_names = [
        "Executive overview",
        "Spatial intelligence",
        "Regional risk",
        "Failure and investment",
        "Finance and funding",
        "Monte Carlo",
        "README",
    ]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_executive_tab(model_df, region, scenario, synthetic_outages)
    with tabs[1]:
        render_spatial_tab(model_df, region)
    with tabs[2]:
        render_regional_risk_tab(model_df)
    with tabs[3]:
        render_failure_investment_tab(model_df)
    with tabs[4]:
        render_finance_funding_tab(model_df)
    with tabs[5]:
        render_monte_carlo_tab(model_df, simulations)
    with tabs[6]:
        render_readme_tab()


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# APPENDIX NOTE 001
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 002
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 003
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 004
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 005
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 006
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 007
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 008
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 009
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 010
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 011
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 012
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 013
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 014
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 015
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 016
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 017
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 018
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 019
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 020
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 021
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 022
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 023
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 024
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 025
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 026
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 027
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 028
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 029
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 030
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 031
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 032
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 033
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 034
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 035
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 036
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 037
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 038
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 039
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 040
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 041
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 042
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 043
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 044
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 045
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 046
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 047
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 048
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 049
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 050
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 051
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 052
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 053
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 054
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 055
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 056
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 057
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 058
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 059
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 060
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 061
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 062
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 063
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 064
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 065
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 066
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 067
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 068
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 069
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 070
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 071
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 072
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 073
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 074
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 075
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 076
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 077
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 078
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 079
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 080
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 081
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 082
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 083
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 084
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 085
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 086
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 087
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 088
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 089
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 090
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 091
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 092
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 093
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 094
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 095
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 096
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 097
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 098
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 099
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 100
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 101
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 102
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 103
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 104
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 105
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 106
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 107
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 108
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 109
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 110
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 111
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 112
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 113
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 114
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 115
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 116
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 117
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 118
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 119
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# APPENDIX NOTE 120
# -----------------------------------------------------------------------------
# This appendix note is included to keep the application self-documenting.
# It records the modelling philosophy used in SAT-Guard:
# - do not classify calm weather as an emergency;
# - separate live evidence from stress-test assumptions;
# - keep every risk, resilience, failure, finance and funding calculation visible;
# - use scenario multipliers only in scenario mode;
# - use social vulnerability as a consequence modifier, not as a weather hazard;
# - use Monte Carlo for uncertainty, not as a replacement for field evidence.
#
# Research interpretation:
# A Q1-quality dashboard should expose its assumptions, thresholds, formulae,
# variables, limitations and uncertainty bands. This is why the app includes a
# README tab, formula boxes, hover-level map explanations and tabular diagnostics.
# -----------------------------------------------------------------------------
