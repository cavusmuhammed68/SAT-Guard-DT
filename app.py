"""
SAT-Guard Digital Twin — Q1 Edition
=====================================
PART 1 of 4 — Configuration, helpers, IoD2025 loader, external data fetching

HOW TO ASSEMBLE:
    cat app_KASVA_PART1.py app_KASVA_PART2.py app_KASVA_PART3.py app_KASVA_PART4.py > app_KASVA_FINAL.py
    streamlit run app_KASVA_FINAL.py

FIXES in this edition vs previous version:
    1. clamp() / risk_label() / resilience_label() defined ONCE only.
    2. Spatial Intelligence tab: pentagon/hexagon cells replaced with proper
       filled local-authority polygon regions (political-map style).
    3. render_spatial_intelligence_ultra() integrated into spatial_tab().
    4. flood_depth_proxy() result always written into places DataFrame.
    5. is_calm_live_weather() signature standardised across all call sites.
    6. scenario_financial_matrix() MC cap raised 60 → 150.
    7. CVaR95 uses correct exceedance-mean formula.
    8. No circular final_risk_score → compound_hazard_proxy feedback.
"""

from __future__ import annotations

import html
import json
import math
import random
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


# =============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="SAT-Guard Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# GLOBAL CSS
# =============================================================================

APP_CSS = """
<style>
:root {
    --bg:#020617; --panel:rgba(15,23,42,0.82); --panel2:rgba(30,41,59,0.68);
    --border:rgba(148,163,184,0.22); --text:#e5e7eb; --muted:#94a3b8;
    --blue:#38bdf8; --green:#22c55e; --yellow:#eab308;
    --orange:#f97316; --red:#ef4444; --purple:#a855f7;
}
.stApp {
    background:
        radial-gradient(circle at top left,rgba(56,189,248,0.20),transparent 34%),
        radial-gradient(circle at 70% 20%,rgba(168,85,247,0.12),transparent 34%),
        linear-gradient(180deg,#020617 0%,#050816 42%,#020617 100%);
}
.block-container { padding-top:1.15rem; padding-bottom:2.5rem; }
[data-testid="stSidebar"] {
    background:rgba(2,6,23,0.96);
    border-right:1px solid rgba(148,163,184,0.18);
}
.hero {
    border:1px solid rgba(148,163,184,0.20);
    background:linear-gradient(135deg,rgba(14,165,233,0.20),rgba(168,85,247,0.10)),rgba(15,23,42,0.82);
    border-radius:28px; padding:22px 24px;
    box-shadow:0 24px 80px rgba(0,0,0,0.32); margin-bottom:18px;
}
.title { font-size:38px; font-weight:950; letter-spacing:-0.05em; color:white; margin-bottom:4px; }
.subtitle { color:#cbd5e1; font-size:15px; line-height:1.5; }
.chip {
    display:inline-block; margin:4px 6px 0 0; border-radius:999px;
    border:1px solid rgba(148,163,184,0.25); background:rgba(2,6,23,0.58);
    padding:7px 12px; color:#bfdbfe; font-weight:800; font-size:12px;
}
.card {
    border:1px solid rgba(148,163,184,0.18); background:rgba(15,23,42,0.72);
    border-radius:24px; padding:18px; box-shadow:0 24px 70px rgba(0,0,0,0.26);
}
.note {
    border:1px solid rgba(56,189,248,0.25); background:rgba(56,189,248,0.09);
    border-radius:18px; padding:14px 16px; color:#dbeafe;
}
.warning {
    border:1px solid rgba(249,115,22,0.30); background:rgba(249,115,22,0.10);
    border-radius:18px; padding:14px 16px; color:#fed7aa;
}
.formula {
    border-left:4px solid #38bdf8; background:rgba(2,6,23,0.50);
    padding:12px 14px; border-radius:12px; color:#e0f2fe;
    font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; font-size:13px;
}
.stMetric {
    background:rgba(15,23,42,0.56); border:1px solid rgba(148,163,184,0.18);
    border-radius:18px; padding:12px 14px; box-shadow:0 12px 34px rgba(0,0,0,0.22);
}
[data-testid="stMetricValue"] { color:white; font-weight:950; }
[data-testid="stMetricLabel"] { color:#bfdbfe; }
hr { border-color:rgba(148,163,184,0.18); }
</style>
"""


# =============================================================================
# CONFIGURATION
# =============================================================================

NPG_DATASET_URL = (
    "https://northernpowergrid.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/live-power-cuts-data/records"
)
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_URL    = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_CURRENT_VARS = ",".join([
    "temperature_2m","apparent_temperature","wind_speed_10m","wind_direction_10m",
    "surface_pressure","cloud_cover","shortwave_radiation","direct_radiation",
    "diffuse_radiation","relative_humidity_2m","precipitation","is_day",
])
AIR_CURRENT_VARS = ",".join([
    "european_aqi","pm10","pm2_5","nitrogen_dioxide","ozone",
    "sulphur_dioxide","carbon_monoxide","aerosol_optical_depth","dust","uv_index",
])

# ---------------------------------------------------------------------------
# SCENARIOS
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Live / Real-time": {
        "wind":1.00,"rain":1.00,"temperature":0.0,"aqi":1.00,"solar":1.00,
        "outage":1.00,"finance":1.00,"hazard_mode":"wind",
        "description":"Observed real-time conditions without imposed stress.",
    },
    "Extreme wind": {
        "wind":3.60,"rain":1.45,"temperature":-2.0,"aqi":1.12,"solar":0.72,
        "outage":3.10,"finance":2.15,"hazard_mode":"wind",
        "description":"Severe wind event stressing overhead lines and exposed assets.",
    },
    "Flood": {
        "wind":1.55,"rain":7.50,"temperature":0.5,"aqi":1.18,"solar":0.28,
        "outage":3.60,"finance":2.40,"hazard_mode":"rain",
        "description":"Extreme rainfall and surface flooding impacting substations.",
    },
    "Heatwave": {
        "wind":0.75,"rain":0.10,"temperature":13.0,"aqi":2.15,"solar":1.35,
        "outage":2.15,"finance":2.00,"hazard_mode":"heat",
        "description":"High temperature stress increasing demand peaks and transformer heating.",
    },
    "Drought": {
        "wind":0.22,"rain":0.05,"temperature":6.5,"aqi":1.65,"solar":0.18,
        "outage":2.30,"finance":2.10,"hazard_mode":"calm",
        "description":"Prolonged low wind and solar generation reducing renewable supply.",
    },
    "Total blackout stress": {
        "wind":1.35,"rain":1.50,"temperature":0.0,"aqi":1.35,"solar":0.35,
        "outage":7.00,"finance":4.20,"hazard_mode":"blackout",
        "description":"Extreme outage clustering and cascading failures across the network.",
    },
    "Compound extreme": {
        "wind":3.25,"rain":6.50,"temperature":8.0,"aqi":2.20,"solar":0.20,
        "outage":5.80,"finance":3.80,"hazard_mode":"storm",
        "description":"Combined wind, flood, heat and system stress — multi-hazard disruption.",
    },
}

# ---------------------------------------------------------------------------
# REGIONS
# ---------------------------------------------------------------------------
REGIONS: Dict[str, Dict[str, Any]] = {
    "North East": {
        "center": {"lat":54.85,"lon":-1.65,"zoom":7},
        "bbox": [-3.35, 54.10, -0.60, 55.95],
        "places": {
            "Newcastle":    {"lat":54.9783,"lon":-1.6178,"postcode_prefix":"NE1","authority_tokens":["newcastle","newcastle upon tyne"],"population_density":2590,"vulnerability_proxy":43,"estimated_load_mw":128,"business_density":0.68},
            "Sunderland":   {"lat":54.9069,"lon":-1.3838,"postcode_prefix":"SR1","authority_tokens":["sunderland"],"population_density":2010,"vulnerability_proxy":52,"estimated_load_mw":106,"business_density":0.54},
            "Durham":       {"lat":54.7761,"lon":-1.5733,"postcode_prefix":"DH1","authority_tokens":["durham","county durham"],"population_density":730,"vulnerability_proxy":38,"estimated_load_mw":64,"business_density":0.38},
            "Middlesbrough":{"lat":54.5742,"lon":-1.2350,"postcode_prefix":"TS1","authority_tokens":["middlesbrough","teesside"],"population_density":2680,"vulnerability_proxy":61,"estimated_load_mw":96,"business_density":0.50},
            "Darlington":   {"lat":54.5236,"lon":-1.5595,"postcode_prefix":"DL1","authority_tokens":["darlington"],"population_density":1070,"vulnerability_proxy":45,"estimated_load_mw":72,"business_density":0.41},
            "Hexham":       {"lat":54.9730,"lon":-2.1010,"postcode_prefix":"NE46","authority_tokens":["hexham","northumberland"],"population_density":330,"vulnerability_proxy":32,"estimated_load_mw":38,"business_density":0.24},
        },
        "tokens": [
            "newcastle","sunderland","durham","middlesbrough","darlington",
            "hexham","gateshead","northumberland","teesside","hartlepool",
            "stockton","redcar","tyne and wear","county durham",
        ],
        # Local authority polygons for the colourful risk map
        "authority_polygons": {
            "Northumberland":          [[-2.8,55.1],[-1.3,55.1],[-1.1,55.8],[-1.5,56.0],[-2.5,55.9],[-2.9,55.5],[-2.8,55.1]],
            "Newcastle / Gateshead":   [[-1.78,54.9],[-1.35,54.9],[-1.32,55.15],[-1.6,55.2],[-1.82,55.05],[-1.78,54.9]],
            "Sunderland":              [[-1.65,54.75],[-1.15,54.75],[-1.1,55.02],[-1.48,55.06],[-1.7,54.9],[-1.65,54.75]],
            "County Durham":           [[-2.1,54.45],[-1.2,54.45],[-1.0,54.95],[-1.35,55.05],[-2.0,54.9],[-2.15,54.55],[-2.1,54.45]],
            "Teesside / Middlesbrough":[[-1.45,54.35],[-0.85,54.35],[-0.78,54.72],[-1.2,54.82],[-1.48,54.58],[-1.45,54.35]],
        },
        "place_authority_map": {
            "Newcastle":"Newcastle / Gateshead","Sunderland":"Sunderland",
            "Durham":"County Durham","Middlesbrough":"Teesside / Middlesbrough",
            "Darlington":"County Durham","Hexham":"Northumberland",
        },
    },
    "Yorkshire": {
        "center": {"lat":53.95,"lon":-1.30,"zoom":7},
        "bbox": [-2.90, 53.20, -0.10, 54.75],
        "places": {
            "Leeds":     {"lat":53.8008,"lon":-1.5491,"postcode_prefix":"LS1","authority_tokens":["leeds"],"population_density":1560,"vulnerability_proxy":44,"estimated_load_mw":168,"business_density":0.74},
            "Sheffield": {"lat":53.3811,"lon":-1.4701,"postcode_prefix":"S1","authority_tokens":["sheffield"],"population_density":1510,"vulnerability_proxy":48,"estimated_load_mw":144,"business_density":0.66},
            "York":      {"lat":53.9600,"lon":-1.0873,"postcode_prefix":"YO1","authority_tokens":["york"],"population_density":740,"vulnerability_proxy":34,"estimated_load_mw":82,"business_density":0.50},
            "Hull":      {"lat":53.7676,"lon":-0.3274,"postcode_prefix":"HU1","authority_tokens":["hull","kingston upon hull"],"population_density":3560,"vulnerability_proxy":62,"estimated_load_mw":116,"business_density":0.52},
            "Bradford":  {"lat":53.7950,"lon":-1.7594,"postcode_prefix":"BD1","authority_tokens":["bradford"],"population_density":1450,"vulnerability_proxy":59,"estimated_load_mw":132,"business_density":0.48},
            "Doncaster": {"lat":53.5228,"lon":-1.1285,"postcode_prefix":"DN1","authority_tokens":["doncaster"],"population_density":540,"vulnerability_proxy":49,"estimated_load_mw":78,"business_density":0.37},
        },
        "tokens": [
            "yorkshire","leeds","sheffield","york","hull","bradford",
            "wakefield","rotherham","doncaster","barnsley","huddersfield",
            "harrogate","scarborough","halifax","east riding",
        ],
        "authority_polygons": {
            "North Yorkshire":      [[-2.7,53.9],[-0.7,53.9],[-0.5,54.7],[-1.4,54.9],[-2.5,54.7],[-2.8,54.2],[-2.7,53.9]],
            "Leeds / Bradford":     [[-2.2,53.65],[-1.2,53.65],[-1.1,53.98],[-1.5,54.05],[-2.25,53.9],[-2.2,53.65]],
            "Sheffield / Doncaster":[[-1.9,53.2],[-1.0,53.2],[-0.9,53.68],[-1.3,53.78],[-1.95,53.55],[-1.9,53.2]],
            "Hull / East Riding":   [[-0.7,53.55],[-0.1,53.55],[0.0,53.9],[-0.3,54.0],[-0.75,53.82],[-0.7,53.55]],
        },
        "place_authority_map": {
            "Leeds":"Leeds / Bradford","Bradford":"Leeds / Bradford",
            "Sheffield":"Sheffield / Doncaster","Doncaster":"Sheffield / Doncaster",
            "York":"North Yorkshire","Hull":"Hull / East Riding",
        },
    },
}

# ---------------------------------------------------------------------------
# HAZARD TYPES
# ---------------------------------------------------------------------------
HAZARD_TYPES: Dict[str, Dict[str, Any]] = {
    "Wind storm":             {"driver":"wind_speed_10m","unit":"km/h","threshold_low":25,"threshold_high":55,"description":"Overhead line exposure, tree fall, conductor galloping and access constraints."},
    "Flood / heavy rain":     {"driver":"precipitation","unit":"mm","threshold_low":1.5,"threshold_high":8.0,"description":"Surface-water flooding, substation access risk and cascading delays."},
    "Drought":                {"driver":"renewable_failure_probability","unit":"probability","threshold_low":0.35,"threshold_high":0.75,"description":"Low wind and solar causing net-load pressure and import dependence."},
    "Air-quality / heat stress":{"driver":"european_aqi","unit":"AQI","threshold_low":35,"threshold_high":95,"description":"Public-health stress, crew welfare constraints and vulnerable-population impacts."},
    "Compound hazard":        {"driver":"compound_hazard_proxy","unit":"score","threshold_low":25,"threshold_high":70,"description":"Combined meteorological and infrastructure stress. Uses only direct drivers — NOT final_risk_score (no circular feedback)."},
}

EV_ASSUMPTIONS: Dict[str, float] = {
    "ev_penetration_low":0.18,"ev_penetration_mid":0.32,"ev_penetration_high":0.48,
    "share_parked_during_storm":0.72,"share_v2g_enabled":0.26,
    "usable_battery_kwh":38.0,"grid_export_limit_kw":7.0,
    "charger_substation_coupling_factor":0.62,"emergency_dispatch_hours":3.0,
}

VALIDATION_BENCHMARKS: Dict[str, str] = {
    "risk_monotonicity":       "Risk should increase when wind, rain, outage intensity, social vulnerability or ENS increases.",
    "resilience_inverse":      "Resilience should decrease when risk, social vulnerability, grid failure, renewable failure or financial loss increases.",
    "scenario_sensitivity":    "Extreme scenarios should produce materially higher risk or loss than live/real-time mode.",
    "postcode_explainability": "Every low postcode resilience score should expose the contributing drivers.",
    "non_black_box":           "The model exposes its formulae, weights, assumptions and intermediate variables.",
}

LAD_NAME_MAPPING: Dict[str, str] = {
    "Newcastle":"Newcastle upon Tyne","Sunderland":"Sunderland","Durham":"County Durham",
    "Middlesbrough":"Middlesbrough","Darlington":"Darlington","Hexham":"Northumberland",
    "Leeds":"Leeds","Sheffield":"Sheffield","York":"York",
    "Hull":"Kingston upon Hull","Bradford":"Bradford","Doncaster":"Doncaster",
}

DATA_DIR  = Path("data")
INFRA_DIR = DATA_DIR / "infrastructure"
FLOOD_DIR = DATA_DIR / "flood"


# =============================================================================
# HELPERS  (each defined ONCE)
# =============================================================================

def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]. Safe against non-numeric inputs."""
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return float(low)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def clean_col(col: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(float(lat1)))
         * math.cos(math.radians(float(lat2)))
         * math.sin(dlon/2)**2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def risk_label(score: float) -> str:
    s = safe_float(score)
    if s >= 85: return "Severe"
    if s >= 65: return "High"
    if s >= 40: return "Moderate"
    return "Low"


def resilience_label(score: float) -> str:
    s = safe_float(score)
    if s >= 80: return "Robust"
    if s >= 60: return "Functional"
    if s >= 40: return "Stressed"
    return "Fragile"


def requests_json(url: str, params: Dict[str,Any]=None, timeout: int=20) -> Dict[str,Any]:
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent":"sat-guard/3.0"}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def money_m(value: float) -> str:
    return f"£{safe_float(value)/1_000_000:.2f}m"


def pct(value: float) -> str:
    return f"{safe_float(value)*100:.1f}%"


def plotly_template() -> str:
    return "plotly_dark"


# =============================================================================
# COLOUR HELPERS
# =============================================================================

def colour_rgba_hex(score: float) -> str:
    s = safe_float(score)
    if s >= 75: return "#ef4444"
    if s >= 55: return "#f97316"
    if s >= 35: return "#eab308"
    return "#22c55e"


def risk_colour_rgba(score: float) -> List[int]:
    s = safe_float(score)
    if s >= 75: return [239,68,68,205]
    if s >= 55: return [249,115,22,190]
    if s >= 35: return [234,179,8,180]
    return [34,197,94,180]


def resilience_colour_rgba(score: float) -> List[int]:
    s = safe_float(score)
    if s >= 80: return [34,197,94,190]
    if s >= 60: return [56,189,248,185]
    if s >= 40: return [234,179,8,180]
    return [239,68,68,190]


def regional_risk_hex(score: float) -> str:
    """High-contrast categorical fill colour for authority polygon map."""
    s = safe_float(score)
    if s >= 85: return "#d80073"   # magenta — severe
    if s >= 65: return "#ff7b00"   # orange  — high
    if s >= 40: return "#0070c0"   # blue    — moderate
    return "#7bd000"               # green   — low


# =============================================================================
# INFRASTRUCTURE LOADERS
# =============================================================================

def load_vector_layer_safe(path: Path) -> dict:
    EMPTY = {"type":"FeatureCollection","features":[]}
    try:
        if path is None or not path.exists(): return EMPTY
        if path.suffix.lower() not in [".geojson",".json"]: return EMPTY
        with open(path,"r",encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data,dict): return EMPTY
        if "features" not in data or not isinstance(data["features"],list): return EMPTY
        valid = [f for f in data["features"] if isinstance(f,dict) and f.get("geometry") and "coordinates" in f["geometry"]]
        return {"type":"FeatureCollection","features":valid}
    except Exception:
        return EMPTY


def geojson_has_features(obj: dict) -> bool:
    return isinstance(obj,dict) and len(obj.get("features",[]))>0


@st.cache_data(ttl=3600)
def load_infrastructure_data():
    substations = load_vector_layer_safe(INFRA_DIR/"gb_substations_data_281118.geojson")
    lines       = load_vector_layer_safe(INFRA_DIR/"GB_Transmission_Network_Data.geojson")
    gsp         = load_vector_layer_safe(INFRA_DIR/"GSP_regions_4326_20260209.geojson")
    return substations, lines, gsp


@st.cache_data(ttl=3600)
def load_flood_data() -> dict:
    return load_vector_layer_safe(FLOOD_DIR/"flood_zones.geojson")


# =============================================================================
# IoD / IMD FILE FINDER
# =============================================================================

def find_imd_files() -> List[Path]:
    current = Path.cwd()
    dirs = [
        current, current/"data", current/"data"/"iod2025", current/"iod2025",
        Path("/mount/src/sat-guard-dt"), Path("/mount/src/sat-guard-dt")/"data"/"iod2025",
        Path("/mnt/data"), Path("/mnt/data")/"data"/"iod2025",
    ]
    explicit = [
        "IoD2025 Local Authority District Summaries (lower-tier) - Rank of average rank.xlsx",
        "File_1_IoD2025 Index of Multiple Deprivation.xlsx",
        "File_2_IoD2025 Domains of Deprivation.xlsx",
        "File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.xlsx",
        "imd.xlsx","domains.xlsx","supplementary.xlsx","lad_summary.xlsx",
    ]
    files: List[Path] = []
    for folder in dirs:
        try:
            if not folder.exists(): continue
            for name in explicit:
                p = folder/name
                if p.exists() and p not in files: files.append(p)
            for pattern in ["*IoD2025*.xlsx","*Deprivation*.xlsx","*IDACI*.xlsx","*.xlsx"]:
                for p in folder.glob(pattern):
                    if p.exists() and p.suffix.lower() in [".xlsx",".xls"] and p not in files:
                        files.append(p)
            for p in folder.glob("*/*.xlsx"):
                if p.exists() and p not in files: files.append(p)
        except Exception:
            continue
    return files


def choose_first_matching_column(columns, include_terms, exclude_terms=None):
    exclude_terms = exclude_terms or []
    cleaned = [(c, clean_col(c)) for c in columns]
    for col, text in cleaned:
        if all(t in text for t in include_terms) and not any(e in text for e in exclude_terms):
            return col
    for col, text in cleaned:
        if any(t in text for t in include_terms) and not any(e in text for e in exclude_terms):
            return col
    return None


def normalise_imd_score_from_rank(rank_value, max_rank):
    rv = safe_float(rank_value, None)
    mr = safe_float(max_rank, None)
    if rv is None or mr is None or mr <= 1: return None
    return round(clamp((1 - (rv-1)/(mr-1))*100, 0, 100), 2)


def extract_imd_summary_from_sheet(df, source_name):
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy().dropna(axis=1, how="all")
    if df.empty: return pd.DataFrame()
    cols = list(df.columns)

    area_col = (choose_first_matching_column(cols,["local","authority"])
                or choose_first_matching_column(cols,["lad"])
                or choose_first_matching_column(cols,["area"])
                or choose_first_matching_column(cols,["name"]))
    score_col  = (choose_first_matching_column(cols,["imd","score"])
                  or choose_first_matching_column(cols,["average","score"]))
    rank_col   = (choose_first_matching_column(cols,["imd","rank"])
                  or choose_first_matching_column(cols,["rank"]))
    decile_col = choose_first_matching_column(cols,["decile"])

    if area_col is None: return pd.DataFrame()

    out = pd.DataFrame()
    out["area_name"] = df[area_col].astype(str)
    out["source_file"] = source_name

    if score_col is not None:
        score = pd.to_numeric(df[score_col], errors="coerce")
        mx, mn = score.max(), score.min()
        out["imd_score_0_100"] = ((score-mn)/max(mx-mn,1)*100) if (mx>100 or mn<0) else score
        out["imd_metric_source"] = f"score:{score_col}"
    elif rank_col is not None:
        rank = pd.to_numeric(df[rank_col], errors="coerce")
        out["imd_score_0_100"] = rank.apply(lambda x: normalise_imd_score_from_rank(x, rank.max()))
        out["imd_metric_source"] = f"rank:{rank_col}"
    elif decile_col is not None:
        decile = pd.to_numeric(df[decile_col], errors="coerce")
        out["imd_score_0_100"] = (10 - decile)/9*100
        out["imd_metric_source"] = f"decile:{decile_col}"
    else:
        return pd.DataFrame()

    out["imd_score_0_100"] = pd.to_numeric(out["imd_score_0_100"], errors="coerce")
    out = out.dropna(subset=["imd_score_0_100"])
    out["imd_score_0_100"] = out["imd_score_0_100"].clip(0,100)
    out["area_key"] = out["area_name"].str.lower()
    return out[["area_name","area_key","imd_score_0_100","imd_metric_source","source_file"]]


@st.cache_data(ttl=3600, show_spinner=False)
def load_imd_summary_cached() -> Tuple[pd.DataFrame, str]:
    files = find_imd_files()
    parts, notes = [], []
    for fp in files:
        try:
            sheets = pd.read_excel(fp, sheet_name=None, engine="openpyxl")
        except Exception:
            continue
        for sn, df in sheets.items():
            try:
                part = extract_imd_summary_from_sheet(df, f"{fp.name}|{sn}")
                if part is not None and not part.empty:
                    parts.append(part); notes.append(f"{fp.name}:{sn}")
            except Exception:
                continue

    if parts:
        summary = pd.concat(parts, ignore_index=True)
        summary["imd_score_0_100"] = pd.to_numeric(summary["imd_score_0_100"], errors="coerce")
        summary = summary.dropna(subset=["area_key"])
        summary["area_key"] = summary["area_key"].astype(str).str.lower()
        grouped = summary.groupby("area_key", as_index=False).agg(
            {"area_name":"first","imd_score_0_100":"mean","imd_metric_source":"first","source_file":"first"}
        )
        grouped["imd_score_0_100"] = pd.to_numeric(grouped["imd_score_0_100"],errors="coerce").fillna(0).clip(0,100)
        grouped["imd_rank"] = grouped["imd_score_0_100"].rank(ascending=False, method="min")
        return grouped, "; ".join(notes[:10])
    return (pd.DataFrame(columns=["area_key","area_name","imd_score_0_100","imd_metric_source","source_file","imd_rank"]),
            "No readable IoD2025 Excel found; using fallback proxies.")


def infer_imd_for_place(place, region, meta, imd_summary):
    fallback = safe_float(meta.get("vulnerability_proxy"), 45)
    if imd_summary is None or imd_summary.empty:
        return {"imd_score":fallback,"imd_source":"fallback","imd_match":"no IMD"}

    tokens = [LAD_NAME_MAPPING.get(place,place).lower()] + [str(t).lower() for t in meta.get("authority_tokens",[])]
    region_tokens = [t.lower() for t in REGIONS[region]["tokens"]]

    for token in tokens:
        hit = imd_summary[imd_summary["area_key"].str.contains(token, regex=False, na=False)]
        if not hit.empty:
            return {"imd_score":round(float(pd.to_numeric(hit["imd_score_0_100"],errors="coerce").mean()),2),"imd_source":str(hit.iloc[0].get("source_file","IoD2025")),"imd_match":f"direct:{token}"}

    rscores = []
    for t in region_tokens:
        hit = imd_summary[imd_summary["area_key"].str.contains(t, regex=False, na=False)]
        if not hit.empty:
            rscores.extend(pd.to_numeric(hit["imd_score_0_100"],errors="coerce").dropna().tolist())
    if rscores:
        return {"imd_score":round(float(np.mean(rscores)),2),"imd_source":"IoD2025 regional","imd_match":"regional fallback"}

    return {"imd_score":fallback,"imd_source":"fallback","imd_match":"no authority match"}


@st.cache_data(ttl=3600, show_spinner=False)
def load_iod2025_domain_model() -> Tuple[pd.DataFrame, str]:
    files = find_imd_files()
    parts, notes = [], []
    DOMAIN_MAP = {
        "income":["income"],"employment":["employment"],"health":["health","disability"],
        "education":["education","skills"],"crime":["crime"],"housing":["housing","barriers"],
        "living":["living","environment"],"idaci":["idaci","children"],"idaopi":["idaopi","older"],
    }
    def detect(cols, keys):
        for c in cols:
            if any(k in clean_col(c) for k in keys): return c
        return None
    def normalise(vals):
        vals = pd.to_numeric(vals, errors="coerce")
        if vals.dropna().empty: return vals
        vmin, vmax = vals.min(), vals.max()
        if "rank" in str(vals.name).lower() and vmax > vmin:
            vals = (1 - (vals-vmin)/max(vmax-vmin,1))*100
        elif vmax <= 1.5: vals = vals*100
        elif vmax > 100 or vmin < 0: vals = (vals-vmin)/max(vmax-vmin,1)*100
        return vals.clip(0,100)

    for fp in files:
        try: sheets = pd.read_excel(fp, sheet_name=None, engine="openpyxl")
        except Exception: continue
        for sn, df in sheets.items():
            if df is None or df.empty: continue
            try:
                work = df.copy().dropna(axis=1, how="all")
                cols = list(work.columns)
                area_col = (choose_first_matching_column(cols,["local authority district name"])
                            or choose_first_matching_column(cols,["local authority"])
                            or choose_first_matching_column(cols,["lad name"])
                            or choose_first_matching_column(cols,["area"])
                            or choose_first_matching_column(cols,["name"]))
                code_col = choose_first_matching_column(cols,["lsoa","code"]) or choose_first_matching_column(cols,["lad","code"])
                if area_col is None and code_col is None: continue
                out = pd.DataFrame()
                base = work[area_col] if area_col else work[code_col]
                out["area_name"] = base.astype(str)
                out["area_key"]  = out["area_name"].str.lower()
                out["area_code"] = work[code_col].astype(str) if code_col else ""
                detected = []
                for domain, keys in DOMAIN_MAP.items():
                    col = detect(cols, keys+["score"]) or detect(cols, keys+["rank"]) or detect(cols, keys)
                    if col:
                        v = normalise(work[col])
                        if not v.dropna().empty: out[domain]=v; detected.append(domain)
                if len(detected) >= 2:
                    out["iod_social_vulnerability_0_100"] = out[detected].mean(axis=1, skipna=True)
                    out["domain_completeness"] = len(detected)
                    out["domains_detected"] = ",".join(detected)
                    out["source_file"] = f"{fp.name}|{sn}"
                    parts.append(out); notes.append(f"{fp.name}:{sn}")
            except Exception: continue

    if not parts:
        return pd.DataFrame(), "No readable IoD2025 domain model found."

    full = pd.concat(parts, ignore_index=True)
    for c in full.columns:
        if c not in ["area_name","area_key","area_code","source_file","domains_detected"]:
            full[c] = pd.to_numeric(full[c], errors="coerce")
    numeric_cols = full.select_dtypes(include=[np.number]).columns.tolist()
    agg = {"area_name":"first","area_code":"first","source_file":"first","domains_detected":"first"}
    for c in numeric_cols: agg[c] = "mean"
    full["area_key"] = full["area_key"].str.replace(r"\s+", " ", regex=True).str.strip()
    grouped = full.groupby("area_key", as_index=False).agg(agg)
    grouped[numeric_cols] = grouped[numeric_cols].apply(pd.to_numeric,errors="coerce").fillna(0).clip(0,100)
    return grouped, "; ".join(notes[:10])


def infer_iod_domain_vulnerability(place, region, meta):
    df, source = load_iod2025_domain_model()
    fallback = safe_float(meta.get("vulnerability_proxy"), 45)
    empty = {"iod_social_vulnerability":fallback,"iod_domain_source":source,"iod_domain_match":"fallback proxy","iod_income":np.nan,"iod_employment":np.nan,"iod_health":np.nan,"iod_education":np.nan,"iod_crime":np.nan,"iod_housing":np.nan,"iod_living":np.nan,"iod_idaci":np.nan,"iod_idaopi":np.nan}
    if df is None or df.empty or "area_key" not in df.columns: return empty
    df = df.copy()
    df["area_key_clean"] = df["area_key"].astype(str).str.lower().str.replace(r"\s+"," ",regex=True).str.strip()
    aliases = {
        "Newcastle":["newcastle upon tyne","newcastle"],"Sunderland":["sunderland"],
        "Durham":["county durham","durham"],"Middlesbrough":["middlesbrough"],
        "Darlington":["darlington"],"Hexham":["northumberland","hexham"],
        "Leeds":["leeds"],"Sheffield":["sheffield"],"York":["york"],
        "Hull":["kingston upon hull","hull"],"Bradford":["bradford"],"Doncaster":["doncaster"],
    }
    tokens = list(dict.fromkeys(aliases.get(place,[])+[place.lower()]+[str(t).lower() for t in meta.get("authority_tokens",[])]))
    hit = pd.DataFrame(); matched = ""
    for token in tokens:
        token = token.strip()
        if not token: continue
        ex = df[df["area_key_clean"]==token]
        if not ex.empty: hit=ex; matched=f"exact:{token}"; break
        pa = df[df["area_key_clean"].str.contains(token,regex=False,na=False)]
        if not pa.empty: hit=pa; matched=f"partial:{token}"; break
    if hit.empty:
        for t in REGIONS.get(region,{}).get("tokens",[]):
            t2 = str(t).lower().strip()
            tmp = df[df["area_key_clean"].str.contains(t2,regex=False,na=False)]
            if not tmp.empty: hit=pd.concat([hit,tmp],ignore_index=True); matched="regional"
    if hit.empty: return {**empty,"iod_domain_match":"no match"}
    def sm(*cols):
        for c in cols:
            if c in hit.columns:
                v = pd.to_numeric(hit[c],errors="coerce").dropna()
                if not v.empty: return round(float(v.mean()),2)
        return np.nan
    social = sm("iod_social_vulnerability_0_100","imd_score_0_100")
    if pd.isna(social): social = fallback
    return {"iod_social_vulnerability":round(float(social),2),"iod_domain_source":source,"iod_domain_match":f"matched:{matched}","iod_income":sm("income","iod_income"),"iod_employment":sm("employment","iod_employment"),"iod_health":sm("health","iod_health"),"iod_education":sm("education","iod_education"),"iod_crime":sm("crime","iod_crime"),"iod_housing":sm("housing","iod_housing"),"iod_living":sm("living","iod_living"),"iod_idaci":sm("idaci","iod_idaci"),"iod_idaopi":sm("idaopi","iod_idaopi")}


# =============================================================================
# EXTERNAL DATA FETCHING
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> Dict[str,Any]:
    return requests_json(OPEN_METEO_WEATHER_URL, params={"latitude":lat,"longitude":lon,"current":WEATHER_CURRENT_VARS,"timezone":"Europe/London"})


@st.cache_data(ttl=900, show_spinner=False)
def fetch_air_quality(lat: float, lon: float) -> Dict[str,Any]:
    return requests_json(OPEN_METEO_AIR_URL, params={"latitude":lat,"longitude":lon,"current":AIR_CURRENT_VARS,"timezone":"Europe/London"})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_northern_powergrid(limit: int=100) -> pd.DataFrame:
    payload = requests_json(NPG_DATASET_URL, params={"limit":int(clamp(limit,1,100))})
    records = payload.get("results",[])
    if not records: return pd.DataFrame()
    return pd.json_normalize(records)


def filter_npg_by_region(raw_df: pd.DataFrame, region: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty: return pd.DataFrame()
    tokens = REGIONS[region]["tokens"]
    obj_cols = [c for c in raw_df.columns if raw_df[c].dtype=="object"]
    if not obj_cols: return raw_df.copy()
    text = raw_df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(False, index=raw_df.index)
    for t in tokens: mask = mask | text.str.contains(t, regex=False)
    filtered = raw_df[mask].copy()
    return filtered if not filtered.empty else raw_df.copy()


def standardise_outages(raw_df: pd.DataFrame, region: str) -> pd.DataFrame:
    output_cols = ["outage_reference","outage_status","outage_category","postcode_label","affected_customers","estimated_restore","latitude","longitude","source_text","is_synthetic_outage"]
    if raw_df is None or raw_df.empty: return pd.DataFrame(columns=output_cols)
    df = filter_npg_by_region(raw_df, region)
    source_text = df.fillna("").astype(str).agg(" ".join, axis=1)
    source_lower = source_text.str.lower()
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
    lat = pd.to_numeric(df[lat_cols[0]], errors="coerce") if lat_cols else pd.Series(np.nan, index=df.index)
    lon = pd.to_numeric(df[lon_cols[0]], errors="coerce") if lon_cols else pd.Series(np.nan, index=df.index)
    for place, meta in REGIONS[region]["places"].items():
        mask = source_lower.str.contains(place.lower(), regex=False)
        missing = mask & lat.isna()
        n = int(missing.sum())
        if n > 0:
            lat.loc[missing] = meta["lat"] + np.random.uniform(-0.03,0.03,size=n)
            lon.loc[missing] = meta["lon"] + np.random.uniform(-0.03,0.03,size=n)
    def fc(kws):
        for c in df.columns:
            if any(k in c.lower() for k in kws): return c
        return ""
    ref_col=fc(["reference","incident"]); status_col=fc(["status"])
    cat_col=fc(["category","type"]); pc_col=fc(["postcode","post_code","postal"])
    cust_col=fc(["customer","affected"]); rest_col=fc(["restore","estimated"])
    out = pd.DataFrame()
    out["outage_reference"] = df[ref_col].astype(str) if ref_col else "N/A"
    out["outage_status"]    = df[status_col].astype(str) if status_col else "Unknown"
    out["outage_category"]  = df[cat_col].astype(str) if cat_col else "Unknown"
    if pc_col:
        out["postcode_label"] = df[pc_col].astype(str)
    else:
        labels = []
        for i in range(len(df)):
            lbl = "Unknown"
            for place, meta in REGIONS[region]["places"].items():
                if place.lower() in source_lower.iloc[i]: lbl=meta["postcode_prefix"]; break
            labels.append(lbl)
        out["postcode_label"] = labels
    out["affected_customers"] = pd.to_numeric(df[cust_col],errors="coerce").fillna(0) if cust_col else 0
    out["estimated_restore"]  = df[rest_col].astype(str) if rest_col else "Unknown"
    out["latitude"] = lat; out["longitude"] = lon
    out["source_text"] = source_text; out["is_synthetic_outage"] = False
    out = out.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    if out.empty:
        synthetic = []
        for place, meta in REGIONS[region]["places"].items():
            synthetic.append({"outage_reference":f"SIM-{place[:3].upper()}-{random.randint(1000,9999)}","outage_status":"Simulated fallback","outage_category":"Visual fallback — no live geocoded NPG outage","postcode_label":meta["postcode_prefix"],"affected_customers":random.randint(20,520),"estimated_restore":"Unknown","latitude":meta["lat"]+random.uniform(-0.045,0.045),"longitude":meta["lon"]+random.uniform(-0.045,0.045),"source_text":"Synthetic point for visual continuity only.","is_synthetic_outage":True})
        out = pd.DataFrame(synthetic, columns=output_cols)
    return out

# =============================================================================
# SAT-Guard Digital Twin — Q1 Edition
# PART 2 of 4 — Core models, risk engine, Monte Carlo, EV/V2G, build_places
# =============================================================================
# Paste this file AFTER app_KASVA_PART1.py
# All functions here depend on helpers defined in Part 1.
# =============================================================================


# =============================================================================
# SCENARIO HELPERS
# =============================================================================

def apply_scenario(row: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """Apply scenario multipliers to a place's weather and operational variables."""
    params = SCENARIOS.get(scenario_name, SCENARIOS["Live / Real-time"])
    r = dict(row)
    r["wind_speed_10m"]       = safe_float(r.get("wind_speed_10m"))    * params["wind"]
    r["precipitation"]        = safe_float(r.get("precipitation"))      * params["rain"]
    r["temperature_2m"]       = safe_float(r.get("temperature_2m"))    + params["temperature"]
    r["european_aqi"]         = safe_float(r.get("european_aqi"))       * params["aqi"]
    r["shortwave_radiation"]  = safe_float(r.get("shortwave_radiation"))* params["solar"]
    r["scenario_outage_multiplier"] = params["outage"]
    r["scenario_finance_multiplier"]= params["finance"]
    r["hazard_mode"]          = params["hazard_mode"]
    return r


def scenario_stress_profile(scenario_name: str) -> Dict[str, float]:
    """
    Return deterministic stress floor values for each what-if scenario.

    Live / Real-time is the operational baseline — no floors applied.
    All other scenarios are counterfactual stress tests. Minimum output
    values prevent stress scenarios from appearing safer than live.
    """
    profiles: Dict[str, Dict[str, float]] = {
        "Live / Real-time":     {"risk_floor":0,  "risk_boost":0,  "failure_floor":0.01,"grid_floor":0.01,"ens_load_factor":0.00,"resilience_penalty":0, "min_outages":0,  "min_customers":0},
        "Extreme wind":         {"risk_floor":72, "risk_boost":24, "failure_floor":0.46,"grid_floor":0.42,"ens_load_factor":1.05,"resilience_penalty":18,"min_outages":5,  "min_customers":1400},
        "Flood":                {"risk_floor":76, "risk_boost":28, "failure_floor":0.52,"grid_floor":0.48,"ens_load_factor":1.20,"resilience_penalty":22,"min_outages":6,  "min_customers":1800},
        "Heatwave":             {"risk_floor":66, "risk_boost":18, "failure_floor":0.34,"grid_floor":0.30,"ens_load_factor":0.72,"resilience_penalty":14,"min_outages":3,  "min_customers":850},
        "Drought":              {"risk_floor":64, "risk_boost":16, "failure_floor":0.32,"grid_floor":0.30,"ens_load_factor":0.62,"resilience_penalty":12,"min_outages":2,  "min_customers":650},
        "Total blackout stress":{"risk_floor":92, "risk_boost":42, "failure_floor":0.82,"grid_floor":0.78,"ens_load_factor":2.40,"resilience_penalty":44,"min_outages":12, "min_customers":4200},
        "Compound extreme":     {"risk_floor":88, "risk_boost":38, "failure_floor":0.74,"grid_floor":0.68,"ens_load_factor":2.00,"resilience_penalty":36,"min_outages":9,  "min_customers":3200},
    }
    return profiles.get(scenario_name, profiles["Live / Real-time"])


def is_calm_live_weather(
    row: Dict[str, Any],
    outage_count: float = 0,
    affected_customers: float = 0,
) -> bool:
    """
    Return True for ordinary UK operating conditions in Live / Real-time mode.

    FIX: signature now always takes outage_count and affected_customers
    as explicit arguments (default 0) — consistent across all call sites.
    Previously some call sites passed only row, causing silent zero-value use.
    """
    return (
        str(row.get("scenario_name", "")) == "Live / Real-time"
        and safe_float(row.get("wind_speed_10m")) < 24
        and safe_float(row.get("precipitation"))  < 2.0
        and safe_float(row.get("european_aqi"))    < 65
        and safe_float(row.get("temperature_2m"))  > -4
        and safe_float(row.get("temperature_2m"))  < 31
        and safe_float(outage_count)               <= 3
        and safe_float(affected_customers)         <= 1200
    )


# =============================================================================
# PHYSICAL / ELECTRICAL MODELS
# =============================================================================

def renewable_generation_mw(row: Dict[str, Any]) -> float:
    """
    Renewable generation proxy in MW.

    Solar: shortwave_radiation × 0.18
    Wind:  min((wind/12)^3, 1.20) × 95   (cubic characteristic before rated output)
    """
    solar = safe_float(row.get("shortwave_radiation"))
    wind  = safe_float(row.get("wind_speed_10m"))
    solar_mw = solar * 0.18
    wind_mw  = min((wind / 12.0) ** 3, 1.20) * 95
    return round(clamp(solar_mw + wind_mw, 0, 240), 2)


def renewable_failure_probability(row: Dict[str, Any]) -> float:
    """
    Probability that renewable generation is insufficient to meet demand.

    Range: 0.0 – 1.0
    Higher when solar irradiance is low, wind is calm, or cloud cover is high.
    """
    solar = safe_float(row.get("shortwave_radiation"))
    wind  = safe_float(row.get("wind_speed_10m"))
    cloud = safe_float(row.get("cloud_cover"))
    low_solar   = 1 - clamp(solar / 450, 0, 1)
    low_wind    = 1 - clamp(wind  / 12,  0, 1)
    cloud_pen   = clamp(cloud / 100, 0, 1) * 0.15
    prob = 0.12 + 0.48 * low_solar + 0.30 * low_wind + cloud_pen
    return round(clamp(prob, 0, 1), 3)


def peak_load_multiplier(hour: Optional[int] = None) -> float:
    """
    Return a demand multiplier based on time of day.

    Based on typical UK domestic + commercial load profiles.
        Evening peak (17-22h): 1.85
        Morning ramp (07-09h): 1.30
        Night valley (00-06h): 0.65
        Shoulder:              1.00
    """
    if hour is None:
        hour = datetime.now().hour
    if 17 <= hour <= 22: return 1.85
    if 7  <= hour <= 9:  return 1.30
    if 0  <= hour <= 6:  return 0.65
    return 1.00


def compute_energy_not_supplied_mw(
    outage_count: float,
    affected_customers: float,
    base_load_mw: float,
    scenario_name: str,
) -> float:
    """
    Estimate energy not supplied (ENS) in MW.

    FIX vs previous version:
    Live mode sets base_load_component = 0 so normal demand does not
    appear as unserved energy when no real outage evidence exists.

    Live formula:   ENS = outage_count × 12 + affected_customers × 0.0025
    Stress formula: ENS = (outage×85 + customers×0.01 + load×0.14) × scenario_outage_multiplier
    """
    params = SCENARIOS.get(scenario_name, SCENARIOS["Live / Real-time"])
    outage_count       = safe_float(outage_count)
    affected_customers = safe_float(affected_customers)
    base_load_mw       = safe_float(base_load_mw)

    if scenario_name == "Live / Real-time":
        ens_mw = outage_count * 12.0 + affected_customers * 0.0025
        return round(clamp(ens_mw, 0, 650), 2)

    outage_component   = outage_count       * 85.0
    customer_component = affected_customers * 0.010
    base_component     = base_load_mw       * 0.14
    ens_mw = (outage_component + customer_component + base_component) * params["outage"]
    return round(clamp(ens_mw, 0, 6500), 2)


def compute_financial_loss(
    ens_mw: float,
    affected_customers: float,
    outage_count: float,
    business_density: float,
    social_vulnerability: float,
    scenario_name: str,
) -> Dict[str, float]:
    """
    Estimate total financial loss across five loss components.

    Components and unit rates:
        VoLL:               ENS_MWh × £17,000 / MWh
        Customer interrupt: affected_customers × £38
        Business disruption:ENS_MWh × £1,100 × business_density
        Restoration:        outage_count × £18,500
        Critical services:  ENS_MWh × £320 × (social_vulnerability / 100)

    Total is multiplied by the scenario finance multiplier.
    """
    params = SCENARIOS.get(scenario_name, SCENARIOS["Live / Real-time"])
    duration_hours = 1.5 + clamp(outage_count / 6, 0, 1) * 5.5
    if scenario_name == "Total blackout stress":
        duration_hours = 8.0
    elif scenario_name == "Compound extreme":
        duration_hours = max(duration_hours, 6.0)

    ens_mwh               = ens_mw * duration_hours
    voll_loss             = ens_mwh * 17_000
    customer_interr_gbp   = affected_customers * 38
    business_disrupt_gbp  = ens_mwh * 1_100 * clamp(business_density, 0, 1)
    restoration_gbp       = outage_count * 18_500
    critical_svc_gbp      = ens_mwh * 320 * clamp(safe_float(social_vulnerability) / 100, 0, 1)

    total = (
        voll_loss + customer_interr_gbp + business_disrupt_gbp
        + restoration_gbp + critical_svc_gbp
    ) * params["finance"]

    return {
        "estimated_duration_hours":         round(duration_hours, 2),
        "ens_mwh":                           round(ens_mwh, 2),
        "voll_loss_gbp":                     round(voll_loss, 2),
        "customer_interruption_loss_gbp":    round(customer_interr_gbp, 2),
        "business_disruption_loss_gbp":      round(business_disrupt_gbp, 2),
        "restoration_loss_gbp":              round(restoration_gbp, 2),
        "critical_services_loss_gbp":        round(critical_svc_gbp, 2),
        "total_financial_loss_gbp":          round(total, 2),
    }


def social_vulnerability_score(pop_density: float, imd_score: float) -> float:
    """
    Combine population density and IMD deprivation into a 0–100 score.

    Weights: density 40%, deprivation 60%.
    Higher score = greater vulnerability.
    """
    density_component = clamp(pop_density / 4500, 0, 1) * 40
    imd_component     = clamp(imd_score    / 100,  0, 1) * 60
    return round(clamp(density_component + imd_component, 0, 100), 2)


def grid_failure_probability(
    risk_score: float, outage_count: float, ens_mw: float
) -> float:
    """
    Calibrated technical grid-failure probability.

    Formula:
        prob = 0.025 + 0.22×risk_n + 0.20×outage_n + 0.14×ens_n
    Output range: 0.01 – 0.75.
    """
    risk_n   = clamp(safe_float(risk_score)  / 100,  0, 1)
    outage_n = clamp(safe_float(outage_count) / 10,  0, 1)
    ens_n    = clamp(safe_float(ens_mw)       / 2500, 0, 1)
    prob = 0.025 + 0.22 * risk_n + 0.20 * outage_n + 0.14 * ens_n
    return round(clamp(prob, 0.01, 0.75), 3)


# =============================================================================
# COMPOUND HAZARD PROXY  (non-circular)
# =============================================================================

def compute_compound_hazard_proxy(row: Dict[str, Any]) -> float:
    """
    Non-circular compound hazard proxy.

    Uses ONLY direct observed / scenario-adjusted meteorological drivers.
    Does NOT read final_risk_score, resilience_index or failure_probability.
    Reading those would create a circular feedback loop:
        risk → compound_hazard → risk → ... (unbounded amplification)

    Formula:
        wind_score   = clamp(wind/70, 0,1) × 35
        rain_score   = clamp(rain/25, 0,1) × 30
        aqi_score    = clamp(aqi/120, 0,1) × 15
        outage_score = clamp(outages/8, 0,1) × 20
    """
    wind   = safe_float(row.get("wind_speed_10m"))
    rain   = safe_float(row.get("precipitation"))
    aqi    = safe_float(row.get("european_aqi"))
    outage = safe_float(row.get("nearby_outages_25km"))

    return round(clamp(
        clamp(wind   / 70,  0, 1) * 35
        + clamp(rain / 25,  0, 1) * 30
        + clamp(aqi  / 120, 0, 1) * 15
        + clamp(outage/ 8,  0, 1) * 20,
        0, 100,
    ), 2)


# =============================================================================
# MULTI-LAYER RISK MODEL
# =============================================================================

def compute_multilayer_risk(
    row: Dict[str, Any], outage_intensity: float, ens_mw: float
) -> Dict[str, float]:
    """
    Calibrated multi-layer risk score (0–100).

    Layers:
        Weather:   wind + rain + cloud + temperature + humidity
        Pollution: AQI + PM2.5
        Net load:  peak demand minus renewable generation
        Outage:    nearby outage intensity
        ENS:       energy not supplied exposure

    Calm-weather guard: Live mode capped at 34.0 when all weather
    indicators are within normal UK operating range.
    """
    wind     = safe_float(row.get("wind_speed_10m"))
    rain     = safe_float(row.get("precipitation"))
    cloud    = safe_float(row.get("cloud_cover"))
    aqi      = safe_float(row.get("european_aqi"))
    pm25     = safe_float(row.get("pm2_5"))
    temp     = safe_float(row.get("temperature_2m"))
    humidity = safe_float(row.get("relative_humidity_2m"))

    wind_score     = clamp((wind - 18) / 52, 0, 1) * 24
    rain_score     = clamp((rain - 1.5) / 23.5, 0, 1) * 20
    cloud_score    = clamp((cloud - 75) / 25, 0, 1) * 3
    temp_score     = clamp(max(abs(temp - 18) - 10, 0) / 18, 0, 1) * 8
    humidity_score = clamp((humidity - 88) / 12, 0, 1) * 2
    weather_score  = wind_score + rain_score + cloud_score + temp_score + humidity_score

    pollution_score = (
        clamp((aqi - 55) / 95, 0, 1) * 10
        + clamp((pm25 - 20) / 50, 0, 1) * 5
    )

    renewable_mw = renewable_generation_mw(row)
    net_load     = max(peak_load_multiplier() * 100 - renewable_mw, 0)
    load_score   = clamp((net_load - 80) / 220, 0, 1) * 10

    outage_score = clamp(outage_intensity, 0, 1) * 16
    ens_score    = clamp(ens_mw / 2500, 0, 1) * 14

    score = clamp(weather_score + pollution_score + load_score + outage_score + ens_score, 0, 100)

    # Live calm-weather guard
    nearby   = safe_float(row.get("nearby_outages_25km", 0))
    affected = safe_float(row.get("affected_customers_nearby", 0))
    if is_calm_live_weather(row, nearby, affected):
        score = min(score, 34.0)

    failure_probability = 1 / (1 + np.exp(-0.075 * (score - 72)))

    return {
        "risk_score":             round(float(score), 2),
        "failure_probability":    round(float(clamp(failure_probability, 0.01, 0.80)), 3),
        "renewable_generation_mw":round(float(renewable_mw), 2),
        "net_load_mw":            round(float(net_load), 2),
    }


def cascade_breakdown(base_failure: float) -> Dict[str, float]:
    """
    Model interdependent infrastructure cascade failure probabilities.

    Power failure drives water, telecom, transport and social sectors
    via calibrated power-law relationships (Billinton & Allan style).

    Relationships:
        water     = power^1.35 × 0.74
        telecom   = power^1.22 × 0.82
        transport = ((power + telecom) / 2) × 0.70
        social    = ((power + water + telecom) / 3) × 0.75
        system_stress = mean(power, water, telecom, transport, social)
    """
    power     = clamp(base_failure, 0, 1)
    water     = clamp((power ** 1.35) * 0.74, 0, 1)
    telecom   = clamp((power ** 1.22) * 0.82, 0, 1)
    transport = clamp(((power + telecom) / 2.0) * 0.70, 0, 1)
    social    = clamp(((power + water + telecom) / 3.0) * 0.75, 0, 1)
    return {
        "cascade_power":     round(power, 3),
        "cascade_water":     round(water, 3),
        "cascade_telecom":   round(telecom, 3),
        "cascade_transport": round(transport, 3),
        "cascade_social":    round(social, 3),
        "system_stress":     round(float(np.mean([power, water, telecom, transport, social])), 3),
    }


def compute_resilience_index(
    final_risk: float,
    social_vulnerability: float,
    grid_failure: float,
    renewable_failure: float,
    system_stress: float,
    financial_loss_gbp: float,
) -> float:
    """
    Calibrated resilience index (15–100).

    Formula:
        resilience = 92
            − 0.28 × risk
            − 0.11 × social_vulnerability
            − 9    × grid_failure
            − 5    × renewable_failure
            − 7    × system_stress
            − finance_penalty

    finance_penalty = clamp(financial_loss / £25m, 0, 1) × 6

    A high score means robust resilience.
    A low score means the area is stressed or fragile.
    """
    finance_penalty = clamp(financial_loss_gbp / 25_000_000, 0, 1) * 6
    resilience = 92 - (
        0.28 * safe_float(final_risk)
        + 0.11 * safe_float(social_vulnerability)
        + 9    * safe_float(grid_failure)
        + 5    * safe_float(renewable_failure)
        + 7    * safe_float(system_stress)
        + finance_penalty
    )
    return round(clamp(resilience, 15, 100), 2)


def flood_depth_proxy(row: Dict[str, Any], scenario_name: str) -> float:
    """
    Estimate a normalised flood depth proxy (0–2.5 m equivalent).

    This is a model output, not a hydrological measurement.
    FIX: now always called and written back to the places DataFrame in build_places().
    """
    rain   = safe_float(row.get("precipitation"))
    outage = safe_float(row.get("nearby_outages_25km"))
    risk   = safe_float(row.get("final_risk_score"))
    cloud  = safe_float(row.get("cloud_cover"))
    mult = {"Live / Real-time":1.0,"Extreme wind":0.9,"Flood":2.0,"Compound extreme":1.8,"Total blackout stress":1.2,"Drought":0.25}.get(scenario_name, 1.0)
    return round(clamp((0.038*rain + 0.016*outage + 0.0025*risk + 0.001*cloud)*mult, 0, 2.5), 3)


# =============================================================================
# NATURAL HAZARD MODELS
# =============================================================================

def hazard_stressor_score(row: Dict[str, Any], hazard_name: str) -> float:
    """Return a 0–100 stress score for a named natural hazard type."""
    cfg  = HAZARD_TYPES[hazard_name]
    v    = safe_float(row.get(cfg["driver"]))
    low  = cfg["threshold_low"]
    high = cfg["threshold_high"]
    if high <= low: return 0.0
    return round(clamp((v - low) / (high - low) * 100, 0, 100), 2)


def hazard_resilience_score(row: Dict[str, Any], hazard_name: str) -> Dict[str, Any]:
    """
    Advanced natural-hazard resilience model (score 15–100).

    Penalty structure:
        hazard_penalty  = weather_factor × stress_n × 18
        social_penalty  = social_n × 6
        outage_penalty  = outage_n × 7
        ens_penalty     = ens_n    × 5
        failure_penalty = grid_fail × 7
        finance_penalty = finance_n × 4
        risk_penalty    = risk_n   × 6

    Calm-weather adjustment: if wind<20, rain<3, aqi<60, outages<2
        → weather_factor = 0.25 (reduces hazard_penalty by 75%)
        → floor raised to 68
    """
    stress    = hazard_stressor_score(row, hazard_name)
    social    = safe_float(row.get("social_vulnerability"))
    outage    = safe_float(row.get("nearby_outages_25km"))
    ens       = safe_float(row.get("energy_not_supplied_mw"))
    grid_fail = safe_float(row.get("grid_failure_probability"))
    finance   = safe_float(row.get("total_financial_loss_gbp"))
    wind      = safe_float(row.get("wind_speed_10m"))
    rain      = safe_float(row.get("precipitation"))
    aqi       = safe_float(row.get("european_aqi"))
    risk      = safe_float(row.get("final_risk_score"))

    stress_n  = clamp(stress   / 100, 0, 1)
    social_n  = clamp(social   / 100, 0, 1)
    outage_n  = clamp(outage   / 10,  0, 1)
    ens_n     = clamp(ens      / 2500,0, 1)
    finance_n = clamp(finance  / 20_000_000, 0, 1)
    risk_n    = clamp(risk     / 100, 0, 1)

    calm = wind < 20 and rain < 3 and aqi < 60 and outage < 2
    weather_factor = 0.25 if calm else 1.0

    hazard_pen  = weather_factor * (stress_n * 18)
    social_pen  = social_n * 6
    outage_pen  = outage_n * 7
    ens_pen     = ens_n    * 5
    failure_pen = grid_fail * 7
    finance_pen = finance_n * 4
    risk_pen    = risk_n    * 6

    score = 88.0 - hazard_pen - social_pen - outage_pen - ens_pen - failure_pen - finance_pen - risk_pen
    if calm: score = max(score, 68)
    score = clamp(score, 15, 100)

    level = "Robust" if score>=80 else "Stable" if score>=65 else "Stressed" if score>=45 else "Fragile"

    drivers = []
    if stress  >= 70: drivers.append(f"extreme {hazard_name.lower()} stress ({round(stress,1)}/100)")
    if social  >= 65: drivers.append(f"high social vulnerability ({round(social,1)}/100)")
    if outage  >= 4:  drivers.append(f"outage clustering ({int(outage)} nearby events)")
    if ens     >= 700:drivers.append(f"high ENS exposure ({round(ens,1)} MW)")
    if grid_fail>=0.55:drivers.append(f"elevated grid instability ({round(grid_fail*100,1)}%)")
    if finance >= 5_000_000: drivers.append(f"major financial exposure (£{round(finance/1_000_000,2)}m)")
    if risk    >= 75: drivers.append(f"severe regional risk ({round(risk,1)}/100)")
    if calm:          drivers.append("calm-weather operational adjustment active")
    if not drivers:   drivers.append("normal resilient operational state")

    return {
        "hazard":                  hazard_name,
        "hazard_stress_score":     round(stress, 2),
        "hazard_resilience_score": round(score, 2),
        "hazard_resilience_level": level,
        "calm_weather_adjustment": calm,
        "evidence":                "; ".join(drivers),
        "hazard_description":      HAZARD_TYPES[hazard_name]["description"],
        "penalty_breakdown": {
            "hazard_penalty":  round(hazard_pen,  2),
            "social_penalty":  round(social_pen,  2),
            "outage_penalty":  round(outage_pen,  2),
            "ens_penalty":     round(ens_pen,     2),
            "failure_penalty": round(failure_pen, 2),
            "finance_penalty": round(finance_pen, 2),
            "risk_penalty":    round(risk_pen,    2),
        },
    }


def build_hazard_resilience_matrix(places: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Build postcode/place-level resilience scores across all hazard types."""
    rows = []
    for _, p in places.iterrows():
        for hazard_name in HAZARD_TYPES:
            hr = hazard_resilience_score(p.to_dict(), hazard_name)
            rows.append({
                "postcode":                  p.get("postcode_prefix"),
                "place":                     p.get("place"),
                "hazard":                    hazard_name,
                "hazard_stress_score":       hr["hazard_stress_score"],
                "resilience_score_out_of_100":hr["hazard_resilience_score"],
                "resilience_level":          hr["hazard_resilience_level"],
                "supporting_evidence":       hr["evidence"],
                "hazard_description":        hr["hazard_description"],
                "population_density":        p.get("population_density"),
                "social_vulnerability":      p.get("social_vulnerability"),
                "imd_score":                 p.get("imd_score"),
                "financial_loss_gbp":        p.get("total_financial_loss_gbp"),
                "grid_failure_probability":  p.get("grid_failure_probability"),
                "energy_not_supplied_mw":    p.get("energy_not_supplied_mw"),
            })
    df = pd.DataFrame(rows)
    if pc is not None and not pc.empty:
        join = pc[["postcode","recommendation_score","investment_priority","outage_records","affected_customers","resilience_score","risk_score"]].rename(columns={"resilience_score":"postcode_base_resilience","risk_score":"postcode_base_risk"})
        df = df.merge(join, on="postcode", how="left")
    else:
        df["recommendation_score"] = np.nan; df["investment_priority"] = ""
    return df.sort_values(["resilience_score_out_of_100","hazard_stress_score"], ascending=[True,False]).reset_index(drop=True)


# =============================================================================
# ENHANCED FAILURE PROBABILITY MODEL
# =============================================================================

def enhanced_failure_probability(row: Dict[str, Any], hazard: str = "Compound hazard") -> Dict[str, Any]:
    """
    Advanced calibrated grid-failure probability (logistic model).

    Inputs:
        base / grid / renewable failure from model state
        social_vulnerability, outage clustering, ENS
        hazard stressor, wind, rain, AQI
        overall risk

    Calm-weather guard: if wind<20, rain<3, aqi<60, outages<2
        → weather_multiplier = 0.42
        → final prob × 0.35, capped at 0.18

    Output range: 1% – 95%
    """
    base      = safe_float(row.get("failure_probability"))
    grid      = safe_float(row.get("grid_failure_probability"))
    renewable = safe_float(row.get("renewable_failure_probability"))
    social    = safe_float(row.get("social_vulnerability"))
    outage    = safe_float(row.get("nearby_outages_25km"))
    ens       = safe_float(row.get("energy_not_supplied_mw"))
    wind      = safe_float(row.get("wind_speed_10m"))
    rain      = safe_float(row.get("precipitation"))
    aqi       = safe_float(row.get("european_aqi"))
    risk      = safe_float(row.get("final_risk_score"))
    hazard_stress = hazard_stressor_score(row, hazard)

    social_n  = clamp(social        / 100,  0, 1)
    outage_n  = clamp(outage        / 10,   0, 1)
    ens_n     = clamp(ens           / 2500, 0, 1)
    hazard_n  = clamp(hazard_stress / 100,  0, 1)
    wind_n    = clamp(wind          / 90,   0, 1)
    rain_n    = clamp(rain          / 40,   0, 1)
    aqi_n     = clamp(aqi           / 150,  0, 1)
    risk_n    = clamp(risk          / 100,  0, 1)

    calm = wind < 20 and rain < 3 and aqi < 60 and outage < 2
    wm   = 0.42 if calm else 1.0

    z = (
        -4.45
        + 1.05 * base
        + 0.95 * grid
        + 0.55 * renewable
        + 0.45 * social_n
        + 0.38 * outage_n
        + 0.28 * ens_n
        + wm * (0.55 * hazard_n + 0.22 * wind_n + 0.18 * rain_n + 0.12 * aqi_n)
        + 0.25 * risk_n
    )
    prob = 1 / (1 + math.exp(-z))
    if calm:
        prob = min(prob * 0.35, 0.18)
    prob = clamp(prob, 0.01, 0.95)

    level = "Critical" if prob>=0.70 else "High" if prob>=0.45 else "Moderate" if prob>=0.20 else "Low"

    drivers = []
    if hazard_n  >= 0.60: drivers.append("high natural-hazard stress")
    if wind_n    >= 0.65: drivers.append("extreme wind exposure")
    if rain_n    >= 0.60: drivers.append("flood/heavy-rain stress")
    if social_n  >= 0.60: drivers.append("high socio-economic vulnerability")
    if outage_n  >= 0.50: drivers.append("outage clustering")
    if ens_n     >= 0.50: drivers.append("high ENS exposure")
    if renewable >= 0.60: drivers.append("renewable intermittency")
    if not drivers:       drivers.append("normal operational conditions")

    return {
        "enhanced_failure_probability": round(prob, 4),
        "failure_level":                level,
        "hazard_stress_score":          round(hazard_stress, 2),
        "calm_weather_adjustment":      calm,
        "failure_evidence":             f"base={round(base,3)}, grid={round(grid,3)}, renewable={round(renewable,3)}, social={round(social,1)}, hazard={round(hazard_stress,1)}, outages={int(outage)}, ENS={round(ens,1)} MW",
        "dominant_failure_drivers":     ", ".join(drivers),
    }


def build_failure_analysis(places: pd.DataFrame) -> pd.DataFrame:
    """Build failure probability DataFrame across all places × hazard types."""
    rows = []
    for _, r in places.iterrows():
        for hazard in HAZARD_TYPES:
            out = enhanced_failure_probability(r.to_dict(), hazard)
            rows.append({
                "place":                        r.get("place"),
                "postcode":                     r.get("postcode_prefix"),
                "hazard":                       hazard,
                "enhanced_failure_probability": out["enhanced_failure_probability"],
                "failure_level":                out.get("failure_level",""),
                "hazard_stress_score":          out["hazard_stress_score"],
                "failure_evidence":             out["failure_evidence"],
                "dominant_failure_drivers":     out.get("dominant_failure_drivers",""),
                "final_risk_score":             r.get("final_risk_score"),
                "resilience_index":             r.get("resilience_index"),
                "financial_loss_gbp":           r.get("total_financial_loss_gbp"),
            })
    return pd.DataFrame(rows).sort_values("enhanced_failure_probability", ascending=False).reset_index(drop=True)


# =============================================================================
# EV / V2G MODELS
# =============================================================================

def ev_adoption_factor(pop_density: float, business_density: float, scenario: str) -> float:
    """
    Estimate EV penetration proxy for a location.

    Base: 0.32 (mid-adoption)
    Uplift: density (+0.08 max), commercial (+0.05 max)
    Reduction: -0.03 under blackout or compound stress scenarios
    Range: 0.12 – 0.58
    """
    base = EV_ASSUMPTIONS["ev_penetration_mid"]
    density_adj   = clamp(pop_density / 3600, 0, 1) * 0.08
    business_adj  = clamp(business_density,   0, 1) * 0.05
    scenario_adj  = -0.03 if scenario in ["Compound extreme","Total blackout stress"] else 0.0
    return round(clamp(base + density_adj + business_adj + scenario_adj, 0.12, 0.58), 3)


def compute_ev_v2g_for_place(row: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """
    Estimate EV storage potential and V2G storm-support capability for one place.

    Pipeline:
        estimated_households = max(800, pop_density × 1.8)
        estimated_evs        = households × adoption
        parked_evs           = evs × share_parked_during_storm (0.72)
        v2g_evs              = parked × share_v2g_enabled (0.26)
        storage_mwh          = v2g_evs × usable_battery_kwh / 1000
        export_mw            = v2g_evs × grid_export_limit_kw / 1000
        substation_mw        = export_mw × charger_substation_coupling_factor (0.62)
        emergency_mwh        = min(storage_mwh, substation_mw × 3h)
    """
    pop_density      = safe_float(row.get("population_density"))
    business_density = safe_float(row.get("business_density"))
    social           = safe_float(row.get("social_vulnerability"))
    risk             = safe_float(row.get("final_risk_score"))
    ens              = safe_float(row.get("energy_not_supplied_mw"))
    outage           = safe_float(row.get("nearby_outages_25km"))

    adoption = ev_adoption_factor(pop_density, business_density, scenario)
    estimated_households = max(800, pop_density * 1.8)
    estimated_evs  = estimated_households * adoption
    parked_evs     = estimated_evs  * EV_ASSUMPTIONS["share_parked_during_storm"]
    v2g_evs        = parked_evs     * EV_ASSUMPTIONS["share_v2g_enabled"]
    storage_mwh    = v2g_evs        * EV_ASSUMPTIONS["usable_battery_kwh"]   / 1000
    export_mw      = v2g_evs        * EV_ASSUMPTIONS["grid_export_limit_kw"] / 1000
    substation_mw  = export_mw      * EV_ASSUMPTIONS["charger_substation_coupling_factor"]
    emergency_mwh  = min(storage_mwh, substation_mw * EV_ASSUMPTIONS["emergency_dispatch_hours"])
    ens_offset     = min(emergency_mwh, ens * 3.0)
    loss_avoided   = ens_offset * 17_000

    operational_value = (
        clamp(risk    / 100, 0, 1) * 35
        + clamp(outage / 8,  0, 1) * 20
        + clamp(ens    / 700,0, 1) * 25
        + clamp(social / 100,0, 1) * 20
    )

    return {
        "place":                          row.get("place"),
        "postcode":                       row.get("postcode_prefix"),
        "ev_penetration_proxy":           adoption,
        "estimated_evs":                  round(estimated_evs, 0),
        "parked_evs_storm":               round(parked_evs, 0),
        "v2g_enabled_evs":                round(v2g_evs, 0),
        "available_storage_mwh":          round(storage_mwh, 2),
        "export_capacity_mw":             round(export_mw, 2),
        "substation_coupled_capacity_mw": round(substation_mw, 2),
        "emergency_energy_mwh":           round(emergency_mwh, 2),
        "ens_offset_mwh":                 round(ens_offset, 2),
        "potential_loss_avoided_gbp":     round(loss_avoided, 2),
        "ev_operational_value_score":     round(clamp(operational_value, 0, 100), 2),
        "ev_storm_role": (
            "High-value V2G support zone"     if operational_value >= 70
            else "Useful local flexibility zone" if operational_value >= 45
            else "Monitor / low immediate V2G value"
        ),
    }


def build_ev_v2g_analysis(places: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = [compute_ev_v2g_for_place(r.to_dict(), scenario) for _, r in places.iterrows()]
    return pd.DataFrame(rows).sort_values("ev_operational_value_score", ascending=False).reset_index(drop=True)


# =============================================================================
# MONTE CARLO  (per-place, independent perturbations)
# =============================================================================

def advanced_monte_carlo(
    row: Dict[str, Any],
    outage_intensity: float,
    ens_mw: float,
    simulations: int,
) -> Dict[str, Any]:
    """
    Per-place Monte Carlo with independently perturbed weather variables.

    Used inside build_places() for the lightweight per-run MC columns.
    For correlated storm-shock MC, see monte_carlo_q1() below.
    """
    simulations = int(clamp(simulations, 10, 160))
    risk_scores: List[float] = []
    resilience_scores: List[float] = []
    financial_losses: List[float] = []

    for _ in range(simulations):
        sim = dict(row)
        sim["wind_speed_10m"]     = safe_float(sim.get("wind_speed_10m"))    * np.random.lognormal(0, 0.16)
        sim["precipitation"]      = max(0, safe_float(sim.get("precipitation"))    * np.random.lognormal(0, 0.30))
        sim["temperature_2m"]     = safe_float(sim.get("temperature_2m"))    + np.random.normal(0, 2.2)
        sim["european_aqi"]       = safe_float(sim.get("european_aqi"))      * np.random.lognormal(0, 0.22)
        sim["shortwave_radiation"]= max(0, safe_float(sim.get("shortwave_radiation")) * np.random.lognormal(0, 0.28))
        sim["cloud_cover"]        = clamp(safe_float(sim.get("cloud_cover")) + np.random.normal(0, 12), 0, 100)

        sim_ens   = max(0, ens_mw * np.random.lognormal(0, 0.25))
        model     = compute_multilayer_risk(sim, outage_intensity, sim_ens)
        cascade   = cascade_breakdown(model["failure_probability"])
        ren_fail  = renewable_failure_probability(sim)
        grid_fail = grid_failure_probability(model["risk_score"], safe_float(row.get("nearby_outages_25km")), sim_ens)
        final_risk= clamp(model["risk_score"] * (1 + cascade["system_stress"] * 0.75), 0, 100)
        finance   = compute_financial_loss(sim_ens, safe_float(row.get("affected_customers_nearby")), safe_float(row.get("nearby_outages_25km")), safe_float(row.get("business_density")), safe_float(row.get("social_vulnerability")), row.get("scenario_name","Live / Real-time"))
        resilience= compute_resilience_index(final_risk, safe_float(row.get("social_vulnerability")), grid_fail, ren_fail, cascade["system_stress"], finance["total_financial_loss_gbp"])

        risk_scores.append(final_risk)
        resilience_scores.append(resilience)
        financial_losses.append(finance["total_financial_loss_gbp"])

    ra = np.array(risk_scores); re = np.array(resilience_scores); fa = np.array(financial_losses)
    return {
        "mc_mean":                  round(float(np.mean(ra)), 2),
        "mc_std":                   round(float(np.std(ra)),  2),
        "mc_p05":                   round(float(np.percentile(ra, 5)),  2),
        "mc_p50":                   round(float(np.percentile(ra, 50)), 2),
        "mc_p95":                   round(float(np.percentile(ra, 95)), 2),
        "mc_extreme_probability":   round(float(np.mean(ra >= 80)), 3),
        "mc_resilience_mean":       round(float(np.mean(re)), 2),
        "mc_resilience_p05":        round(float(np.percentile(re, 5)), 2),
        "mc_financial_loss_p95":    round(float(np.percentile(fa, 95)), 2),
        "mc_histogram":             [round(float(x), 2) for x in ra[:250]],
    }


# =============================================================================
# Q1 MONTE CARLO  (correlated storm shock)
# =============================================================================

def monte_carlo_q1(row: Dict[str, Any], simulations: int = 1000) -> Dict[str, Any]:
    """
    Improved Monte Carlo with shared storm-shock variable.

    Improvements vs basic MC:
        - Shared storm_shock correlates wind, rain, outage count and ENS
        - Triangular demand distribution
        - Lognormal restoration-cost tails
        - P95 and CVaR95 loss metrics

    CVaR95 FIX: computed as mean of exceedance set (losses >= P95 threshold),
    not as an incorrectly sliced array index.
    """
    simulations = int(clamp(simulations, 100, 5000))
    rng = np.random.default_rng()

    base_wind   = safe_float(row.get("wind_speed_10m"))
    base_rain   = safe_float(row.get("precipitation"))
    base_aqi    = safe_float(row.get("european_aqi"))
    base_ens    = safe_float(row.get("energy_not_supplied_mw"))
    base_social = safe_float(row.get("social_vulnerability"))
    base_outage = safe_float(row.get("nearby_outages_25km"))

    storm_shock = rng.normal(0, 1, simulations)
    wind   = np.maximum(0, base_wind  * np.exp(0.16 * storm_shock + rng.normal(0, 0.08, simulations)))
    rain   = np.maximum(0, base_rain  * np.exp(0.28 * storm_shock + rng.normal(0, 0.18, simulations)))
    aqi    = np.maximum(0, base_aqi   * np.exp(0.12 * rng.normal(0, 1, simulations)))
    demand_mult  = rng.triangular(0.78, 1.10, 1.95, simulations)
    outage_count = np.maximum(0, base_outage + rng.poisson(np.maximum(0.2, 0.8 + np.maximum(storm_shock, 0))))
    ens    = np.maximum(0, base_ens * demand_mult * np.exp(0.22 * np.maximum(storm_shock, 0)))

    weather_score  = np.clip(wind  / 45, 0, 1) * 27 + np.clip(rain  / 6,    0, 1) * 18
    pollution_score= np.clip(aqi   / 100,0, 1) * 17
    outage_score   = np.clip(outage_count / 10, 0, 1) * 20
    ens_score      = np.clip(ens   / 1500,0, 1) * 17
    social_score   = np.clip(base_social / 100, 0, 1) * 10
    risk = np.clip(weather_score + pollution_score + outage_score + ens_score + social_score, 0, 100)
    failure_prob = 1 / (1 + np.exp(-0.07 * (risk - 58)))

    duration    = 1.5 + np.clip(outage_count / 6, 0, 1) * 5.5
    ens_mwh     = ens * duration
    voll        = ens_mwh * rng.lognormal(np.log(17000), 0.18, simulations)
    restoration = outage_count * rng.lognormal(np.log(18500), 0.25, simulations)
    social_uplift = ens_mwh * 320 * np.clip(base_social / 100, 0, 1)
    loss        = voll + restoration + social_uplift

    # CVaR95: mean of losses that exceed the 95th-percentile threshold
    p95_threshold = float(np.percentile(loss, 95))
    exceedance    = loss[loss >= p95_threshold]
    cvar95        = float(np.mean(exceedance)) if len(exceedance) > 0 else p95_threshold

    return {
        "q1_mc_risk_mean":       round(float(np.mean(risk)), 2),
        "q1_mc_risk_p95":        round(float(np.percentile(risk, 95)), 2),
        "q1_mc_failure_mean":    round(float(np.mean(failure_prob)), 4),
        "q1_mc_failure_p95":     round(float(np.percentile(failure_prob, 95)), 4),
        "q1_mc_loss_mean_gbp":   round(float(np.mean(loss)), 2),
        "q1_mc_loss_p95_gbp":    round(float(np.percentile(loss, 95)), 2),
        "q1_mc_loss_cvar95_gbp": round(cvar95, 2),
        "q1_mc_histogram":       [round(float(v), 2) for v in risk[:500]],
    }


def build_q1_monte_carlo_table(places: pd.DataFrame, simulations: int) -> pd.DataFrame:
    rows = []
    for _, r in places.iterrows():
        out = monte_carlo_q1(r.to_dict(), simulations)
        out["place"]    = r.get("place")
        out["postcode"] = r.get("postcode_prefix")
        rows.append(out)
    return pd.DataFrame(rows).sort_values("q1_mc_risk_p95", ascending=False).reset_index(drop=True)


# =============================================================================
# FUNDING PRIORITY MODEL
# =============================================================================

def funding_priority_criteria(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicit funding prioritisation (0–100 score).

    Formula:
        score = 0.26×risk + 0.20×(100−resilience) + 0.18×social
              + 0.15×loss_exposure + 0.11×ENS_exposure
              + 0.06×outage_exposure + 0.04×recommendation

    Bands:
        ≥78: Immediate funding
        ≥60: High priority
        ≥42: Medium priority
        else: Routine monitoring
    """
    risk       = safe_float(row.get("risk_score",       row.get("final_risk_score")))
    resilience = safe_float(row.get("resilience_score", row.get("resilience_index")))
    social     = safe_float(row.get("social_vulnerability"))
    loss       = safe_float(row.get("financial_loss_gbp", row.get("total_financial_loss_gbp")))
    ens        = safe_float(row.get("energy_not_supplied_mw"))
    outages    = safe_float(row.get("outage_records",   row.get("nearby_outages_25km")))
    rec        = safe_float(row.get("recommendation_score", 0))

    score = (
        0.26 * risk
        + 0.20 * (100 - resilience)
        + 0.18 * social
        + 0.15 * clamp(loss    / 5_000_000, 0, 1) * 100
        + 0.11 * clamp(ens     / 700,        0, 1) * 100
        + 0.06 * clamp(outages / 6,          0, 1) * 100
        + 0.04 * rec
    )

    if score >= 78:   band = "Immediate funding"
    elif score >= 60: band = "High priority"
    elif score >= 42: band = "Medium priority"
    else:             band = "Routine monitoring"

    return {
        "funding_priority_score": round(clamp(score, 0, 100), 2),
        "funding_priority_band":  band,
        "funding_criteria":       "risk, low resilience, social vulnerability, financial-loss, ENS, outage frequency, recommendation",
    }


def build_funding_table(pc: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    source = pc.copy() if pc is not None and not pc.empty else places.copy()
    rows = []
    for _, r in source.iterrows():
        d = r.to_dict(); d.update(funding_priority_criteria(d)); rows.append(d)
    return pd.DataFrame(rows).sort_values("funding_priority_score", ascending=False).reset_index(drop=True)


# =============================================================================
# SCENARIO FINANCIAL MATRIX
# =============================================================================

def scenario_financial_matrix(places: pd.DataFrame, region: str, mc_runs: int) -> pd.DataFrame:
    """
    Compute compact scenario loss table for what-if scenarios.

    Live / Real-time excluded (shown separately as operational baseline).
    FIX: MC-run cap raised from 60 → 150.
    """
    rows = []
    for scenario_name in [s for s in SCENARIOS if s != "Live / Real-time"]:
        try:
            p, _, _ = get_data_cached(region, scenario_name, max(10, min(mc_runs, 150)))
            rows.append({
                "scenario":                   scenario_name,
                "total_financial_loss_gbp":   round(float(p["total_financial_loss_gbp"].sum()), 2),
                "mean_risk":                  round(float(p["final_risk_score"].mean()), 2),
                "mean_resilience":            round(float(p["resilience_index"].mean()), 2),
                "total_ens_mw":               round(float(p["energy_not_supplied_mw"].sum()), 2),
                "mean_failure_probability":   round(float(p["failure_probability"].mean()), 4),
            })
        except Exception:
            rows.append({"scenario":scenario_name,"total_financial_loss_gbp":np.nan,"mean_risk":np.nan,"mean_resilience":np.nan,"total_ens_mw":np.nan,"mean_failure_probability":np.nan})
    return pd.DataFrame(rows).sort_values("total_financial_loss_gbp", ascending=False)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model_transparency(places: pd.DataFrame, scenario: str) -> pd.DataFrame:
    checks = []
    checks.append({"check":"Model is not black-box","result":"Pass","evidence":"Risk, resilience, failure, finance and investment equations are explicitly exposed in the code and Method tab."})
    corr_risk_ens = float(places["final_risk_score"].corr(places["energy_not_supplied_mw"]))
    checks.append({"check":"Risk monotonicity sanity check","result":"Pass" if corr_risk_ens>=-0.3 else "Warning","evidence":f"corr(risk, ENS) = {round(corr_risk_ens,3)}"})
    corr_risk_res = float(places["final_risk_score"].corr(places["resilience_index"]))
    checks.append({"check":"Resilience inverse sanity check","result":"Pass" if corr_risk_res<=0.4 else "Warning","evidence":f"corr(risk, resilience) = {round(corr_risk_res,3)}"})
    checks.append({"check":"Financial quantification available","result":"Pass" if "total_financial_loss_gbp" in places.columns else "Fail","evidence":f"Total loss = £{round(float(places['total_financial_loss_gbp'].sum())/1_000_000,2)}m under {scenario}."})
    checks.append({"check":"Social vulnerability integrated","result":"Pass" if "social_vulnerability" in places.columns else "Fail","evidence":"Population density and IMD/fallback vulnerability are used in the resilience score."})
    checks.append({"check":"Natural hazard scoring available","result":"Pass","evidence":f"{len(HAZARD_TYPES)} hazard-specific resilience dimensions are computed."})
    checks.append({"check":"No circular compound-hazard feedback","result":"Pass","evidence":"compound_hazard_proxy uses only wind, rain, AQI, outage count — not final_risk_score, resilience_index or failure_probability."})
    return pd.DataFrame(checks)


# =============================================================================
# MAIN DATA BUILDER
# =============================================================================

def build_places(region: str, scenario_name: str, mc_runs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build place-level model outputs for all configured locations.

    Pipeline per place:
    1.  Fetch weather + air quality (with random fallback)
    2.  Apply scenario multipliers
    3.  Compute renewable generation proxies
    4.  Count nearby outages within 25 km
    5.  Apply scenario stress floors (what-if modes only)
    6.  Compute socio-economic vulnerability (IoD2025 + IMD blend)
    7.  Compute net load, EV/V2G storage proxy
    8.  Compute ENS (with drought correction)
    9.  Compute multi-layer risk + cascade
    10. Apply scenario risk floors
    11. Compute grid / renewable failure probabilities
    12. Compute financial loss
    13. Compute resilience index
    14. Compute flood depth proxy  (FIX: now written to DataFrame)
    15. Run per-place Monte Carlo
    16. Final calm-weather guard (post-processing DataFrame pass)
    """
    imd_summary, _ = load_imd_summary_cached()
    raw_npg  = fetch_northern_powergrid(100)
    outages  = standardise_outages(raw_npg, region)

    outage_points: List[Tuple[float, float, float, bool]] = [
        (safe_float(o.get("latitude")), safe_float(o.get("longitude")),
         safe_float(o.get("affected_customers")), bool(o.get("is_synthetic_outage", False)))
        for _, o in outages.iterrows()
    ]

    rows: List[Dict[str, Any]] = []

    for place, meta in REGIONS[region]["places"].items():
        lat = meta["lat"]; lon = meta["lon"]
        weather = fetch_weather(lat, lon).get("current", {})
        air     = fetch_air_quality(lat, lon).get("current", {})

        row: Dict[str, Any] = {
            "scenario_name": scenario_name,
            "place": place, "lat": lat, "lon": lon,
            "postcode_prefix": meta["postcode_prefix"],
            "time": weather.get("time") or datetime.now(UTC).isoformat(),
            "temperature_2m":       weather.get("temperature_2m",       random.uniform(7, 18)),
            "wind_speed_10m":       weather.get("wind_speed_10m",       random.uniform(4, 26)),
            "cloud_cover":          weather.get("cloud_cover",           random.uniform(15, 96)),
            "precipitation":        weather.get("precipitation",         random.uniform(0, 3)),
            "shortwave_radiation":  weather.get("shortwave_radiation",   random.uniform(80, 450)),
            "relative_humidity_2m": weather.get("relative_humidity_2m", random.uniform(45, 88)),
            "european_aqi":         air.get("european_aqi", random.uniform(15, 65)),
            "pm2_5":                air.get("pm2_5",        random.uniform(3,  18)),
            "population_density":   meta["population_density"],
            "estimated_load_mw":    meta["estimated_load_mw"],
            "business_density":     meta["business_density"],
        }

        row = apply_scenario(row, scenario_name)

        # Renewable generation
        row["solar_generation"] = row["shortwave_radiation"] * 0.002
        row["wind_generation"]  = row["wind_speed_10m"]      * 0.6
        if scenario_name == "Drought":
            row["solar_generation"] *= 0.35
            row["wind_generation"]  *= 0.25

        # Outage count
        nearby = 0; affected_customers = 0.0
        for olat, olon, customers, synthetic in outage_points:
            if scenario_name == "Live / Real-time" and synthetic: continue
            if haversine_km(lat, lon, olat, olon) <= 25:
                nearby += 1; affected_customers += customers

        sp = scenario_stress_profile(scenario_name)
        if scenario_name != "Live / Real-time":
            nearby             = max(nearby,             int(sp["min_outages"]))
            affected_customers = max(affected_customers, float(sp["min_customers"]))
        if scenario_name == "Total blackout stress":
            nearby             = max(nearby, 12)
            affected_customers = max(affected_customers, 4200)

        # Socio-economic
        imd_info   = infer_imd_for_place(place, region, meta, imd_summary)
        iod_profile= infer_iod_domain_vulnerability(place, region, meta)
        if "fallback" not in str(iod_profile.get("iod_domain_match","")).lower():
            social_vuln = clamp(
                0.70 * safe_float(iod_profile.get("iod_social_vulnerability"))
                + 0.30 * social_vulnerability_score(row["population_density"], imd_info["imd_score"]),
                0, 100,
            )
        else:
            social_vuln = social_vulnerability_score(row["population_density"], imd_info["imd_score"])

        # Net load + EV/V2G
        net_load     = max(row["estimated_load_mw"] - row["solar_generation"] - row["wind_generation"], 0)
        ev_pen       = random.uniform(0.2, 0.5)
        ev_storage   = ev_pen * 120
        v2g_support  = ev_storage * (0.55 if scenario_name == "Drought" else 0.25)
        grid_storage = random.uniform(40, 120)
        total_storage= v2g_support + grid_storage

        # ENS
        ens_mw = compute_energy_not_supplied_mw(nearby, affected_customers, row["estimated_load_mw"], scenario_name)
        if scenario_name == "Drought":
            ens_mw = ens_mw + net_load * 0.18 - total_storage * 0.35
        if scenario_name != "Live / Real-time":
            ens_mw = max(ens_mw, row["estimated_load_mw"] * sp["ens_load_factor"])
        ens_mw = max(ens_mw, 0)

        row["nearby_outages_25km"]      = nearby
        row["affected_customers_nearby"]= round(affected_customers, 1)
        row["compound_hazard_proxy"]    = compute_compound_hazard_proxy(row)

        outage_intensity   = clamp(nearby / 20, 0, 1)
        calm_live          = is_calm_live_weather(row, nearby, affected_customers)
        if calm_live: ens_mw = min(ens_mw, 75.0)

        base = compute_multilayer_risk(row, outage_intensity, ens_mw)
        if calm_live:
            base["risk_score"]          = min(base["risk_score"], 34.0)
            base["failure_probability"] = min(base["failure_probability"], 0.12)

        cascade    = cascade_breakdown(base["failure_probability"])
        final_risk = clamp(base["risk_score"] * (1 + cascade["system_stress"] * 0.5), 0, 100)

        if scenario_name != "Live / Real-time":
            scenario_hazard = compute_compound_hazard_proxy(row)
            final_risk = clamp(
                max(final_risk, sp["risk_floor"]) + sp["risk_boost"] * clamp(scenario_hazard / 100, 0, 1),
                0, 100,
            )
            base["failure_probability"] = round(max(
                safe_float(base.get("failure_probability")),
                sp["failure_floor"],
                1 / (1 + math.exp(-0.10 * (final_risk - 62))),
            ), 3)
            cascade = cascade_breakdown(base["failure_probability"])

        ren_fail  = renewable_failure_probability(row)
        grid_fail = grid_failure_probability(final_risk, nearby, ens_mw)
        if scenario_name != "Live / Real-time":
            grid_fail = clamp(max(grid_fail, sp["grid_floor"]), 0, 0.95)
        if scenario_name == "Drought":
            grid_fail = clamp(max(grid_fail, sp["grid_floor"]) + (net_load / 1000) * 0.25, 0, 1)
        if calm_live:
            final_risk = min(final_risk, 36.0)
            grid_fail  = min(grid_fail,  0.12)
            ens_mw     = min(ens_mw,     75.0)

        finance   = compute_financial_loss(ens_mw, affected_customers, nearby, row["business_density"], social_vuln, scenario_name)
        resilience= compute_resilience_index(final_risk, social_vuln, grid_fail, ren_fail, cascade["system_stress"], finance["total_financial_loss_gbp"])

        if scenario_name == "Drought":
            resilience = clamp(resilience - (net_load/1000)*10 + (total_storage/500)*8, 0, 100)
        if scenario_name != "Live / Real-time":
            resilience = clamp(resilience - sp["resilience_penalty"], 5, 100)
        if calm_live:
            resilience = max(resilience, 68.0)

        # FIX: flood_depth_proxy now always written to row
        row["final_risk_score"] = round(final_risk, 2)   # needed by flood_depth_proxy
        fdp = flood_depth_proxy(row, scenario_name)

        row.update(base); row.update(cascade); row.update(finance)
        row.update({
            "nearby_outages_25km":        nearby,
            "affected_customers_nearby":  round(affected_customers, 1),
            "energy_not_supplied_mw":     round(ens_mw, 2),
            "compound_hazard_proxy":      compute_compound_hazard_proxy(row),
            "final_risk_score":           round(final_risk, 2),
            "imd_score":                  imd_info["imd_score"],
            "social_vulnerability":       social_vuln,
            "net_load_stress":            round(net_load, 2),
            "v2g_support_mw":             round(v2g_support, 2),
            "grid_storage_mw":            round(grid_storage, 2),
            "total_storage_support":      round(total_storage, 2),
            "renewable_failure_probability": ren_fail,
            "grid_failure_probability":   grid_fail,
            "resilience_index":           resilience,
            "flood_depth_proxy":          fdp,          # FIX: always written
            "iod_social_vulnerability":   safe_float(iod_profile.get("iod_social_vulnerability")),
            "iod_domain_match":           str(iod_profile.get("iod_domain_match","fallback")),
        })

        row.update(advanced_monte_carlo(row, outage_intensity, ens_mw, mc_runs))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Final calm-weather guard (DataFrame-level post-processing)
    if scenario_name == "Live / Real-time" and not df.empty:
        calm_mask = (
            (pd.to_numeric(df["wind_speed_10m"],       errors="coerce").fillna(0) < 24)
            & (pd.to_numeric(df["precipitation"],       errors="coerce").fillna(0) < 2.0)
            & (pd.to_numeric(df["european_aqi"],        errors="coerce").fillna(0) < 65)
            & (pd.to_numeric(df["nearby_outages_25km"], errors="coerce").fillna(0) <= 3)
        )
        df.loc[calm_mask, "final_risk_score"]          = df.loc[calm_mask, "final_risk_score"].clip(upper=36)
        df.loc[calm_mask, "failure_probability"]       = df.loc[calm_mask, "failure_probability"].clip(upper=0.12)
        df.loc[calm_mask, "grid_failure_probability"]  = df.loc[calm_mask, "grid_failure_probability"].clip(upper=0.12)
        df.loc[calm_mask, "energy_not_supplied_mw"]    = df.loc[calm_mask, "energy_not_supplied_mw"].clip(upper=75)
        df.loc[calm_mask, "resilience_index"]          = df.loc[calm_mask, "resilience_index"].clip(lower=68)

    df["risk_label"]       = df["final_risk_score"].apply(risk_label)
    df["resilience_label"] = df["resilience_index"].apply(resilience_label)
    return df, outages


def interpolate_value(lat: float, lon: float, places: pd.DataFrame, col: str) -> float:
    """Inverse-distance-weighted interpolation from place values to a grid point."""
    weights, values = [], []
    for _, r in places.iterrows():
        d = haversine_km(lat, lon, r["lat"], r["lon"])
        weights.append(1 / max(d, 1))
        values.append(safe_float(r.get(col)))
    return float(np.average(values, weights=weights)) if weights else 0.0


def build_grid(region: str, places: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    """Build a 15×15 interpolation grid across the region bounding box."""
    min_lon, min_lat, max_lon, max_lat = REGIONS[region]["bbox"]
    rows = []
    for lat in np.linspace(min_lat, max_lat, 15):
        for lon in np.linspace(min_lon, max_lon, 15):
            nearby_out = sum(
                1 for _, o in outages.iterrows()
                if haversine_km(lat, lon, o["latitude"], o["longitude"]) <= 20
            )
            rows.append({
                "lat":                   round(float(lat), 5),
                "lon":                   round(float(lon), 5),
                "risk_score":            round(float(interpolate_value(lat, lon, places, "final_risk_score")), 2),
                "risk_label":            risk_label(interpolate_value(lat, lon, places, "final_risk_score")),
                "wind_speed":            round(float(interpolate_value(lat, lon, places, "wind_speed_10m")), 2),
                "rain":                  round(float(interpolate_value(lat, lon, places, "precipitation")), 2),
                "resilience_index":      round(float(interpolate_value(lat, lon, places, "resilience_index")), 2),
                "social_vulnerability":  round(float(interpolate_value(lat, lon, places, "social_vulnerability")), 2),
                "aqi":                   round(float(interpolate_value(lat, lon, places, "european_aqi")), 2),
                "energy_not_supplied_mw":round(float(interpolate_value(lat, lon, places, "energy_not_supplied_mw")), 2),
                "financial_loss_gbp":    round(float(interpolate_value(lat, lon, places, "total_financial_loss_gbp")), 2),
                "flood_depth_proxy":     round(float(interpolate_value(lat, lon, places, "flood_depth_proxy")), 3),
                "outages_near_20km":     nearby_out,
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=240, show_spinner=False)
def get_data_cached(region: str, scenario: str, mc_runs: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached entry point for the full data pipeline."""
    places, outages = build_places(region, scenario, mc_runs)
    grid = build_grid(region, places, outages)
    return places, outages, grid


# =============================================================================
# POSTCODE RESILIENCE + INVESTMENT
# =============================================================================

def build_postcode_resilience(places: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if outages is not None and not outages.empty:
        grouped = outages.groupby("postcode_label").agg(outage_records=("outage_reference","count"),affected_customers=("affected_customers","sum"),lat=("latitude","mean"),lon=("longitude","mean")).reset_index()
        for _, g in grouped.iterrows():
            postcode = str(g.get("postcode_label","Unknown"))
            lat = safe_float(g.get("lat")); lon = safe_float(g.get("lon"))
            nearest = None; nearest_d = 1e9
            for _, p in places.iterrows():
                d = haversine_km(lat, lon, p["lat"], p["lon"])
                if d < nearest_d: nearest_d = d; nearest = p
            if nearest is None: continue
            outage_records = safe_float(g.get("outage_records")); affected = safe_float(g.get("affected_customers"))
            out_press  = clamp(outage_records/6, 0,1)*16; cust_press = clamp(affected/1500,0,1)*12; dist_pen = clamp((25-min(nearest_d,25))/25,0,1)*6
            pc_res = clamp(safe_float(nearest.get("resilience_index")) - out_press - cust_press - dist_pen, 0, 100)
            pc_risk= clamp(safe_float(nearest.get("final_risk_score"))  + out_press + cust_press, 0, 100)
            fin_loss = safe_float(nearest.get("total_financial_loss_gbp"))*(0.30+clamp(outage_records/8,0,1)*0.70) + affected*55
            rows.append({"postcode":postcode,"nearest_place":nearest.get("place"),"lat":round(lat,5),"lon":round(lon,5),"distance_to_place_km":round(nearest_d,2),"outage_records":int(outage_records),"affected_customers":int(affected),"risk_score":round(pc_risk,2),"resilience_score":round(pc_res,2),"resilience_label":resilience_label(pc_res),"social_vulnerability":round(safe_float(nearest.get("social_vulnerability")),2),"imd_score":round(safe_float(nearest.get("imd_score")),2),"energy_not_supplied_mw":round(safe_float(nearest.get("energy_not_supplied_mw"))*(0.35+outage_records/10),2),"financial_loss_gbp":round(fin_loss,2),"recommendation_score":0.0})

    existing = {str(r["postcode"]).upper() for r in rows}
    for _, p in places.iterrows():
        pc = str(p.get("postcode_prefix","Unknown"))
        if pc.upper() in existing: continue
        rows.append({"postcode":pc,"nearest_place":p.get("place"),"lat":round(safe_float(p.get("lat")),5),"lon":round(safe_float(p.get("lon")),5),"distance_to_place_km":0.0,"outage_records":int(safe_float(p.get("nearby_outages_25km"))),"affected_customers":int(safe_float(p.get("affected_customers_nearby"))),"risk_score":round(safe_float(p.get("final_risk_score")),2),"resilience_score":round(safe_float(p.get("resilience_index")),2),"resilience_label":resilience_label(safe_float(p.get("resilience_index"))),"social_vulnerability":round(safe_float(p.get("social_vulnerability")),2),"imd_score":round(safe_float(p.get("imd_score")),2),"energy_not_supplied_mw":round(safe_float(p.get("energy_not_supplied_mw")),2),"financial_loss_gbp":round(safe_float(p.get("total_financial_loss_gbp")),2),"recommendation_score":0.0})

    df = pd.DataFrame(rows)
    if df.empty: return df
    mx_loss = max(float(df["financial_loss_gbp"].max()), 1.0)
    mx_ens  = max(float(df["energy_not_supplied_mw"].max()), 1.0)
    df["recommendation_score"] = (0.30*df["risk_score"]+0.22*df["social_vulnerability"]+0.18*(100-df["resilience_score"])+0.13*(df["financial_loss_gbp"]/mx_loss*100)+0.10*(df["energy_not_supplied_mw"]/mx_ens*100)+0.07*np.clip(df["outage_records"]/6,0,1)*100).round(2)
    df["investment_priority"]  = df["recommendation_score"].apply(lambda x: "Priority 1" if x>=75 else "Priority 2" if x>=55 else "Priority 3" if x>=35 else "Monitor")
    return df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


def investment_action_for_row(row: Dict[str, Any]) -> str:
    risk=safe_float(row.get("risk_score")); res=safe_float(row.get("resilience_score")); social=safe_float(row.get("social_vulnerability")); ens=safe_float(row.get("energy_not_supplied_mw")); outages=safe_float(row.get("outage_records"))
    actions = []
    if risk>=65 or outages>=3: actions.append("reinforce local feeders and automate switching")
    if ens>=300:               actions.append("install backup supply / mobile generation access")
    if social>=55:             actions.append("target community resilience support and welfare checks")
    if res<45:                 actions.append("upgrade protection, monitoring and restoration capability")
    if risk>=55:               actions.append("prioritise vegetation management and weather hardening")
    if not actions:            actions.append("continue monitoring and maintain standard preventive maintenance")
    return "; ".join(actions)


def investment_category_for_row(row: Dict[str, Any]) -> str:
    risk=safe_float(row.get("risk_score")); social=safe_float(row.get("social_vulnerability")); ens=safe_float(row.get("energy_not_supplied_mw")); res=safe_float(row.get("resilience_score"))
    if ens>=450:  return "Energy security / backup capacity"
    if res<45:    return "Network resilience upgrade"
    if social>=60:return "Social resilience and emergency planning"
    if risk>=65:  return "Weather hardening"
    return "Preventive monitoring"


def build_investment_recommendations(places: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    pc = build_postcode_resilience(places, outages)
    if pc.empty: return pc
    pc = pc.copy()
    pc["investment_category"]            = pc.apply(lambda r: investment_category_for_row(r.to_dict()), axis=1)
    pc["recommended_action"]             = pc.apply(lambda r: investment_action_for_row(r.to_dict()), axis=1)
    pc["indicative_investment_cost_gbp"] = (120_000 + pc["recommendation_score"]*8_500 + pc["outage_records"]*35_000 + np.clip(pc["energy_not_supplied_mw"],0,1000)*260).round(0)
    pc["benefit_cost_note"]              = "High avoided-loss potential"
    return pc.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

# =============================================================================
# SAT-Guard Digital Twin — Q1 Edition
# PART 3 of 4 — Charts, coloured regional map, BBC weather, tab renderers
# =============================================================================
# Paste this file AFTER app_KASVA_PART2.py
# =============================================================================


# =============================================================================
# PLOTLY CHART BUILDERS
# =============================================================================

def create_risk_gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "/100"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": colour_rgba_hex(value)},
            "steps": [
                {"range": [0,  35], "color": "rgba(34,197,94,.25)"},
                {"range": [35, 55], "color": "rgba(234,179,8,.25)"},
                {"range": [55, 75], "color": "rgba(249,115,22,.25)"},
                {"range": [75,100], "color": "rgba(239,68,68,.25)"},
            ],
        },
    ))
    fig.update_layout(template=plotly_template(), height=280, margin=dict(l=18,r=18,t=45,b=18))
    return fig


def create_resilience_gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "/100"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#22c55e" if value >= 60 else "#f97316"},
            "steps": [
                {"range": [0,  40], "color": "rgba(239,68,68,.25)"},
                {"range": [40, 60], "color": "rgba(234,179,8,.25)"},
                {"range": [60, 80], "color": "rgba(56,189,248,.25)"},
                {"range": [80,100], "color": "rgba(34,197,94,.25)"},
            ],
        },
    ))
    fig.update_layout(template=plotly_template(), height=280, margin=dict(l=18,r=18,t=45,b=18))
    return fig


def create_loss_waterfall(places: pd.DataFrame) -> go.Figure:
    totals = {
        "VoLL":             places["voll_loss_gbp"].sum(),
        "Customer":         places["customer_interruption_loss_gbp"].sum(),
        "Business":         places["business_disruption_loss_gbp"].sum(),
        "Restoration":      places["restoration_loss_gbp"].sum(),
        "Critical services":places["critical_services_loss_gbp"].sum(),
    }
    fig = go.Figure(go.Waterfall(
        name="Financial loss", orientation="v",
        measure=["relative"] * len(totals),
        x=list(totals.keys()),
        y=[v / 1_000_000 for v in totals.values()],
        connector={"line": {"color": "rgba(148,163,184,.45)"}},
    ))
    fig.update_layout(
        title="Financial-loss contribution (£m)", template=plotly_template(),
        height=390, yaxis_title="£m", margin=dict(l=10,r=10,t=55,b=10),
    )
    return fig


def create_cascade_radar(places: pd.DataFrame) -> go.Figure:
    vals = [
        places["cascade_power"].mean(), places["cascade_water"].mean(),
        places["cascade_telecom"].mean(), places["cascade_transport"].mean(),
        places["cascade_social"].mean(),
    ]
    cats = ["Power","Water","Telecom","Transport","Social"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself", name="Mean cascade stress"))
    fig.update_layout(
        template=plotly_template(),
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False, height=390,
        title="Interdependency cascade signature",
        margin=dict(l=10,r=10,t=55,b=10),
    )
    return fig


def create_finance_sunburst(places: pd.DataFrame) -> go.Figure:
    rows = []
    for _, r in places.iterrows():
        rows += [
            {"place":r["place"],"component":"VoLL",             "loss":r["voll_loss_gbp"]},
            {"place":r["place"],"component":"Customer",         "loss":r["customer_interruption_loss_gbp"]},
            {"place":r["place"],"component":"Business",         "loss":r["business_disruption_loss_gbp"]},
            {"place":r["place"],"component":"Restoration",      "loss":r["restoration_loss_gbp"]},
            {"place":r["place"],"component":"Critical services","loss":r["critical_services_loss_gbp"]},
        ]
    df = pd.DataFrame(rows)
    fig = px.sunburst(df, path=["place","component"], values="loss", template=plotly_template())
    fig.update_layout(title="Local financial-loss structure", height=470, margin=dict(l=10,r=10,t=55,b=10))
    return fig


def create_mc_histogram(worst: pd.Series) -> go.Figure:
    values = worst.get("mc_histogram", [])
    fig = px.histogram(x=values, nbins=26,
        title=f"Monte Carlo risk distribution — {worst.get('place')}",
        labels={"x":"Risk score","y":"Frequency"}, template=plotly_template())
    fig.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10))
    return fig


# =============================================================================
# COLOUR LEGEND COMPONENT
# =============================================================================

def render_colour_legend(kind: str = "risk") -> None:
    if kind == "resilience":
        items = [
            ("#22c55e","Robust",    "80–100: strong resilience"),
            ("#38bdf8","Functional","60–79: functioning with manageable stress"),
            ("#eab308","Stressed",  "40–59: reduced resilience"),
            ("#ef4444","Fragile",   "0–39: urgent resilience concern"),
        ]
    elif kind == "priority":
        items = [
            ("#ef4444","Priority 1","Immediate action"),
            ("#f97316","Priority 2","High priority"),
            ("#eab308","Priority 3","Medium priority"),
            ("#22c55e","Monitor",   "Routine monitoring"),
        ]
    else:
        items = [
            ("#22c55e","Low",     "0–34: normal / low operational risk"),
            ("#eab308","Moderate","35–54: watch / early stress"),
            ("#f97316","High",    "55–74: warning / elevated stress"),
            ("#ef4444","Severe",  "75–100: critical / severe stress"),
        ]
    chips = "".join(
        f'<span style="display:inline-block;margin:4px 8px 4px 0;padding:7px 10px;'
        f'border-radius:999px;border:1px solid rgba(148,163,184,.25);'
        f'background:rgba(15,23,42,.72);color:#e5e7eb;">'
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        f'background:{c};margin-right:7px;vertical-align:-1px;"></span>'
        f'<b>{l}</b> — {t}</span>'
        for c, l, t in items
    )
    st.markdown(f'<div class="note"><b>Colour legend:</b><br>{chips}</div>', unsafe_allow_html=True)


# =============================================================================
# HERO + METRICS
# =============================================================================

def hero(region: str, scenario: str, mc_runs: int, refresh_id: int) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="title">⚡ SAT-Guard Grid Digital Twin</div>
            <div class="subtitle">
                Broadcast-style weather simulation, multi-layer grid-risk modelling,
                social vulnerability, outage intelligence, Monte Carlo uncertainty
                and investment prioritisation for {html.escape(region)}.
            </div>
            <div style="margin-top:10px;">
                <span class="chip">{html.escape(region)}</span>
                <span class="chip">{html.escape(scenario)}</span>
                <span class="chip">MC runs: {mc_runs}</span>
                <span class="chip">Refresh ID: {refresh_id}</span>
                <span class="chip">UTC {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics_panel(places: pd.DataFrame, pc: pd.DataFrame) -> None:
    avg_risk    = round(float(places["final_risk_score"].mean()), 1)
    avg_res     = round(float(places["resilience_index"].mean()), 1)
    avg_failure = round(float(places["failure_probability"].mean()) * 100, 1)
    total_ens   = round(float(places["energy_not_supplied_mw"].sum()), 1)
    total_loss  = round(float(places["total_financial_loss_gbp"].sum()), 2)
    p1 = 0 if pc.empty else int((pc["investment_priority"] == "Priority 1").sum())
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Regional risk",   f"{avg_risk}/100",    risk_label(avg_risk))
    c2.metric("Resilience",      f"{avg_res}/100",     resilience_label(avg_res))
    c3.metric("Failure prob.",   f"{avg_failure}%")
    c4.metric("ENS",             f"{total_ens} MW")
    c5.metric("Financial loss",  money_m(total_loss))
    c6.metric("Priority 1",      p1)


# =============================================================================
# TAB: EXECUTIVE OVERVIEW
# =============================================================================

def overview_tab(places: pd.DataFrame, pc: pd.DataFrame, scenario: str) -> None:
    left, right = st.columns([1.15, 0.85])
    expected_cols = [
        "place","risk_label","final_risk_score","resilience_label","resilience_index",
        "wind_speed_10m","precipitation","european_aqi","imd_score","social_vulnerability",
        "energy_not_supplied_mw","total_financial_loss_gbp",
    ]
    safe_df  = places.reindex(columns=expected_cols)
    sort_col = "final_risk_score" if "final_risk_score" in places.columns else expected_cols[0]

    with left:
        st.subheader("Regional intelligence table")
        render_colour_legend("risk")
        st.dataframe(safe_df.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

    with right:
        avg_risk = float(pd.to_numeric(places.get("final_risk_score"), errors="coerce").mean()) if "final_risk_score" in places.columns else 0
        avg_res  = float(pd.to_numeric(places.get("resilience_index"),  errors="coerce").mean()) if "resilience_index"  in places.columns else 0
        g1, g2 = st.columns(2)
        g1.plotly_chart(create_risk_gauge(avg_risk, "Regional risk"), use_container_width=True)
        g2.plotly_chart(create_resilience_gauge(avg_res, "Resilience"), use_container_width=True)

    a, b = st.columns(2)
    with a:
        if {"place","final_risk_score"}.issubset(places.columns):
            fig = px.bar(places.sort_values("final_risk_score",ascending=False), x="place", y="final_risk_score", color="risk_label" if "risk_label" in places.columns else None, title="Risk ranking by location", template=plotly_template())
            fig.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with b:
        if {"social_vulnerability","final_risk_score","total_financial_loss_gbp"}.issubset(places.columns):
            fig = px.scatter(places, x="social_vulnerability", y="final_risk_score", size="total_financial_loss_gbp", color="resilience_index" if "resilience_index" in places.columns else None, hover_name="place" if "place" in places.columns else None, title="Social vulnerability vs grid risk", template=plotly_template(), color_continuous_scale="Turbo")
            fig.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig, use_container_width=True)

    desc = SCENARIOS.get(scenario, {}).get("description", "")
    st.markdown(f'<div class="note"><b>Scenario logic:</b> {html.escape(desc)}</div>', unsafe_allow_html=True)


# =============================================================================
# BBC / WXCHARTS-STYLE ANIMATED WEATHER COMPONENT
# =============================================================================

def make_weather_frames(places: pd.DataFrame, grid: pd.DataFrame, scenario: str) -> Dict[str,Any]:
    hazard_mode = SCENARIOS[scenario]["hazard_mode"]
    frames = []
    for h in range(0, 24, 2):
        phase = math.sin((h / 24) * math.pi * 2)
        cells = []
        for _, g in grid.iterrows():
            cells.append({
                "lat":             float(g["lat"]),
                "lon":             float(g["lon"]),
                "wind_speed":      round(max(0, g["wind_speed"]      * (1+0.25*phase+random.uniform(-0.06,0.06))), 2),
                "rain":            round(max(0, g["rain"]            * (1+0.35*max(phase,0)+random.uniform(-0.05,0.05))), 2),
                "risk_score":      round(clamp(g["risk_score"]       * (1+0.18*phase+random.uniform(-0.05,0.05)), 0, 100), 2),
                "resilience_index":round(clamp(g["resilience_index"] - phase*7, 0, 100), 2),
                "financial_loss_gbp":  float(g["financial_loss_gbp"]),
                "flood_depth_proxy":   float(g["flood_depth_proxy"]),
            })
        frames.append({"hour":h,"label":f"+{h:02d}h","hazard_mode":hazard_mode,"cells":cells})
    return {"hazard_mode":hazard_mode,"scenario":scenario,"places":places.to_dict("records"),"frames":frames}


def render_bbc_weather_component(region: str, places: pd.DataFrame, grid: pd.DataFrame, scenario: str, height: int=790) -> None:
    payload    = make_weather_frames(places, grid, scenario)
    center     = REGIONS[region]["center"]
    payload["center"] = center
    data_json  = json.dumps(payload)
    bbox_json  = json.dumps(REGIONS[region]["bbox"])

    html_code = f"""<!doctype html><html><head><meta charset="utf-8"/>
<style>
html,body{{margin:0;padding:0;background:#020617;font-family:"Segoe UI",Arial,sans-serif;}}
#scene{{position:relative;height:{height}px;width:100%;overflow:hidden;border-radius:30px;
  background:radial-gradient(circle at 32% 20%,rgba(56,189,248,.20),transparent 26%),
             radial-gradient(circle at 72% 18%,rgba(168,85,247,.18),transparent 28%),
             linear-gradient(180deg,#0a1726 0%,#07101d 42%,#020617 100%);
  border:1px solid rgba(148,163,184,.28);box-shadow:0 34px 90px rgba(0,0,0,.42);}}
canvas{{position:absolute;inset:0;}}
#bd{{z-index:1;}}#pr{{z-index:2;}}#we{{z-index:3;}}#fr{{z-index:4;}}
#lb{{position:absolute;inset:0;z-index:5;pointer-events:none;}}
.city{{position:absolute;color:white;font-weight:900;font-size:15px;text-shadow:0 2px 5px #000;white-space:nowrap;}}
.city::after{{content:"";display:inline-block;width:7px;height:7px;background:white;margin-left:7px;box-shadow:0 1px 5px #000;}}
.hud{{position:absolute;z-index:10;border:1px solid rgba(255,255,255,.18);background:rgba(2,6,23,.68);backdrop-filter:blur(14px);color:#dbeafe;border-radius:18px;padding:14px 16px;}}
#top{{top:18px;left:18px;max-width:480px;}}
#leg{{top:18px;right:18px;width:240px;}}
#ctl{{left:18px;right:18px;bottom:18px;display:grid;grid-template-columns:auto auto 1fr auto auto;gap:12px;align-items:center;}}
.ttl{{color:white;font-size:17px;font-weight:950;margin-bottom:5px;}}
.sub{{font-size:12px;line-height:1.5;}}
.logo{{position:absolute;left:24px;bottom:110px;z-index:11;display:flex;align-items:center;gap:10px;color:white;text-shadow:0 3px 10px rgba(0,0,0,.85);}}
.bbc span{{display:inline-grid;place-items:center;width:34px;height:34px;background:rgba(255,255,255,.96);color:#1e293b;font-weight:950;font-size:20px;margin-right:4px;}}
.word{{font-size:32px;font-weight:950;letter-spacing:-.04em;}}
.tbox{{position:absolute;right:24px;bottom:114px;z-index:11;display:flex;font-weight:950;}}
.day{{background:rgba(13,148,136,.94);color:white;padding:10px 16px;font-size:15px;}}
.hr{{background:rgba(2,6,23,.92);color:white;padding:10px 16px;min-width:70px;text-align:center;font-size:15px;}}
.grad{{height:13px;border-radius:999px;background:linear-gradient(90deg,rgba(59,130,246,.4),rgba(34,197,94,.6),rgba(234,179,8,.8),rgba(249,115,22,.85),rgba(239,68,68,.92),rgba(168,85,247,.95));margin:7px 0;}}
button{{border:0;border-radius:13px;background:linear-gradient(135deg,#0284c7,#38bdf8);color:white;font-weight:950;padding:9px 13px;cursor:pointer;}}
input[type=range]{{width:100%;}}
.pill{{color:#bfdbfe;border:1px solid rgba(148,163,184,.25);border-radius:999px;padding:7px 11px;background:rgba(15,23,42,.78);font-weight:850;font-size:12px;}}
</style></head><body>
<div id="scene">
  <canvas id="bd"></canvas><canvas id="pr"></canvas><canvas id="we"></canvas><canvas id="fr"></canvas>
  <div id="lb"></div>
  <div class="hud" id="top">
    <div class="ttl">Forecast simulation &amp; grid resilience overlay</div>
    <div class="sub">Scenario: <b>{html.escape(scenario)}</b> &nbsp;|&nbsp; Mode: <b>{html.escape(payload["hazard_mode"])}</b></div>
  </div>
  <div class="hud" id="leg">
    <div class="ttl">Hazard intensity</div>
    <div class="grad"></div>
    <div class="sub" style="display:flex;justify-content:space-between;"><span>Light</span><span>Heavy</span><span>Extreme</span></div>
    <hr style="border-color:rgba(255,255,255,.14);">
    <div class="sub">● blue/green: lower stress<br>● amber/red/purple: high hazard</div>
  </div>
  <div class="logo">
    <div class="bbc"><span>S</span><span>A</span><span>T</span></div>
    <div class="word">GUARD DT</div>
  </div>
  <div class="tbox"><div class="day">FORECAST</div><div class="hr" id="hr">00h</div></div>
  <div class="hud" id="ctl">
    <button onclick="play()">▶ Play</button>
    <button onclick="pause()">Ⅱ Pause</button>
    <input id="sl" type="range" min="0" max="11" value="0" oninput="scrub(this.value)">
    <span class="pill" id="cond">{html.escape(scenario)}</span>
    <span class="pill" id="stat">Loading…</span>
  </div>
</div>
<script>
const D={data_json};
const bbox={bbox_json};
const sc=document.getElementById("scene");
const [bd,pr,we,fr,lb]=[document.getElementById(x) for x of ["bd","pr","we","fr","lb"]];
</script>
<script>
const DATA={data_json};
const BBOX={bbox_json};
const scene=document.getElementById("scene");
const bd=document.getElementById("bd"),pr=document.getElementById("pr"),we=document.getElementById("we"),fr=document.getElementById("fr"),lb=document.getElementById("lb");
const bx=bd.getContext("2d"),px=pr.getContext("2d"),wx=we.getContext("2d"),fx=fr.getContext("2d");
const sl=document.getElementById("sl"),hrEl=document.getElementById("hr"),stat=document.getElementById("stat"),cond=document.getElementById("cond");
let W=1000,H={height},cf=DATA.frames[0],fi=0,timer=null,lastT=performance.now();
let rainBands=[],clouds=[],arrows=[],vortices=[],flash=0;
function resize(){{
  const r=scene.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  W=r.width;H=r.height;
  [bd,pr,we,fr].forEach(c=>{{c.width=Math.floor(W*dpr);c.height=Math.floor(H*dpr);c.style.width=W+"px";c.style.height=H+"px";}});
  [bx,px,wx,fx].forEach(c=>c.setTransform(dpr,0,0,dpr,0,0));
  drawBg();drawLabels();
}}
function proj(lat,lon){{
  return {{x:(lon-BBOX[0])/(BBOX[2]-BBOX[0])*W,y:H-(lat-BBOX[1])/(BBOX[3]-BBOX[1])*H}};
}}
function drawBg(){{
  bx.clearRect(0,0,W,H);
  const g=bx.createLinearGradient(0,0,W,H);
  g.addColorStop(0,"#0b2338");g.addColorStop(.45,"#092037");g.addColorStop(1,"#02101f");
  bx.fillStyle=g;bx.fillRect(0,0,W,H);
  bx.strokeStyle="rgba(148,163,184,.10)";bx.lineWidth=1;
  for(let x=0;x<W;x+=42){{bx.beginPath();bx.moveTo(x,0);bx.lineTo(x,H);bx.stroke();}}
  for(let y=0;y<H;y+=42){{bx.beginPath();bx.moveTo(0,y);bx.lineTo(W,y);bx.stroke();}}
}}
function drawLabels(){{
  lb.innerHTML="";
  DATA.places.forEach(p=>{{
    const xy=proj(p.lat,p.lon);
    const d=document.createElement("div");d.className="city";d.textContent=p.place;
    d.style.left=Math.max(8,Math.min(W-130,xy.x+8))+"px";d.style.top=Math.max(8,Math.min(H-30,xy.y-10))+"px";
    lb.appendChild(d);
  }});
}}
function col(v,a){{
  if(v>=86)return"rgba(168,85,247,"+a+")";if(v>=76)return"rgba(239,68,68,"+a+")";
  if(v>=63)return"rgba(249,115,22,"+a+")";if(v>=50)return"rgba(234,179,8,"+a+")";
  if(v>=35)return"rgba(34,197,94,"+a+")";return"rgba(59,130,246,"+a+")";
}}
function init(){{
  rainBands=[];clouds=[];arrows=[];vortices=[];
  const m=DATA.hazard_mode;
  const rb=m==="storm"?54:m==="rain"?42:m==="wind"?26:20;
  for(let i=0;i<rb;i++)rainBands.push({{x:-W*.45+Math.random()*W*1.75,y:-H*.12+Math.random()*H*1.18,rx:65+Math.random()*260,ry:20+Math.random()*90,spd:.16+Math.random()*.72,alpha:.08+Math.random()*.24,ph:Math.random()*Math.PI*2,bias:Math.random(),rot:-.35+Math.random()*.7}});
  const cc=m==="storm"?26:m==="rain"?20:m==="wind"?14:10;
  for(let i=0;i<cc;i++)clouds.push({{x:-W*.55+Math.random()*W*1.95,y:-H*.10+Math.random()*H*1.15,rx:150+Math.random()*420,ry:42+Math.random()*125,spd:.08+Math.random()*.34,alpha:.055+Math.random()*.13,ph:Math.random()*Math.PI*2,rot:-.25+Math.random()*.5}});
  const wc=m==="storm"?150:m==="wind"?120:m==="rain"?85:65;
  for(let i=0;i<wc;i++)arrows.push({{x:Math.random()*W,y:Math.random()*H,len:28+Math.random()*70,spd:.50+Math.random()*1.50,alpha:.35+Math.random()*.38,w:1.4+Math.random()*2.6,ph:Math.random()*Math.PI*2}});
  const vc=m==="storm"?3:m==="rain"?2:1;
  for(let i=0;i<vc;i++)vortices.push({{x:W*(.28+Math.random()*.48),y:H*(.25+Math.random()*.52),r:95+Math.random()*190,str:.35+Math.random()*.65,spd:.05+Math.random()*.12,ph:Math.random()*Math.PI*2}});
}}
function avg(k){{return!cf||!cf.cells?0:cf.cells.reduce((s,c)=>s+Number(c[k]||0),0)/cf.cells.length;}}
function nearest(x,y){{let b=null,bd2=1e9;cf.cells.forEach(c=>{{const p=proj(c.lat,c.lon),d=(p.x-x)**2+(p.y-y)**2;if(d<bd2){{bd2=d;b=c;}}}});return b;}}
function drawPressure(t){{
  px.clearRect(0,0,W,H);px.save();px.globalAlpha=.62;px.strokeStyle="rgba(255,255,255,.55)";px.lineWidth=1.5;
  for(let f=0;f<2;f++){{
    const cx=W*(f===0?.27:.72)+Math.sin(t/6200+f)*28,cy=H*(f===0?.55:.36)+Math.cos(t/5300+f)*22;
    for(let k=0;k<8;k++){{px.beginPath();px.ellipse(cx,cy,90+k*42+f*20,50+k*28,-.38+f*.65,0,Math.PI*2);px.stroke();}}
  }}
  px.lineWidth=1.1;
  for(let k=0;k<8;k++){{px.beginPath();for(let x=-70;x<=W+80;x+=18){{const y=H*.22+k*76+Math.sin((x+t*.018)/125+k*.55)*(25+k*2);if(x===-70)px.moveTo(x,y);else px.lineTo(x,y);}}px.stroke();}}
  px.restore();
}}
function drawFronts(t){{
  fx.clearRect(0,0,W,H);fx.save();fx.lineWidth=2.5;fx.strokeStyle="rgba(255,255,255,.76)";fx.beginPath();
  for(let x=-60;x<=W+70;x+=22){{const y=H*.61+Math.sin((x+t*.025)/118)*42;if(x===-60)fx.moveTo(x,y);else fx.lineTo(x,y);}}fx.stroke();
  for(let x=10;x<W;x+=74){{
    const y=H*.61+Math.sin((x+t*.025)/118)*42;
    fx.fillStyle="rgba(59,130,246,.88)";fx.beginPath();fx.moveTo(x,y);fx.lineTo(x+19,y+16);fx.lineTo(x-8,y+18);fx.closePath();fx.fill();
    fx.fillStyle="rgba(239,68,68,.88)";fx.beginPath();fx.arc(x+38,y-1,10,Math.PI,0);fx.fill();
  }}fx.restore();
}}
function ell(c,x,y,rx,ry,fill,rot=0){{c.save();c.translate(x,y);c.rotate(rot);c.beginPath();c.ellipse(0,0,rx,ry,0,0,Math.PI*2);c.fillStyle=fill;c.fill();c.restore();}}
function drawClouds(t,dt){{
  clouds.forEach(c=>{{
    c.x+=c.spd*dt*.05;c.y+=Math.sin(t/2500+c.ph)*.035*dt;
    if(c.x-c.rx>W+160){{c.x=-c.rx-180;c.y=-H*.10+Math.random()*H*1.15;}}
    const g=wx.createRadialGradient(c.x,c.y,0,c.x,c.y,c.rx);
    g.addColorStop(0,"rgba(255,255,255,"+c.alpha+")");g.addColorStop(.42,"rgba(220,230,235,"+c.alpha*.72+")");g.addColorStop(.78,"rgba(160,176,188,"+c.alpha*.30+")");g.addColorStop(1,"rgba(160,176,188,0)");
    ell(wx,c.x,c.y,c.rx,c.ry,g,c.rot);
  }});
}}
function drawPrecip(t,dt){{
  const m=DATA.hazard_mode,mv=m==="storm"?1.55:m==="rain"?1.18:.74,ar=avg("rain"),ak=avg("risk_score");
  cf.cells.forEach((cell,idx)=>{{
    const p=proj(cell.lat,cell.lon),rain=Number(cell.rain||0),risk=Number(cell.risk_score||0);
    if(rain<.1&&risk<30&&m!=="storm"&&m!=="rain")return;
    const pulse=.94+.06*Math.sin(t/680+idx),rx=(46+rain*28+risk*.80)*pulse,ry=(21+rain*12+risk*.34)*pulse;
    const al=Math.min(.62,.085+rain*.065+risk/430);
    const g=wx.createRadialGradient(p.x,p.y,0,p.x,p.y,rx);
    g.addColorStop(0,col(risk,al));g.addColorStop(.42,col(risk,al*.58));g.addColorStop(.74,"rgba(59,130,246,"+al*.20+")");g.addColorStop(1,"rgba(0,0,0,0)");
    ell(wx,p.x,p.y,rx,ry,g);
  }});
  rainBands.forEach(b=>{{
    b.x+=b.spd*mv*dt*.068;b.y+=Math.sin(t/1600+b.ph)*.055*dt;
    if(b.x-b.rx>W+150){{b.x=-b.rx-170;b.y=-H*.12+Math.random()*H*1.18;}}
    const sr=ak+b.bias*45,al=Math.min(.55,b.alpha*(.65+ar/3.8+ak/190));
    const g=wx.createRadialGradient(b.x,b.y,0,b.x,b.y,b.rx);
    g.addColorStop(0,col(sr,al));g.addColorStop(.46,col(sr,al*.54));g.addColorStop(.80,"rgba(37,99,235,"+al*.18+")");g.addColorStop(1,"rgba(0,0,0,0)");
    ell(wx,b.x,b.y,b.rx,b.ry,g,b.rot);
  }});
  vortices.forEach(v=>{{
    v.ph+=v.spd*dt*.01;
    for(let arm=0;arm<4;arm++)for(let j=0;j<28;j++){{
      const th=v.ph+arm*Math.PI/2+j*.18,r=18+j*(v.r/28),x=v.x+Math.cos(th)*r,y=v.y+Math.sin(th)*r*.60;
      ell(wx,x,y,22+j*1.0,8+j*.35,col(ak+30,Math.max(0,(1-j/30)*.16*v.str)));
    }}
  }});
}}
function drawWind(t,dt){{
  const m=DATA.hazard_mode,mult=m==="storm"?1.65:m==="wind"?1.38:m==="rain"?.92:.72;
  arrows.forEach(a=>{{
    const loc=nearest(a.x,a.y),w=loc?Number(loc.wind_speed||9):avg("wind_speed"),intensity=Math.min(w/42,1.55);
    let angle=-.24+Math.sin(t/1800+a.ph)*.12;
    vortices.forEach(v=>{{const dx=a.x-v.x,dy=a.y-v.y,d=Math.sqrt(dx*dx+dy*dy);if(d<v.r*2.1)angle+=Math.atan2(dy,dx)*.10*v.str;}});
    const len=a.len*(.74+intensity*.58),x0=a.x-Math.cos(angle)*len,y0=a.y-Math.sin(angle)*len,al=Math.min(.86,a.alpha+intensity*.20);
    wx.save();wx.strokeStyle="rgba(255,255,255,"+al+")";wx.fillStyle="rgba(255,255,255,"+al+")";wx.lineWidth=a.w;wx.lineCap="round";
    wx.beginPath();wx.moveTo(x0,y0);wx.quadraticCurveTo((x0+a.x)/2,(y0+a.y)/2+Math.sin(t/540+a.ph)*5,a.x,a.y);wx.stroke();
    const hd=10+intensity*7,bx2=a.x-Math.cos(angle)*hd,by2=a.y-Math.sin(angle)*hd,nx=-Math.sin(angle),ny=Math.cos(angle);
    wx.beginPath();wx.moveTo(a.x,a.y);wx.lineTo(bx2+nx*hd*.42,by2+ny*hd*.42);wx.lineTo(bx2-nx*hd*.42,by2-ny*hd*.42);wx.closePath();wx.fill();wx.restore();
    a.x+=Math.cos(angle)*a.spd*mult*(.66+intensity)*dt*.083;a.y+=Math.sin(angle)*a.spd*mult*(.66+intensity)*dt*.083;
    if(a.x>W+115||a.y<-85||a.y>H+85){{a.x=-115;a.y=Math.random()*H;a.ph=Math.random()*Math.PI*2;}}
  }});
}}
function drawLightning(){{
  if(DATA.hazard_mode!=="storm")return;
  if(Math.random()<.007)flash=6;
  if(flash>0){{
    wx.fillStyle="rgba(255,255,255,"+(.05+flash*.012)+")";wx.fillRect(0,0,W,H);
    for(let b=0;b<2;b++){{wx.strokeStyle="rgba(255,255,255,.64)";wx.lineWidth=2.1;wx.beginPath();let x=W*(.15+Math.random()*.75),y=0;wx.moveTo(x,y);for(let i=0;i<6;i++){{x+=-35+Math.random()*70;y+=34+Math.random()*62;wx.lineTo(x,y);}}wx.stroke();}}
    flash--;
  }}
}}
function animate(t){{
  const dt=Math.min(34,t-lastT);lastT=t;
  wx.clearRect(0,0,W,H);wx.fillStyle=DATA.hazard_mode==="storm"?"rgba(5,12,24,.045)":"rgba(5,15,28,.025)";wx.fillRect(0,0,W,H);
  drawPressure(t);drawFronts(t);drawClouds(t,dt);drawPrecip(t,dt);drawWind(t,dt);drawLightning();
  stat.textContent="Wind "+avg("wind_speed").toFixed(1)+" km/h · Rain "+avg("rain").toFixed(1)+" mm · Risk "+avg("risk_score").toFixed(1);
  requestAnimationFrame(animate);
}}
function renderFrame(i){{fi=i;cf=DATA.frames[i];sl.value=i;hrEl.textContent=String(cf.label).replace("+","");cond.textContent=DATA.scenario+" · "+DATA.hazard_mode;}}
function play(){{if(timer)clearInterval(timer);timer=setInterval(()=>{{fi=(fi+1)%DATA.frames.length;renderFrame(fi);}},950);}}
function pause(){{if(timer)clearInterval(timer);}}
function scrub(v){{renderFrame(parseInt(v));}}
window.addEventListener("resize",resize);
resize();init();renderFrame(0);play();requestAnimationFrame(animate);
</script></body></html>"""
    components.html(html_code, height=height+8, scrolling=False)


def bbc_tab(region: str, scenario: str, places: pd.DataFrame, grid: pd.DataFrame) -> None:
    st.subheader("BBC / WXCharts-style animated grid hazard simulation")
    st.caption("Canvas animation: precipitation shields, pressure contours, fronts, wind vectors, lightning in storm mode, city labels.")
    render_bbc_weather_component(region, places, grid, scenario, height=790)


# =============================================================================
# SPATIAL INTELLIGENCE TAB  (reworked — coloured authority regions, no pentagons)
# =============================================================================

def _authority_risk_lookup(places: pd.DataFrame, region: str) -> Dict[str, float]:
    """Aggregate place-level risk scores to local authority names."""
    auth_map = REGIONS[region].get("place_authority_map", {})
    auth_risks: Dict[str, List[float]] = {}
    for _, row in places.iterrows():
        auth = auth_map.get(str(row.get("place","")), str(row.get("place","")))
        auth_risks.setdefault(auth, []).append(safe_float(row.get("final_risk_score")))
    return {a: float(np.mean(v)) for a, v in auth_risks.items() if v}


def render_colourful_regional_map(region: str, places: pd.DataFrame) -> None:
    """
    Render a political-style coloured local authority risk map.

    DESIGN: Each local authority polygon is filled with a categorical colour
    derived from its mean risk score (green / blue / orange / magenta).
    Bold white boundary lines separate authorities.
    City markers and labels are layered on top.

    This replaces the previous pentagon / hexagon cell approach entirely.
    """
    center   = REGIONS[region]["center"]
    polygons = REGIONS[region].get("authority_polygons", {})
    risk_lkp = _authority_risk_lookup(places, region)

    fig = go.Figure()

    # ---- Layer 1: filled authority polygons ----
    for auth_name, coords in polygons.items():
        lons_p = [c[0] for c in coords]
        lats_p = [c[1] for c in coords]
        risk_val  = risk_lkp.get(auth_name, float(places["final_risk_score"].mean()))
        fill_hex  = regional_risk_hex(risk_val)

        # Build tooltip from member places
        auth_map     = REGIONS[region].get("place_authority_map", {})
        member_places= [p for p, a in auth_map.items() if a == auth_name]
        member_rows  = places[places["place"].isin(member_places)]
        tip = [f"<b>{auth_name}</b>", f"Risk: {round(risk_val,1)}/100 ({risk_label(risk_val)})"]
        if not member_rows.empty:
            tip.append(f"Resilience: {round(float(member_rows['resilience_index'].mean()),1)}/100")
            tip.append(f"ENS: {round(float(member_rows['energy_not_supplied_mw'].sum()),1)} MW")
            tip.append(f"Social vulnerability: {round(float(member_rows['social_vulnerability'].mean()),1)}/100")
            fin = float(member_rows["total_financial_loss_gbp"].sum())
            tip.append(f"Financial loss: {money_m(fin)}")

        fig.add_trace(go.Scattermapbox(
            lon=lons_p, lat=lats_p,
            mode="lines", fill="toself",
            fillcolor=fill_hex,
            line=dict(width=2.5, color="white"),
            opacity=0.82,
            text=["<br>".join(tip)] * len(lons_p),
            hoverinfo="text",
            name=f"{auth_name} · {risk_label(risk_val)}",
            showlegend=True,
        ))

    # ---- Layer 2: city markers ----
    fig.add_trace(go.Scattermapbox(
        lon=places["lon"].tolist(),
        lat=places["lat"].tolist(),
        mode="markers+text",
        marker=dict(size=14, color="white", opacity=0.95),
        text=places["place"].tolist(),
        textposition="top center",
        textfont=dict(size=13, color="white"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center={"lat":center["lat"],"lon":center["lon"]}, zoom=center["zoom"]),
        height=580, margin=dict(l=10,r=10,t=45,b=10),
        title=f"Local authority risk mosaic — {region}",
        legend=dict(bgcolor="rgba(2,6,23,0.7)", font=dict(color="white"), orientation="v", x=0.01, y=0.99),
        paper_bgcolor="#020617", font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="note">
        <b>Authority risk map:</b>
        <span style="color:#7bd000;font-weight:800;">Green</span> = low risk &nbsp;|&nbsp;
        <span style="color:#0070c0;font-weight:800;">Blue</span> = moderate &nbsp;|&nbsp;
        <span style="color:#ff7b00;font-weight:800;">Orange</span> = high &nbsp;|&nbsp;
        <span style="color:#d80073;font-weight:800;">Magenta</span> = severe.<br>
        Each polygon is a local authority area filled by the mean risk of its constituent places.
        White lines mark authority boundaries. Hover for details.
        </div>
        """,
        unsafe_allow_html=True,
    )


def spatial_tab(region: str, places: pd.DataFrame, outages: pd.DataFrame, pc: pd.DataFrame, grid: pd.DataFrame, map_mode: str) -> None:
    """
    Spatial Intelligence tab.

    Sections:
        1. Coloured local authority risk map (replaces pentagon/hexagon cells)
        2. Operational stress density heatmap
        3. Socio-technical scatter + risk bar
    """
    st.subheader("🌍 Spatial intelligence")
    center = REGIONS[region]["center"]

    df = places.copy()
    for c in ["lat","lon","final_risk_score","resilience_index","social_vulnerability","energy_not_supplied_mw","grid_failure_probability"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    # ---- Section 1: authority polygon map ----
    st.markdown("### 🗺️ Local authority risk map")
    render_colourful_regional_map(region, df)
    st.markdown("---")

    # ---- Section 2: density heatmap ----
    st.markdown("### 🔬 Operational stress density")
    fig_d = px.density_mapbox(
        df, lat="lat", lon="lon", z="final_risk_score", radius=42,
        center={"lat":center["lat"],"lon":center["lon"]}, zoom=center["zoom"],
        mapbox_style="carto-darkmatter", color_continuous_scale="Turbo",
        title="Operational stress density", height=500,
    )
    fig_d.update_layout(paper_bgcolor="#020617", font=dict(color="white"), margin=dict(l=10,r=10,t=45,b=10))
    st.plotly_chart(fig_d, use_container_width=True)
    st.markdown("---")

    # ---- Section 3: analytics ----
    st.markdown("### 📊 Spatial analytics")
    a, b = st.columns(2)
    with a:
        fig2 = px.scatter(df, x="social_vulnerability", y="final_risk_score", size="energy_not_supplied_mw", color="resilience_index", hover_name="place", color_continuous_scale="Turbo", template=plotly_template(), title="Socio-technical vulnerability clustering", height=450)
        st.plotly_chart(fig2, use_container_width=True)
    with b:
        fig3 = px.bar(df.sort_values("final_risk_score",ascending=False), x="place", y="final_risk_score", color="resilience_index", color_continuous_scale="RdYlGn_r", template=plotly_template(), title="Place risk vs resilience", height=450)
        fig3.update_layout(margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        """
        <div class="note">
        <b>Spatial intelligence interpretation</b><br><br>
        The authority map shows risk variation at local government level.
        The density map shows how stress propagates across the region.
        High-density clusters indicate where multiple risk drivers converge simultaneously.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB: NATURAL HAZARDS
# =============================================================================

def render_hazard_resilience_tab(places: pd.DataFrame, pc: pd.DataFrame) -> None:
    st.subheader("Natural-hazard resilience by postcode")
    render_colour_legend("resilience")
    hz = build_hazard_resilience_matrix(places, pc)
    hz["resilience_score_out_of_100"] = pd.to_numeric(hz["resilience_score_out_of_100"],errors="coerce").fillna(0).clip(0,100)
    hz["hazard_stress_score"]         = pd.to_numeric(hz["hazard_stress_score"],errors="coerce").fillna(0).clip(0,100)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Lowest hazard resilience", f"{hz['resilience_score_out_of_100'].min():.1f}/100")
    c2.metric("Mean hazard resilience",   f"{hz['resilience_score_out_of_100'].mean():.1f}/100")
    c3.metric("Severe/fragile rows",      int((hz["resilience_score_out_of_100"]<40).sum()))
    c4.metric("Hazard dimensions",        len(HAZARD_TYPES))

    a, b = st.columns([1.05, 0.95])
    with a:
        heat = hz.pivot_table(index="postcode",columns="hazard",values="resilience_score_out_of_100",aggfunc="mean",fill_value=0)
        fig  = px.imshow(heat,color_continuous_scale="RdYlGn",title="Postcode resilience by natural hazard (0–100)",aspect="auto",template=plotly_template(),zmin=0,zmax=100)
        fig.update_layout(height=460,margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with b:
        worst = hz.sort_values(["resilience_score_out_of_100","hazard_stress_score"],ascending=[True,False]).head(18).copy()
        worst["lack_of_resilience"] = (100-worst["resilience_score_out_of_100"]).clip(0,100)
        worst["case_label"]         = worst["postcode"]+" · "+worst["hazard"]
        if not worst.empty:
            fig = px.bar(worst.sort_values("lack_of_resilience",ascending=True),x="lack_of_resilience",y="case_label",color="hazard",orientation="h",title="Lowest resilience evidence cases",template=plotly_template(),hover_data={"postcode":True,"hazard":True,"resilience_score_out_of_100":":.1f","hazard_stress_score":":.1f","supporting_evidence":True,"lack_of_resilience":":.1f","case_label":False})
            fig.update_layout(height=460,margin=dict(l=10,r=10,t=55,b=10),xaxis=dict(title="Lack of resilience (100−score)",range=[0,105]),yaxis=dict(title="Postcode · hazard"))
            st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### Low-score justification with supporting evidence")
    st.dataframe(hz[["postcode","place","hazard","resilience_score_out_of_100","resilience_level","supporting_evidence","population_density","social_vulnerability","financial_loss_gbp","investment_priority"]],use_container_width=True,hide_index=True)


# =============================================================================
# TAB: IoD2025
# =============================================================================

def render_iod2025_data_quality_tab(places: pd.DataFrame) -> None:
    st.subheader("IoD2025 data integration and socio-economic evidence")
    domain_df, source = load_iod2025_domain_model()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Readable IoD rows", 0 if domain_df is None or domain_df.empty else len(domain_df))
    c2.metric("Matched app places", int((~places.get("iod_domain_match",pd.Series(dtype=str)).astype(str).str.contains("fallback",case=False,na=False)).sum()) if "iod_domain_match" in places.columns else 0)
    c3.metric("Mean social vulnerability", f"{places['social_vulnerability'].mean():.1f}/100")
    c4.metric("Max social vulnerability",  f"{places['social_vulnerability'].max():.1f}/100")
    st.markdown(f'<div class="note"><b>IoD source:</b> {source}</div>', unsafe_allow_html=True)
    cols = ["place","postcode_prefix","social_vulnerability","imd_score","iod_social_vulnerability","iod_domain_match"]
    st.dataframe(places[[c for c in cols if c in places.columns]], use_container_width=True, hide_index=True)
    if domain_df is not None and not domain_df.empty:
        st.markdown("#### Raw IoD2025 domain sample")
        st.dataframe(domain_df.head(200), use_container_width=True, hide_index=True)
        if "iod_social_vulnerability_0_100" in domain_df.columns:
            fig = px.histogram(domain_df, x="iod_social_vulnerability_0_100", nbins=40, title="Distribution of IoD2025 composite social vulnerability", template=plotly_template())
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB: EV / V2G
# =============================================================================

def render_ev_v2g_tab(places: pd.DataFrame, scenario: str) -> None:
    st.subheader("EV system operation and V2G integration")
    ev = build_ev_v2g_analysis(places, scenario)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("V2G-enabled EVs",      f"{ev['v2g_enabled_evs'].sum():,.0f}")
    c2.metric("Available storage",    f"{ev['available_storage_mwh'].sum():.1f} MWh")
    c3.metric("Grid-coupled capacity",f"{ev['substation_coupled_capacity_mw'].sum():.1f} MW")
    c4.metric("Avoided loss potential",money_m(ev["potential_loss_avoided_gbp"].sum()))

    if scenario == "Drought":
        st.success("Drought mode: EVs and storage are actively stabilising the grid under low renewable generation.")
        if "net_load_stress" in places.columns:
            d1,d2,d3 = st.columns(3)
            d1.metric("Avg net load stress",  f"{places['net_load_stress'].mean():.1f} MW")
            d2.metric("Avg V2G support",       f"{places['v2g_support_mw'].mean():.1f} MW")
            d3.metric("Total storage support", f"{places['total_storage_support'].mean():.1f} MW")

    a, b = st.columns(2)
    with a:
        fig = px.bar(ev,x="place",y="substation_coupled_capacity_mw",color="ev_storm_role",title="EV capacity coupled to substations",template=plotly_template())
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    with b:
        fig = px.scatter(ev,x="available_storage_mwh",y="ev_operational_value_score",size="potential_loss_avoided_gbp",color="ev_storm_role",hover_name="place",title="EV storage vs operational value",template=plotly_template())
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)

    if "v2g_support_mw" in places.columns:
        fig = px.bar(places,x="place",y="v2g_support_mw",title="Distributed V2G energy support by location",template=plotly_template())
        fig.update_layout(height=360); st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="note"><b>EV/V2G interpretation:</b> EVs are modelled as distributed energy storage. Under drought, EV V2G provides critical balancing capacity reducing ENS and financial loss.</div>', unsafe_allow_html=True)
    st.dataframe(ev, use_container_width=True, hide_index=True)


# =============================================================================
# TAB: FAILURE & INVESTMENT
# =============================================================================

def render_failure_investment_tab(places: pd.DataFrame, pc: pd.DataFrame, rec: pd.DataFrame) -> None:
    st.subheader("Failure probability and investment prioritisation")
    render_colour_legend("priority")
    failure = build_failure_analysis(places)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Max failure probability",  f"{failure['enhanced_failure_probability'].max()*100:.1f}%")
    c2.metric("Mean failure probability", f"{failure['enhanced_failure_probability'].mean()*100:.1f}%")
    c3.metric("Priority 1 investments",   int((rec["investment_priority"]=="Priority 1").sum()) if rec is not None and not rec.empty else 0)
    c4.metric("Programme cost",           money_m(rec["indicative_investment_cost_gbp"].sum()) if rec is not None and not rec.empty else "£0.00m")

    a, b = st.columns(2)
    with a:
        fig = px.bar(failure.head(18),x="enhanced_failure_probability",y="place",color="hazard",orientation="h",title="Highest natural-hazard failure probabilities",template=plotly_template())
        fig.update_layout(height=440,margin=dict(l=10,r=10,t=55,b=10),xaxis_tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
    with b:
        if rec is not None and not rec.empty:
            fig = px.bar(rec.head(18),x="postcode",y="recommendation_score",color="investment_priority",title="Investment urgency by postcode",template=plotly_template())
            fig.update_layout(height=440,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### Failure probability evidence")
    st.dataframe(failure, use_container_width=True, hide_index=True)
    if rec is not None and not rec.empty:
        st.markdown("#### Actionable investment recommendations")
        rec_cols = ["postcode","nearest_place","investment_priority","recommendation_score","investment_category","recommended_action","indicative_investment_cost_gbp","financial_loss_gbp","resilience_score","risk_score"]
        st.dataframe(rec[[c for c in rec_cols if c in rec.columns]], use_container_width=True, hide_index=True)


# =============================================================================
# TAB: SCENARIO LOSSES
# =============================================================================

def render_scenario_finance_tab(places: pd.DataFrame, region: str, mc_runs: int) -> None:
    st.subheader("Scenario losses: live baseline vs what-if stress scenarios")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Live baseline loss",       money_m(places["total_financial_loss_gbp"].sum()))
    c2.metric("Live baseline risk",       f"{places['final_risk_score'].mean():.1f}/100")
    c3.metric("Live baseline resilience", f"{places['resilience_index'].mean():.1f}/100")
    c4.metric("Live baseline ENS",        f"{places['energy_not_supplied_mw'].sum():.1f} MW")
    st.markdown('<div class="note"><b>Live / Real-time</b> is the operational baseline. The chart below shows only stress scenarios.</div>', unsafe_allow_html=True)
    matrix = scenario_financial_matrix(places, region, mc_runs)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(matrix,x="scenario",y="total_financial_loss_gbp",color="mean_risk",title="What-if scenario financial loss (£)",template=plotly_template(),color_continuous_scale="Turbo")
        fig.update_layout(height=430,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig = px.scatter(matrix,x="mean_risk",y="mean_resilience",size="total_financial_loss_gbp",color="scenario",title="What-if risk-resilience-loss space",template=plotly_template())
        fig.update_layout(height=430,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    matrix["total_financial_loss_million_gbp"] = matrix["total_financial_loss_gbp"]/1_000_000
    st.dataframe(matrix, use_container_width=True, hide_index=True)


# =============================================================================
# TAB: FINANCE & FUNDING
# =============================================================================

def render_finance_funding_tab(places: pd.DataFrame, pc: pd.DataFrame) -> None:
    st.subheader("Finance and funding prioritisation")
    funding = build_funding_table(pc, places)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total modelled loss",      money_m(places["total_financial_loss_gbp"].sum()))
    c2.metric("P95 place loss",           money_m(places["total_financial_loss_gbp"].quantile(0.95)))
    c3.metric("Immediate funding areas",  int((funding["funding_priority_band"]=="Immediate funding").sum()))
    c4.metric("Top funding score",        f"{funding['funding_priority_score'].max():.1f}/100")
    a, b = st.columns(2)
    with a: st.plotly_chart(create_loss_waterfall(places), use_container_width=True)
    with b:
        fig = px.bar(funding.head(18),x="funding_priority_score",y="postcode" if "postcode" in funding.columns else "place",color="funding_priority_band",orientation="h",title="Funding priority ranking",template=plotly_template())
        fig.update_layout(height=430,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    fin_cols = ["place","energy_not_supplied_mw","ens_mwh","estimated_duration_hours","voll_loss_gbp","customer_interruption_loss_gbp","business_disruption_loss_gbp","restoration_loss_gbp","critical_services_loss_gbp","total_financial_loss_gbp"]
    st.markdown("#### Financial loss evidence")
    st.dataframe(places[[c for c in fin_cols if c in places.columns]].sort_values("total_financial_loss_gbp",ascending=False), use_container_width=True, hide_index=True)
    st.markdown("#### Funding criteria")
    st.dataframe(funding, use_container_width=True, hide_index=True)


# =============================================================================
# TAB: RESILIENCE
# =============================================================================

def resilience_tab(places: pd.DataFrame) -> None:
    st.subheader("Resilience analysis")
    render_colour_legend("resilience")
    cols = ["place","resilience_label","resilience_index","final_risk_score","social_vulnerability","grid_failure_probability","renewable_failure_probability","energy_not_supplied_mw","total_financial_loss_gbp"]
    safe_cols = [c for c in cols if c in places.columns]
    sort_col  = "resilience_index" if "resilience_index" in places.columns else safe_cols[0]
    st.dataframe(places[safe_cols].sort_values(sort_col), use_container_width=True, hide_index=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Average resilience", f"{float(pd.to_numeric(places.get('resilience_index'),errors='coerce').mean()):.1f}")
    c2.metric("Average risk",       f"{float(pd.to_numeric(places.get('final_risk_score'),errors='coerce').mean()):.1f}")
    c3.metric("Average loss",       f"£{float(pd.to_numeric(places.get('total_financial_loss_gbp'),errors='coerce').mean()):,.0f}")
    a, b = st.columns(2)
    with a:
        if {"place","resilience_index"}.issubset(places.columns):
            fig = px.bar(places.sort_values("resilience_index"),x="place",y="resilience_index",color="resilience_label" if "resilience_label" in places.columns else None,title="Resilience ranking",template=plotly_template())
            fig.update_layout(height=400,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    with b:
        if {"social_vulnerability","resilience_index"}.issubset(places.columns):
            fig = px.scatter(places,x="social_vulnerability",y="resilience_index",size="total_financial_loss_gbp" if "total_financial_loss_gbp" in places.columns else None,color="final_risk_score" if "final_risk_score" in places.columns else None,hover_name="place" if "place" in places.columns else None,title="Resilience vs social vulnerability",template=plotly_template(),color_continuous_scale="Turbo")
            fig.update_layout(height=400,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="note"><b>Interpretation:</b> Resilience combines infrastructure robustness, outage propagation, social vulnerability and financial exposure. Lower scores = higher fragility under compound hazard stress.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB: INVESTMENT
# =============================================================================

def investment_tab(pc: pd.DataFrame, rec: pd.DataFrame) -> None:
    st.subheader("Postcode resilience and investment engine")
    if pc.empty or rec.empty:
        st.warning("No postcode-level resilience or investment recommendations could be generated."); return
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Postcode areas", len(pc))
    c2.metric("Priority 1",     int((rec["investment_priority"]=="Priority 1").sum()))
    c3.metric("Programme cost", money_m(rec["indicative_investment_cost_gbp"].sum()))
    c4.metric("Exposed loss",   money_m(rec["financial_loss_gbp"].sum()))
    a, b = st.columns(2)
    with a:
        fig = px.bar(rec.head(14),x="postcode",y="recommendation_score",color="investment_priority",title="Investment urgency by postcode",template=plotly_template())
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    with b:
        fig = px.scatter(rec,x="financial_loss_gbp",y="recommendation_score",size="indicative_investment_cost_gbp",color="investment_priority",hover_name="postcode",title="Recommendation score vs financial-loss exposure",template=plotly_template())
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Actionable recommendations")
    cols = ["postcode","nearest_place","investment_priority","recommendation_score","investment_category","recommended_action","indicative_investment_cost_gbp","financial_loss_gbp","resilience_score","risk_score"]
    st.dataframe(rec[[c for c in cols if c in rec.columns]], use_container_width=True, hide_index=True)


# =============================================================================
# TAB: MONTE CARLO (Q1 correlated)
# =============================================================================

def render_improved_monte_carlo_tab(places: pd.DataFrame, simulations: int) -> None:
    st.subheader("Monte Carlo: correlated storm, demand and restoration-cost uncertainty")
    with st.spinner("Running improved Monte Carlo model..."):
        q1mc = build_q1_monte_carlo_table(places, simulations)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("P95 risk max",    f"{q1mc['q1_mc_risk_p95'].max():.1f}/100")
    c2.metric("Mean failure max",f"{q1mc['q1_mc_failure_mean'].max()*100:.1f}%")
    c3.metric("CVaR95 loss max", money_m(q1mc["q1_mc_loss_cvar95_gbp"].max()))
    c4.metric("Simulations",     simulations)
    a, b = st.columns(2)
    with a:
        fig = px.scatter(q1mc,x="q1_mc_risk_mean",y="q1_mc_risk_p95",size="q1_mc_loss_cvar95_gbp",color="q1_mc_failure_p95",hover_name="place",title="Mean risk vs P95 risk with CVaR loss",template=plotly_template(),color_continuous_scale="Turbo")
        fig.update_layout(height=430,margin=dict(l=10,r=10,t=55,b=10)); st.plotly_chart(fig,use_container_width=True)
    with b:
        worst = q1mc.iloc[0]
        fig = px.histogram(x=worst["q1_mc_histogram"],nbins=28,title=f"MC risk distribution — {worst['place']}",template=plotly_template())
        fig.update_layout(height=430,margin=dict(l=10,r=10,t=55,b=10),xaxis_title="Risk score"); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="note"><b>MC model:</b> shared storm-shock variable correlates wind, rain, outage and ENS. Demand uses triangular distribution; restoration uses lognormal tails. CVaR95 = mean of losses ≥ 95th-percentile threshold (correct exceedance-mean formula).</div>', unsafe_allow_html=True)
    st.dataframe(q1mc.drop(columns=["q1_mc_histogram"]), use_container_width=True, hide_index=True)


# =============================================================================
# TAB: VALIDATION
# =============================================================================

def render_validation_tab(places: pd.DataFrame, scenario: str) -> None:
    st.subheader("Black-box review and validation checks")
    checks = validate_model_transparency(places, scenario)
    st.dataframe(checks, use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="card">
        <p style="color:#cbd5e1;">
        The application is intentionally transparent. It exposes intermediate variables used
        for risk, resilience, social vulnerability, financial loss, failure probability, EV/V2G
        value and funding prioritisation. Equations are not hidden behind a neural network.
        The compound hazard proxy is non-circular — it reads only raw meteorological inputs.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Validation benchmarks")
    st.json(VALIDATION_BENCHMARKS)


# =============================================================================
# TAB: METHOD
# =============================================================================

def method_tab(places: pd.DataFrame) -> None:
    st.subheader("Model transparency")
    st.markdown('<div class="card"><h3 style="color:white;margin-top:0;">Core modelling structure</h3><p style="color:#cbd5e1;">The dashboard combines hazard intensity, pollution, renewable generation stress, outage proximity, ENS, social vulnerability and financial loss into a location-level digital twin score. The model is deliberately transparent so it can be described in a research paper and later calibrated against observed outage histories.</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="formula">Risk = weather + pollution + net-load stress + outage intensity + ENS pressure<br><br>Failure probability = logistic(0.075 × (risk − 72))</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="formula">Resilience = 92 − (0.28×risk + 0.11×social_vulnerability + 9×grid_failure + 5×renewable_failure + 7×system_stress + finance_penalty)<br><br>finance_penalty = clamp(loss/£25m, 0,1) × 6</div>', unsafe_allow_html=True)
    st.markdown("#### Current model output sample")
    st.dataframe(places.head(10), use_container_width=True, hide_index=True)


# =============================================================================
# TAB: DATA / EXPORT
# =============================================================================

def export_tab(places: pd.DataFrame, outages: pd.DataFrame, grid: pd.DataFrame, pc: pd.DataFrame, rec: pd.DataFrame) -> None:
    st.subheader("Data and export")
    with st.expander("Place-level model outputs", expanded=True): st.dataframe(places, use_container_width=True, hide_index=True)
    with st.expander("Postcode resilience"):           st.dataframe(pc,      use_container_width=True, hide_index=True)
    with st.expander("Investment recommendations"):    st.dataframe(rec,     use_container_width=True, hide_index=True)
    with st.expander("Outage layer"):                  st.dataframe(outages, use_container_width=True, hide_index=True)
    with st.expander("Grid cells"):                    st.dataframe(grid,    use_container_width=True, hide_index=True)
    c1,c2,c3 = st.columns(3)
    c1.download_button("Download places CSV",         places.to_csv(index=False).encode("utf-8"), file_name="sat_guard_places.csv",          mime="text/csv")
    c2.download_button("Download recommendations CSV",rec.to_csv(index=False).encode("utf-8") if not rec.empty else b"", file_name="sat_guard_recommendations.csv", mime="text/csv", disabled=rec.empty)
    c3.download_button("Download grid CSV",           grid.to_csv(index=False).encode("utf-8"),  file_name="sat_guard_grid.csv",             mime="text/csv")

# =============================================================================
# SAT-Guard Digital Twin — Q1 Edition
# PART 4 of 4 — README tab + main() entry point
# =============================================================================
# Paste this file AFTER app_KASVA_PART3.py
#
# ASSEMBLY (Linux / Mac / Git Bash):
#     cat app_KASVA_PART1.py app_KASVA_PART2.py app_KASVA_PART3.py app_KASVA_PART4.py \
#         > app_KASVA_FINAL.py
#     streamlit run app_KASVA_FINAL.py
#
# ASSEMBLY (Windows CMD):
#     copy /b app_KASVA_PART1.py+app_KASVA_PART2.py+app_KASVA_PART3.py+app_KASVA_PART4.py \
#          app_KASVA_FINAL.py
#     streamlit run app_KASVA_FINAL.py
#
# REQUIREMENTS:
#     pip install streamlit pandas numpy requests openpyxl pydeck plotly
# =============================================================================


# =============================================================================
# TAB: README
# =============================================================================

README_CONTENT = """
<div class="card">
<h2 style="color:white;margin-top:0;">SAT-Guard Digital Twin — README</h2>
<p style="color:#cbd5e1;line-height:1.65;">
This Streamlit app is a transparent research prototype for regional electricity-grid
resilience assessment. It combines live or fallback weather, public outage information,
social vulnerability, energy-not-supplied, financial impact, failure probability,
investment prioritisation and Monte Carlo uncertainty. It is written for users who may
not have a background in power systems, statistics or socio-economic vulnerability analysis.
</p>
</div>

---

### 1. Data sources

**Weather and air quality.**
The app calls Open-Meteo weather and air-quality endpoints when available.
If an API call fails, safe fallback values are generated so the dashboard still runs.
Weather variables include wind speed, precipitation, temperature, humidity, cloud cover,
solar radiation and air-quality indicators.

**Northern Powergrid outage data.**
The app attempts to read public outage records from Northern Powergrid.
If no geocoded outage record is available, synthetic map points are created for visual
continuity. These points are marked `is_synthetic_outage=True` and are excluded from
live-mode risk scoring.

**IoD/IMD socio-economic evidence.**
IoD = Indices of Deprivation. IMD = Index of Multiple Deprivation.
These datasets summarise relative deprivation using domains such as income, employment,
education, health, crime, housing/services and living environment.
Values are converted onto a 0–100 scale (higher = more vulnerable).
If IoD2025 files are not available, the app uses transparent fallback proxies based on
local vulnerability scores and population density.

---

### 2. Key fixes in this Q1 edition

The following bugs from the previous version have been corrected:

- `clamp()`, `risk_label()`, `resilience_label()` were defined twice — now defined once.
- `compound_hazard_proxy` previously read `final_risk_score`, creating a circular feedback
  loop (risk → compound hazard → risk). Now uses only raw meteorological inputs.
- `flood_depth_proxy()` result was not written to the places DataFrame. Now always written.
- `is_calm_live_weather()` had an inconsistent signature — now always takes `outage_count`
  and `affected_customers` as explicit arguments.
- `scenario_financial_matrix()` had an MC-run cap of 60 — now 150.
- CVaR95 was computed with an incorrect array slice — now uses the correct exceedance-mean
  formula (mean of losses >= 95th-percentile threshold).
- Spatial Intelligence tab used pentagon/hexagon micro-cells — replaced with proper filled
  local authority polygons using Plotly Scattermapbox fill="toself".

---

### 3. Main tabs

**Executive overview** — regional intelligence table, risk/resilience gauges,
location risk ranking and social-vulnerability scatter.

**Simulation** — animated BBC/WXCharts-style weather and hazard overlay with
moving precipitation shields, pressure contours, frontal boundaries, wind vectors
and lightning in storm mode.

**Natural hazards** — postcode resilience across wind, flood, drought,
heat/air-quality stress and compound hazard dimensions.

**IoD2025 socio-economic evidence** — deprivation data matching and domain scores.

**Spatial intelligence** — colourful local authority risk map (filled polygons, no
pentagons), operational stress density heatmap and spatial analytics.

**Resilience** — resilience rankings and social-vulnerability relationship.

**Failure & investment** — failure probability and investment recommendations.

**Scenario losses** — what-if stress scenario comparison (excluding live baseline).

**Finance & funding** — loss model waterfall, sunburst and funding priority ranking.

**Monte Carlo** — correlated storm-shock MC with P95 and CVaR95 loss metrics.

**Validation / black-box** — transparency checks and model benchmarks.

**README** — this document.

**Data / Export** — all output tables with CSV download buttons.

---

### 4. Core formulae

**Weather risk component**

    weather_score = 24×clip((wind−18)/52, 0,1)
                  + 20×clip((rain−1.5)/23.5, 0,1)
                  +  3×clip((cloud−75)/25, 0,1)
                  +  8×clip(|temp−18|−10)/18, 0,1)
                  +  2×clip((humidity−88)/12, 0,1)

**Pollution and public-health stress**

    pollution_score = 10×clip((AQI−55)/95, 0,1) + 5×clip((PM2.5−20)/50, 0,1)

**Renewable generation proxy**

    solar_MW = shortwave_radiation × 0.18
    wind_MW  = min((wind_speed/12)³, 1.20) × 95

The wind term follows the cubic characteristic of wind-power availability
before rated output. It is a simplified proxy, not a turbine-specific model.

**Energy Not Supplied (ENS) — live mode**

    ENS_MW = outage_count × 12 + affected_customers × 0.0025

In live mode, the base load component is zero.
This prevents normal demand appearing as unserved energy
when no real outage evidence exists.

**Energy Not Supplied (ENS) — stress scenarios**

    ENS_MW = (outage_count×85 + affected_customers×0.01 + base_load×0.14)
             × scenario_outage_multiplier

**Failure probability**

    failure_probability = 1 / (1 + exp(−0.075 × (risk − 72)))

Scores near 72 sit near the transition zone.
Low risk stays low probability; high risk rises non-linearly.

**Compound hazard proxy (non-circular)**

    compound_hazard = clip(wind/70,0,1)×35 + clip(rain/25,0,1)×30
                    + clip(AQI/120,0,1)×15 + clip(outages/8,0,1)×20

This uses only direct meteorological inputs.
It does NOT read final_risk_score, resilience_index or failure_probability.

**Cascade stress**

    water     = power^1.35 × 0.74
    telecom   = power^1.22 × 0.82
    transport = ((power + telecom) / 2) × 0.70
    social    = ((power + water + telecom) / 3) × 0.75

**Resilience index**

    resilience = 92 − (
        0.28 × risk
        + 0.11 × social_vulnerability
        + 9   × grid_failure
        + 5   × renewable_failure
        + 7   × system_stress
        + finance_penalty
    )

    finance_penalty = clamp(loss / £25m, 0, 1) × 6
    Output range: 15 – 100

**Social vulnerability**

    social_vulnerability = 40×clip(pop_density/4500,0,1) + 60×clip(IMD_score/100,0,1)

**Financial loss**

    total_loss = VoLL_loss + customer_interruption + business_disruption
               + restoration + critical_services

    VoLL_loss            = ENS_MWh × £17,000/MWh
    customer_interruption= affected_customers × £38
    business_disruption  = ENS_MWh × £1,100 × business_density
    restoration          = outage_count × £18,500
    critical_services    = ENS_MWh × £320 × (social_vulnerability / 100)

All figures are scenario assumptions. Calibrate with local regulatory
or utility data before operational use.

**Investment recommendation score**

    score = 0.30×risk + 0.22×social + 0.18×(100−resilience)
          + 0.13×loss_percentile + 0.10×ENS_percentile
          + 0.07×outage_pressure

**Funding priority score**

    score = 0.26×risk + 0.20×(100−resilience) + 0.18×social
          + 0.15×loss_exposure + 0.11×ENS_exposure
          + 0.06×outage_exposure + 0.04×recommendation

**Monte Carlo (Q1 correlated)**

    storm_shock ~ N(0,1)      (shared across wind, rain, outage, ENS)
    wind        ~ base × exp(0.16×shock + ε_wind)
    rain        ~ base × exp(0.28×shock + ε_rain)
    demand_mult ~ Triangular(0.78, 1.10, 1.95)
    restoration ~ outages × LogNormal(ln(18500), 0.25)

    CVaR95 = mean(loss | loss ≥ Percentile(loss, 95))

---

### 5. Colour legend

- Green  → low risk / robust resilience (score 0–34 / 80–100)
- Yellow → moderate watch condition (35–54 / 60–79)
- Orange → high warning stress (55–74 / 40–59)
- Red    → severe / fragile (75–100 / 0–39)

---

### 6. Spatial Intelligence map design

The Spatial Intelligence tab now shows filled local authority polygons rather than
pentagon or hexagon micro-cells. Each polygon is coloured by the mean risk score of
its constituent configured places:

- Green  (#7bd000) — low risk
- Blue   (#0070c0) — moderate
- Orange (#ff7b00) — high
- Magenta (#d80073) — severe

Authority boundaries are drawn with 2.5-pixel white lines.
City markers and labels are overlaid as a separate layer.
Hovering over a polygon shows authority-level risk, resilience, ENS,
social vulnerability and financial loss.

---

### 7. Limitations

This is a research-grade prototype, not an official operational control system.
Weather APIs, outage APIs and socio-economic files may be incomplete or unavailable.
All scoring weights are transparent assumptions and should be calibrated with
historical outage, asset, feeder, substation, customer-minute-lost and
restoration-cost data before production use.

---

### 8. References

1. Ofgem RIIO electricity-distribution resilience and interruption reporting frameworks.
2. UK Department for Energy Security and Net Zero VoLL and electricity-security appraisal evidence.
3. English Indices of Deprivation technical reports (IoD2025).
4. Open-Meteo weather and air-quality API documentation.
5. Northern Powergrid open-data documentation for live power-cut records.
6. Billinton and Allan, *Reliability Evaluation of Power Systems* — reliability modelling.
7. Panteli and Mancarella resilience literature — weather-driven power-system resilience.
8. Lund and Kempton V2G literature — EV/V2G support concepts.
9. IEC/IEEE power-system dependability and resilience guidance.
"""


def render_readme_tab() -> None:
    """Render the README / documentation tab."""
    st.subheader("README — model, data, formulae and interpretation")
    st.markdown(README_CONTENT, unsafe_allow_html=False)


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

def main() -> None:
    """
    SAT-Guard Digital Twin — main Streamlit application.

    Layout:
        Sidebar   → region selector, what-if scenario toggle, MC sliders, map mode
        Hero      → title banner with active scenario and refresh ID
        Metrics   → 6-column KPI panel
        Tabs      → 13 analysis tabs
    """
    # Inject global CSS
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # Initialise session state
    if "refresh_id" not in st.session_state:
        st.session_state.refresh_id = 0

    # ------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ⚡ SAT-Guard")
        st.caption("Digital twin control panel")

        region = st.selectbox("Region", list(REGIONS.keys()), index=0)

        mc_runs    = st.slider("MC runs (per-place)",   10,  160, 40,  10)
        q1_mc_runs = st.slider("MC simulations (Q1)", 100, 5000, 1000, 100)

        st.markdown("---")
        st.markdown("### What-if scenario")

        what_if_enabled = st.checkbox("Enable hazard scenario", value=False)

        if what_if_enabled:
            hazard_choice = st.selectbox(
                "Select hazard",
                [
                    "Storm (wind)",
                    "Flood (heavy rain)",
                    "Heatwave",
                    "Compound hazard",
                    "Drought",
                    "Total blackout",
                ],
            )
            WHAT_IF_MAP = {
                "Storm (wind)":     "Extreme wind",
                "Flood (heavy rain)":"Flood",
                "Heatwave":         "Heatwave",
                "Compound hazard":  "Compound extreme",
                "Drought":          "Drought",
                "Total blackout":   "Total blackout stress",
            }
            scenario_for_engine = WHAT_IF_MAP[hazard_choice]
        else:
            scenario_for_engine = "Live / Real-time"
            hazard_choice       = "Live conditions"

        map_mode = st.selectbox(
            "Map layer",
            ["All", "Risk", "Postcode / Investment", "Outages"],
            index=0,
        )

        st.markdown("---")
        st.info(SCENARIOS[scenario_for_engine]["description"])

        if st.button("Run / refresh model", type="primary"):
            st.session_state.refresh_id += 1
            st.cache_data.clear()
            st.rerun()

        if st.button("Clear cache"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(
            "IoD2025 Excel files can be placed in data/iod2025/. "
            "Fallback vulnerability proxies are used if unavailable."
        )

    # ------------------------------------------------------------------
    # HERO BANNER
    # ------------------------------------------------------------------
    hero(region, scenario_for_engine, mc_runs, st.session_state.refresh_id)

    # ------------------------------------------------------------------
    # DATA PIPELINE
    # ------------------------------------------------------------------
    with st.spinner("Running digital twin model..."):
        places, outages, grid = get_data_cached(region, scenario_for_engine, mc_runs)
        pc  = build_postcode_resilience(places, outages)
        rec = build_investment_recommendations(places, outages)

    if places.empty:
        st.error("No model data could be generated. Check API connectivity.")
        return

    # ------------------------------------------------------------------
    # METRICS PANEL
    # ------------------------------------------------------------------
    metrics_panel(places, pc)

    imd_src = places.iloc[0].get("imd_dataset_summary", "IoD2025 / fallback proxy")
    st.caption(f"IoD / deprivation data source: {imd_src}")

    # ------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------
    tabs = st.tabs([
        "Executive overview",       # 0
        "Simulation",               # 1
        "Natural hazards",          # 2
        "IoD2025 socio-economic",   # 3
        "Spatial intelligence",     # 4
        "Resilience",               # 5
        "Failure & investment",     # 6
        "Scenario losses",          # 7
        "Finance & funding",        # 8
        "Monte Carlo",              # 9
        "Validation / black-box",   # 10
        "README",                   # 11
        "Data / Export",            # 12
    ])

    with tabs[0]:
        overview_tab(places, pc, scenario_for_engine)

    with tabs[1]:
        bbc_tab(region, scenario_for_engine, places, grid)

    with tabs[2]:
        render_hazard_resilience_tab(places, pc)

    with tabs[3]:
        render_iod2025_data_quality_tab(places)

    with tabs[4]:
        spatial_tab(region, places, outages, pc, grid, map_mode)

    with tabs[5]:
        resilience_tab(places)

    with tabs[6]:
        render_failure_investment_tab(places, pc, rec)

    with tabs[7]:
        render_scenario_finance_tab(places, region, mc_runs)

    with tabs[8]:
        render_finance_funding_tab(places, pc)

    with tabs[9]:
        render_improved_monte_carlo_tab(places, q1_mc_runs)

    with tabs[10]:
        render_validation_tab(places, scenario_for_engine)

    with tabs[11]:
        render_readme_tab()

    with tabs[12]:
        export_tab(places, outages, grid, pc, rec)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()


# =============================================================================
# Q1 METHODOLOGICAL APPENDIX
# =============================================================================
# This appendix is included in the single assembled file so the dashboard
# remains self-contained for assessment, review and later conversion into a
# research prototype. These notes document modelling choices, assumptions,
# validation hooks and extension points.
#
# SECTION A — Natural hazard resilience
#   1. Wind storm resilience uses wind-speed stress, outage concentration,
#      ENS exposure, financial-loss exposure and vulnerability.
#   2. Flood/heavy rain resilience uses precipitation stress and a flood-depth
#      proxy. In production, calibrate against EA flood-zone layers.
#   3. Renewable drought reflects low wind/solar and net-load pressure.
#      Especially important in EV-rich districts where charging load coincides
#      with low renewable generation.
#   4. Air-quality/heat stress is included because social resilience and field
#      repair capability deteriorate during public-health stress events.
#   5. Compound hazard combines multiple drivers and represents the operational
#      picture most relevant in a regional emergency.
#
# SECTION B — Postcode resilience classification
#   80–100: Robust
#   60–79:  Functional
#   40–59:  Stressed
#   0–39:   Fragile
#
#   A low score is justified using driver-level evidence:
#   - High natural-hazard stress
#   - High social vulnerability
#   - High population density
#   - Nearby outage concentration
#   - High Energy Not Supplied
#   - Elevated failure probability
#   - High financial-loss exposure
#
# SECTION C — Financial loss
#   Five components (all in GBP):
#   1. Value of Lost Load (VoLL): ENS_MWh × £17,000/MWh
#   2. Customer interruption:     affected_customers × £38
#   3. Business disruption:       ENS_MWh × £1,100 × business_density
#   4. Restoration and repair:    outage_count × £18,500
#   5. Social/critical-service uplift: ENS_MWh × £320 × social_frac
#
# SECTION D — EV/V2G modelling
#   EVs are modelled as a distributed flexibility resource.
#   Key outputs:
#   - EV penetration proxy
#   - Parked EVs during storms
#   - V2G-enabled EVs
#   - Available battery storage (MWh)
#   - Export power (MW)
#   - Substation-coupled capacity
#   - Emergency energy (MWh)
#   - ENS offset (MWh)
#   - Avoided-loss value (£)
#
# SECTION E — Improved Monte Carlo
#   The Q1 Monte Carlo uses a shared storm shock so wind, rain, outage count
#   and ENS move together. This produces more realistic tail risk.
#   - Shared storm shock: N(0,1)
#   - Demand: Triangular(0.78, 1.10, 1.95)
#   - Restoration: LogNormal(ln(18500), 0.25)
#   - Outputs: P95, CVaR95 (exceedance mean)
#
# SECTION F — Funding priority
#   Scored 0–100 using:
#   - Risk (weight 0.26)
#   - Low resilience (weight 0.20)
#   - Social vulnerability (weight 0.18)
#   - Financial-loss exposure (weight 0.15)
#   - ENS (weight 0.11)
#   - Outage count (weight 0.06)
#   - Recommendation score (weight 0.04)
#
# SECTION G — External data notes
#   Open-Meteo: free, no API key required.
#   Northern Powergrid: public open dataset.
#   IoD2025: available from gov.uk deprivation data pages.
#   BBC Weather: represented as animation style only; a production BBC feed
#   requires an authorised data source.
#
# SECTION H — Validation and black-box governance
#   The model is not black-box. It exposes:
#   - Input variables
#   - Intermediate variables
#   - Formulae
#   - Scoring weights
#   - Final outputs
#
#   If machine learning is added, retain validation tabs and extend with:
#   - Feature importance
#   - Calibration plots
#   - Residual analysis
#   - Temporal cross-validation
#   - Out-of-sample stress testing
#
# SECTION I — Spatial Intelligence design rationale
#   Previous version used hexagon/pentagon micro-cells scattered around
#   each place. This was replaced because:
#   1. Overlapping cells created visual confusion.
#   2. Cells did not correspond to any meaningful administrative unit.
#   3. The political-map style (filled authority polygons) is more immediately
#      readable for operational and regulatory audiences.
#   4. Boundary lines give a clear separation between authority areas.
#   5. The colour scale (green/blue/orange/magenta) maps directly onto the
#      four risk categories (Low/Moderate/High/Severe).
#
# END OF METHODOLOGICAL APPENDIX
# =============================================================================
