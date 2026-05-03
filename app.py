"""
SAT-Guard Q1 Advanced Streamlit Dashboard
========================================

Single-file Streamlit application for a Q1-quality digital twin dashboard.

This file is intentionally self-contained:
- Live/fallback weather and outage ingestion
- Scenario-based grid risk and resilience modelling
- Monte Carlo uncertainty
- Financial loss estimation
- Postcode resilience and investment recommendation engine
- Advanced Streamlit UI
- Plotly analytics
- PyDeck spatial visualisation
- BBC/WXCharts-inspired animated weather component using embedded HTML/CSS/JS

Run:
    pip install streamlit pandas numpy requests openpyxl pydeck plotly
    streamlit run streamlit_app_q1.py

Recommended Streamlit Cloud main file:
    streamlit_app_q1.py

Recommended requirements.txt:
    streamlit
    pandas
    numpy
    requests
    openpyxl
    pydeck
    plotly
"""

from __future__ import annotations

import base64
import html
import json
import math
import random
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SAT-Guard Q1 Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# GLOBAL STYLE
# =============================================================================

APP_CSS = """
<style>
:root {
    --bg: #020617;
    --panel: rgba(15, 23, 42, 0.82);
    --panel2: rgba(30, 41, 59, 0.68);
    --border: rgba(148, 163, 184, 0.22);
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
        radial-gradient(circle at 70% 20%, rgba(168,85,247,0.12), transparent 34%),
        linear-gradient(180deg, #020617 0%, #050816 42%, #020617 100%);
}
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2.5rem;
}
[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.96);
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}
.q1-hero {
    border: 1px solid rgba(148,163,184,0.20);
    background:
        linear-gradient(135deg, rgba(14,165,233,0.20), rgba(168,85,247,0.10)),
        rgba(15,23,42,0.82);
    border-radius: 28px;
    padding: 22px 24px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.32);
    margin-bottom: 18px;
}
.q1-title {
    font-size: 38px;
    font-weight: 950;
    letter-spacing: -0.05em;
    color: white;
    margin-bottom: 4px;
}
.q1-subtitle {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.5;
}
.q1-chip {
    display: inline-block;
    margin: 4px 6px 0 0;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.25);
    background: rgba(2,6,23,0.58);
    padding: 7px 12px;
    color: #bfdbfe;
    font-weight: 800;
    font-size: 12px;
}
.q1-card {
    border: 1px solid rgba(148,163,184,0.18);
    background: rgba(15,23,42,0.72);
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.26);
}
.q1-note {
    border: 1px solid rgba(56,189,248,0.25);
    background: rgba(56,189,248,0.09);
    border-radius: 18px;
    padding: 14px 16px;
    color: #dbeafe;
}
.q1-warning {
    border: 1px solid rgba(249,115,22,0.30);
    background: rgba(249,115,22,0.10);
    border-radius: 18px;
    padding: 14px 16px;
    color: #fed7aa;
}
.q1-formula {
    border-left: 4px solid #38bdf8;
    background: rgba(2,6,23,0.50);
    padding: 12px 14px;
    border-radius: 12px;
    color: #e0f2fe;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
}
.stMetric {
    background: rgba(15, 23, 42, 0.56);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 12px 14px;
    box-shadow: 0 12px 34px rgba(0,0,0,0.22);
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
# CONFIGURATION
# =============================================================================

NPG_DATASET_URL = (
    "https://northernpowergrid.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/live-power-cuts-data/records"
)

OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_CURRENT_VARS = ",".join([
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "relative_humidity_2m",
    "precipitation",
    "is_day",
])

AIR_CURRENT_VARS = ",".join([
    "european_aqi",
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
])

SCENARIOS = {
    "Live / Real-time": {
        "wind": 1.00,
        "rain": 1.00,
        "temperature": 0.0,
        "aqi": 1.00,
        "solar": 1.00,
        "outage": 1.00,
        "finance": 1.00,
        "hazard_mode": "wind",
        "description": "Current data without deliberate stress multipliers.",
    },
    "Extreme wind": {
        "wind": 1.95,
        "rain": 1.10,
        "temperature": -1.0,
        "aqi": 1.05,
        "solar": 0.82,
        "outage": 1.35,
        "finance": 1.25,
        "hazard_mode": "wind",
        "description": "Strong wind stress affecting overhead lines and local network assets.",
    },
    "Flood cascade": {
        "wind": 1.25,
        "rain": 3.25,
        "temperature": 0.0,
        "aqi": 1.12,
        "solar": 0.42,
        "outage": 1.90,
        "finance": 1.55,
        "hazard_mode": "rain",
        "description": "Heavy precipitation, flood-depth proxy and infrastructure disruption.",
    },
    "Renewable collapse": {
        "wind": 0.32,
        "rain": 0.40,
        "temperature": 2.0,
        "aqi": 1.20,
        "solar": 0.10,
        "outage": 1.12,
        "finance": 1.35,
        "hazard_mode": "calm",
        "description": "Low wind and low solar generation causing net-load stress.",
    },
    "Total blackout stress": {
        "wind": 1.15,
        "rain": 1.25,
        "temperature": 0.0,
        "aqi": 1.15,
        "solar": 0.55,
        "outage": 4.00,
        "finance": 2.50,
        "hazard_mode": "blackout",
        "description": "Forced high outage concentration and severe grid-failure stress.",
    },
    "Compound extreme": {
        "wind": 2.00,
        "rain": 2.75,
        "temperature": 6.0,
        "aqi": 1.85,
        "solar": 0.34,
        "outage": 2.75,
        "finance": 2.00,
        "hazard_mode": "storm",
        "description": "Combined weather, pollution, outage and renewable-disruption scenario.",
    },
}

REGIONS = {
    "North East": {
        "center": {"lat": 54.85, "lon": -1.65, "zoom": 7},
        "bbox": [-3.35, 54.10, -0.60, 55.95],
        "places": {
            "Newcastle": {
                "lat": 54.9783, "lon": -1.6178,
                "postcode_prefix": "NE1",
                "authority_tokens": ["newcastle", "newcastle upon tyne"],
                "population_density": 2590,
                "vulnerability_proxy": 43,
                "estimated_load_mw": 128,
                "business_density": 0.68,
            },
            "Sunderland": {
                "lat": 54.9069, "lon": -1.3838,
                "postcode_prefix": "SR1",
                "authority_tokens": ["sunderland"],
                "population_density": 2010,
                "vulnerability_proxy": 52,
                "estimated_load_mw": 106,
                "business_density": 0.54,
            },
            "Durham": {
                "lat": 54.7761, "lon": -1.5733,
                "postcode_prefix": "DH1",
                "authority_tokens": ["durham", "county durham"],
                "population_density": 730,
                "vulnerability_proxy": 38,
                "estimated_load_mw": 64,
                "business_density": 0.38,
            },
            "Middlesbrough": {
                "lat": 54.5742, "lon": -1.2350,
                "postcode_prefix": "TS1",
                "authority_tokens": ["middlesbrough", "teesside"],
                "population_density": 2680,
                "vulnerability_proxy": 61,
                "estimated_load_mw": 96,
                "business_density": 0.50,
            },
            "Darlington": {
                "lat": 54.5236, "lon": -1.5595,
                "postcode_prefix": "DL1",
                "authority_tokens": ["darlington"],
                "population_density": 1070,
                "vulnerability_proxy": 45,
                "estimated_load_mw": 72,
                "business_density": 0.41,
            },
            "Hexham": {
                "lat": 54.9730, "lon": -2.1010,
                "postcode_prefix": "NE46",
                "authority_tokens": ["hexham", "northumberland"],
                "population_density": 330,
                "vulnerability_proxy": 32,
                "estimated_load_mw": 38,
                "business_density": 0.24,
            },
        },
        "tokens": [
            "newcastle", "sunderland", "durham", "middlesbrough", "darlington",
            "hexham", "gateshead", "northumberland", "teesside", "hartlepool",
            "stockton", "redcar", "tyne and wear", "county durham",
        ],
    },
    "Yorkshire": {
        "center": {"lat": 53.95, "lon": -1.30, "zoom": 7},
        "bbox": [-2.90, 53.20, -0.10, 54.75],
        "places": {
            "Leeds": {
                "lat": 53.8008, "lon": -1.5491,
                "postcode_prefix": "LS1",
                "authority_tokens": ["leeds"],
                "population_density": 1560,
                "vulnerability_proxy": 44,
                "estimated_load_mw": 168,
                "business_density": 0.74,
            },
            "Sheffield": {
                "lat": 53.3811, "lon": -1.4701,
                "postcode_prefix": "S1",
                "authority_tokens": ["sheffield"],
                "population_density": 1510,
                "vulnerability_proxy": 48,
                "estimated_load_mw": 144,
                "business_density": 0.66,
            },
            "York": {
                "lat": 53.9600, "lon": -1.0873,
                "postcode_prefix": "YO1",
                "authority_tokens": ["york"],
                "population_density": 740,
                "vulnerability_proxy": 34,
                "estimated_load_mw": 82,
                "business_density": 0.50,
            },
            "Hull": {
                "lat": 53.7676, "lon": -0.3274,
                "postcode_prefix": "HU1",
                "authority_tokens": ["hull", "kingston upon hull"],
                "population_density": 3560,
                "vulnerability_proxy": 62,
                "estimated_load_mw": 116,
                "business_density": 0.52,
            },
            "Bradford": {
                "lat": 53.7950, "lon": -1.7594,
                "postcode_prefix": "BD1",
                "authority_tokens": ["bradford"],
                "population_density": 1450,
                "vulnerability_proxy": 59,
                "estimated_load_mw": 132,
                "business_density": 0.48,
            },
            "Doncaster": {
                "lat": 53.5228, "lon": -1.1285,
                "postcode_prefix": "DN1",
                "authority_tokens": ["doncaster"],
                "population_density": 540,
                "vulnerability_proxy": 49,
                "estimated_load_mw": 78,
                "business_density": 0.37,
            },
        },
        "tokens": [
            "yorkshire", "leeds", "sheffield", "york", "hull", "bradford",
            "wakefield", "rotherham", "doncaster", "barnsley", "huddersfield",
            "harrogate", "scarborough", "halifax", "east riding",
        ],
    },
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def clamp(value: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if pd.isna(value):
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
    radius = 6371.0
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(float(lat1)))
        * math.cos(math.radians(float(lat2)))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def risk_label(score: float) -> str:
    score = safe_float(score)
    if score >= 75:
        return "Severe"
    if score >= 55:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def resilience_label(score: float) -> str:
    score = safe_float(score)
    if score >= 80:
        return "Robust"
    if score >= 60:
        return "Functional"
    if score >= 40:
        return "Stressed"
    return "Fragile"


def requests_json(url: str, params: Dict[str, Any] = None, timeout: int = 20) -> Dict[str, Any]:
    try:
        headers = {"User-Agent": "sat-guard-q1-streamlit/2.0"}
        response = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def money_m(value: float) -> str:
    return f"£{safe_float(value) / 1_000_000:.2f}m"


def pct(value: float) -> str:
    return f"{safe_float(value) * 100:.1f}%"


# =============================================================================
# IMD / IOD2025 DATASET LOADER
# =============================================================================

def find_imd_files() -> List[Path]:
    current = Path.cwd()
    possible_dirs = [
        current,
        current / "data",
        Path("/mnt/data"),
    ]

    explicit = [
        "IoD2025 Local Authority District Summaries (lower-tier) - Rank of average rank.xlsx",
        "File_1_IoD2025 Index of Multiple Deprivation.xlsx",
        "File_2_IoD2025 Domains of Deprivation.xlsx",
        "File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.xlsx",
    ]

    files = []

    for folder in possible_dirs:
        if not folder.exists():
            continue

        for name in explicit:
            p = folder / name
            if p.exists():
                files.append(p)

        for p in folder.glob("*IoD2025*.xlsx"):
            if p not in files:
                files.append(p)

        for p in folder.glob("*Deprivation*.xlsx"):
            if p not in files:
                files.append(p)

    return files


def choose_first_matching_column(columns: List[Any], include_terms: List[str], exclude_terms: List[str] = None) -> Any:
    exclude_terms = exclude_terms or []
    cleaned = [(c, clean_col(c)) for c in columns]

    for col, text in cleaned:
        if all(term in text for term in include_terms) and not any(ex in text for ex in exclude_terms):
            return col

    for col, text in cleaned:
        if any(term in text for term in include_terms) and not any(ex in text for ex in exclude_terms):
            return col

    return None


def normalise_imd_score_from_rank(rank_value: float, max_rank: float) -> float:
    rank_value = safe_float(rank_value, None)
    max_rank = safe_float(max_rank, None)

    if rank_value is None or max_rank is None or max_rank <= 1:
        return None

    return round(clamp((1 - (rank_value - 1) / (max_rank - 1)) * 100, 0, 100), 2)


def extract_imd_summary_from_sheet(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame()

    cols = list(df.columns)

    area_col = choose_first_matching_column(cols, ["local", "authority"])
    if area_col is None:
        area_col = choose_first_matching_column(cols, ["lad"])
    if area_col is None:
        area_col = choose_first_matching_column(cols, ["district"])
    if area_col is None:
        area_col = choose_first_matching_column(cols, ["area"])
    if area_col is None:
        area_col = choose_first_matching_column(cols, ["name"])

    score_col = choose_first_matching_column(cols, ["imd", "score"])
    if score_col is None:
        score_col = choose_first_matching_column(cols, ["index", "multiple", "deprivation", "score"])
    if score_col is None:
        score_col = choose_first_matching_column(cols, ["average", "score"])

    rank_col = choose_first_matching_column(cols, ["imd", "rank"])
    if rank_col is None:
        rank_col = choose_first_matching_column(cols, ["rank", "average", "rank"])
    if rank_col is None:
        rank_col = choose_first_matching_column(cols, ["rank"])

    decile_col = choose_first_matching_column(cols, ["decile"])

    if area_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["area_name"] = df[area_col].astype(str)
    out["source_file"] = source_name
    out["source_area_col"] = str(area_col)

    if score_col is not None:
        score = pd.to_numeric(df[score_col], errors="coerce")
        if score.dropna().empty:
            out["imd_score_0_100"] = np.nan
        else:
            max_val = score.max()
            min_val = score.min()
            if max_val > 100 or min_val < 0:
                out["imd_score_0_100"] = ((score - min_val) / max(max_val - min_val, 1)) * 100
            else:
                out["imd_score_0_100"] = score
        out["imd_metric_source"] = f"score: {score_col}"

    elif rank_col is not None:
        rank = pd.to_numeric(df[rank_col], errors="coerce")
        max_rank = rank.max()
        out["imd_score_0_100"] = rank.apply(lambda x: normalise_imd_score_from_rank(x, max_rank))
        out["imd_metric_source"] = f"rank converted: {rank_col}"

    elif decile_col is not None:
        decile = pd.to_numeric(df[decile_col], errors="coerce")
        out["imd_score_0_100"] = (10 - decile) / 9 * 100
        out["imd_metric_source"] = f"decile converted: {decile_col}"

    else:
        return pd.DataFrame()

    out["imd_score_0_100"] = pd.to_numeric(out["imd_score_0_100"], errors="coerce")
    out = out.dropna(subset=["imd_score_0_100"])
    out["imd_score_0_100"] = out["imd_score_0_100"].clip(0, 100)
    out["area_key"] = out["area_name"].str.lower()

    return out[["area_name", "area_key", "imd_score_0_100", "imd_metric_source", "source_file", "source_area_col"]]


@st.cache_data(ttl=3600, show_spinner=False)
def load_imd_summary_cached() -> Tuple[pd.DataFrame, str]:
    files = find_imd_files()
    all_parts = []
    source_notes = []

    for file_path in files:
        try:
            sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        except Exception:
            continue

        for sheet_name, df in sheets.items():
            try:
                part = extract_imd_summary_from_sheet(df, f"{file_path.name} | {sheet_name}")
                if not part.empty:
                    all_parts.append(part)
                    source_notes.append(f"{file_path.name}:{sheet_name}")
            except Exception:
                continue

    if all_parts:
        summary = pd.concat(all_parts, ignore_index=True)
        grouped = (
            summary.groupby("area_key")
            .agg(
                area_name=("area_name", "first"),
                imd_score_0_100=("imd_score_0_100", "mean"),
                imd_metric_source=("imd_metric_source", "first"),
                source_file=("source_file", "first"),
            )
            .reset_index()
        )
        source = "; ".join(source_notes[:8])
    else:
        grouped = pd.DataFrame(columns=["area_key", "area_name", "imd_score_0_100", "imd_metric_source", "source_file"])
        source = "No readable IoD2025 Excel summary found; using configured fallback proxies."

    return grouped, source


def infer_imd_for_place(place: str, region: str, meta: Dict[str, Any], imd_summary: pd.DataFrame) -> Dict[str, Any]:
    fallback = safe_float(meta.get("vulnerability_proxy"), 45)

    if imd_summary is None or imd_summary.empty:
        return {
            "imd_score": fallback,
            "imd_source": "fallback proxy",
            "imd_match": "no IMD Excel match",
        }

    tokens = [place.lower()] + [t.lower() for t in meta.get("authority_tokens", [])]
    region_tokens = [t.lower() for t in REGIONS[region]["tokens"]]

    for token in tokens:
        hit = imd_summary[imd_summary["area_key"].str.contains(token, regex=False, na=False)]
        if not hit.empty:
            return {
                "imd_score": round(float(hit["imd_score_0_100"].mean()), 2),
                "imd_source": str(hit.iloc[0].get("source_file", "IoD2025")),
                "imd_match": f"matched token: {token}",
            }

    regional_scores = []
    for token in region_tokens:
        hit = imd_summary[imd_summary["area_key"].str.contains(token, regex=False, na=False)]
        if not hit.empty:
            regional_scores.extend(hit["imd_score_0_100"].dropna().tolist())

    if regional_scores:
        return {
            "imd_score": round(float(np.mean(regional_scores)), 2),
            "imd_source": "IoD2025 regional token average",
            "imd_match": "regional average",
        }

    return {
        "imd_score": fallback,
        "imd_source": "fallback proxy",
        "imd_match": "no authority match",
    }


# =============================================================================
# EXTERNAL DATA FETCHING
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": WEATHER_CURRENT_VARS,
        "timezone": "Europe/London",
    }
    return requests_json(OPEN_METEO_WEATHER_URL, params=params)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_air_quality(lat: float, lon: float) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": AIR_CURRENT_VARS,
        "timezone": "Europe/London",
    }
    return requests_json(OPEN_METEO_AIR_URL, params=params)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_northern_powergrid(limit: int = 100) -> pd.DataFrame:
    payload = requests_json(NPG_DATASET_URL, params={"limit": int(clamp(limit, 1, 100))})
    records = payload.get("results", [])
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def filter_npg_by_region(raw_df: pd.DataFrame, region: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    tokens = REGIONS[region]["tokens"]
    object_cols = [c for c in raw_df.columns if raw_df[c].dtype == "object"]

    if not object_cols:
        return raw_df.copy()

    text = raw_df[object_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(False, index=raw_df.index)

    for token in tokens:
        mask = mask | text.str.contains(token, regex=False)

    filtered = raw_df[mask].copy()
    return filtered if not filtered.empty else raw_df.copy()


def standardise_outages(raw_df: pd.DataFrame, region: str) -> pd.DataFrame:
    output_cols = [
        "outage_reference",
        "outage_status",
        "outage_category",
        "postcode_label",
        "affected_customers",
        "estimated_restore",
        "latitude",
        "longitude",
        "source_text",
    ]

    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=output_cols)

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
        if int(missing.sum()) > 0:
            lat.loc[missing] = meta["lat"] + np.random.uniform(-0.03, 0.03, size=int(missing.sum()))
            lon.loc[missing] = meta["lon"] + np.random.uniform(-0.03, 0.03, size=int(missing.sum()))

    def find_col(keywords: List[str]) -> str:
        for c in df.columns:
            low = c.lower()
            for k in keywords:
                if k in low:
                    return c
        return ""

    ref_col = find_col(["reference", "incident"])
    status_col = find_col(["status"])
    category_col = find_col(["category", "type"])
    postcode_col = find_col(["postcode", "post_code", "postal"])
    customer_col = find_col(["customer", "affected"])
    restore_col = find_col(["restore", "estimated"])

    out = pd.DataFrame()
    out["outage_reference"] = df[ref_col].astype(str) if ref_col else "N/A"
    out["outage_status"] = df[status_col].astype(str) if status_col else "Unknown"
    out["outage_category"] = df[category_col].astype(str) if category_col else "Unknown"

    if postcode_col:
        out["postcode_label"] = df[postcode_col].astype(str)
    else:
        labels = []
        for i in range(len(df)):
            text_i = source_lower.iloc[i]
            label = "Unknown"
            for place, meta in REGIONS[region]["places"].items():
                if place.lower() in text_i:
                    label = meta["postcode_prefix"]
                    break
            labels.append(label)
        out["postcode_label"] = labels

    out["affected_customers"] = pd.to_numeric(df[customer_col], errors="coerce").fillna(0) if customer_col else 0
    out["estimated_restore"] = df[restore_col].astype(str) if restore_col else "Unknown"
    out["latitude"] = lat
    out["longitude"] = lon
    out["source_text"] = source_text

    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    if out.empty:
        synthetic = []
        for place, meta in REGIONS[region]["places"].items():
            synthetic.append({
                "outage_reference": f"SIM-{place[:3].upper()}-{random.randint(1000, 9999)}",
                "outage_status": "Simulated fallback",
                "outage_category": "Visual fallback when live coordinates unavailable",
                "postcode_label": meta["postcode_prefix"],
                "affected_customers": random.randint(20, 520),
                "estimated_restore": "Unknown",
                "latitude": meta["lat"] + random.uniform(-0.045, 0.045),
                "longitude": meta["lon"] + random.uniform(-0.045, 0.045),
                "source_text": "Synthetic point generated because no live geocoded NPG outage was available.",
            })
        out = pd.DataFrame(synthetic, columns=output_cols)

    return out


# =============================================================================
# CORE MODELS
# =============================================================================

def apply_scenario(row: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    params = SCENARIOS.get(scenario_name, SCENARIOS["Live / Real-time"])
    r = dict(row)

    r["wind_speed_10m"] = safe_float(r.get("wind_speed_10m")) * params["wind"]
    r["precipitation"] = safe_float(r.get("precipitation")) * params["rain"]
    r["temperature_2m"] = safe_float(r.get("temperature_2m")) + params["temperature"]
    r["european_aqi"] = safe_float(r.get("european_aqi")) * params["aqi"]
    r["shortwave_radiation"] = safe_float(r.get("shortwave_radiation")) * params["solar"]
    r["scenario_outage_multiplier"] = params["outage"]
    r["scenario_finance_multiplier"] = params["finance"]
    r["hazard_mode"] = params["hazard_mode"]

    return r


def renewable_generation_mw(row: Dict[str, Any]) -> float:
    solar = safe_float(row.get("shortwave_radiation"))
    wind = safe_float(row.get("wind_speed_10m"))

    solar_mw = solar * 0.18
    wind_mw = min((wind / 12.0) ** 3, 1.20) * 95
    return round(clamp(solar_mw + wind_mw, 0, 240), 2)


def renewable_failure_probability(row: Dict[str, Any]) -> float:
    solar = safe_float(row.get("shortwave_radiation"))
    wind = safe_float(row.get("wind_speed_10m"))
    cloud = safe_float(row.get("cloud_cover"))

    low_solar = 1 - clamp(solar / 450, 0, 1)
    low_wind = 1 - clamp(wind / 12, 0, 1)
    cloud_penalty = clamp(cloud / 100, 0, 1) * 0.15

    probability = 0.12 + 0.48 * low_solar + 0.30 * low_wind + cloud_penalty
    return round(clamp(probability, 0, 1), 3)


def peak_load_multiplier(hour: int = None) -> float:
    if hour is None:
        hour = datetime.now().hour
    if 17 <= hour <= 22:
        return 1.85
    if 7 <= hour <= 9:
        return 1.30
    if 0 <= hour <= 6:
        return 0.65
    return 1.00


def compute_energy_not_supplied_mw(
    outage_count: float,
    affected_customers: float,
    base_load_mw: float,
    scenario_name: str,
) -> float:
    params = SCENARIOS.get(scenario_name, SCENARIOS["Live / Real-time"])

    outage_component = safe_float(outage_count) * 100.0
    customer_component = safe_float(affected_customers) * 0.014
    base_component = safe_float(base_load_mw) * 0.18

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
    params = SCENARIOS.get(scenario_name, SCENARIOS["Live / Real-time"])

    duration_hours = 1.5 + clamp(outage_count / 6, 0, 1) * 5.5
    if scenario_name == "Total blackout stress":
        duration_hours = 8.0
    elif scenario_name == "Compound extreme":
        duration_hours = max(duration_hours, 6.0)

    ens_mwh = ens_mw * duration_hours

    value_of_lost_load_gbp_per_mwh = 17000
    customer_interruption_gbp = affected_customers * 38
    business_disruption_gbp = ens_mwh * 1100 * clamp(business_density, 0, 1)
    restoration_gbp = outage_count * 18500
    critical_services_uplift_gbp = ens_mwh * 320 * clamp(social_vulnerability / 100, 0, 1)

    voll_loss = ens_mwh * value_of_lost_load_gbp_per_mwh

    total = (
        voll_loss
        + customer_interruption_gbp
        + business_disruption_gbp
        + restoration_gbp
        + critical_services_uplift_gbp
    ) * params["finance"]

    return {
        "estimated_duration_hours": round(duration_hours, 2),
        "ens_mwh": round(ens_mwh, 2),
        "voll_loss_gbp": round(voll_loss, 2),
        "customer_interruption_loss_gbp": round(customer_interruption_gbp, 2),
        "business_disruption_loss_gbp": round(business_disruption_gbp, 2),
        "restoration_loss_gbp": round(restoration_gbp, 2),
        "critical_services_loss_gbp": round(critical_services_uplift_gbp, 2),
        "total_financial_loss_gbp": round(total, 2),
    }


def social_vulnerability_score(pop_density: float, imd_score: float) -> float:
    density_component = clamp(pop_density / 4500, 0, 1) * 40
    imd_component = clamp(imd_score / 100, 0, 1) * 60
    return round(clamp(density_component + imd_component, 0, 100), 2)


def grid_failure_probability(risk_score: float, outage_count: float, ens_mw: float) -> float:
    return round(clamp(
        clamp(risk_score / 100, 0, 1) * 0.50
        + clamp(outage_count / 10, 0, 1) * 0.30
        + clamp(ens_mw / 1200, 0, 1) * 0.20,
        0,
        1,
    ), 3)


def compute_multilayer_risk(row: Dict[str, Any], outage_intensity: float, ens_mw: float) -> Dict[str, float]:
    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    cloud = safe_float(row.get("cloud_cover"))
    aqi = safe_float(row.get("european_aqi"))
    pm25 = safe_float(row.get("pm2_5"))
    temp = safe_float(row.get("temperature_2m"))
    humidity = safe_float(row.get("relative_humidity_2m"))

    weather_score = (
        clamp(wind / 45, 0, 1) * 27
        + clamp(rain / 6, 0, 1) * 18
        + clamp(cloud / 100, 0, 1) * 7
        + clamp(abs(temp - 18) / 20, 0, 1) * 8
        + clamp(humidity / 100, 0, 1) * 4
    )

    pollution_score = (
        clamp(aqi / 100, 0, 1) * 17
        + clamp(pm25 / 60, 0, 1) * 9
    )

    renewable_mw = renewable_generation_mw(row)
    net_load = peak_load_multiplier() * 100 - renewable_mw
    load_score = clamp(net_load / 220, 0, 1) * 14

    outage_score = clamp(outage_intensity, 0, 1) * 20
    ens_score = clamp(ens_mw / 1500, 0, 1) * 17

    score = clamp(weather_score + pollution_score + load_score + outage_score + ens_score, 0, 100)
    failure_probability = 1 / (1 + np.exp(-0.065 * (score - 60)))

    return {
        "risk_score": round(float(score), 2),
        "failure_probability": round(float(failure_probability), 3),
        "renewable_generation_mw": round(float(renewable_mw), 2),
        "net_load_mw": round(float(net_load), 2),
    }


def cascade_breakdown(base_failure: float) -> Dict[str, float]:
    power = clamp(base_failure, 0, 1)
    water = clamp((power ** 1.35) * 0.74, 0, 1)
    telecom = clamp((power ** 1.22) * 0.82, 0, 1)
    transport = clamp(((power + telecom) / 2.0) * 0.70, 0, 1)
    social = clamp(((power + water + telecom) / 3.0) * 0.75, 0, 1)

    return {
        "cascade_power": round(power, 3),
        "cascade_water": round(water, 3),
        "cascade_telecom": round(telecom, 3),
        "cascade_transport": round(transport, 3),
        "cascade_social": round(social, 3),
        "system_stress": round(float(np.mean([power, water, telecom, transport, social])), 3),
    }


def compute_resilience_index(
    final_risk: float,
    social_vulnerability: float,
    grid_failure: float,
    renewable_failure: float,
    system_stress: float,
    financial_loss_gbp: float,
) -> float:
    finance_penalty = clamp(financial_loss_gbp / 15_000_000, 0, 1) * 10

    resilience = 100 - (
        0.42 * safe_float(final_risk)
        + 0.20 * safe_float(social_vulnerability)
        + 17 * safe_float(grid_failure)
        + 10 * safe_float(renewable_failure)
        + 12 * safe_float(system_stress)
        + finance_penalty
    )

    return round(clamp(resilience, 0, 100), 2)


def flood_depth_proxy(row: Dict[str, Any], scenario_name: str) -> float:
    rain = safe_float(row.get("precipitation"))
    outages = safe_float(row.get("nearby_outages_25km"))
    risk = safe_float(row.get("final_risk_score"))
    cloud = safe_float(row.get("cloud_cover"))

    multiplier = {
        "Live / Real-time": 1.0,
        "Extreme wind": 0.9,
        "Flood cascade": 2.0,
        "Renewable collapse": 0.25,
        "Total blackout stress": 1.2,
        "Compound extreme": 1.8,
    }.get(scenario_name, 1.0)

    return round(clamp((0.038 * rain + 0.016 * outages + 0.0025 * risk + 0.001 * cloud) * multiplier, 0, 2.5), 3)


def advanced_monte_carlo(row: Dict[str, Any], outage_intensity: float, ens_mw: float, simulations: int) -> Dict[str, Any]:
    simulations = int(clamp(simulations, 10, 160))
    risk_scores = []
    resilience_scores = []
    financial_losses = []

    for _ in range(simulations):
        sim = dict(row)
        sim["wind_speed_10m"] = safe_float(sim.get("wind_speed_10m")) * np.random.lognormal(mean=0, sigma=0.16)
        sim["precipitation"] = max(0, safe_float(sim.get("precipitation")) * np.random.lognormal(mean=0, sigma=0.30))
        sim["temperature_2m"] = safe_float(sim.get("temperature_2m")) + np.random.normal(0, 2.2)
        sim["european_aqi"] = safe_float(sim.get("european_aqi")) * np.random.lognormal(mean=0, sigma=0.22)
        sim["shortwave_radiation"] = max(0, safe_float(sim.get("shortwave_radiation")) * np.random.lognormal(mean=0, sigma=0.28))
        sim["cloud_cover"] = clamp(safe_float(sim.get("cloud_cover")) + np.random.normal(0, 12), 0, 100)

        sim_ens = max(0, ens_mw * np.random.lognormal(mean=0, sigma=0.25))
        model = compute_multilayer_risk(sim, outage_intensity, sim_ens)
        cascade = cascade_breakdown(model["failure_probability"])
        renewable_fail = renewable_failure_probability(sim)
        grid_fail = grid_failure_probability(model["risk_score"], safe_float(row.get("nearby_outages_25km")), sim_ens)
        final_risk = clamp(model["risk_score"] * (1 + cascade["system_stress"] * 0.75), 0, 100)

        finance = compute_financial_loss(
            sim_ens,
            safe_float(row.get("affected_customers_nearby")),
            safe_float(row.get("nearby_outages_25km")),
            safe_float(row.get("business_density")),
            safe_float(row.get("social_vulnerability")),
            row.get("scenario_name", "Live / Real-time"),
        )

        resilience = compute_resilience_index(
            final_risk,
            safe_float(row.get("social_vulnerability")),
            grid_fail,
            renewable_fail,
            cascade["system_stress"],
            finance["total_financial_loss_gbp"],
        )

        risk_scores.append(final_risk)
        resilience_scores.append(resilience)
        financial_losses.append(finance["total_financial_loss_gbp"])

    risk_arr = np.array(risk_scores)
    res_arr = np.array(resilience_scores)
    fin_arr = np.array(financial_losses)

    return {
        "mc_mean": round(float(np.mean(risk_arr)), 2),
        "mc_std": round(float(np.std(risk_arr)), 2),
        "mc_p05": round(float(np.percentile(risk_arr, 5)), 2),
        "mc_p50": round(float(np.percentile(risk_arr, 50)), 2),
        "mc_p95": round(float(np.percentile(risk_arr, 95)), 2),
        "mc_extreme_probability": round(float(np.mean(risk_arr >= 80)), 3),
        "mc_resilience_mean": round(float(np.mean(res_arr)), 2),
        "mc_resilience_p05": round(float(np.percentile(res_arr, 5)), 2),
        "mc_financial_loss_p95": round(float(np.percentile(fin_arr, 95)), 2),
        "mc_histogram": [round(float(x), 2) for x in risk_arr[:250]],
    }


# =============================================================================
# DATA BUILDING
# =============================================================================

def build_places(region: str, scenario_name: str, mc_runs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    imd_summary, imd_source = load_imd_summary_cached()

    raw_npg = fetch_northern_powergrid(100)
    outages = standardise_outages(raw_npg, region)

    outage_points = []
    for _, o in outages.iterrows():
        outage_points.append((
            safe_float(o.get("latitude")),
            safe_float(o.get("longitude")),
            safe_float(o.get("affected_customers")),
        ))

    rows = []

    for place, meta in REGIONS[region]["places"].items():
        lat = meta["lat"]
        lon = meta["lon"]

        weather = fetch_weather(lat, lon).get("current", {})
        air = fetch_air_quality(lat, lon).get("current", {})

        row = {
            "scenario_name": scenario_name,
            "place": place,
            "lat": lat,
            "lon": lon,
            "postcode_prefix": meta["postcode_prefix"],
            "time": weather.get("time") or datetime.now(UTC).isoformat(),
            "temperature_2m": weather.get("temperature_2m", random.uniform(7, 18)),
            "apparent_temperature": weather.get("apparent_temperature", random.uniform(7, 18)),
            "wind_speed_10m": weather.get("wind_speed_10m", random.uniform(4, 26)),
            "wind_direction_10m": weather.get("wind_direction_10m", random.uniform(0, 360)),
            "surface_pressure": weather.get("surface_pressure", random.uniform(990, 1030)),
            "cloud_cover": weather.get("cloud_cover", random.uniform(15, 96)),
            "shortwave_radiation": weather.get("shortwave_radiation", random.uniform(0, 550)),
            "direct_radiation": weather.get("direct_radiation", random.uniform(0, 350)),
            "diffuse_radiation": weather.get("diffuse_radiation", random.uniform(0, 170)),
            "relative_humidity_2m": weather.get("relative_humidity_2m", random.uniform(55, 95)),
            "precipitation": weather.get("precipitation", random.uniform(0, 3)),
            "is_day": weather.get("is_day", 1),
            "european_aqi": air.get("european_aqi", random.uniform(15, 65)),
            "pm10": air.get("pm10", random.uniform(5, 30)),
            "pm2_5": air.get("pm2_5", random.uniform(3, 18)),
            "nitrogen_dioxide": air.get("nitrogen_dioxide", random.uniform(5, 45)),
            "ozone": air.get("ozone", random.uniform(20, 90)),
            "uv_index": air.get("uv_index", random.uniform(0, 5)),
            "population_density": meta["population_density"],
            "estimated_load_mw": meta["estimated_load_mw"],
            "business_density": meta["business_density"],
        }

        row = apply_scenario(row, scenario_name)

        nearby = 0
        affected_customers = 0.0

        for olat, olon, customers in outage_points:
            if haversine_km(lat, lon, olat, olon) <= 25:
                nearby += 1
                affected_customers += customers

        if scenario_name == "Total blackout stress":
            nearby = max(nearby, 10)
            affected_customers = max(affected_customers, 3000)

        imd_info = infer_imd_for_place(place, region, meta, imd_summary)
        social_vuln = social_vulnerability_score(row["population_density"], imd_info["imd_score"])

        outage_intensity = clamp((nearby / 20) * row.get("scenario_outage_multiplier", 1.0), 0, 1)

        ens_mw = compute_energy_not_supplied_mw(
            nearby,
            affected_customers,
            row["estimated_load_mw"],
            scenario_name,
        )

        base = compute_multilayer_risk(row, outage_intensity, ens_mw)
        cascade = cascade_breakdown(base["failure_probability"])

        final_risk = clamp(
            base["risk_score"] * (1 + cascade["system_stress"] * SCENARIOS[scenario_name]["outage"] * 0.55),
            0,
            100,
        )

        renewable_fail = renewable_failure_probability(row)
        grid_fail = grid_failure_probability(final_risk, nearby, ens_mw)

        finance = compute_financial_loss(
            ens_mw=ens_mw,
            affected_customers=affected_customers,
            outage_count=nearby,
            business_density=row["business_density"],
            social_vulnerability=social_vuln,
            scenario_name=scenario_name,
        )

        resilience = compute_resilience_index(
            final_risk,
            social_vuln,
            grid_fail,
            renewable_fail,
            cascade["system_stress"],
            finance["total_financial_loss_gbp"],
        )

        row.update(base)
        row.update(cascade)
        row.update(finance)
        row.update({
            "nearby_outages_25km": nearby,
            "affected_customers_nearby": round(affected_customers, 1),
            "outage_intensity": round(outage_intensity, 3),
            "energy_not_supplied_mw": ens_mw,
            "final_risk_score": round(final_risk, 2),
            "risk_label": risk_label(final_risk),
            "imd_score": imd_info["imd_score"],
            "imd_source": imd_info["imd_source"],
            "imd_match": imd_info["imd_match"],
            "imd_dataset_summary": imd_source,
            "social_vulnerability": social_vuln,
            "renewable_failure_probability": renewable_fail,
            "grid_failure_probability": grid_fail,
            "resilience_index": resilience,
            "resilience_label": resilience_label(resilience),
        })

        row["flood_depth_proxy"] = flood_depth_proxy(row, scenario_name)

        mc = advanced_monte_carlo(row, outage_intensity, ens_mw, mc_runs)
        row.update(mc)

        rows.append(row)

    return pd.DataFrame(rows), outages


def interpolate_value(lat: float, lon: float, places: pd.DataFrame, col: str) -> float:
    weights = []
    values = []

    for _, r in places.iterrows():
        d = haversine_km(lat, lon, r["lat"], r["lon"])
        weights.append(1 / max(d, 1))
        values.append(safe_float(r.get(col)))

    if not weights:
        return 0.0

    return float(np.average(values, weights=weights))


def build_grid(region: str, places: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = REGIONS[region]["bbox"]

    lats = np.linspace(min_lat, max_lat, 15)
    lons = np.linspace(min_lon, max_lon, 15)

    rows = []

    for lat in lats:
        for lon in lons:
            nearby_outages = 0
            for _, o in outages.iterrows():
                if haversine_km(lat, lon, o["latitude"], o["longitude"]) <= 20:
                    nearby_outages += 1

            risk = interpolate_value(lat, lon, places, "final_risk_score")
            wind = interpolate_value(lat, lon, places, "wind_speed_10m")
            rain = interpolate_value(lat, lon, places, "precipitation")
            resilience = interpolate_value(lat, lon, places, "resilience_index")
            social = interpolate_value(lat, lon, places, "social_vulnerability")
            aqi = interpolate_value(lat, lon, places, "european_aqi")
            ens = interpolate_value(lat, lon, places, "energy_not_supplied_mw")
            loss = interpolate_value(lat, lon, places, "total_financial_loss_gbp")
            flood = interpolate_value(lat, lon, places, "flood_depth_proxy")

            rows.append({
                "lat": round(float(lat), 5),
                "lon": round(float(lon), 5),
                "risk_score": round(float(risk), 2),
                "risk_label": risk_label(risk),
                "wind_speed": round(float(wind), 2),
                "rain": round(float(rain), 2),
                "resilience_index": round(float(resilience), 2),
                "social_vulnerability": round(float(social), 2),
                "aqi": round(float(aqi), 2),
                "energy_not_supplied_mw": round(float(ens), 2),
                "financial_loss_gbp": round(float(loss), 2),
                "flood_depth_proxy": round(float(flood), 3),
                "outages_near_20km": nearby_outages,
            })

    return pd.DataFrame(rows)


@st.cache_data(ttl=240, show_spinner=False)
def get_data_cached(region: str, scenario: str, mc_runs: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    places, outages = build_places(region, scenario, mc_runs)
    grid = build_grid(region, places, outages)
    return places, outages, grid


# =============================================================================
# POSTCODE RESILIENCE + INVESTMENT RECOMMENDATION MODELS
# =============================================================================

def build_postcode_resilience(places: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if outages is not None and not outages.empty:
        grouped = (
            outages.groupby("postcode_label")
            .agg(
                outage_records=("outage_reference", "count"),
                affected_customers=("affected_customers", "sum"),
                lat=("latitude", "mean"),
                lon=("longitude", "mean"),
            )
            .reset_index()
        )

        for _, g in grouped.iterrows():
            postcode = str(g.get("postcode_label", "Unknown"))
            lat = safe_float(g.get("lat"))
            lon = safe_float(g.get("lon"))

            nearest = None
            nearest_d = 1e9
            for _, p in places.iterrows():
                d = haversine_km(lat, lon, p["lat"], p["lon"])
                if d < nearest_d:
                    nearest_d = d
                    nearest = p

            if nearest is None:
                continue

            outage_records = safe_float(g.get("outage_records"))
            affected = safe_float(g.get("affected_customers"))

            outage_pressure = clamp(outage_records / 6, 0, 1) * 16
            customer_pressure = clamp(affected / 1500, 0, 1) * 12
            distance_penalty = clamp((25 - min(nearest_d, 25)) / 25, 0, 1) * 6

            base_resilience = safe_float(nearest.get("resilience_index"))
            postcode_resilience = clamp(
                base_resilience - outage_pressure - customer_pressure - distance_penalty,
                0,
                100,
            )

            postcode_risk = clamp(
                safe_float(nearest.get("final_risk_score")) + outage_pressure + customer_pressure,
                0,
                100,
            )

            financial_loss = (
                safe_float(nearest.get("total_financial_loss_gbp")) * (0.30 + clamp(outage_records / 8, 0, 1) * 0.70)
                + affected * 55
            )

            rows.append({
                "postcode": postcode,
                "nearest_place": nearest.get("place"),
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "distance_to_place_km": round(nearest_d, 2),
                "outage_records": int(outage_records),
                "affected_customers": int(affected),
                "risk_score": round(postcode_risk, 2),
                "resilience_score": round(postcode_resilience, 2),
                "resilience_label": resilience_label(postcode_resilience),
                "social_vulnerability": round(safe_float(nearest.get("social_vulnerability")), 2),
                "imd_score": round(safe_float(nearest.get("imd_score")), 2),
                "energy_not_supplied_mw": round(safe_float(nearest.get("energy_not_supplied_mw")) * (0.35 + outage_records / 10), 2),
                "financial_loss_gbp": round(financial_loss, 2),
                "recommendation_score": 0.0,
            })

    existing = {str(r["postcode"]).upper() for r in rows}
    for _, p in places.iterrows():
        pc = str(p.get("postcode_prefix", "Unknown"))
        if pc.upper() in existing:
            continue

        rows.append({
            "postcode": pc,
            "nearest_place": p.get("place"),
            "lat": round(safe_float(p.get("lat")), 5),
            "lon": round(safe_float(p.get("lon")), 5),
            "distance_to_place_km": 0.0,
            "outage_records": int(safe_float(p.get("nearby_outages_25km"))),
            "affected_customers": int(safe_float(p.get("affected_customers_nearby"))),
            "risk_score": round(safe_float(p.get("final_risk_score")), 2),
            "resilience_score": round(safe_float(p.get("resilience_index")), 2),
            "resilience_label": resilience_label(safe_float(p.get("resilience_index"))),
            "social_vulnerability": round(safe_float(p.get("social_vulnerability")), 2),
            "imd_score": round(safe_float(p.get("imd_score")), 2),
            "energy_not_supplied_mw": round(safe_float(p.get("energy_not_supplied_mw")), 2),
            "financial_loss_gbp": round(safe_float(p.get("total_financial_loss_gbp")), 2),
            "recommendation_score": 0.0,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    max_loss = max(float(df["financial_loss_gbp"].max()), 1.0)
    max_ens = max(float(df["energy_not_supplied_mw"].max()), 1.0)

    df["recommendation_score"] = (
        0.30 * df["risk_score"]
        + 0.22 * df["social_vulnerability"]
        + 0.18 * (100 - df["resilience_score"])
        + 0.13 * (df["financial_loss_gbp"] / max_loss * 100)
        + 0.10 * (df["energy_not_supplied_mw"] / max_ens * 100)
        + 0.07 * np.clip(df["outage_records"] / 6, 0, 1) * 100
    ).round(2)

    df["investment_priority"] = df["recommendation_score"].apply(
        lambda x: "Priority 1" if x >= 75 else "Priority 2" if x >= 55 else "Priority 3" if x >= 35 else "Monitor"
    )

    return df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


def investment_action_for_row(row: Dict[str, Any]) -> str:
    risk = safe_float(row.get("risk_score"))
    resilience = safe_float(row.get("resilience_score"))
    social = safe_float(row.get("social_vulnerability"))
    ens = safe_float(row.get("energy_not_supplied_mw"))
    outages = safe_float(row.get("outage_records"))

    actions = []

    if risk >= 65 or outages >= 3:
        actions.append("reinforce local feeders and automate switching")
    if ens >= 300:
        actions.append("install backup supply / mobile generation access")
    if social >= 55:
        actions.append("target community resilience support and welfare checks")
    if resilience < 45:
        actions.append("upgrade protection, monitoring and restoration capability")
    if risk >= 55:
        actions.append("prioritise vegetation management and weather hardening")

    if not actions:
        actions.append("continue monitoring and maintain standard preventive maintenance")

    return "; ".join(actions)


def investment_category_for_row(row: Dict[str, Any]) -> str:
    risk = safe_float(row.get("risk_score"))
    social = safe_float(row.get("social_vulnerability"))
    ens = safe_float(row.get("energy_not_supplied_mw"))
    resilience = safe_float(row.get("resilience_score"))

    if ens >= 450:
        return "Energy security / backup capacity"
    if resilience < 45:
        return "Network resilience upgrade"
    if social >= 60:
        return "Social resilience and emergency planning"
    if risk >= 65:
        return "Weather hardening"
    return "Preventive monitoring"


def build_investment_recommendations(places: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    pc = build_postcode_resilience(places, outages)
    if pc.empty:
        return pc

    pc = pc.copy()
    pc["investment_category"] = pc.apply(lambda r: investment_category_for_row(r.to_dict()), axis=1)
    pc["recommended_action"] = pc.apply(lambda r: investment_action_for_row(r.to_dict()), axis=1)

    pc["indicative_investment_cost_gbp"] = (
        120000
        + pc["recommendation_score"] * 8500
        + pc["outage_records"] * 35000
        + np.clip(pc["energy_not_supplied_mw"], 0, 1000) * 260
    ).round(0)

    pc["benefit_cost_note"] = "High avoided-loss potential"

    return pc.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


# =============================================================================
# VISUAL FUNCTIONS
# =============================================================================

def colour_rgba_hex(score: float) -> str:
    score = safe_float(score)
    if score >= 75:
        return "#ef4444"
    if score >= 55:
        return "#f97316"
    if score >= 35:
        return "#eab308"
    return "#22c55e"


def risk_colour(score: float) -> List[int]:
    score = safe_float(score)
    if score >= 75:
        return [239, 68, 68, 205]
    if score >= 55:
        return [249, 115, 22, 190]
    if score >= 35:
        return [234, 179, 8, 180]
    return [34, 197, 94, 180]


def resilience_colour(score: float) -> List[int]:
    score = safe_float(score)
    if score >= 80:
        return [34, 197, 94, 190]
    if score >= 60:
        return [56, 189, 248, 185]
    if score >= 40:
        return [234, 179, 8, 180]
    return [239, 68, 68, 190]


def priority_colour(priority: str) -> List[int]:
    if priority == "Priority 1":
        return [239, 68, 68, 210]
    if priority == "Priority 2":
        return [249, 115, 22, 195]
    if priority == "Priority 3":
        return [234, 179, 8, 185]
    return [34, 197, 94, 165]


def plotly_template():
    return "plotly_dark"


def create_risk_gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "/100"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour_rgba_hex(value)},
            "steps": [
                {"range": [0, 35], "color": "rgba(34,197,94,.25)"},
                {"range": [35, 55], "color": "rgba(234,179,8,.25)"},
                {"range": [55, 75], "color": "rgba(249,115,22,.25)"},
                {"range": [75, 100], "color": "rgba(239,68,68,.25)"},
            ],
        },
    ))
    fig.update_layout(template=plotly_template(), height=280, margin=dict(l=18, r=18, t=45, b=18))
    return fig


def create_resilience_gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "/100"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#22c55e" if value >= 60 else "#f97316"},
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,.25)"},
                {"range": [40, 60], "color": "rgba(234,179,8,.25)"},
                {"range": [60, 80], "color": "rgba(56,189,248,.25)"},
                {"range": [80, 100], "color": "rgba(34,197,94,.25)"},
            ],
        },
    ))
    fig.update_layout(template=plotly_template(), height=280, margin=dict(l=18, r=18, t=45, b=18))
    return fig


def create_loss_waterfall(places: pd.DataFrame) -> go.Figure:
    totals = {
        "VoLL": places["voll_loss_gbp"].sum(),
        "Customer": places["customer_interruption_loss_gbp"].sum(),
        "Business": places["business_disruption_loss_gbp"].sum(),
        "Restoration": places["restoration_loss_gbp"].sum(),
        "Critical services": places["critical_services_loss_gbp"].sum(),
    }
    fig = go.Figure(go.Waterfall(
        name="Financial loss",
        orientation="v",
        measure=["relative"] * len(totals),
        x=list(totals.keys()),
        y=[v / 1_000_000 for v in totals.values()],
        connector={"line": {"color": "rgba(148,163,184,.45)"}},
    ))
    fig.update_layout(
        title="Financial-loss contribution (£m)",
        template=plotly_template(),
        height=390,
        yaxis_title="£m",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def create_cascade_radar(places: pd.DataFrame) -> go.Figure:
    vals = [
        places["cascade_power"].mean(),
        places["cascade_water"].mean(),
        places["cascade_telecom"].mean(),
        places["cascade_transport"].mean(),
        places["cascade_social"].mean(),
    ]
    cats = ["Power", "Water", "Telecom", "Transport", "Social"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name="Mean cascade stress",
    ))
    fig.update_layout(
        template=plotly_template(),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=390,
        title="Interdependency cascade signature",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def create_finance_sunburst(places: pd.DataFrame) -> go.Figure:
    rows = []
    for _, r in places.iterrows():
        rows.extend([
            {"place": r["place"], "component": "VoLL", "loss": r["voll_loss_gbp"]},
            {"place": r["place"], "component": "Customer", "loss": r["customer_interruption_loss_gbp"]},
            {"place": r["place"], "component": "Business", "loss": r["business_disruption_loss_gbp"]},
            {"place": r["place"], "component": "Restoration", "loss": r["restoration_loss_gbp"]},
            {"place": r["place"], "component": "Critical services", "loss": r["critical_services_loss_gbp"]},
        ])
    df = pd.DataFrame(rows)
    fig = px.sunburst(df, path=["place", "component"], values="loss", template=plotly_template())
    fig.update_layout(title="Local financial-loss structure", height=470, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def create_mc_histogram(worst: pd.Series) -> go.Figure:
    values = worst.get("mc_histogram", [])
    fig = px.histogram(
        x=values,
        nbins=26,
        title=f"Monte Carlo risk distribution — {worst.get('place')}",
        labels={"x": "Risk score", "y": "Frequency"},
        template=plotly_template(),
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def render_pydeck_map(region: str, places: pd.DataFrame, outages: pd.DataFrame, pc: pd.DataFrame, grid: pd.DataFrame, map_mode: str) -> None:
    try:
        import pydeck as pdk
    except Exception:
        st.map(places.rename(columns={"lat": "latitude", "lon": "longitude"}))
        return

    center = REGIONS[region]["center"]

    places_map = places.copy()
    places_map["color"] = places_map["final_risk_score"].apply(risk_colour)
    places_map["radius"] = 4500 + places_map["final_risk_score"].clip(0, 100) * 130

    grid_map = grid.copy()
    grid_map["color"] = grid_map["risk_score"].apply(lambda x: [56, 189, 248, int(35 + clamp(x, 0, 100) * 1.2)])
    grid_map["radius"] = 2800 + grid_map["risk_score"].clip(0, 100) * 60

    pc_map = pc.copy()
    if not pc_map.empty:
        pc_map["color"] = pc_map["investment_priority"].apply(priority_colour)
        pc_map["radius"] = 3200 + pc_map["recommendation_score"].clip(0, 100) * 105

    layers = []

    if map_mode in ["All", "Risk"]:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=grid_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.38,
            )
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=places_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.88,
            )
        )

    if map_mode in ["All", "Postcode / Investment"] and not pc_map.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pc_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.77,
            )
        )

    if map_mode in ["All", "Outages"] and not outages.empty:
        outages_map = outages.copy()
        outages_map["color"] = [[255, 255, 255, 225]] * len(outages_map)
        outages_map["radius"] = 2500 + pd.to_numeric(outages_map["affected_customers"], errors="coerce").fillna(0).clip(0, 1000) * 16
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=outages_map,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.66,
            )
        )

    tooltip = {
        "html": """
        <b>{place}{postcode}{outage_reference}</b><br/>
        Risk: {final_risk_score}{risk_score}<br/>
        Resilience: {resilience_index}{resilience_score}<br/>
        ENS: {energy_not_supplied_mw} MW<br/>
        Financial loss: £{total_financial_loss_gbp}{financial_loss_gbp}<br/>
        Priority: {investment_priority}
        """,
        "style": {"backgroundColor": "rgba(15,23,42,0.95)", "color": "white"},
    }

    view_state = pdk.ViewState(
        latitude=center["lat"],
        longitude=center["lon"],
        zoom=center["zoom"] - 1,
        pitch=42,
        bearing=-8,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=layers,
            tooltip=tooltip,
        ),
        use_container_width=True,
        height=680,
    )


# =============================================================================
# BBC / WXCHARTS STYLE ANIMATED COMPONENT
# =============================================================================

def make_weather_frames(places: pd.DataFrame, grid: pd.DataFrame, scenario: str) -> Dict[str, Any]:
    hazard_mode = SCENARIOS[scenario]["hazard_mode"]
    frames = []

    for h in range(0, 24, 2):
        phase = math.sin((h / 24) * math.pi * 2)
        frame_cells = []

        for _, g in grid.iterrows():
            wind_factor = 1 + 0.25 * phase + random.uniform(-0.06, 0.06)
            risk_factor = 1 + 0.18 * phase + random.uniform(-0.05, 0.05)
            rain_factor = 1 + 0.35 * max(phase, 0) + random.uniform(-0.05, 0.05)

            frame_cells.append({
                "lat": float(g["lat"]),
                "lon": float(g["lon"]),
                "wind_speed": round(max(0, g["wind_speed"] * wind_factor), 2),
                "rain": round(max(0, g["rain"] * rain_factor), 2),
                "risk_score": round(clamp(g["risk_score"] * risk_factor, 0, 100), 2),
                "resilience_index": round(clamp(g["resilience_index"] - (phase * 7), 0, 100), 2),
                "financial_loss_gbp": float(g["financial_loss_gbp"]),
                "flood_depth_proxy": float(g["flood_depth_proxy"]),
            })

        frames.append({
            "hour": h,
            "label": f"+{h:02d}h",
            "hazard_mode": hazard_mode,
            "cells": frame_cells,
        })

    return {
        "hazard_mode": hazard_mode,
        "scenario": scenario,
        "places": places.to_dict("records"),
        "frames": frames,
    }


def render_bbc_weather_component(region: str, places: pd.DataFrame, grid: pd.DataFrame, scenario: str, height: int = 790) -> None:
    payload = make_weather_frames(places, grid, scenario)
    center = REGIONS[region]["center"]
    payload["center"] = center
    data_json = json.dumps(payload)

    html_code = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
html, body {{
    margin:0;
    padding:0;
    background:#020617;
    font-family: "Segoe UI", Arial, sans-serif;
}}
#scene {{
    position:relative;
    height:{height}px;
    width:100%;
    overflow:hidden;
    border-radius:30px;
    background:
        radial-gradient(circle at 32% 20%, rgba(56,189,248,.20), transparent 26%),
        radial-gradient(circle at 72% 18%, rgba(168,85,247,.18), transparent 28%),
        linear-gradient(180deg, #0a1726 0%, #07101d 42%, #020617 100%);
    border:1px solid rgba(148,163,184,.28);
    box-shadow:0 34px 90px rgba(0,0,0,.42);
}}
canvas {{
    position:absolute;
    inset:0;
}}
#backdrop {{
    z-index:1;
}}
#pressure {{
    z-index:2;
}}
#weather {{
    z-index:3;
}}
#front {{
    z-index:4;
}}
#labels {{
    position:absolute;
    inset:0;
    z-index:5;
    pointer-events:none;
}}
.city {{
    position:absolute;
    color:white;
    font-weight:900;
    font-size:15px;
    text-shadow:0 2px 5px #000, 0 0 4px #000;
    white-space:nowrap;
}}
.city::after {{
    content:"";
    display:inline-block;
    width:7px;
    height:7px;
    background:white;
    margin-left:7px;
    box-shadow:0 1px 5px #000;
}}
.hud {{
    position:absolute;
    z-index:10;
    border:1px solid rgba(255,255,255,.18);
    background:rgba(2,6,23,.68);
    backdrop-filter:blur(14px);
    color:#dbeafe;
    border-radius:18px;
    padding:14px 16px;
}}
#top {{
    top:18px;
    left:18px;
    max-width:520px;
}}
#legend {{
    top:18px;
    right:18px;
    width:260px;
}}
#controls {{
    left:18px;
    right:18px;
    bottom:18px;
    display:grid;
    grid-template-columns:auto auto 1fr auto auto;
    gap:12px;
    align-items:center;
}}
.title {{
    color:white;
    font-size:18px;
    font-weight:950;
    margin-bottom:5px;
}}
.sub {{
    font-size:12px;
    line-height:1.5;
}}
.blocks {{
    position:absolute;
    left:24px;
    bottom:110px;
    z-index:11;
    display:flex;
    align-items:center;
    gap:10px;
    color:white;
    text-shadow:0 3px 10px rgba(0,0,0,.85);
}}
.bbc span {{
    display:inline-grid;
    place-items:center;
    width:35px;
    height:35px;
    background:rgba(255,255,255,.96);
    color:#1e293b;
    font-weight:950;
    font-size:22px;
    margin-right:5px;
}}
.word {{
    font-size:34px;
    font-weight:950;
    letter-spacing:-.04em;
}}
.timebox {{
    position:absolute;
    right:24px;
    bottom:114px;
    z-index:11;
    display:flex;
    box-shadow:0 8px 22px rgba(0,0,0,.45);
    font-weight:950;
}}
.day {{
    background:rgba(13,148,136,.94);
    color:white;
    padding:11px 18px;
    font-size:16px;
    letter-spacing:.04em;
}}
.hour {{
    background:rgba(2,6,23,.92);
    color:white;
    padding:11px 18px;
    min-width:80px;
    text-align:center;
    font-size:16px;
}}
.gradient {{
    height:14px;
    border-radius:999px;
    background:linear-gradient(90deg, rgba(59,130,246,.38), rgba(37,99,235,.56), rgba(34,197,94,.66), rgba(234,179,8,.80), rgba(249,115,22,.86), rgba(239,68,68,.92), rgba(168,85,247,.95));
    margin:8px 0;
}}
button {{
    border:0;
    border-radius:14px;
    background:linear-gradient(135deg,#0284c7,#38bdf8);
    color:white;
    font-weight:950;
    padding:10px 14px;
    cursor:pointer;
}}
input[type=range] {{
    width:100%;
}}
.pill {{
    color:#bfdbfe;
    border:1px solid rgba(148,163,184,.25);
    border-radius:999px;
    padding:8px 12px;
    background:rgba(15,23,42,.78);
    font-weight:850;
    font-size:12px;
}}
</style>
</head>
<body>
<div id="scene">
    <canvas id="backdrop"></canvas>
    <canvas id="pressure"></canvas>
    <canvas id="weather"></canvas>
    <canvas id="front"></canvas>
    <div id="labels"></div>

    <div class="hud" id="top">
        <div class="title">Q1 forecast simulation and grid resilience overlay</div>
        <div class="sub">Scenario: <b>{html.escape(scenario)}</b><br>Visual mode: <b>{html.escape(payload["hazard_mode"])}</b><br>Animated precipitation, pressure contours, wind vectors, fronts and local risk intensity.</div>
    </div>

    <div class="hud" id="legend">
        <div class="title">Hazard intensity</div>
        <div class="gradient"></div>
        <div class="sub" style="display:flex;justify-content:space-between;"><span>Light</span><span>Heavy</span><span>Extreme</span></div>
        <hr style="border-color:rgba(255,255,255,.14);">
        <div class="sub">● blue/green: lower stress<br>● amber/red/purple: high grid hazard</div>
    </div>

    <div class="blocks">
        <div class="bbc"><span>B</span><span>B</span><span>C</span></div>
        <div class="word">WEATHER</div>
    </div>

    <div class="timebox">
        <div class="day">FRIDAY</div>
        <div class="hour" id="hour">00h</div>
    </div>

    <div class="hud" id="controls">
        <button onclick="play()">▶ Play</button>
        <button onclick="pause()">Ⅱ Pause</button>
        <input id="slider" type="range" min="0" max="11" value="0" oninput="scrub(this.value)">
        <span class="pill" id="condition">{html.escape(scenario)}</span>
        <span class="pill" id="stats">Initialising</span>
    </div>
</div>

<script>
const data = {data_json};
const scene = document.getElementById("scene");
const backdrop = document.getElementById("backdrop");
const pressure = document.getElementById("pressure");
const weather = document.getElementById("weather");
const front = document.getElementById("front");
const labels = document.getElementById("labels");
const bctx = backdrop.getContext("2d");
const pctx = pressure.getContext("2d");
const wctx = weather.getContext("2d");
const fctx = front.getContext("2d");
const slider = document.getElementById("slider");
const hour = document.getElementById("hour");
const stats = document.getElementById("stats");
const condition = document.getElementById("condition");

let W = 1000, H = {height};
let currentFrame = data.frames[0];
let frameIndex = 0;
let timer = null;
let playing = true;
let lastT = performance.now();
let rainBands = [];
let cloudShields = [];
let windArrows = [];
let vortices = [];
let lightningFlash = 0;

function resize() {{
    const rect = scene.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    W = rect.width;
    H = rect.height;
    [backdrop, pressure, weather, front].forEach(c => {{
        c.width = Math.floor(W * dpr);
        c.height = Math.floor(H * dpr);
        c.style.width = W + "px";
        c.style.height = H + "px";
    }});
    [bctx, pctx, wctx, fctx].forEach(ctx => ctx.setTransform(dpr,0,0,dpr,0,0));
    drawBackdrop();
    layoutLabels();
}}
function project(lat, lon) {{
    const bbox = {json.dumps(REGIONS[region]["bbox"])};
    const minLon = bbox[0], minLat = bbox[1], maxLon = bbox[2], maxLat = bbox[3];
    const x = (lon - minLon) / (maxLon - minLon) * W;
    const y = H - (lat - minLat) / (maxLat - minLat) * H;
    return {{x, y}};
}}
function drawBackdrop() {{
    bctx.clearRect(0,0,W,H);
    const grad = bctx.createLinearGradient(0,0,W,H);
    grad.addColorStop(0,"#0b2338");
    grad.addColorStop(.45,"#092037");
    grad.addColorStop(1,"#02101f");
    bctx.fillStyle = grad;
    bctx.fillRect(0,0,W,H);

    bctx.strokeStyle = "rgba(148,163,184,.10)";
    bctx.lineWidth = 1;
    for (let x=0; x<W; x+=42) {{
        bctx.beginPath(); bctx.moveTo(x,0); bctx.lineTo(x,H); bctx.stroke();
    }}
    for (let y=0; y<H; y+=42) {{
        bctx.beginPath(); bctx.moveTo(0,y); bctx.lineTo(W,y); bctx.stroke();
    }}

    data.frames[0].cells.forEach(c => {{
        const p = project(c.lat,c.lon);
        const r = 18 + c.risk_score * .85;
        const grd = bctx.createRadialGradient(p.x,p.y,0,p.x,p.y,r);
        grd.addColorStop(0, colour(c.risk_score,.28));
        grd.addColorStop(1, "rgba(0,0,0,0)");
        bctx.fillStyle = grd;
        bctx.beginPath();
        bctx.ellipse(p.x,p.y,r,r*.62,0,0,Math.PI*2);
        bctx.fill();
    }});
}}
function layoutLabels() {{
    labels.innerHTML = "";
    data.places.forEach(p => {{
        const xy = project(p.lat,p.lon);
        const d = document.createElement("div");
        d.className = "city";
        d.textContent = p.place;
        d.style.left = Math.max(8, Math.min(W-130, xy.x + 8)) + "px";
        d.style.top = Math.max(8, Math.min(H-30, xy.y - 10)) + "px";
        labels.appendChild(d);
    }});
}}
function colour(v,a) {{
    if (v >= 86) return "rgba(168,85,247,"+a+")";
    if (v >= 76) return "rgba(239,68,68,"+a+")";
    if (v >= 63) return "rgba(249,115,22,"+a+")";
    if (v >= 50) return "rgba(234,179,8,"+a+")";
    if (v >= 35) return "rgba(34,197,94,"+a+")";
    return "rgba(59,130,246,"+a+")";
}}
function initWeather() {{
    rainBands = []; cloudShields = []; windArrows = []; vortices = [];
    const mode = data.hazard_mode;
    const rb = mode === "storm" ? 54 : mode === "rain" ? 42 : mode === "wind" ? 26 : 20;
    for(let i=0;i<rb;i++) rainBands.push({{
        x:-W*.45 + Math.random()*W*1.75,
        y:-H*.12 + Math.random()*H*1.18,
        rx:65+Math.random()*260,
        ry:20+Math.random()*90,
        speed:.16+Math.random()*.72,
        alpha:.08+Math.random()*.24,
        phase:Math.random()*Math.PI*2,
        bias:Math.random(),
        rotate:-.35+Math.random()*.7
    }});
    const cc = mode === "storm" ? 26 : mode === "rain" ? 20 : mode === "wind" ? 14 : 10;
    for(let i=0;i<cc;i++) cloudShields.push({{
        x:-W*.55 + Math.random()*W*1.95,
        y:-H*.10 + Math.random()*H*1.15,
        rx:150+Math.random()*420,
        ry:42+Math.random()*125,
        speed:.08+Math.random()*.34,
        alpha:.055+Math.random()*.13,
        phase:Math.random()*Math.PI*2,
        rotate:-.25+Math.random()*.5
    }});
    const wc = mode === "storm" ? 150 : mode === "wind" ? 120 : mode === "rain" ? 85 : 65;
    for(let i=0;i<wc;i++) windArrows.push({{
        x:Math.random()*W, y:Math.random()*H,
        len:28+Math.random()*70, speed:.50+Math.random()*1.50,
        alpha:.35+Math.random()*.38, width:1.4+Math.random()*2.6,
        phase:Math.random()*Math.PI*2
    }});
    const vc = mode === "storm" ? 3 : mode === "rain" ? 2 : 1;
    for(let i=0;i<vc;i++) vortices.push({{
        x:W*(.28+Math.random()*.48),
        y:H*(.25+Math.random()*.52),
        radius:95+Math.random()*190,
        strength:.35+Math.random()*.65,
        speed:.05+Math.random()*.12,
        phase:Math.random()*Math.PI*2
    }});
}}
function avg(key) {{
    if(!currentFrame || !currentFrame.cells) return 0;
    return currentFrame.cells.reduce((s,c)=>s+Number(c[key]||0),0)/currentFrame.cells.length;
}}
function nearestCell(x,y) {{
    let best=null, bestD=1e9;
    currentFrame.cells.forEach(c => {{
        const p=project(c.lat,c.lon);
        const d=(p.x-x)*(p.x-x)+(p.y-y)*(p.y-y);
        if(d<bestD) {{bestD=d;best=c;}}
    }});
    return best;
}}
function drawPressure(t) {{
    pctx.clearRect(0,0,W,H);
    pctx.save();
    pctx.globalAlpha=.62;
    pctx.strokeStyle="rgba(255,255,255,.55)";
    pctx.lineWidth=1.5;
    for(let family=0;family<2;family++) {{
        const cx=W*(family===0?.27:.72)+Math.sin(t/6200+family)*28;
        const cy=H*(family===0?.55:.36)+Math.cos(t/5300+family)*22;
        for(let k=0;k<8;k++) {{
            const rx=90+k*42+family*20;
            const ry=50+k*28;
            pctx.beginPath();
            pctx.ellipse(cx,cy,rx,ry,-.38+family*.65,0,Math.PI*2);
            pctx.stroke();
        }}
    }}
    pctx.lineWidth=1.1;
    for(let k=0;k<8;k++) {{
        pctx.beginPath();
        for(let x=-70;x<=W+80;x+=18) {{
            const y=H*.22+k*76+Math.sin((x+t*.018)/125+k*.55)*(25+k*2);
            if(x===-70)pctx.moveTo(x,y); else pctx.lineTo(x,y);
        }}
        pctx.stroke();
    }}
    pctx.restore();
}}
function drawFronts(t) {{
    fctx.clearRect(0,0,W,H);
    fctx.save();
    fctx.lineWidth=2.5;
    fctx.strokeStyle="rgba(255,255,255,.76)";
    fctx.beginPath();
    const baseY=H*.61;
    for(let x=-60;x<=W+70;x+=22) {{
        const y=baseY+Math.sin((x+t*.025)/118)*42;
        if(x===-60)fctx.moveTo(x,y); else fctx.lineTo(x,y);
    }}
    fctx.stroke();
    for(let x=10;x<W;x+=74) {{
        const y=baseY+Math.sin((x+t*.025)/118)*42;
        fctx.fillStyle="rgba(59,130,246,.88)";
        fctx.beginPath(); fctx.moveTo(x,y); fctx.lineTo(x+19,y+16); fctx.lineTo(x-8,y+18); fctx.closePath(); fctx.fill();
        fctx.fillStyle="rgba(239,68,68,.88)";
        fctx.beginPath(); fctx.arc(x+38,y-1,10,Math.PI,0); fctx.fill();
    }}
    fctx.restore();
}}
function ellipse(ctx,x,y,rx,ry,fill,rot=0) {{
    ctx.save(); ctx.translate(x,y); ctx.rotate(rot); ctx.beginPath(); ctx.ellipse(0,0,rx,ry,0,0,Math.PI*2); ctx.fillStyle=fill; ctx.fill(); ctx.restore();
}}
function drawClouds(t,dt) {{
    cloudShields.forEach(c => {{
        c.x += c.speed*dt*.05;
        c.y += Math.sin(t/2500+c.phase)*.035*dt;
        if(c.x-c.rx>W+160) {{ c.x=-c.rx-180; c.y=-H*.10+Math.random()*H*1.15; }}
        const grad=wctx.createRadialGradient(c.x,c.y,0,c.x,c.y,c.rx);
        grad.addColorStop(0,"rgba(255,255,255,"+c.alpha+")");
        grad.addColorStop(.42,"rgba(220,230,235,"+c.alpha*.72+")");
        grad.addColorStop(.78,"rgba(160,176,188,"+c.alpha*.30+")");
        grad.addColorStop(1,"rgba(160,176,188,0)");
        ellipse(wctx,c.x,c.y,c.rx,c.ry,grad,c.rotate);
    }});
}}
function drawPrecip(t,dt) {{
    const mode=data.hazard_mode;
    const movement=mode==="storm"?1.55:mode==="rain"?1.18:.74;
    const avgRain=avg("rain"), avgRisk=avg("risk_score");
    currentFrame.cells.forEach((cell,idx) => {{
        const p=project(cell.lat,cell.lon);
        const rain=Number(cell.rain||0), risk=Number(cell.risk_score||0);
        if(rain<.1 && risk<30 && mode!=="storm" && mode!=="rain") return;
        const pulse=.94+.06*Math.sin(t/680+idx);
        const rx=(46+rain*28+risk*.80)*pulse;
        const ry=(21+rain*12+risk*.34)*pulse;
        const alpha=Math.min(.62,.085+rain*.065+risk/430);
        const grad=wctx.createRadialGradient(p.x,p.y,0,p.x,p.y,rx);
        grad.addColorStop(0,colour(risk,alpha));
        grad.addColorStop(.42,colour(risk,alpha*.58));
        grad.addColorStop(.74,"rgba(59,130,246,"+alpha*.20+")");
        grad.addColorStop(1,"rgba(0,0,0,0)");
        ellipse(wctx,p.x,p.y,rx,ry,grad);
    }});
    rainBands.forEach(b => {{
        b.x += b.speed*movement*dt*.068;
        b.y += Math.sin(t/1600+b.phase)*.055*dt;
        if(b.x-b.rx>W+150) {{ b.x=-b.rx-170; b.y=-H*.12+Math.random()*H*1.18; }}
        const syntheticRisk=avgRisk+b.bias*45;
        const alpha=Math.min(.55,b.alpha*(.65+avgRain/3.8+avgRisk/190));
        const grad=wctx.createRadialGradient(b.x,b.y,0,b.x,b.y,b.rx);
        grad.addColorStop(0,colour(syntheticRisk,alpha));
        grad.addColorStop(.46,colour(syntheticRisk,alpha*.54));
        grad.addColorStop(.80,"rgba(37,99,235,"+alpha*.18+")");
        grad.addColorStop(1,"rgba(0,0,0,0)");
        ellipse(wctx,b.x,b.y,b.rx,b.ry,grad,b.rotate);
    }});
    vortices.forEach(v => {{
        v.phase += v.speed*dt*.01;
        for(let arm=0;arm<4;arm++) {{
            for(let j=0;j<28;j++) {{
                const theta=v.phase+arm*Math.PI/2+j*.18;
                const r=18+j*(v.radius/28);
                const x=v.x+Math.cos(theta)*r;
                const y=v.y+Math.sin(theta)*r*.60;
                const a=Math.max(0,(1-j/30)*.16*v.strength);
                ellipse(wctx,x,y,22+j*1.0,8+j*.35,colour(avgRisk+30,a));
            }}
        }}
    }});
}}
function drawWind(t,dt) {{
    const mode=data.hazard_mode;
    const mult=mode==="storm"?1.65:mode==="wind"?1.38:mode==="rain"?.92:.72;
    windArrows.forEach(a => {{
        const local=nearestCell(a.x,a.y);
        const w=local?Number(local.wind_speed||9):avg("wind_speed");
        const intensity=Math.min(w/42,1.55);
        let angle=-.24+Math.sin(t/1800+a.phase)*.12;
        vortices.forEach(v => {{
            const dx=a.x-v.x, dy=a.y-v.y, d=Math.sqrt(dx*dx+dy*dy);
            if(d<v.radius*2.1) angle += Math.atan2(dy,dx)*.10*v.strength;
        }});
        const len=a.len*(.74+intensity*.58);
        const x0=a.x-Math.cos(angle)*len, y0=a.y-Math.sin(angle)*len;
        const alpha=Math.min(.86,a.alpha+intensity*.20);
        wctx.save();
        wctx.strokeStyle="rgba(255,255,255,"+alpha+")";
        wctx.fillStyle="rgba(255,255,255,"+alpha+")";
        wctx.lineWidth=a.width;
        wctx.lineCap="round";
        wctx.beginPath(); wctx.moveTo(x0,y0);
        wctx.quadraticCurveTo((x0+a.x)/2,(y0+a.y)/2+Math.sin(t/540+a.phase)*5,a.x,a.y);
        wctx.stroke();
        const head=10+intensity*7;
        const bx=a.x-Math.cos(angle)*head, by=a.y-Math.sin(angle)*head;
        const nx=-Math.sin(angle), ny=Math.cos(angle);
        wctx.beginPath(); wctx.moveTo(a.x,a.y); wctx.lineTo(bx+nx*head*.42,by+ny*head*.42); wctx.lineTo(bx-nx*head*.42,by-ny*head*.42); wctx.closePath(); wctx.fill();
        wctx.restore();
        a.x += Math.cos(angle)*a.speed*mult*(.66+intensity)*dt*.083;
        a.y += Math.sin(angle)*a.speed*mult*(.66+intensity)*dt*.083;
        if(a.x>W+115 || a.y<-85 || a.y>H+85) {{ a.x=-115; a.y=Math.random()*H; a.phase=Math.random()*Math.PI*2; }}
    }});
}}
function drawLightning(W,H) {{
    if(data.hazard_mode!=="storm") return;
    if(Math.random()<.007) lightningFlash=6;
    if(lightningFlash>0) {{
        wctx.fillStyle="rgba(255,255,255,"+(.05+lightningFlash*.012)+")";
        wctx.fillRect(0,0,W,H);
        for(let bolt=0; bolt<2; bolt++) {{
            wctx.strokeStyle="rgba(255,255,255,.64)";
            wctx.lineWidth=2.1; wctx.beginPath();
            let x=W*(.15+Math.random()*.75), y=0; wctx.moveTo(x,y);
            for(let i=0;i<6;i++) {{ x += -35+Math.random()*70; y += 34+Math.random()*62; wctx.lineTo(x,y); }}
            wctx.stroke();
        }}
        lightningFlash--;
    }}
}}
function animate(t) {{
    const dt=Math.min(34,t-lastT); lastT=t;
    wctx.clearRect(0,0,W,H);
    wctx.fillStyle=data.hazard_mode==="storm"?"rgba(5,12,24,.045)":"rgba(5,15,28,.025)";
    wctx.fillRect(0,0,W,H);
    drawPressure(t);
    drawFronts(t);
    drawClouds(t,dt);
    drawPrecip(t,dt);
    drawWind(t,dt);
    drawLightning(W,H);
    stats.textContent = "Wind " + avg("wind_speed").toFixed(1) + " km/h · Rain " + avg("rain").toFixed(1) + " mm · Risk " + avg("risk_score").toFixed(1);
    requestAnimationFrame(animate);
}}
function renderFrame(i) {{
    frameIndex=i; currentFrame=data.frames[i]; slider.value=i;
    hour.textContent=String(currentFrame.label).replace("+","");
    condition.textContent=data.scenario+" · "+data.hazard_mode;
}}
function play() {{
    if(timer) clearInterval(timer);
    playing=true;
    timer=setInterval(()=>{{ frameIndex=(frameIndex+1)%data.frames.length; renderFrame(frameIndex); }},950);
}}
function pause() {{
    playing=false; if(timer) clearInterval(timer);
}}
function scrub(v) {{
    renderFrame(parseInt(v));
}}
window.addEventListener("resize",resize);
resize(); initWeather(); renderFrame(0); play(); requestAnimationFrame(animate);
</script>
</body>
</html>
"""
    components.html(html_code, height=height + 8, scrolling=False)


# =============================================================================
# UI PANELS
# =============================================================================

def hero(region: str, scenario: str, mc_runs: int, refresh_id: int) -> None:
    st.markdown(
        f"""
        <div class="q1-hero">
            <div class="q1-title">⚡ SAT-Guard Q1 Grid Digital Twin</div>
            <div class="q1-subtitle">
                Broadcast-style weather simulation, multi-layer grid-risk modelling, social vulnerability,
                outage intelligence, Monte Carlo uncertainty and investment prioritisation for {html.escape(region)}.
            </div>
            <div style="margin-top:10px;">
                <span class="q1-chip">{html.escape(region)}</span>
                <span class="q1-chip">{html.escape(scenario)}</span>
                <span class="q1-chip">MC runs: {mc_runs}</span>
                <span class="q1-chip">Refresh ID: {refresh_id}</span>
                <span class="q1-chip">UTC {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics_panel(places: pd.DataFrame, pc: pd.DataFrame) -> None:
    avg_risk = round(float(places["final_risk_score"].mean()), 1)
    avg_res = round(float(places["resilience_index"].mean()), 1)
    avg_failure = round(float(places["failure_probability"].mean()) * 100, 1)
    total_ens = round(float(places["energy_not_supplied_mw"].sum()), 1)
    total_loss = round(float(places["total_financial_loss_gbp"].sum()), 2)
    p1 = 0 if pc.empty else int((pc["investment_priority"] == "Priority 1").sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Regional risk", f"{avg_risk}/100", risk_label(avg_risk))
    c2.metric("Resilience", f"{avg_res}/100", resilience_label(avg_res))
    c3.metric("Failure prob.", f"{avg_failure}%")
    c4.metric("ENS", f"{total_ens} MW")
    c5.metric("Financial loss", money_m(total_loss))
    c6.metric("Priority 1", p1)


def overview_tab(places: pd.DataFrame, pc: pd.DataFrame, scenario: str) -> None:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Regional intelligence table")
        display_cols = [
            "place", "risk_label", "final_risk_score", "resilience_label", "resilience_index",
            "wind_speed_10m", "precipitation", "european_aqi", "imd_score",
            "social_vulnerability", "energy_not_supplied_mw", "total_financial_loss_gbp",
        ]
        st.dataframe(
            places[display_cols].sort_values("final_risk_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        avg_risk = float(places["final_risk_score"].mean())
        avg_res = float(places["resilience_index"].mean())
        g1, g2 = st.columns(2)
        g1.plotly_chart(create_risk_gauge(avg_risk, "Regional risk"), use_container_width=True)
        g2.plotly_chart(create_resilience_gauge(avg_res, "Resilience"), use_container_width=True)

    a, b = st.columns(2)
    with a:
        fig = px.bar(
            places.sort_values("final_risk_score", ascending=False),
            x="place",
            y="final_risk_score",
            color="risk_label",
            title="Risk ranking by location",
            template=plotly_template(),
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.scatter(
            places,
            x="social_vulnerability",
            y="final_risk_score",
            size="total_financial_loss_gbp",
            color="resilience_index",
            hover_name="place",
            title="Social vulnerability vs grid risk",
            template=plotly_template(),
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="q1-note">
            <b>Scenario logic:</b> {html.escape(SCENARIOS[scenario]["description"])}
            The deterministic model is combined with Monte Carlo perturbations over wind, rain, temperature,
            AQI, solar radiation, cloud cover and energy-not-supplied uncertainty.
        </div>
        """,
        unsafe_allow_html=True,
    )


def bbc_tab(region: str, scenario: str, places: pd.DataFrame, grid: pd.DataFrame) -> None:
    st.subheader("BBC / WXCharts-style animated grid hazard simulation")
    st.caption("Canvas-based animation embedded inside Streamlit: moving precipitation shields, pressure contours, frontal boundaries, wind vectors, lightning for storm mode, and city labels.")
    render_bbc_weather_component(region, places, grid, scenario, height=790)


def spatial_tab(region: str, places: pd.DataFrame, outages: pd.DataFrame, pc: pd.DataFrame, grid: pd.DataFrame, map_mode: str) -> None:
    st.subheader("3D operational map")
    render_pydeck_map(region, places, outages, pc, grid, map_mode)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Highest-risk grid cells")
        st.dataframe(grid.sort_values("risk_score", ascending=False).head(40), use_container_width=True, hide_index=True)
    with b:
        st.markdown("#### Outage layer")
        st.dataframe(outages.head(100), use_container_width=True, hide_index=True)


def resilience_tab(places: pd.DataFrame) -> None:
    st.subheader("Resilience and cascade diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_cascade_radar(places), use_container_width=True)
    with c2:
        fig = px.scatter(
            places,
            x="grid_failure_probability",
            y="resilience_index",
            size="energy_not_supplied_mw",
            color="social_vulnerability",
            hover_name="place",
            template=plotly_template(),
            title="Grid failure vs resilience",
            color_continuous_scale="Plasma",
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    cols = [
        "place", "resilience_label", "resilience_index", "imd_score", "imd_match",
        "social_vulnerability", "grid_failure_probability", "renewable_failure_probability",
        "cascade_power", "cascade_water", "cascade_telecom", "cascade_transport",
        "cascade_social", "total_financial_loss_gbp",
    ]
    st.dataframe(places[cols].sort_values("resilience_index"), use_container_width=True, hide_index=True)


def investment_tab(pc: pd.DataFrame, rec: pd.DataFrame) -> None:
    st.subheader("Postcode resilience and investment engine")

    if pc.empty or rec.empty:
        st.warning("No postcode-level resilience or investment recommendations could be generated.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Postcode areas", len(pc))
    c2.metric("Priority 1", int((rec["investment_priority"] == "Priority 1").sum()))
    c3.metric("Programme cost", money_m(rec["indicative_investment_cost_gbp"].sum()))
    c4.metric("Exposed loss", money_m(rec["financial_loss_gbp"].sum()))

    a, b = st.columns([1.0, 1.0])
    with a:
        fig = px.bar(
            rec.head(14),
            x="postcode",
            y="recommendation_score",
            color="investment_priority",
            title="Investment urgency by postcode",
            template=plotly_template(),
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.scatter(
            rec,
            x="financial_loss_gbp",
            y="recommendation_score",
            size="indicative_investment_cost_gbp",
            color="investment_priority",
            hover_name="postcode",
            title="Recommendation score vs financial-loss exposure",
            template=plotly_template(),
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Actionable recommendations")
    cols = [
        "postcode", "nearest_place", "investment_priority", "recommendation_score",
        "investment_category", "recommended_action", "indicative_investment_cost_gbp",
        "financial_loss_gbp", "resilience_score", "risk_score",
    ]
    st.dataframe(rec[cols], use_container_width=True, hide_index=True)


def finance_tab(places: pd.DataFrame) -> None:
    st.subheader("Financial loss model")
    a, b = st.columns([1, 1])
    with a:
        st.plotly_chart(create_loss_waterfall(places), use_container_width=True)
    with b:
        st.plotly_chart(create_finance_sunburst(places), use_container_width=True)

    fin_cols = [
        "place", "energy_not_supplied_mw", "ens_mwh", "estimated_duration_hours",
        "voll_loss_gbp", "customer_interruption_loss_gbp", "business_disruption_loss_gbp",
        "restoration_loss_gbp", "critical_services_loss_gbp", "total_financial_loss_gbp",
    ]
    st.dataframe(
        places[fin_cols].sort_values("total_financial_loss_gbp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def monte_carlo_tab(places: pd.DataFrame) -> None:
    st.subheader("Monte Carlo uncertainty")

    worst = places.sort_values("mc_p95", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Worst P95 risk", worst["mc_p95"], worst["place"])
    c2.metric("Extreme probability", f"{round(worst['mc_extreme_probability'] * 100, 1)}%")
    c3.metric("P95 financial loss", money_m(worst["mc_financial_loss_p95"]))
    c4.metric("Mean MC resilience", worst["mc_resilience_mean"])

    a, b = st.columns([1, 1])
    with a:
        st.plotly_chart(create_mc_histogram(worst), use_container_width=True)
    with b:
        fig = px.scatter(
            places,
            x="mc_mean",
            y="mc_p95",
            size="mc_financial_loss_p95",
            color="mc_extreme_probability",
            hover_name="place",
            title="Mean risk vs P95 tail risk",
            template=plotly_template(),
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    mc_cols = [
        "place", "mc_mean", "mc_std", "mc_p05", "mc_p50", "mc_p95",
        "mc_extreme_probability", "mc_resilience_mean", "mc_resilience_p05",
        "mc_financial_loss_p95",
    ]
    st.dataframe(places[mc_cols].sort_values("mc_p95", ascending=False), use_container_width=True, hide_index=True)


def method_tab(places: pd.DataFrame) -> None:
    st.subheader("Model transparency")
    st.markdown(
        """
        <div class="q1-card">
        <h3 style="color:white;margin-top:0;">Core modelling structure</h3>
        <p style="color:#cbd5e1;">
        The dashboard combines hazard intensity, pollution, renewable generation stress,
        outage proximity, Energy Not Supplied, social vulnerability and financial loss into a
        location-level digital twin score. The model is deliberately transparent so it can be
        described in a research paper and later calibrated against observed outage histories.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="q1-formula">
            Risk = weather + pollution + net-load stress + outage intensity + ENS pressure<br><br>
            Failure probability = logistic(0.065 × (risk - 60))
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="q1-formula">
            Resilience = 100 − risk penalty − social vulnerability penalty − grid failure penalty
            − renewable failure penalty − cascade stress − finance penalty
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("#### Current model output sample")
    st.dataframe(places.head(10), use_container_width=True, hide_index=True)


def export_tab(places: pd.DataFrame, outages: pd.DataFrame, grid: pd.DataFrame, pc: pd.DataFrame, rec: pd.DataFrame) -> None:
    st.subheader("Data and export")
    with st.expander("Place-level model outputs", expanded=True):
        st.dataframe(places, use_container_width=True, hide_index=True)
    with st.expander("Postcode resilience"):
        st.dataframe(pc, use_container_width=True, hide_index=True)
    with st.expander("Investment recommendations"):
        st.dataframe(rec, use_container_width=True, hide_index=True)
    with st.expander("Outage layer"):
        st.dataframe(outages, use_container_width=True, hide_index=True)
    with st.expander("Grid cells"):
        st.dataframe(grid, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download places CSV",
        places.to_csv(index=False).encode("utf-8"),
        file_name="sat_guard_q1_places.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download recommendations CSV",
        rec.to_csv(index=False).encode("utf-8") if not rec.empty else b"",
        file_name="sat_guard_q1_recommendations.csv",
        mime="text/csv",
        disabled=rec.empty,
    )
    c3.download_button(
        "Download grid CSV",
        grid.to_csv(index=False).encode("utf-8"),
        file_name="sat_guard_q1_grid.csv",
        mime="text/csv",
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    if "refresh_id" not in st.session_state:
        st.session_state.refresh_id = 0

    with st.sidebar:
        st.markdown("## ⚡ SAT-Guard Q1")
        st.caption("Digital twin control panel")

        region = st.selectbox("Region", list(REGIONS.keys()), index=0)
        scenario = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
        mc_runs = st.slider("Monte Carlo runs", min_value=10, max_value=160, value=40, step=10)
        map_mode = st.selectbox("Map layer", ["All", "Risk", "Postcode / Investment", "Outages"], index=0)

        st.markdown("---")
        st.info(SCENARIOS[scenario]["description"])

        if st.button("Run / refresh model", type="primary"):
            st.session_state.refresh_id += 1
            st.cache_data.clear()
            st.rerun()

        if st.button("Clear cache"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption("Put IoD2025 Excel files in the same folder or data/ folder. Fallback vulnerability proxies are used when files are absent.")

    hero(region, scenario, mc_runs, st.session_state.refresh_id)

    with st.spinner("Running Q1 digital twin model..."):
        places, outages, grid = get_data_cached(region, scenario, mc_runs)
        pc = build_postcode_resilience(places, outages)
        rec = build_investment_recommendations(places, outages)

    if places.empty:
        st.error("No model data could be generated.")
        return

    metrics_panel(places, pc)

    imd_source = places.iloc[0].get("imd_dataset_summary", "Unknown")
    st.caption(f"IoD / deprivation data source: {imd_source}")

    tabs = st.tabs([
        "Executive overview",
        "BBC simulation",
        "Spatial intelligence",
        "Resilience",
        "Investment",
        "Finance",
        "Monte Carlo",
        "Method",
        "Data / Export",
    ])

    with tabs[0]:
        overview_tab(places, pc, scenario)

    with tabs[1]:
        bbc_tab(region, scenario, places, grid)

    with tabs[2]:
        spatial_tab(region, places, outages, pc, grid, map_mode)

    with tabs[3]:
        resilience_tab(places)

    with tabs[4]:
        investment_tab(pc, rec)

    with tabs[5]:
        finance_tab(places)

    with tabs[6]:
        monte_carlo_tab(places)

    with tabs[7]:
        method_tab(places)

    with tabs[8]:
        export_tab(places, outages, grid, pc, rec)


if __name__ == "__main__":
    main()
