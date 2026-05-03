"""
SAT-Guard / North East & Yorkshire Grid Digital Twin
Streamlit single-file version
====================================================

This version removes Flask routes and converts the application into a deployable
Streamlit dashboard.

Main features:
- Region and scenario controls
- Open-Meteo weather and air-quality ingestion with cache
- Northern Powergrid outage ingestion with fallback points
- IoD2025 / deprivation Excel loader with fallback proxies
- Multi-layer risk, resilience, financial loss and Monte Carlo model
- Postcode resilience and investment recommendations
- Streamlit maps, charts, tables and CSV exports

Run locally:
    pip install streamlit pandas numpy requests openpyxl pydeck
    streamlit run streamlit_app.py

Streamlit Cloud:
    Main file path: streamlit_app.py

Recommended requirements.txt:
    streamlit
    pandas
    numpy
    requests
    openpyxl
    pydeck
"""

import math
import random
import re
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="SAT-Guard Grid Digital Twin",
    page_icon="⚡",
    layout="wide",
)


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

NASA_LAYERS = {
    "VIIRS NOAA-20 True Colour": (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "VIIRS_NOAA20_CorrectedReflectance_TrueColor/default/"
        "{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    ),
    "MODIS Terra True Colour": (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "MODIS_Terra_CorrectedReflectance_TrueColor/default/"
        "{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    ),
    "NOAA-20 Night Lights": (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "VIIRS_NOAA20_DayNightBand_At_Sensor_Radiance/default/"
        "{date}/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png"
    ),
    "Black Marble Night Lights": (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "BlackMarble_2016/default/{date}/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png"
    ),
}

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
    return max(low, min(high, float(value)))


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


def get_nasa_tile_url(layer_name: str, date_string: str = None) -> str:
    if layer_name not in NASA_LAYERS:
        layer_name = "VIIRS NOAA-20 True Colour"
    if date_string is None:
        date_string = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")
    return NASA_LAYERS[layer_name].replace("{date}", date_string)


def requests_json(url: str, params: Dict[str, Any] = None, timeout: int = 20) -> Dict[str, Any]:
    try:
        headers = {"User-Agent": "sat-guard-streamlit/1.0"}
        response = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


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
# MODELS
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
    simulations = int(clamp(simulations, 10, 120))
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

    lats = np.linspace(min_lat, max_lat, 13)
    lons = np.linspace(min_lon, max_lon, 13)

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
# VISUAL HELPERS
# =============================================================================

def risk_colour(score: float) -> List[int]:
    score = safe_float(score)
    if score >= 75:
        return [239, 68, 68, 190]
    if score >= 55:
        return [249, 115, 22, 180]
    if score >= 35:
        return [234, 179, 8, 170]
    return [34, 197, 94, 170]


def priority_colour(priority: str) -> List[int]:
    if priority == "Priority 1":
        return [239, 68, 68, 190]
    if priority == "Priority 2":
        return [249, 115, 22, 180]
    if priority == "Priority 3":
        return [234, 179, 8, 170]
    return [34, 197, 94, 160]


def add_colour_columns(places: pd.DataFrame, pc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    places_map = places.copy()
    places_map["color"] = places_map["final_risk_score"].apply(risk_colour)
    places_map["radius"] = 6000 + places_map["final_risk_score"].clip(0, 100) * 120

    pc_map = pc.copy()
    if not pc_map.empty:
        pc_map["color"] = pc_map["investment_priority"].apply(priority_colour)
        pc_map["radius"] = 4000 + pc_map["recommendation_score"].clip(0, 100) * 110

    return places_map, pc_map


def render_pydeck_map(region: str, places: pd.DataFrame, outages: pd.DataFrame, pc: pd.DataFrame, map_mode: str) -> None:
    try:
        import pydeck as pdk
    except Exception:
        st.map(places.rename(columns={"lat": "latitude", "lon": "longitude"}))
        return

    places_map, pc_map = add_colour_columns(places, pc)

    center = REGIONS[region]["center"]

    layers = []

    if map_mode in ["Risk", "All"]:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=places_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.80,
            )
        )

    if map_mode in ["Postcode / Investment", "All"] and not pc_map.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pc_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.70,
            )
        )

    if map_mode in ["Outages", "All"] and not outages.empty:
        outages_map = outages.copy()
        outages_map["color"] = [[255, 255, 255, 210]] * len(outages_map)
        outages_map["radius"] = 4200 + pd.to_numeric(outages_map["affected_customers"], errors="coerce").fillna(0).clip(0, 1000) * 15
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=outages_map,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.65,
            )
        )

    tooltip = {
        "html": """
        <b>{place}{postcode}{outage_reference}</b><br/>
        Risk: {final_risk_score}{risk_score}<br/>
        Resilience: {resilience_index}{resilience_score}<br/>
        Financial loss: £{total_financial_loss_gbp}{financial_loss_gbp}<br/>
        Priority: {investment_priority}
        """,
        "style": {"backgroundColor": "rgba(15, 23, 42, 0.95)", "color": "white"},
    }

    view_state = pdk.ViewState(
        latitude=center["lat"],
        longitude=center["lon"],
        zoom=center["zoom"] - 1,
        pitch=35,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=layers,
            tooltip=tooltip,
        ),
        use_container_width=True,
    )


def format_money_m(value: float) -> str:
    return f"£{safe_float(value) / 1_000_000:.2f}m"


def show_metric_cards(places: pd.DataFrame, pc: pd.DataFrame, scenario: str) -> None:
    avg_risk = round(float(places["final_risk_score"].mean()), 1)
    avg_res = round(float(places["resilience_index"].mean()), 1)
    avg_failure = round(float(places["failure_probability"].mean()) * 100, 1)
    total_ens = round(float(places["energy_not_supplied_mw"].sum()), 1)
    total_loss = round(float(places["total_financial_loss_gbp"].sum()), 2)
    p1 = 0 if pc.empty else int((pc["investment_priority"] == "Priority 1").sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Scenario", scenario)
    c2.metric("Regional risk", avg_risk)
    c3.metric("Resilience", f"{avg_res}/100")
    c4.metric("Failure probability", f"{avg_failure}%")
    c5.metric("ENS", f"{total_ens} MW")
    c6.metric("Financial loss", format_money_m(total_loss))
    st.caption(f"Priority 1 postcode areas: {p1}")


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem;}
        .stMetric {background: rgba(15,23,42,0.04); border-radius: 14px; padding: 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚡ SAT-Guard Grid Digital Twin")
    st.caption("North East & Yorkshire · weather, outage, deprivation, resilience, financial-loss and investment decision support")

    with st.sidebar:
        st.header("Controls")
        region = st.selectbox("Region", list(REGIONS.keys()), index=0)
        scenario = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
        mc_runs = st.slider("Monte Carlo runs", min_value=10, max_value=120, value=30, step=10)
        map_mode = st.selectbox("Map layer", ["All", "Risk", "Postcode / Investment", "Outages"], index=0)

        st.info(SCENARIOS[scenario]["description"])

        run_model = st.button("Run / refresh model", type="primary")

        st.divider()
        st.caption("IoD2025 Excel files can be placed in the same folder or a data/ folder. If not found, fallback vulnerability proxies are used.")

        if st.button("Clear Streamlit cache"):
            st.cache_data.clear()
            st.rerun()

    if run_model:
        st.cache_data.clear()

    with st.spinner("Running digital twin model..."):
        places, outages, grid = get_data_cached(region, scenario, mc_runs)
        pc = build_postcode_resilience(places, outages)
        rec = build_investment_recommendations(places, outages)

    if places.empty:
        st.error("No place-level model data could be generated.")
        return

    show_metric_cards(places, pc, scenario)

    imd_source = places.iloc[0].get("imd_dataset_summary", "Unknown") if not places.empty else "Unknown"

    tabs = st.tabs([
        "Overview",
        "Map",
        "Resilience",
        "Postcodes",
        "Investment",
        "Finance",
        "Monte Carlo",
        "Data / Export",
    ])

    with tabs[0]:
        left, right = st.columns([1.15, 0.85])

        with left:
            st.subheader("Regional intelligence")
            display_cols = [
                "place",
                "final_risk_score",
                "risk_label",
                "resilience_index",
                "resilience_label",
                "wind_speed_10m",
                "precipitation",
                "european_aqi",
                "social_vulnerability",
                "energy_not_supplied_mw",
                "total_financial_loss_gbp",
            ]
            st.dataframe(
                places[display_cols].sort_values("final_risk_score", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader("Risk distribution")
            st.bar_chart(places.set_index("place")["final_risk_score"])

            st.subheader("Resilience distribution")
            st.bar_chart(places.set_index("place")["resilience_index"])

        st.info(f"IoD / deprivation data source: {imd_source}")

    with tabs[1]:
        st.subheader("Spatial view")
        render_pydeck_map(region, places, outages, pc, map_mode)

        st.caption("Risk markers are city/town-level model outputs. Postcode markers are postcode-prefix level resilience/investment estimates.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Grid cells")
            st.dataframe(grid.sort_values("risk_score", ascending=False).head(50), use_container_width=True, hide_index=True)

        with col_b:
            st.subheader("Outages")
            st.dataframe(outages.head(100), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Resilience diagnostics")
        res_cols = [
            "place",
            "resilience_label",
            "resilience_index",
            "imd_score",
            "imd_match",
            "social_vulnerability",
            "grid_failure_probability",
            "renewable_failure_probability",
            "total_financial_loss_gbp",
        ]
        st.dataframe(
            places[res_cols].sort_values("resilience_index"),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(places.set_index("place")["grid_failure_probability"])
        with c2:
            st.bar_chart(places.set_index("place")["renewable_failure_probability"])

    with tabs[3]:
        st.subheader("Postcode-level resilience")
        if pc.empty:
            st.warning("No postcode resilience data could be generated.")
        else:
            top_cols = [
                "postcode",
                "nearest_place",
                "resilience_label",
                "resilience_score",
                "risk_score",
                "social_vulnerability",
                "outage_records",
                "affected_customers",
                "energy_not_supplied_mw",
                "financial_loss_gbp",
                "investment_priority",
                "recommendation_score",
            ]
            st.dataframe(pc[top_cols], use_container_width=True, hide_index=True)
            st.bar_chart(pc.set_index("postcode")["recommendation_score"])

    with tabs[4]:
        st.subheader("Prioritised investment recommendations")
        if rec.empty:
            st.warning("No investment recommendations could be generated.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Priority 1 areas", int((rec["investment_priority"] == "Priority 1").sum()))
            c2.metric("Indicative programme cost", format_money_m(rec["indicative_investment_cost_gbp"].sum()))
            c3.metric("Exposed financial loss", format_money_m(rec["financial_loss_gbp"].sum()))

            investment_cols = [
                "postcode",
                "nearest_place",
                "investment_priority",
                "recommendation_score",
                "investment_category",
                "recommended_action",
                "indicative_investment_cost_gbp",
                "financial_loss_gbp",
            ]
            st.dataframe(rec[investment_cols], use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Financial loss model")
        fin_cols = [
            "place",
            "energy_not_supplied_mw",
            "ens_mwh",
            "estimated_duration_hours",
            "voll_loss_gbp",
            "customer_interruption_loss_gbp",
            "business_disruption_loss_gbp",
            "restoration_loss_gbp",
            "critical_services_loss_gbp",
            "total_financial_loss_gbp",
        ]
        st.dataframe(
            places[fin_cols].sort_values("total_financial_loss_gbp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(places.set_index("place")["total_financial_loss_gbp"])

    with tabs[6]:
        st.subheader("Monte Carlo uncertainty")
        worst = places.sort_values("mc_p95", ascending=False).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Worst P95 risk", worst["mc_p95"], worst["place"])
        c2.metric("Extreme probability", f"{round(worst['mc_extreme_probability'] * 100, 1)}%")
        c3.metric("P95 financial loss", format_money_m(worst["mc_financial_loss_p95"]))
        c4.metric("Mean MC resilience", worst["mc_resilience_mean"])

        hist_values = worst.get("mc_histogram", [])
        if hist_values:
            hist_df = pd.DataFrame({"risk": hist_values})
            counts = pd.cut(hist_df["risk"], bins=np.linspace(0, 100, 25)).value_counts().sort_index()
            counts.index = counts.index.astype(str)
            st.bar_chart(counts)

        mc_cols = [
            "place",
            "mc_mean",
            "mc_std",
            "mc_p05",
            "mc_p50",
            "mc_p95",
            "mc_extreme_probability",
            "mc_financial_loss_p95",
        ]
        st.dataframe(
            places[mc_cols].sort_values("mc_p95", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[7]:
        st.subheader("Raw model outputs")

        st.markdown("#### Places")
        st.dataframe(places, use_container_width=True, hide_index=True)

        st.markdown("#### Outages")
        st.dataframe(outages, use_container_width=True, hide_index=True)

        st.markdown("#### Grid")
        st.dataframe(grid, use_container_width=True, hide_index=True)

        st.download_button(
            "Download place-level CSV",
            data=places.to_csv(index=False).encode("utf-8"),
            file_name="sat_guard_places.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download postcode recommendations CSV",
            data=rec.to_csv(index=False).encode("utf-8") if not rec.empty else b"",
            file_name="sat_guard_postcode_recommendations.csv",
            mime="text/csv",
            disabled=rec.empty,
        )


if __name__ == "__main__":
    main()
