"""
North East & Yorkshire Grid Digital Twin — IMD + Financial Loss + BBC-style Hazard Animation
==========================================================================================

This is a single-file Flask application.

Main upgrades:
- Uses IoD2025 / deprivation Excel files for social vulnerability when available.
- Keeps robust fallback vulnerability values if the Excel schema cannot be matched.
- Adds financial loss model:
    * Energy Not Supplied (MW and MWh)
    * Value of Lost Load (VoLL)
    * Customer interruption loss
    * Business disruption loss
    * Critical-service/social-vulnerability uplift
    * Restoration and repair cost
- Adds BBC-weather-inspired hazard simulation:
    * Working Leaflet map
    * Canvas overlay with moving wind streaks
    * Rain particles for flood/storm scenarios
    * Pulsing hazard glow cells
    * Play / pause / scrub controls
- Keeps Northern Powergrid outage layer, postcode popups, NASA GIBS overlays,
  resilience model, renewable/grid failure model, and Monte Carlo uncertainty.

Run:
    pip install flask pandas numpy requests openpyxl
    python app.py

Open:
    http://localhost:5000

For IMD / deprivation:
    Put the IoD2025 Excel files in the same folder as app.py, for example:
    - File_1_IoD2025 Index of Multiple Deprivation.xlsx
    - File_2_IoD2025 Domains of Deprivation.xlsx
    - File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.xlsx
    - IoD2025 Local Authority District Summaries (lower-tier) - Rank of average rank.xlsx
"""

import json
import math
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from flask import Flask, Response, jsonify, render_template_string, request


app = Flask(__name__)

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

DATA_CACHE = {
    "key": None,
    "timestamp": None,
    "places": None,
    "outages": None,
    "grid": None,
}

IMD_CACHE = {
    "loaded": False,
    "summary": None,
    "source": None,
}

CACHE_SECONDS = 240


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
        date_string = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    return NASA_LAYERS[layer_name].replace("{date}", date_string)


def requests_json(url: str, params: Dict[str, Any] = None, timeout: int = 12) -> Dict[str, Any]:
    try:
        headers = {"User-Agent": "flask-grid-digital-twin-imd-bbc/4.0"}
        response = requests.get(url, params=params or {}, headers=headers, timeout=timeout, verify=False)
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
    """
    Rank is usually 1 = most deprived. Convert to 0-100 vulnerability.
    """
    rank_value = safe_float(rank_value, None)
    max_rank = safe_float(max_rank, None)

    if rank_value is None or max_rank is None or max_rank <= 1:
        return None

    return round(clamp((1 - (rank_value - 1) / (max_rank - 1)) * 100, 0, 100), 2)


def extract_imd_summary_from_sheet(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Remove empty-looking columns
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame()

    cols = list(df.columns)

    area_col = choose_first_matching_column(
        cols,
        ["local", "authority"],
    )
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
        # If score has large range, normalise roughly.
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
        # decile 1 = most deprived; convert to vulnerability
        out["imd_score_0_100"] = (10 - decile) / 9 * 100
        out["imd_metric_source"] = f"decile converted: {decile_col}"

    else:
        return pd.DataFrame()

    out["imd_score_0_100"] = pd.to_numeric(out["imd_score_0_100"], errors="coerce")
    out = out.dropna(subset=["imd_score_0_100"])
    out["imd_score_0_100"] = out["imd_score_0_100"].clip(0, 100)
    out["area_key"] = out["area_name"].str.lower()

    return out[["area_name", "area_key", "imd_score_0_100", "imd_metric_source", "source_file", "source_area_col"]]


def load_imd_summary() -> Tuple[pd.DataFrame, str]:
    """
    Loads all uploaded IoD2025 spreadsheets and extracts a flexible summary.

    The uploaded files may contain different sheets and schemas. This loader:
    - scans each sheet
    - tries to infer local authority / area name
    - tries to infer IMD score, rank, decile, or average-rank style measure
    - converts ranks/deciles into 0-100 vulnerability score
    """
    if IMD_CACHE["loaded"]:
        return IMD_CACHE["summary"].copy(), IMD_CACHE["source"]

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
        # Average duplicate area matches from multiple sheets.
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

    IMD_CACHE["loaded"] = True
    IMD_CACHE["summary"] = grouped.copy()
    IMD_CACHE["source"] = source

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

    # Regional soft match
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

def fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": WEATHER_CURRENT_VARS,
        "timezone": "Europe/London",
    }
    return requests_json(OPEN_METEO_WEATHER_URL, params=params)


def fetch_air_quality(lat: float, lon: float) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": AIR_CURRENT_VARS,
        "timezone": "Europe/London",
    }
    return requests_json(OPEN_METEO_AIR_URL, params=params)


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

    if lat_cols:
        lat = pd.to_numeric(df[lat_cols[0]], errors="coerce")
    else:
        lat = pd.Series(np.nan, index=df.index)

    if lon_cols:
        lon = pd.to_numeric(df[lon_cols[0]], errors="coerce")
    else:
        lon = pd.Series(np.nan, index=df.index)

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
            if random.random() < 0.55:
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

    # Transparent prototype assumptions
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
    simulations = int(clamp(simulations, 20, 500))
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
    imd_summary, imd_source = load_imd_summary()

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
            "time": weather.get("time") or datetime.utcnow().isoformat(),
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


def get_data(region: str, scenario: str, mc_runs: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = f"{region}|{scenario}|{mc_runs}"
    now = datetime.utcnow()

    if (
        DATA_CACHE["timestamp"] is not None
        and DATA_CACHE["key"] == key
        and (now - DATA_CACHE["timestamp"]).total_seconds() < CACHE_SECONDS
    ):
        return (
            DATA_CACHE["places"].copy(),
            DATA_CACHE["outages"].copy(),
            DATA_CACHE["grid"].copy(),
        )

    places, outages = build_places(region, scenario, mc_runs)
    grid = build_grid(region, places, outages)

    DATA_CACHE["key"] = key
    DATA_CACHE["timestamp"] = now
    DATA_CACHE["places"] = places.copy()
    DATA_CACHE["outages"] = outages.copy()
    DATA_CACHE["grid"] = grid.copy()

    return places, outages, grid


# =============================================================================
# CONTROLS
# =============================================================================

def get_controls() -> Tuple[str, str, str, str, int]:
    region = request.args.get("region", "North East")
    if region not in REGIONS:
        region = "North East"

    scenario = request.args.get("scenario", "Live / Real-time")
    if scenario not in SCENARIOS:
        scenario = "Live / Real-time"

    layer = request.args.get("layer", "VIIRS NOAA-20 True Colour")
    if layer not in NASA_LAYERS:
        layer = "VIIRS NOAA-20 True Colour"

    page = request.args.get("page", "overview")

    mc = safe_int(request.args.get("mc"), 100)
    mc = int(clamp(mc, 20, 500))

    return region, scenario, layer, page, mc


def make_base_url(region: str, scenario: str, layer: str, mc: int) -> str:
    return f"/?region={region}&scenario={scenario}&layer={layer}&mc={mc}"


# =============================================================================
# TEMPLATE
# =============================================================================

BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>North East & Yorkshire Grid Digital Twin</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <style>
        :root {
            --bg: #020617;
            --panel: rgba(15, 23, 42, 0.88);
            --panel2: rgba(30, 41, 59, 0.82);
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

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: "Segoe UI", Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(56,189,248,0.25), transparent 32%),
                radial-gradient(circle at bottom right, rgba(168,85,247,0.20), transparent 34%),
                #020617;
        }

        .app { display: flex; min-height: 100vh; }

        .sidebar {
            width: 318px;
            min-height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            padding: 24px 18px;
            background: rgba(2,6,23,0.94);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            backdrop-filter: blur(16px);
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 24px;
        }

        .logo {
            width: 56px;
            height: 56px;
            border-radius: 18px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            box-shadow: 0 0 34px rgba(56,189,248,0.42);
            font-size: 27px;
        }

        .brand-title {
            color: white;
            font-size: 22px;
            font-weight: 950;
            letter-spacing: -0.03em;
        }

        .brand-sub {
            color: var(--muted);
            font-size: 12px;
            margin-top: 3px;
        }

        .nav a {
            display: block;
            color: #cbd5e1;
            text-decoration: none;
            font-weight: 820;
            padding: 13px 14px;
            border-radius: 16px;
            margin-bottom: 8px;
            transition: 0.18s ease;
        }

        .nav a:hover {
            color: var(--blue);
            background: rgba(56,189,248,0.14);
            transform: translateX(3px);
        }

        .control {
            margin-top: 18px;
            padding: 16px;
            border-radius: 22px;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(168,85,247,0.10));
        }

        .control label {
            display: block;
            font-size: 13px;
            font-weight: 850;
            margin-top: 12px;
            margin-bottom: 6px;
            color: #bfdbfe;
        }

        select, input {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(2,6,23,0.83);
            color: white;
            padding: 12px;
        }

        button, .btn {
            display: inline-block;
            border: 0;
            border-radius: 15px;
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            color: white;
            font-weight: 950;
            padding: 12px 18px;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
        }

        .control button {
            width: 100%;
            margin-top: 14px;
        }

        .main {
            margin-left: 318px;
            width: calc(100% - 318px);
            padding: 28px;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            margin-bottom: 24px;
        }

        .title {
            font-size: 35px;
            font-weight: 950;
            color: white;
            letter-spacing: -0.04em;
        }

        .subtitle {
            color: var(--muted);
            margin-top: 7px;
            max-width: 980px;
        }

        .chip {
            display: inline-block;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(15,23,42,0.88);
            border: 1px solid var(--border);
            color: #bfdbfe;
            font-size: 13px;
            font-weight: 850;
            margin-left: 8px;
            margin-bottom: 8px;
        }

        .live-chip {
            background: rgba(34,197,94,0.15);
            color: #bbf7d0;
            border-color: rgba(34,197,94,0.35);
        }

        .scenario-chip {
            background: rgba(249,115,22,0.15);
            color: #fed7aa;
            border-color: rgba(249,115,22,0.35);
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 20px;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 18px;
            margin-bottom: 20px;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-bottom: 20px;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 25px;
            padding: 22px;
            box-shadow: 0 24px 60px rgba(0,0,0,0.30);
            backdrop-filter: blur(14px);
            overflow: hidden;
        }

        .bbc-card {
            background:
                linear-gradient(135deg, rgba(14,165,233,0.25), rgba(2,6,23,0.92)),
                radial-gradient(circle at 82% 16%, rgba(250,204,21,0.25), transparent 33%);
        }

        h2, h3 {
            margin-top: 0;
            color: white;
        }

        .metric-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 34px;
            font-weight: 950;
            color: white;
            line-height: 1.08;
        }

        .metric-note {
            color: var(--blue);
            font-size: 13px;
            margin-top: 8px;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 7px 11px;
            font-size: 12px;
            font-weight: 900;
        }

        .Low, .Robust, .Stable {
            background: rgba(34,197,94,0.16);
            color: #86efac;
        }

        .Moderate, .Functional {
            background: rgba(56,189,248,0.16);
            color: #7dd3fc;
        }

        .High, .Stressed {
            background: rgba(249,115,22,0.16);
            color: #fdba74;
        }

        .Priority1 { background: rgba(239,68,68,0.18); color: #fca5a5; }

        .Priority2 { background: rgba(249,115,22,0.16); color: #fdba74; }

        .Priority3 { background: rgba(234,179,8,0.16); color: #fde68a; }

        .Monitor { background: rgba(34,197,94,0.16); color: #86efac; }

        .Severe, .Fragile {
            background: rgba(239,68,68,0.18);
            color: #fca5a5;
        }

        .bar {
            width: 100%;
            height: 10px;
            background: rgba(148,163,184,0.18);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 8px;
        }

        .fill {
            height: 100%;
            background: linear-gradient(90deg, #22c55e, #38bdf8);
        }

        .fill-risk {
            height: 100%;
            background: linear-gradient(90deg, #eab308, #ef4444);
        }

        .fill-purple {
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #a855f7);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }

        th, td {
            padding: 13px;
            border-bottom: 1px solid rgba(148,163,184,0.16);
            text-align: left;
            font-size: 14px;
        }

        th {
            color: #93c5fd;
            font-size: 12px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        tr:hover td {
            background: rgba(56,189,248,0.04);
        }

        .map {
            height: 680px;
            width: 100%;
            border-radius: 24px;
            border: 1px solid var(--border);
            background: #0f172a;
            overflow: hidden;
            position: relative;
        }

        .small-map {
            height: 360px;
            width: 100%;
            border-radius: 20px;
            border: 1px solid var(--border);
            background: #0f172a;
            overflow: hidden;
        }

        .hazard-canvas {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 450;
            pointer-events: none;
        }

        .map-hud {
            position: absolute;
            top: 14px;
            right: 14px;
            z-index: 800;
            background: rgba(2,6,23,0.76);
            border: 1px solid rgba(148,163,184,0.28);
            border-radius: 16px;
            padding: 12px 14px;
            color: white;
            font-size: 13px;
            backdrop-filter: blur(8px);
        }

        .chart {
            height: 245px;
            display: flex;
            align-items: end;
            gap: 11px;
            padding-top: 18px;
        }

        .chart-col {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: end;
            gap: 8px;
            color: var(--muted);
            font-size: 12px;
        }

        .chart-bar {
            width: 100%;
            max-width: 44px;
            border-radius: 14px 14px 4px 4px;
            background: linear-gradient(180deg, #38bdf8, #1d4ed8);
            box-shadow: 0 0 24px rgba(56,189,248,0.25);
        }

        .histogram {
            height: 230px;
            display: flex;
            align-items: end;
            gap: 4px;
            padding-top: 18px;
        }

        .hist-bar {
            flex: 1;
            min-width: 4px;
            border-radius: 6px 6px 0 0;
            background: linear-gradient(180deg, #a855f7, #38bdf8);
        }

        .alert {
            padding: 16px;
            border-radius: 18px;
            border: 1px solid rgba(239,68,68,0.28);
            background: rgba(239,68,68,0.14);
            color: #fecaca;
            margin-bottom: 12px;
        }

        .success {
            border-color: rgba(34,197,94,0.28);
            background: rgba(34,197,94,0.14);
            color: #bbf7d0;
        }

        .map-toolbar {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin: 14px 0;
        }

        .range {
            flex: 1;
            min-width: 220px;
        }


        .metra-scene {
            position: relative;
            height: 760px;
            border-radius: 28px;
            overflow: hidden;
            background: radial-gradient(circle at 35% 20%, rgba(56,189,248,0.12), transparent 32%), #07111f;
            border: 1px solid var(--border);
            box-shadow: 0 30px 80px rgba(0,0,0,0.38);
            perspective: 1200px;
        }

        .metra-stage {
            position: absolute;
            inset: 0;
            transform-origin: 50% 65%;
            transform: rotateX(54deg) rotateZ(-4deg) scale(1.12);
            filter: saturate(1.08) contrast(1.04);
        }

        .metra-stage.flat {
            transform: rotateX(0deg) rotateZ(0deg) scale(1.0);
        }

        .metra-map {
            position: absolute;
            inset: 0;
            z-index: 1;
        }

        .metra-canvas {
            position: absolute;
            inset: 0;
            z-index: 480;
            pointer-events: none;
        }

        .metra-depth-canvas {
            position: absolute;
            inset: 0;
            z-index: 520;
            pointer-events: none;
        }

        .metra-ui {
            position: absolute;
            left: 18px;
            right: 18px;
            bottom: 18px;
            z-index: 900;
            display: grid;
            grid-template-columns: auto auto 1fr auto;
            gap: 12px;
            align-items: center;
            padding: 14px;
            border-radius: 20px;
            border: 1px solid rgba(148,163,184,0.28);
            background: rgba(2,6,23,0.74);
            backdrop-filter: blur(16px);
        }

        .metra-top-hud {
            position: absolute;
            top: 18px;
            left: 18px;
            z-index: 900;
            padding: 14px 16px;
            border-radius: 20px;
            border: 1px solid rgba(148,163,184,0.28);
            background: rgba(2,6,23,0.70);
            backdrop-filter: blur(16px);
            max-width: 380px;
        }

        .metra-title {
            font-size: 15px;
            font-weight: 950;
            color: white;
            margin-bottom: 4px;
        }

        .metra-sub {
            color: #cbd5e1;
            font-size: 12px;
            line-height: 1.45;
        }

        .metra-hour {
            font-size: 24px;
            font-weight: 950;
            color: white;
            min-width: 92px;
        }

        .metra-toggle {
            display: inline-flex;
            gap: 8px;
            align-items: center;
        }

        .metra-toggle input {
            width: auto;
        }

        .metra-legend {
            position: absolute;
            right: 18px;
            top: 18px;
            z-index: 900;
            padding: 12px 14px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.28);
            background: rgba(2,6,23,0.62);
            backdrop-filter: blur(16px);
            font-size: 12px;
            color: #cbd5e1;
        }

        .legend-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 6px 0;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 999px;
            display: inline-block;
        }


        .bbc-forecast-scene {
            position: relative;
            height: 760px;
            border-radius: 28px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: #07111f;
            box-shadow: 0 30px 80px rgba(0,0,0,0.34);
        }

        .bbc-forecast-map {
            position: absolute;
            inset: 0;
            z-index: 1;
        }

        .bbc-weather-canvas {
            position: absolute;
            inset: 0;
            z-index: 450;
            pointer-events: none;
        }

        .bbc-forecast-top {
            position: absolute;
            top: 18px;
            left: 18px;
            z-index: 900;
            padding: 14px 16px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.28);
            background: rgba(2,6,23,0.72);
            backdrop-filter: blur(14px);
            max-width: 430px;
        }

        .bbc-forecast-title {
            font-size: 16px;
            font-weight: 950;
            color: white;
            margin-bottom: 4px;
        }

        .bbc-forecast-sub {
            color: #cbd5e1;
            font-size: 12px;
            line-height: 1.45;
        }

        .bbc-forecast-controls {
            position: absolute;
            left: 18px;
            right: 18px;
            bottom: 18px;
            z-index: 900;
            display: grid;
            grid-template-columns: auto auto 1fr auto auto;
            gap: 12px;
            align-items: center;
            padding: 14px;
            border-radius: 20px;
            border: 1px solid rgba(148,163,184,0.30);
            background: rgba(2,6,23,0.76);
            backdrop-filter: blur(16px);
        }

        .bbc-time-pill {
            min-width: 94px;
            text-align: center;
            font-size: 24px;
            font-weight: 950;
            color: white;
        }

        .bbc-legend {
            position: absolute;
            right: 18px;
            top: 18px;
            z-index: 900;
            width: 230px;
            padding: 13px 14px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.28);
            background: rgba(2,6,23,0.68);
            backdrop-filter: blur(14px);
            color: #cbd5e1;
            font-size: 12px;
        }

        .bbc-gradient {
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(56,189,248,.35), rgba(34,197,94,.45), rgba(234,179,8,.65), rgba(249,115,22,.75), rgba(239,68,68,.85));
            margin: 8px 0;
        }

        .bbc-layer-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #cbd5e1;
            font-size: 12px;
        }

        .bbc-layer-toggle input {
            width: auto;
        }


        .bbc-ref-scene {
            position: relative;
            height: 780px;
            border-radius: 28px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.28);
            background:
                radial-gradient(circle at 18% 22%, rgba(56,189,248,0.15), transparent 30%),
                linear-gradient(180deg, #0a1726, #07101d);
            box-shadow: 0 34px 90px rgba(0,0,0,0.42);
        }

        .bbc-ref-map {
            position: absolute;
            inset: 0;
            z-index: 1;
            filter: saturate(1.18) contrast(1.08) brightness(0.86);
        }

        .bbc-ref-canvas {
            position: absolute;
            inset: 0;
            z-index: 460;
            pointer-events: none;
        }

        .bbc-ref-iso-canvas {
            position: absolute;
            inset: 0;
            z-index: 430;
            pointer-events: none;
        }

        .bbc-brand {
            position: absolute;
            left: 22px;
            bottom: 106px;
            z-index: 920;
            display: flex;
            align-items: center;
            gap: 10px;
            color: white;
            text-shadow: 0 2px 8px rgba(0,0,0,0.78);
        }

        .bbc-blocks {
            display: flex;
            gap: 5px;
        }

        .bbc-blocks span {
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            background: rgba(255,255,255,0.96);
            color: #1e293b;
            font-weight: 950;
            font-size: 21px;
            line-height: 1;
        }

        .bbc-weather-word {
            font-size: 32px;
            font-weight: 950;
            letter-spacing: -0.04em;
        }

        .bbc-ref-top {
            position: absolute;
            top: 18px;
            left: 18px;
            z-index: 930;
            padding: 13px 16px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(3,10,22,0.58);
            backdrop-filter: blur(12px);
            max-width: 470px;
        }

        .bbc-ref-title {
            color: white;
            font-weight: 950;
            font-size: 17px;
            margin-bottom: 4px;
        }

        .bbc-ref-sub {
            color: #dbeafe;
            font-size: 12px;
            line-height: 1.45;
        }

        .bbc-time-box {
            position: absolute;
            right: 24px;
            bottom: 110px;
            z-index: 930;
            display: flex;
            box-shadow: 0 6px 16px rgba(0,0,0,0.35);
            font-weight: 950;
        }

        .bbc-time-day {
            background: rgba(15,118,110,0.93);
            color: white;
            padding: 10px 16px;
            font-size: 16px;
            letter-spacing: 0.03em;
        }

        .bbc-time-hour {
            background: rgba(2,6,23,0.88);
            color: white;
            padding: 10px 16px;
            font-size: 16px;
            min-width: 78px;
            text-align: center;
        }

        .bbc-ref-controls {
            position: absolute;
            left: 18px;
            right: 18px;
            bottom: 18px;
            z-index: 940;
            display: grid;
            grid-template-columns: auto auto 1fr auto auto auto auto auto;
            gap: 10px;
            align-items: center;
            padding: 13px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(2,6,23,0.72);
            backdrop-filter: blur(14px);
        }

        .bbc-ref-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #dbeafe;
            font-size: 12px;
            white-space: nowrap;
        }

        .bbc-ref-toggle input {
            width: auto;
        }

        .bbc-ref-legend {
            position: absolute;
            right: 18px;
            top: 18px;
            z-index: 930;
            width: 240px;
            padding: 12px 14px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(3,10,22,0.54);
            backdrop-filter: blur(12px);
            color: #dbeafe;
            font-size: 12px;
        }

        .bbc-ref-gradient {
            height: 13px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(90,180,255,.35), rgba(37,99,235,.48), rgba(34,197,94,.62), rgba(234,179,8,.78), rgba(239,68,68,.88), rgba(168,85,247,.9));
            margin: 8px 0;
        }

        .city-label-bbc {
            color: white;
            font-size: 16px;
            font-weight: 700;
            text-shadow: 0 2px 5px #000, 0 0 3px #000;
            white-space: nowrap;
        }

        .city-label-bbc::after {
            content: "";
            display: inline-block;
            width: 7px;
            height: 7px;
            margin-left: 6px;
            background: white;
            box-shadow: 0 1px 4px rgba(0,0,0,.65);
        }


        .q1-scene {
            position: relative;
            height: 820px;
            border-radius: 30px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.28);
            background:
                radial-gradient(circle at 30% 18%, rgba(59,130,246,0.22), transparent 28%),
                linear-gradient(180deg, #07111f, #020617);
            box-shadow: 0 38px 96px rgba(0,0,0,0.45);
        }

        .q1-map {
            position: absolute;
            inset: 0;
            z-index: 1;
            filter: saturate(1.28) contrast(1.15) brightness(0.82);
        }

        .q1-pressure-canvas {
            position: absolute;
            inset: 0;
            z-index: 420;
            pointer-events: none;
        }

        .q1-weather-canvas {
            position: absolute;
            inset: 0;
            z-index: 465;
            pointer-events: none;
        }

        .q1-front-canvas {
            position: absolute;
            inset: 0;
            z-index: 510;
            pointer-events: none;
        }

        .q1-brand {
            position: absolute;
            left: 24px;
            bottom: 112px;
            z-index: 950;
            display: flex;
            align-items: center;
            gap: 10px;
            color: white;
            text-shadow: 0 3px 10px rgba(0,0,0,0.85);
        }

        .q1-brand-blocks {
            display: flex;
            gap: 5px;
        }

        .q1-brand-blocks span {
            display: grid;
            place-items: center;
            width: 35px;
            height: 35px;
            background: rgba(255,255,255,0.96);
            color: #1e293b;
            font-weight: 950;
            font-size: 22px;
        }

        .q1-brand-word {
            font-size: 34px;
            font-weight: 950;
            letter-spacing: -0.04em;
        }

        .q1-top-card {
            position: absolute;
            top: 18px;
            left: 18px;
            z-index: 960;
            padding: 14px 17px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(3,10,22,0.62);
            backdrop-filter: blur(14px);
            max-width: 520px;
        }

        .q1-top-title {
            color: white;
            font-weight: 950;
            font-size: 18px;
            margin-bottom: 4px;
        }

        .q1-top-sub {
            color: #dbeafe;
            font-size: 12px;
            line-height: 1.45;
        }

        .q1-time-strip {
            position: absolute;
            right: 24px;
            bottom: 114px;
            z-index: 960;
            display: flex;
            box-shadow: 0 8px 22px rgba(0,0,0,0.45);
            font-weight: 950;
        }

        .q1-day {
            background: rgba(13,148,136,0.94);
            color: white;
            padding: 11px 18px;
            font-size: 16px;
            letter-spacing: 0.04em;
        }

        .q1-hour {
            background: rgba(2,6,23,0.92);
            color: white;
            padding: 11px 18px;
            min-width: 80px;
            text-align: center;
            font-size: 16px;
        }

        .q1-controls {
            position: absolute;
            left: 18px;
            right: 18px;
            bottom: 18px;
            z-index: 970;
            display: grid;
            grid-template-columns: auto auto 1fr auto auto auto auto auto auto;
            gap: 10px;
            align-items: center;
            padding: 13px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(2,6,23,0.74);
            backdrop-filter: blur(16px);
        }

        .q1-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #dbeafe;
            font-size: 12px;
            white-space: nowrap;
        }

        .q1-toggle input {
            width: auto;
        }

        .q1-legend {
            position: absolute;
            right: 18px;
            top: 18px;
            z-index: 960;
            width: 255px;
            padding: 13px 15px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(3,10,22,0.58);
            backdrop-filter: blur(14px);
            color: #dbeafe;
            font-size: 12px;
        }

        .q1-gradient {
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(59,130,246,.38), rgba(37,99,235,.56), rgba(34,197,94,.66), rgba(234,179,8,.80), rgba(249,115,22,.86), rgba(239,68,68,.92), rgba(168,85,247,.95));
            margin: 8px 0;
        }

        .city-label-bbc {
            color: white;
            font-size: 16px;
            font-weight: 800;
            text-shadow: 0 2px 5px #000, 0 0 3px #000;
            white-space: nowrap;
        }

        .city-label-bbc::after {
            content: "";
            display: inline-block;
            width: 7px;
            height: 7px;
            margin-left: 6px;
            background: white;
            box-shadow: 0 1px 4px rgba(0,0,0,.70);
        }

        .wx-watermark {
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            bottom: 103px;
            z-index: 950;
            color: rgba(255,255,255,0.72);
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-shadow: 0 2px 6px rgba(0,0,0,0.8);
        }

        .Priority1 { background: rgba(239,68,68,0.18); color: #fca5a5; }
        .Priority2 { background: rgba(249,115,22,0.16); color: #fdba74; }
        .Priority3 { background: rgba(234,179,8,0.16); color: #fde68a; }
        .Monitor { background: rgba(34,197,94,0.16); color: #86efac; }

        .footer {
            color: #64748b;
            font-size: 12px;
            margin-top: 26px;
        }

        @media(max-width: 1100px) {
            .sidebar {
                position: relative;
                width: 100%;
                min-height: auto;
            }

            .app {
                display: block;
            }

            .main {
                margin-left: 0;
                width: 100%;
            }

            .grid, .grid-2, .grid-3 {
                grid-template-columns: 1fr;
            }

            .topbar {
                display: block;
            }
        }
    </style>
</head>
<body>
<div class="app">
    <aside class="sidebar">
        <div class="brand-row">
            <div class="logo">⚡</div>
            <div>
                <div class="brand-title">Grid Digital Twin</div>
                <div class="brand-sub">NASA · NPG · IoD2025 · BBC-style Hazard</div>
            </div>
        </div>

        <div class="nav">
            <a href="{{ base_url }}">📊 Overview</a>
            <a href="{{ base_url }}&page=simple">🌬️ Simple Map</a>
            <a href="{{ base_url }}&page=bbc">🌧️ BBC Weather Simulation</a>
            <a href="{{ base_url }}&page=resilience">🛡️ Resilience</a>
            <a href="{{ base_url }}&page=postcode">📮 Postcodes</a>
            <a href="{{ base_url }}&page=postcode_resilience">📍 Postcode Resilience</a>
            <a href="{{ base_url }}&page=investment">🏗️ Investment Priorities</a>
            <a href="{{ base_url }}&page=finance">💷 Financial Loss</a>
            <a href="{{ base_url }}&page=montecarlo">🔬 Monte Carlo</a>
            <a href="{{ base_url }}&page=satellite">🛰️ NASA Storyboard</a>
            <a href="{{ base_url }}&page=data">📋 Data</a>
        </div>

        <form class="control" method="get">
            <strong>Controls</strong>

            <label>Region</label>
            <select name="region">
                {% for r in regions %}
                <option value="{{ r }}" {% if r == region %}selected{% endif %}>{{ r }}</option>
                {% endfor %}
            </select>

            <label>Mode / What-if Scenario</label>
            <select name="scenario">
                {% for s in scenarios %}
                <option value="{{ s }}" {% if s == scenario %}selected{% endif %}>{{ s }}</option>
                {% endfor %}
            </select>

            <label>NASA Layer</label>
            <select name="layer">
                {% for l in layers %}
                <option value="{{ l }}" {% if l == layer %}selected{% endif %}>{{ l }}</option>
                {% endfor %}
            </select>

            <label>Monte Carlo runs</label>
            <input type="number" name="mc" min="20" max="500" value="{{ mc }}">

            <input type="hidden" name="page" value="{{ page }}">
            <button>Update</button>
        </form>

        <div class="control">
            <strong>IoD2025 social vulnerability</strong>
            <p style="color:#94a3b8;">
                The app scans uploaded IoD2025 Excel files and uses matched authority scores when possible.
            </p>
        </div>
    </aside>

    <main class="main">
        <div class="topbar">
            <div>
                <div class="title">{{ title }}</div>
                <div class="subtitle">{{ subtitle }}</div>
            </div>
            <div>
                <span class="chip {% if scenario == 'Live / Real-time' %}live-chip{% else %}scenario-chip{% endif %}">
                    {{ scenario }}
                </span>
                <span class="chip">{{ region }}</span>
                <span class="chip">UTC {{ now }}</span>
            </div>
        </div>

        {{ content | safe }}

        <div class="footer">
            Prototype digital twin · Leaflet maps · High-visibility moving hazard canvas · Northern Powergrid feed · NASA GIBS tiles · Open-Meteo weather and air quality · IoD2025-compatible social vulnerability · Financial loss model
        </div>
    </main>
</div>

{{ scripts | safe }}
</body>
</html>
"""


# =============================================================================
# RENDER HELPERS
# =============================================================================

def get_controls() -> Tuple[str, str, str, str, int]:
    region = request.args.get("region", "North East")
    if region not in REGIONS:
        region = "North East"

    scenario = request.args.get("scenario", "Live / Real-time")
    if scenario not in SCENARIOS:
        scenario = "Live / Real-time"

    layer = request.args.get("layer", "VIIRS NOAA-20 True Colour")
    if layer not in NASA_LAYERS:
        layer = "VIIRS NOAA-20 True Colour"

    page = request.args.get("page", "overview")

    mc = safe_int(request.args.get("mc"), 100)
    mc = int(clamp(mc, 20, 500))

    return region, scenario, layer, page, mc


def make_base_url(region: str, scenario: str, layer: str, mc: int) -> str:
    return f"/?region={region}&scenario={scenario}&layer={layer}&mc={mc}"


def render_app(title: str, subtitle: str, content: str, scripts: str = "") -> str:
    region, scenario, layer, page, mc = get_controls()
    return render_template_string(
        BASE_HTML,
        title=title,
        subtitle=subtitle,
        content=content,
        scripts=scripts,
        region=region,
        scenario=scenario,
        layer=layer,
        page=page,
        mc=mc,
        regions=list(REGIONS.keys()),
        scenarios=list(SCENARIOS.keys()),
        layers=list(NASA_LAYERS.keys()),
        base_url=make_base_url(region, scenario, layer, mc),
        now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def places_table(places: pd.DataFrame) -> str:
    rows = ""
    for _, r in places.sort_values("final_risk_score", ascending=False).iterrows():
        rows += f"""
        <tr>
            <td><strong>{r['place']}</strong></td>
            <td>{round(r['wind_speed_10m'],1)} km/h</td>
            <td>{round(r['precipitation'],2)} mm</td>
            <td>{round(r['european_aqi'],1)}</td>
            <td>{round(r['imd_score'],1)}</td>
            <td>{round(r['social_vulnerability'],1)}</td>
            <td>{int(r['nearby_outages_25km'])}</td>
            <td>{round(r['energy_not_supplied_mw'],1)} MW</td>
            <td>£{round(r['total_financial_loss_gbp']/1_000_000,2)}m</td>
            <td><span class="badge {r['risk_label']}">{r['risk_label']}</span></td>
            <td>
                {round(r['final_risk_score'],1)}
                <div class="bar"><div class="fill-risk" style="width:{r['final_risk_score']}%"></div></div>
            </td>
            <td>
                <span class="badge {r['resilience_label']}">{r['resilience_label']}</span>
                <div class="bar"><div class="fill" style="width:{r['resilience_index']}%"></div></div>
            </td>
        </tr>
        """

    return f"""
    <table>
        <tr>
            <th>Place</th>
            <th>Wind</th>
            <th>Rain</th>
            <th>AQI</th>
            <th>IoD/IMD</th>
            <th>Social vuln.</th>
            <th>Outages</th>
            <th>ENS</th>
            <th>Financial loss</th>
            <th>Risk</th>
            <th>Risk score</th>
            <th>Resilience</th>
        </tr>
        {rows}
    </table>
    """


def chart_bars(places: pd.DataFrame, column: str) -> str:
    bars = ""
    for _, r in places.iterrows():
        value = clamp(safe_float(r.get(column)), 0, 100)
        bars += f"""
        <div class="chart-col">
            <div class="chart-bar" style="height:{max(8, value * 2.1)}px"></div>
            <span>{r['place']}</span>
        </div>
        """
    return f'<div class="chart">{bars}</div>'


def histogram(values: List[float]) -> str:
    if not values:
        return "<p>No Monte Carlo data available.</p>"

    counts, _ = np.histogram(values, bins=24, range=(0, 100))
    max_count = max(1, int(max(counts)))
    bars = ""
    for count in counts:
        height = max(4, (count / max_count) * 220)
        bars += f'<div class="hist-bar" style="height:{height}px"></div>'
    return f'<div class="histogram">{bars}</div>'


# =============================================================================
# API ROUTES
# =============================================================================

@app.route("/api/data")
def api_data():
    region, scenario, layer, page, mc = get_controls()
    places, outages, grid = get_data(region, scenario, mc)

    return jsonify({
        "region": region,
        "scenario": scenario,
        "hazard_mode": SCENARIOS[scenario]["hazard_mode"],
        "layer": layer,
        "center": REGIONS[region]["center"],
        "nasa_tile": get_nasa_tile_url(layer),
        "imd_source": IMD_CACHE.get("source") or "",
        "places": places.to_dict("records"),
        "outages": outages.to_dict("records"),
        "grid": grid.to_dict("records"),
    })



@app.route("/api/postcode_resilience")
def api_postcode_resilience():
    region, scenario, layer, page, mc = get_controls()
    places, outages, grid = get_data(region, scenario, mc)
    pc = build_postcode_resilience(places, outages)

    return jsonify({
        "region": region,
        "scenario": scenario,
        "center": REGIONS[region]["center"],
        "postcodes": pc.to_dict("records"),
    })


@app.route("/api/bbc_frames")
def api_bbc_frames():
    region, scenario, layer, page, mc = get_controls()
    places, outages, grid = get_data(region, scenario, mc)

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
                "lat": g["lat"],
                "lon": g["lon"],
                "wind_speed": round(max(0, g["wind_speed"] * wind_factor), 2),
                "rain": round(max(0, g["rain"] * rain_factor), 2),
                "risk_score": round(clamp(g["risk_score"] * risk_factor, 0, 100), 2),
                "resilience_index": round(clamp(g["resilience_index"] - (phase * 7), 0, 100), 2),
                "energy_not_supplied_mw": g["energy_not_supplied_mw"],
                "financial_loss_gbp": g["financial_loss_gbp"],
                "flood_depth_proxy": g["flood_depth_proxy"],
            })

        frames.append({
            "hour": h,
            "label": f"+{h:02d}h",
            "hazard_mode": hazard_mode,
            "cells": frame_cells,
        })

    return jsonify({
        "center": REGIONS[region]["center"],
        "hazard_mode": hazard_mode,
        "scenario": scenario,
        "frames": frames,
        "places": places.to_dict("records"),
        "outages": outages.to_dict("records"),
    })


# =============================================================================
# JAVASCRIPT MAPS
# =============================================================================

def simple_map_script(map_id: str) -> str:
    return f"""
    <script>
    document.addEventListener("DOMContentLoaded", async function() {{
        const response = await fetch("/api/data" + window.location.search);
        const data = await response.json();

        const centre = data.center;
        const map = L.map("{map_id}").setView([centre.lat, centre.lon], centre.zoom);

        L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
            maxZoom: 19,
            attribution: "OpenStreetMap"
        }}).addTo(map);

        L.tileLayer(data.nasa_tile, {{
            attribution: "NASA GIBS",
            opacity: 0.35
        }}).addTo(map);

        function riskColour(v) {{
            if (v >= 75) return "#ef4444";
            if (v >= 55) return "#f97316";
            if (v >= 35) return "#eab308";
            return "#22c55e";
        }}

        data.grid.forEach(g => {{
            const intensity = Math.min(g.wind_speed / 45, 1);
            L.circle([g.lat, g.lon], {{
                radius: 2600 + intensity * 9000,
                color: "#38bdf8",
                fillColor: "#38bdf8",
                fillOpacity: 0.08 + intensity * 0.18,
                weight: 1
            }}).addTo(map).bindPopup(
                "<b>Wind cell</b><br>" +
                "Wind speed: " + g.wind_speed + " km/h<br>" +
                "Rain: " + g.rain + " mm<br>" +
                "Risk: " + g.risk_score + "<br>" +
                "ENS: " + g.energy_not_supplied_mw + " MW<br>" +
                "Financial loss: £" + Math.round(g.financial_loss_gbp).toLocaleString()
            );
        }});

        data.places.forEach(p => {{
            L.circleMarker([p.lat, p.lon], {{
                radius: 11,
                color: "white",
                weight: 2,
                fillColor: riskColour(p.final_risk_score),
                fillOpacity: 0.96
            }}).addTo(map).bindPopup(
                "<b>" + p.place + "</b><br>" +
                "Postcode prefix: " + p.postcode_prefix + "<br>" +
                "IoD/IMD score: " + p.imd_score + "<br>" +
                "Social vulnerability: " + p.social_vulnerability + "<br>" +
                "Wind: " + Math.round(p.wind_speed_10m * 10) / 10 + " km/h<br>" +
                "Risk: " + p.final_risk_score + "<br>" +
                "Resilience: " + p.resilience_index + "<br>" +
                "ENS: " + p.energy_not_supplied_mw + " MW<br>" +
                "Financial loss: £" + Math.round(p.total_financial_loss_gbp).toLocaleString()
            );
        }});

        data.outages.forEach(o => {{
            L.marker([o.latitude, o.longitude]).addTo(map).bindPopup(
                "<b>Northern Powergrid outage</b><br>" +
                "Reference: " + o.outage_reference + "<br>" +
                "Status: " + o.outage_status + "<br>" +
                "Postcode: " + o.postcode_label + "<br>" +
                "Customers affected: " + o.affected_customers + "<br>" +
                "Restore: " + o.estimated_restore
            );
        }});

        setTimeout(() => map.invalidateSize(), 300);
    }});
    </script>
    """


def bbc_hazard_script(map_id: str, canvas_id: str) -> str:
    return f"""
    <script>
    document.addEventListener("DOMContentLoaded", async function() {{
        const response = await fetch("/api/bbc_frames" + window.location.search);
        const data = await response.json();

        const centre = data.center;

        const map = L.map("{map_id}", {{
            zoomControl: true,
            attributionControl: false,
            preferCanvas: true,
            dragging: true,
            scrollWheelZoom: true
        }}).setView([centre.lat, centre.lon], centre.zoom);

        // Broadcast/weather-chart style dark base: strong sea-land contrast.
        L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
            maxZoom: 19,
            opacity: 0.90
        }}).addTo(map);

        const labelLayer = L.layerGroup().addTo(map);
        const postcodeLayer = L.layerGroup().addTo(map);
        const outageLayer = L.layerGroup().addTo(map);

        const weatherCanvas = document.getElementById("{canvas_id}");
        const pressureCanvas = document.getElementById("q1PressureCanvas");
        const frontCanvas = document.getElementById("q1FrontCanvas");

        const wctx = weatherCanvas.getContext("2d");
        const pctx = pressureCanvas.getContext("2d");
        const fctx = frontCanvas.getContext("2d");

        const range = document.getElementById("frameRange");
        const dayLabel = document.getElementById("q1DayLabel");
        const hourLabel = document.getElementById("q1HourLabel");
        const conditionLabel = document.getElementById("q1ConditionLabel");

        const opacitySlider = document.getElementById("weatherOpacity");
        const rainToggle = document.getElementById("rainLayerToggle");
        const cloudToggle = document.getElementById("cloudLayerToggle");
        const windToggle = document.getElementById("windLayerToggle");
        const isobarToggle = document.getElementById("isobarLayerToggle");
        const lightningToggle = document.getElementById("lightningLayerToggle");

        let currentFrame = null;
        let frameIndex = 0;
        let playing = false;
        let timer = null;
        let lastT = performance.now();
        let lightningFlash = 0;

        let rainCells = [];
        let rainBands = [];
        let cloudShields = [];
        let windArrows = [];
        let vortices = [];

        function resizeCanvases() {{
            const container = document.getElementById("q1Scene");
            const rect = container.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;

            [weatherCanvas, pressureCanvas, frontCanvas].forEach(c => {{
                c.width = Math.max(1, Math.floor(rect.width * dpr));
                c.height = Math.max(1, Math.floor(rect.height * dpr));
                c.style.width = rect.width + "px";
                c.style.height = rect.height + "px";
            }});

            wctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            fctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            weatherCanvas._w = rect.width;
            weatherCanvas._h = rect.height;
        }}

        function avg(key) {{
            if (!currentFrame || !currentFrame.cells || currentFrame.cells.length === 0) return 0;
            let total = 0;
            currentFrame.cells.forEach(c => total += Number(c[key] || 0));
            return total / currentFrame.cells.length;
        }}

        function pointFor(c) {{
            return map.latLngToContainerPoint([c.lat, c.lon]);
        }}

        function nearestCell(x, y) {{
            if (!currentFrame || !currentFrame.cells) return null;
            let best = null;
            let bestD = Infinity;
            currentFrame.cells.forEach(c => {{
                const p = pointFor(c);
                const dx = p.x - x;
                const dy = p.y - y;
                const d = dx * dx + dy * dy;
                if (d < bestD) {{
                    bestD = d;
                    best = c;
                }}
            }});
            return best;
        }}

        function precipColour(rain, risk, alpha) {{
            if (risk >= 86 || rain >= 8.0) return "rgba(168,85,247," + alpha + ")";
            if (risk >= 76 || rain >= 5.7) return "rgba(239,68,68," + alpha + ")";
            if (risk >= 63 || rain >= 3.7) return "rgba(249,115,22," + alpha + ")";
            if (risk >= 50 || rain >= 2.2) return "rgba(234,179,8," + alpha + ")";
            if (risk >= 35 || rain >= 0.9) return "rgba(34,197,94," + alpha + ")";
            return "rgba(59,130,246," + alpha + ")";
        }}

        function riskSolid(v) {{
            if (v >= 86) return "#a855f7";
            if (v >= 76) return "#ef4444";
            if (v >= 63) return "#f97316";
            if (v >= 50) return "#eab308";
            if (v >= 35) return "#22c55e";
            return "#3b82f6";
        }}

        function drawEllipse(ctx, x, y, rx, ry, fillStyle, strokeStyle, lw) {{
            ctx.save();
            ctx.beginPath();
            ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
            ctx.fillStyle = fillStyle;
            ctx.fill();
            if (strokeStyle) {{
                ctx.strokeStyle = strokeStyle;
                ctx.lineWidth = lw || 1;
                ctx.stroke();
            }}
            ctx.restore();
        }}

        function initWeather() {{
            resizeCanvases();

            const W = weatherCanvas._w || 1180;
            const H = weatherCanvas._h || 760;
            const mode = data.hazard_mode;

            rainCells = [];
            rainBands = [];
            cloudShields = [];
            windArrows = [];
            vortices = [];

            const rainBandCount = mode === "storm" ? 42 : mode === "rain" ? 35 : mode === "wind" ? 22 : 18;
            for (let i = 0; i < rainBandCount; i++) {{
                rainBands.push({{
                    x: -W * 0.45 + Math.random() * W * 1.75,
                    y: -H * 0.12 + Math.random() * H * 1.18,
                    rx: 65 + Math.random() * 250,
                    ry: 20 + Math.random() * 95,
                    speed: 0.16 + Math.random() * 0.72,
                    alpha: 0.09 + Math.random() * 0.24,
                    phase: Math.random() * Math.PI * 2,
                    bias: Math.random(),
                    rotate: -0.35 + Math.random() * 0.7
                }});
            }}

            const cloudCount = mode === "storm" ? 22 : mode === "rain" ? 18 : mode === "wind" ? 14 : 10;
            for (let i = 0; i < cloudCount; i++) {{
                cloudShields.push({{
                    x: -W * 0.55 + Math.random() * W * 1.95,
                    y: -H * 0.10 + Math.random() * H * 1.15,
                    rx: 150 + Math.random() * 420,
                    ry: 42 + Math.random() * 125,
                    speed: 0.08 + Math.random() * 0.34,
                    alpha: 0.055 + Math.random() * 0.12,
                    phase: Math.random() * Math.PI * 2,
                    rotate: -0.25 + Math.random() * 0.5
                }});
            }}

            const arrowCount = mode === "storm" ? 125 : mode === "wind" ? 105 : mode === "rain" ? 74 : 60;
            for (let i = 0; i < arrowCount; i++) {{
                windArrows.push({{
                    x: Math.random() * W,
                    y: Math.random() * H,
                    len: 28 + Math.random() * 70,
                    speed: 0.50 + Math.random() * 1.50,
                    alpha: 0.35 + Math.random() * 0.38,
                    width: 1.4 + Math.random() * 2.6,
                    phase: Math.random() * Math.PI * 2,
                    curve: Math.random() * 0.8
                }});
            }}

            // Cyclonic spiral centres, like WXCharts storm systems.
            const vortexCount = mode === "storm" ? 3 : mode === "rain" ? 2 : 1;
            for (let i = 0; i < vortexCount; i++) {{
                vortices.push({{
                    x: W * (0.28 + Math.random() * 0.48),
                    y: H * (0.25 + Math.random() * 0.52),
                    radius: 95 + Math.random() * 180,
                    strength: 0.35 + Math.random() * 0.65,
                    speed: 0.05 + Math.random() * 0.12,
                    phase: Math.random() * Math.PI * 2
                }});
            }}
        }}

        function drawPressureAndFronts(W, H, t) {{
            pctx.clearRect(0, 0, W, H);
            fctx.clearRect(0, 0, W, H);

            if (isobarToggle && isobarToggle.checked) {{
                pctx.save();
                pctx.globalAlpha = 0.60;
                pctx.strokeStyle = "rgba(255,255,255,0.56)";
                pctx.lineWidth = 1.6;

                // Multiple contour families.
                for (let family = 0; family < 2; family++) {{
                    const cx = W * (family === 0 ? 0.27 : 0.72) + Math.sin(t / 6200 + family) * 28;
                    const cy = H * (family === 0 ? 0.55 : 0.36) + Math.cos(t / 5300 + family) * 22;

                    for (let k = 0; k < 8; k++) {{
                        const rx = 90 + k * 42 + family * 20;
                        const ry = 50 + k * 28;
                        pctx.beginPath();
                        pctx.ellipse(cx, cy, rx, ry, -0.38 + family * 0.65, 0, Math.PI * 2);
                        pctx.stroke();
                    }}
                }}

                // Long sweeping isobars.
                pctx.lineWidth = 1.2;
                for (let k = 0; k < 7; k++) {{
                    pctx.beginPath();
                    for (let x = -70; x <= W + 80; x += 18) {{
                        const y = H * 0.24 + k * 78 + Math.sin((x + t * 0.018) / 125 + k * 0.55) * (26 + k * 2);
                        if (x === -70) pctx.moveTo(x, y);
                        else pctx.lineTo(x, y);
                    }}
                    pctx.stroke();
                }}

                pctx.restore();
            }}

            // Front line with red semicircles and blue triangles.
            fctx.save();
            fctx.lineWidth = 2.4;
            fctx.strokeStyle = "rgba(255,255,255,0.76)";
            fctx.beginPath();

            const baseY = H * 0.60;
            for (let x = -60; x <= W + 70; x += 22) {{
                const y = baseY + Math.sin((x + t * 0.025) / 118) * 42;
                if (x === -60) fctx.moveTo(x, y);
                else fctx.lineTo(x, y);
            }}
            fctx.stroke();

            for (let x = 10; x < W; x += 74) {{
                const y = baseY + Math.sin((x + t * 0.025) / 118) * 42;

                fctx.fillStyle = "rgba(59,130,246,0.88)";
                fctx.beginPath();
                fctx.moveTo(x, y);
                fctx.lineTo(x + 19, y + 16);
                fctx.lineTo(x - 8, y + 18);
                fctx.closePath();
                fctx.fill();

                fctx.fillStyle = "rgba(239,68,68,0.88)";
                fctx.beginPath();
                fctx.arc(x + 38, y - 1, 10, Math.PI, 0);
                fctx.fill();
            }}
            fctx.restore();
        }}

        function drawClouds(W, H, t, dt) {{
            if (!cloudToggle || !cloudToggle.checked) return;

            cloudShields.forEach(c => {{
                c.x += c.speed * dt * 0.05;
                c.y += Math.sin(t / 2500 + c.phase) * 0.035 * dt;

                if (c.x - c.rx > W + 160) {{
                    c.x = -c.rx - 180;
                    c.y = -H * 0.10 + Math.random() * H * 1.15;
                }}

                wctx.save();
                wctx.translate(c.x, c.y);
                wctx.rotate(c.rotate);

                // Shadow then cloud, broad and soft.
                let shadow = wctx.createRadialGradient(22, 35, 0, 22, 35, c.rx);
                shadow.addColorStop(0, "rgba(0,0,0," + c.alpha * 0.36 + ")");
                shadow.addColorStop(1, "rgba(0,0,0,0)");
                drawEllipse(wctx, 22, 35, c.rx * 0.95, c.ry * 0.85, shadow);

                let cloud = wctx.createRadialGradient(0, 0, 0, 0, 0, c.rx);
                cloud.addColorStop(0, "rgba(255,255,255," + c.alpha + ")");
                cloud.addColorStop(0.40, "rgba(220,230,235," + c.alpha * 0.72 + ")");
                cloud.addColorStop(0.78, "rgba(160,176,188," + c.alpha * 0.30 + ")");
                cloud.addColorStop(1, "rgba(160,176,188,0)");
                drawEllipse(wctx, 0, 0, c.rx, c.ry, cloud);

                wctx.restore();
            }});
        }}

        function drawPrecipitation(W, H, t, dt) {{
            if (!rainToggle || !rainToggle.checked || !currentFrame) return;

            const opacity = Number(opacitySlider ? opacitySlider.value : 0.76);
            const mode = data.hazard_mode;
            const movement = mode === "storm" ? 1.55 : mode === "rain" ? 1.18 : 0.74;
            const avgRain = avg("rain");
            const avgRisk = avg("risk_score");

            // Anchored precipitation over model cells.
            currentFrame.cells.forEach((cell, idx) => {{
                const p = pointFor(cell);
                const rain = Number(cell.rain || 0);
                const risk = Number(cell.risk_score || 0);
                if (rain < 0.1 && risk < 30 && mode !== "storm" && mode !== "rain") return;

                const pulse = 0.94 + 0.06 * Math.sin(t / 680 + idx);
                const rx = (46 + rain * 28 + risk * 0.80) * pulse;
                const ry = (21 + rain * 12 + risk * 0.34) * pulse;
                const alpha = Math.min(0.62, (0.085 + rain * 0.065 + risk / 430) * opacity);

                const grad = wctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, rx);
                grad.addColorStop(0, precipColour(rain, risk, alpha));
                grad.addColorStop(0.42, precipColour(rain, risk, alpha * 0.58));
                grad.addColorStop(0.74, "rgba(59,130,246," + alpha * 0.20 + ")");
                grad.addColorStop(1, "rgba(0,0,0,0)");

                drawEllipse(wctx, p.x, p.y, rx, ry, grad);
            }});

            // Moving rain bands
            rainBands.forEach((b, i) => {{
                b.x += b.speed * movement * dt * 0.068;
                b.y += Math.sin(t / 1600 + b.phase) * 0.055 * dt;

                if (b.x - b.rx > W + 150) {{
                    b.x = -b.rx - 170;
                    b.y = -H * 0.12 + Math.random() * H * 1.18;
                }}

                const syntheticRain = avgRain + b.bias * 5.0;
                const syntheticRisk = avgRisk + b.bias * 45.0;
                const alpha = Math.min(0.55, b.alpha * (0.65 + avgRain / 3.8 + avgRisk / 190) * opacity);

                wctx.save();
                wctx.translate(b.x, b.y);
                wctx.rotate(b.rotate);

                const grad = wctx.createRadialGradient(0, 0, 0, 0, 0, b.rx);
                grad.addColorStop(0, precipColour(syntheticRain, syntheticRisk, alpha));
                grad.addColorStop(0.46, precipColour(syntheticRain, syntheticRisk, alpha * 0.54));
                grad.addColorStop(0.80, "rgba(37,99,235," + alpha * 0.18 + ")");
                grad.addColorStop(1, "rgba(0,0,0,0)");

                drawEllipse(wctx, 0, 0, b.rx, b.ry, grad);

                // Embedded heavier core spots like WXCharts.
                if (b.bias > 0.58) {{
                    const core = wctx.createRadialGradient(0, 0, 0, 0, 0, b.rx * 0.30);
                    core.addColorStop(0, precipColour(syntheticRain + 2.0, syntheticRisk + 20, alpha * 0.90));
                    core.addColorStop(1, "rgba(0,0,0,0)");
                    drawEllipse(wctx, 0, 0, b.rx * 0.32, b.ry * 0.46, core);
                }}

                wctx.restore();
            }});

            // Spiral precipitation around vortices.
            vortices.forEach(v => {{
                v.phase += v.speed * dt * 0.01;
                for (let arm = 0; arm < 4; arm++) {{
                    for (let j = 0; j < 28; j++) {{
                        const theta = v.phase + arm * Math.PI / 2 + j * 0.18;
                        const r = 18 + j * (v.radius / 28);
                        const x = v.x + Math.cos(theta) * r;
                        const y = v.y + Math.sin(theta) * r * 0.60;
                        const a = Math.max(0, (1 - j / 30) * 0.16 * opacity * v.strength);
                        const col = precipColour(avgRain + 3, avgRisk + 30, a);
                        drawEllipse(wctx, x, y, 22 + j * 1.0, 8 + j * 0.35, col);
                    }}
                }}
            }});
        }}

        function drawWindArrows(W, H, t, dt) {{
            if (!windToggle || !windToggle.checked) return;

            const mode = data.hazard_mode;
            const mult = mode === "storm" ? 1.65 : mode === "wind" ? 1.38 : mode === "rain" ? 0.92 : 0.72;

            windArrows.forEach(a => {{
                const local = nearestCell(a.x, a.y);
                const w = local ? Number(local.wind_speed || 9) : avg("wind_speed");
                const intensity = Math.min(w / 42, 1.55);

                // Vortex influence for curved arrows
                let angle = -0.24 + Math.sin(t / 1800 + a.phase) * 0.12;
                vortices.forEach(v => {{
                    const dx = a.x - v.x;
                    const dy = a.y - v.y;
                    const d = Math.sqrt(dx*dx + dy*dy);
                    if (d < v.radius * 2.1) {{
                        angle += Math.atan2(dy, dx) * 0.10 * v.strength;
                    }}
                }});

                const len = a.len * (0.74 + intensity * 0.58);
                const x0 = a.x - Math.cos(angle) * len;
                const y0 = a.y - Math.sin(angle) * len;
                const alpha = Math.min(0.86, a.alpha + intensity * 0.20);

                wctx.save();
                wctx.strokeStyle = "rgba(255,255,255," + alpha + ")";
                wctx.fillStyle = "rgba(255,255,255," + alpha + ")";
                wctx.lineWidth = a.width;
                wctx.lineCap = "round";

                wctx.beginPath();
                wctx.moveTo(x0, y0);
                wctx.quadraticCurveTo(
                    (x0 + a.x) / 2,
                    (y0 + a.y) / 2 + Math.sin(t / 540 + a.phase) * 5,
                    a.x,
                    a.y
                );
                wctx.stroke();

                const head = 10 + intensity * 7;
                const bx = a.x - Math.cos(angle) * head;
                const by = a.y - Math.sin(angle) * head;
                const nx = -Math.sin(angle);
                const ny = Math.cos(angle);

                wctx.beginPath();
                wctx.moveTo(a.x, a.y);
                wctx.lineTo(bx + nx * head * 0.42, by + ny * head * 0.42);
                wctx.lineTo(bx - nx * head * 0.42, by - ny * head * 0.42);
                wctx.closePath();
                wctx.fill();
                wctx.restore();

                a.x += Math.cos(angle) * a.speed * mult * (0.66 + intensity) * dt * 0.083;
                a.y += Math.sin(angle) * a.speed * mult * (0.66 + intensity) * dt * 0.083;

                if (a.x > W + 115 || a.y < -85 || a.y > H + 85) {{
                    a.x = -115;
                    a.y = Math.random() * H;
                    a.phase = Math.random() * Math.PI * 2;
                }}
            }});
        }}

        function drawLightning(W, H) {{
            if (!lightningToggle || !lightningToggle.checked || data.hazard_mode !== "storm") return;

            if (Math.random() < 0.006) lightningFlash = 6;

            if (lightningFlash > 0) {{
                wctx.fillStyle = "rgba(255,255,255," + (0.05 + lightningFlash * 0.012) + ")";
                wctx.fillRect(0, 0, W, H);

                for (let bolt = 0; bolt < 2; bolt++) {{
                    wctx.strokeStyle = "rgba(255,255,255,0.64)";
                    wctx.lineWidth = 2.1;
                    wctx.beginPath();
                    let x = W * (0.15 + Math.random() * 0.75);
                    let y = 0;
                    wctx.moveTo(x, y);
                    for (let i = 0; i < 6; i++) {{
                        x += -35 + Math.random() * 70;
                        y += 34 + Math.random() * 62;
                        wctx.lineTo(x, y);
                    }}
                    wctx.stroke();
                }}

                lightningFlash -= 1;
            }}
        }}

        function drawHUD(W, H) {{
            wctx.save();
            wctx.fillStyle = "rgba(2,6,23,0.58)";
            wctx.strokeStyle = "rgba(255,255,255,0.18)";
            wctx.lineWidth = 1;
            wctx.beginPath();
            wctx.roundRect(18, H - 92, 430, 70, 14);
            wctx.fill();
            wctx.stroke();

            wctx.fillStyle = "rgba(255,255,255,0.96)";
            wctx.font = "700 13px Segoe UI, Arial";
            wctx.fillText("Q1-style forecast simulation", 34, H - 64);

            wctx.font = "12px Segoe UI, Arial";
            wctx.fillStyle = "rgba(219,234,254,0.96)";
            wctx.fillText("WXCharts-style precipitation · BBC-style arrows · isobars/fronts", 34, H - 42);
            wctx.fillText("Avg wind " + avg("wind_speed").toFixed(1) + " km/h · Avg rain " + avg("rain").toFixed(1) + " mm · Risk " + avg("risk_score").toFixed(1), 34, H - 24);
            wctx.restore();
        }}

        function animate(t) {{
            if (!currentFrame) {{
                requestAnimationFrame(animate);
                return;
            }}

            const dt = Math.min(34, t - lastT);
            lastT = t;

            const W = weatherCanvas._w || weatherCanvas.clientWidth || 1180;
            const H = weatherCanvas._h || weatherCanvas.clientHeight || 760;

            wctx.clearRect(0, 0, W, H);

            // Light atmospheric wash; map remains visible.
            wctx.fillStyle = data.hazard_mode === "storm" ? "rgba(5,12,24,0.045)" : "rgba(5,15,28,0.025)";
            wctx.fillRect(0, 0, W, H);

            drawPressureAndFronts(W, H, t);
            drawClouds(W, H, t, dt);
            drawPrecipitation(W, H, t, dt);
            drawWindArrows(W, H, t, dt);
            drawLightning(W, H);
            drawHUD(W, H);

            requestAnimationFrame(animate);
        }}

        function renderFrame(idx) {{
            currentFrame = data.frames[idx];
            frameIndex = idx;
            range.value = idx;

            dayLabel.textContent = "FRIDAY";
            hourLabel.textContent = String(currentFrame.label).replace("+", "");
            conditionLabel.textContent = data.scenario + " · " + data.hazard_mode;

            // Progress storm fields with timeline.
            rainBands.forEach((b, i) => b.x += 30 + i * 1.8);
            cloudShields.forEach((c, i) => c.x += 14 + i * 0.6);
            vortices.forEach(v => v.x += 8);
        }}

        // City labels
        data.places.forEach(p => {{
            const label = L.divIcon({{
                className: "",
                html: "<div class='city-label-bbc'>" + p.place + "</div>",
                iconSize: [130, 30],
                iconAnchor: [0, 14]
            }});
            L.marker([p.lat, p.lon], {{ icon: label }}).addTo(labelLayer).bindPopup(
                "<b>" + p.place + "</b><br>" +
                "Risk: " + p.final_risk_score + "<br>" +
                "Resilience: " + p.resilience_index + "<br>" +
                "IoD/IMD: " + p.imd_score + "<br>" +
                "Financial loss: £" + Math.round(p.total_financial_loss_gbp).toLocaleString()
            );
        }});

        data.outages.forEach(o => {{
            L.circleMarker([o.latitude, o.longitude], {{
                radius: 5,
                color: "white",
                weight: 1,
                fillColor: "#ef4444",
                fillOpacity: 0.95
            }}).addTo(outageLayer).bindPopup(
                "<b>Outage</b><br>" +
                "Postcode: " + o.postcode_label + "<br>" +
                "Customers: " + o.affected_customers
            );
        }});

        // Postcode resilience layer from dedicated endpoint
        fetch("/api/postcode_resilience" + window.location.search)
            .then(r => r.json())
            .then(pcData => {{
                pcData.postcodes.forEach(p => {{
                    const col = p.resilience_score >= 80 ? "#22c55e" :
                                p.resilience_score >= 60 ? "#38bdf8" :
                                p.resilience_score >= 40 ? "#eab308" : "#ef4444";

                    L.circleMarker([p.lat, p.lon], {{
                        radius: 5 + Math.min(7, p.recommendation_score / 14),
                        color: "white",
                        weight: 1.2,
                        fillColor: col,
                        fillOpacity: 0.78
                    }}).addTo(postcodeLayer).bindPopup(
                        "<b>Postcode: " + p.postcode + "</b><br>" +
                        "Resilience: " + p.resilience_score + "<br>" +
                        "Risk: " + p.risk_score + "<br>" +
                        "Priority: " + p.investment_priority + "<br>" +
                        "Financial loss: £" + Math.round(p.financial_loss_gbp).toLocaleString()
                    );
                }});
            }});

        window.playBBC = function() {{
            if (playing) return;
            playing = true;
            timer = setInterval(() => {{
                frameIndex = (frameIndex + 1) % data.frames.length;
                renderFrame(frameIndex);
            }}, 950);
        }};

        window.pauseBBC = function() {{
            playing = false;
            if (timer) clearInterval(timer);
        }};

        window.scrubBBC = function(value) {{
            renderFrame(parseInt(value));
        }};

        window.reseedWeather = function() {{
            initWeather();
        }};

        [opacitySlider, rainToggle, cloudToggle, windToggle, isobarToggle, lightningToggle].forEach(el => {{
            if (el) el.addEventListener("input", function() {{}});
            if (el) el.addEventListener("change", function() {{}});
        }});

        range.max = data.frames.length - 1;
        range.value = 0;

        map.on("move zoom resize", function() {{
            resizeCanvases();
        }});

        window.addEventListener("resize", function() {{
            resizeCanvases();
            map.invalidateSize();
        }});

        resizeCanvases();
        currentFrame = data.frames[0];
        initWeather();
        renderFrame(0);

        setTimeout(() => {{
            map.invalidateSize();
            resizeCanvases();
        }}, 300);

        requestAnimationFrame(animate);

        setTimeout(() => {{
            window.playBBC();
        }}, 800);
    }});
    </script>
    """


def nasa_storyboard_script(prefix: str) -> str:
    return f"""
    <script>
    document.addEventListener("DOMContentLoaded", async function() {{
        const response = await fetch("/api/data" + window.location.search);
        const data = await response.json();
        const centre = data.center;

        const phases = [
            ["{prefix}0", "Pre-event", 0.55],
            ["{prefix}1", "Disturbance", 0.85],
            ["{prefix}2", "Peak failure", 1.20],
            ["{prefix}3", "Recovery", 0.70]
        ];

        function colour(v) {{
            if (v >= 75) return "#ef4444";
            if (v >= 55) return "#f97316";
            if (v >= 35) return "#eab308";
            return "#22c55e";
        }}

        phases.forEach(item => {{
            const mapId = item[0];
            const phase = item[1];
            const mult = item[2];

            const map = L.map(mapId, {{
                zoomControl: false,
                attributionControl: false
            }}).setView([centre.lat, centre.lon], centre.zoom);

            L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
                opacity: 0.25
            }}).addTo(map);

            L.tileLayer(data.nasa_tile, {{
                opacity: 0.82,
                attribution: "NASA GIBS"
            }}).addTo(map);

            data.places.forEach(p => {{
                const risk = Math.min(100, p.final_risk_score * mult);
                L.circleMarker([p.lat, p.lon], {{
                    radius: 8,
                    color: "white",
                    weight: 1.5,
                    fillColor: colour(risk),
                    fillOpacity: 0.95
                }}).addTo(map).bindPopup(
                    "<b>" + p.place + "</b><br>" +
                    phase + "<br>Risk: " + Math.round(risk * 10) / 10
                );

                L.circle([p.lat, p.lon], {{
                    radius: 2500 + risk * 45,
                    color: "#38bdf8",
                    fillColor: "#38bdf8",
                    fillOpacity: 0.08,
                    weight: 1
                }}).addTo(map);
            }});

            if (phase !== "Pre-event") {{
                data.outages.forEach(o => {{
                    L.marker([o.latitude, o.longitude]).addTo(map).bindPopup(
                        "<b>Outage</b><br>Postcode: " + o.postcode_label
                    );
                }});
            }}

            setTimeout(() => map.invalidateSize(), 300);
        }});
    }});
    </script>
    """



# =============================================================================
# POSTCODE RESILIENCE + INVESTMENT RECOMMENDATION MODELS
# =============================================================================

def build_postcode_resilience(places: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    """
    Produces postcode-level resilience estimates.

    This uses available postcode labels from Northern Powergrid outage records.
    If live postcode labels are sparse, it also includes the configured city/town
    postcode prefixes so the page is never empty.

    In a full production system, replace this with a national postcode centroid
    or postcode boundary dataset. This prototype computes resilience using:
    - nearest place-level digital twin risk
    - nearby outage concentration
    - affected customers
    - social vulnerability
    - ENS / financial loss proxy
    """
    rows = []

    # Build rows from live outage postcode labels
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

    # Ensure all configured postcode prefixes appear
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

    # Very transparent indicative cost model
    pc["indicative_investment_cost_gbp"] = (
        120000
        + pc["recommendation_score"] * 8500
        + pc["outage_records"] * 35000
        + np.clip(pc["energy_not_supplied_mw"], 0, 1000) * 260
    ).round(0)

    pc["benefit_cost_note"] = (
        "High avoided-loss potential"
    )

    return pc.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    region, scenario, layer, page, mc = get_controls()
    places, outages, grid = get_data(region, scenario, mc)

    if page == "simple":
        return page_simple(region, scenario, layer, places, outages, grid)

    if page == "bbc":
        return page_bbc(region, scenario, layer, places, outages, grid)

    if page == "resilience":
        return page_resilience(region, scenario, layer, places, outages, grid)

    if page == "postcode":
        return page_postcode(region, scenario, layer, places, outages, grid)

    if page == "postcode_resilience":
        return page_postcode_resilience(region, scenario, layer, places, outages, grid)

    if page == "investment":
        return page_investment(region, scenario, layer, places, outages, grid)

    if page == "finance":
        return page_finance(region, scenario, layer, places, outages, grid)

    if page == "montecarlo":
        return page_montecarlo(region, scenario, layer, places, outages, grid)

    if page == "satellite":
        return page_satellite(region, scenario, layer, places, outages, grid)

    if page == "data":
        return page_data(region, scenario, layer, places, outages, grid)

    return page_overview(region, scenario, layer, places, outages, grid)


def page_overview(region, scenario, layer, places, outages, grid):
    avg_risk = round(float(places["final_risk_score"].mean()), 1)
    avg_res = round(float(places["resilience_index"].mean()), 1)
    avg_failure = round(float(places["failure_probability"].mean()) * 100, 1)
    total_ens = round(float(places["energy_not_supplied_mw"].sum()), 1)
    total_loss = round(float(places["total_financial_loss_gbp"].sum()) / 1_000_000, 2)
    avg_wind = round(float(places["wind_speed_10m"].mean()), 1)
    avg_social = round(float(places["social_vulnerability"].mean()), 1)
    imd_source = places.iloc[0].get("imd_dataset_summary", "Unknown") if not places.empty else "Unknown"

    content = f"""
    <div class="grid">
        {metric_card("Mode", scenario, SCENARIOS[scenario]["description"])}
        {metric_card("Regional Risk", str(avg_risk), "scenario-adjusted risk score")}
        {metric_card("Resilience", f"{avg_res}/100", "higher is better")}
        {metric_card("Financial Loss", f"£{total_loss}m", "estimated regional loss")}
    </div>

    <div class="grid">
        {metric_card("ENS", f"{total_ens} MW", "energy not supplied")}
        {metric_card("Failure Probability", f"{avg_failure}%", "mean predicted probability")}
        {metric_card("Wind Speed", f"{avg_wind} km/h", "weather headline variable")}
        {metric_card("Social Vulnerability", f"{avg_social}/100", "IoD2025/deprivation-aware")}
    </div>

    <div class="card">
        <h2>IoD2025 / deprivation data source</h2>
        <p style="color:#94a3b8;">{imd_source}</p>
    </div>

    <div class="grid-2">
        <div class="card">
            <h2>Regional intelligence</h2>
            {places_table(places)}
        </div>

        <div class="card bbc-card">
            <h2>BBC-style snapshot</h2>
            <div class="metric-label">Headline hazard variable</div>
            <div class="metric-value">🌬️ {avg_wind} km/h</div>
            <div class="metric-note">Risk {avg_risk} · Resilience {avg_res} · ENS {total_ens} MW</div>
            <hr style="border-color:rgba(148,163,184,0.18); margin:18px 0;">
            <h3>Risk distribution</h3>
            {chart_bars(places, "final_risk_score")}
        </div>
    </div>
    """

    return render_app(
        "North East & Yorkshire Grid Digital Twin",
        "Updated version using your IoD2025 deprivation datasets, improved financial loss model and BBC-style moving hazard animation.",
        content,
    )


def page_simple(region, scenario, layer, places, outages, grid):
    content = """
    <div class="card">
        <h2>Simple Map: Wind, Postcode and Outage Layer</h2>
        <p style="color:#94a3b8;">
            Wind cells, postcode outage markers, place markers and NASA overlay.
        </p>
        <div id="simpleMap" class="map"></div>
    </div>
    """

    return render_app(
        "Simple Map",
        "Wind-speed map with postcode markers, Northern Powergrid outage popups and NASA overlay.",
        content,
        simple_map_script("simpleMap"),
    )


def page_bbc(region, scenario, layer, places, outages, grid):
    hazard = SCENARIOS[scenario]["hazard_mode"]

    content = f"""
    <div class="card bbc-card">
        <h2>Weather Simulation</h2>
        <p style="color:#cbd5e1;">
            Professional forecast-style view inspired by BBC Weather and WXCharts:
            dark broadcast basemap, labelled cities, moving precipitation fields, spiral storm bands,
            white wind arrows, isobars/fronts, lightning, time strip and postcode resilience markers.
        </p>

        <div id="q1Scene" class="q1-scene">
            <div id="q1WeatherMap" class="q1-map"></div>
            <canvas id="q1PressureCanvas" class="q1-pressure-canvas"></canvas>
            <canvas id="q1WeatherCanvas" class="q1-weather-canvas"></canvas>
            <canvas id="q1FrontCanvas" class="q1-front-canvas"></canvas>

            <div class="q1-top-card">
                <div class="q1-top-title">Forecast simulation and resilience overlay</div>
                <div class="q1-top-sub">
                    Scenario: {scenario}<br>
                    Visual mode: {hazard}<br>
                    Weather layers remain semi-transparent so investment and postcode markers can still be inspected.
                </div>
            </div>

            <div class="q1-legend">
                <strong>Precipitation / hazard intensity</strong>
                <div class="q1-gradient"></div>
                <div style="display:flex;justify-content:space-between;">
                    <span>Light</span><span>Heavy</span><span>Extreme</span>
                </div>
                <hr style="border-color:rgba(255,255,255,.14);">
                <span style="color:#22c55e;">●</span> resilient postcode&nbsp;&nbsp;
                <span style="color:#ef4444;">●</span> fragile postcode
            </div>

            <div class="q1-brand">
                <div class="q1-brand-blocks"><span>B</span><span>B</span><span>C</span></div>
                <div class="q1-brand-word">WEATHER</div>
            </div>

            <div class="wx-watermark">Q1 GRID RESILIENCE WEATHER MODEL</div>

            <div class="q1-time-strip">
                <div class="q1-day" id="q1DayLabel">FRIDAY</div>
                <div class="q1-hour" id="q1HourLabel">00h</div>
            </div>

            <div class="q1-controls">
                <button onclick="playBBC()">▶ Play</button>
                <button onclick="pauseBBC()">Ⅱ Pause</button>
                <input id="frameRange" class="range" type="range" min="0" max="1" value="0" oninput="scrubBBC(this.value)">
                <span class="chip" id="q1ConditionLabel">{hazard}</span>

                <label class="q1-toggle"><input id="rainLayerToggle" type="checkbox" checked> rain</label>
                <label class="q1-toggle"><input id="cloudLayerToggle" type="checkbox" checked> cloud</label>
                <label class="q1-toggle"><input id="windLayerToggle" type="checkbox" checked> wind</label>
                <label class="q1-toggle"><input id="isobarLayerToggle" type="checkbox" checked> isobars</label>
                <label class="q1-toggle"><input id="lightningLayerToggle" type="checkbox" checked> lightning</label>

                <label class="q1-toggle">
                    opacity
                    <input id="weatherOpacity" type="range" min="0.35" max="0.95" step="0.05" value="0.78">
                </label>
                <button onclick="reseedWeather()">refresh</button>
            </div>
        </div>
    </div>
    """

    return render_app(
        "Weather Simulation",
        "Professional animated forecast map with precipitation, isobars, fronts, wind arrows, postcode resilience and investment context.",
        content,
        bbc_hazard_script("q1WeatherMap", "q1WeatherCanvas"),
    )



def page_resilience(region, scenario, layer, places, outages, grid):
    rows = ""
    for _, r in places.sort_values("resilience_index").iterrows():
        rows += f"""
        <tr>
            <td><strong>{r['place']}</strong></td>
            <td><span class="badge {r['resilience_label']}">{r['resilience_label']}</span></td>
            <td>{round(r['resilience_index'],1)}</td>
            <td>{round(r['imd_score'],1)}</td>
            <td>{r['imd_match']}</td>
            <td>{round(r['social_vulnerability'],1)}</td>
            <td>{round(r['grid_failure_probability']*100,1)}%</td>
            <td>{round(r['renewable_failure_probability']*100,1)}%</td>
            <td>£{round(r['total_financial_loss_gbp']/1_000_000,2)}m</td>
        </tr>
        """

    content = f"""
    <div class="grid">
        {metric_card("Mean Resilience", f"{round(places['resilience_index'].mean(),1)}/100", "higher is better")}
        {metric_card("Lowest Resilience", f"{round(places['resilience_index'].min(),1)}", "priority location")}
        {metric_card("Mean IoD/Social", f"{round(places['social_vulnerability'].mean(),1)}", "deprivation-aware")}
        {metric_card("Mean Grid Failure", f"{round(places['grid_failure_probability'].mean()*100,1)}%", "risk + outages + ENS")}
    </div>

    <div class="card">
        <h2>Resilience diagnostics using IoD2025 / deprivation</h2>
        <table>
            <tr>
                <th>Place</th>
                <th>Level</th>
                <th>Index</th>
                <th>IoD/IMD score</th>
                <th>Dataset match</th>
                <th>Social vulnerability</th>
                <th>Grid failure</th>
                <th>Renewable failure</th>
                <th>Financial loss</th>
            </tr>
            {rows}
        </table>
    </div>
    """

    return render_app(
        "Resilience Level",
        "Resilience index with uploaded IoD2025 deprivation data, social vulnerability, grid failure, renewable failure and financial loss.",
        content,
    )


def page_postcode(region, scenario, layer, places, outages, grid):
    if outages.empty:
        content = """
        <div class="card">
            <h2>Postcode Visuals</h2>
            <div class="alert">No outage postcode data available.</div>
        </div>
        """
        return render_app("Postcode Visuals", "Northern Powergrid postcode distribution.", content)

    grouped = (
        outages.groupby("postcode_label")
        .agg(outages=("outage_reference", "count"), customers=("affected_customers", "sum"))
        .reset_index()
        .sort_values("outages", ascending=False)
    )

    rows = ""
    chart = ""
    max_count = max(1, int(grouped["outages"].max()))

    for _, r in grouped.head(20).iterrows():
        width = (r["outages"] / max_count) * 100
        rows += f"""
        <tr>
            <td><strong>{r['postcode_label']}</strong></td>
            <td>{int(r['outages'])}</td>
            <td>{int(r['customers'])}</td>
            <td><div class="bar"><div class="fill-risk" style="width:{width}%"></div></div></td>
        </tr>
        """

        chart += f"""
        <div class="chart-col">
            <div class="chart-bar" style="height:{max(8, width*2.2)}px"></div>
            <span>{r['postcode_label']}</span>
        </div>
        """

    content = f"""
    <div class="grid-2">
        <div class="card">
            <h2>Postcode outage ranking</h2>
            <table>
                <tr>
                    <th>Postcode</th>
                    <th>Outages</th>
                    <th>Affected customers</th>
                    <th>Intensity</th>
                </tr>
                {rows}
            </table>
        </div>

        <div class="card">
            <h2>Postcode visual intensity</h2>
            <div class="chart">{chart}</div>
            <div class="alert success">Postcode visuals are based on normalised Northern Powergrid records.</div>
        </div>
    </div>
    """

    return render_app("Postcode Visuals", "Postcode-level outage distribution and affected-customer intensity.", content)



def postcode_resilience_map_script(map_id: str) -> str:
    return f"""
    <script>
    document.addEventListener("DOMContentLoaded", async function() {{
        const response = await fetch("/api/postcode_resilience" + window.location.search);
        const data = await response.json();

        const map = L.map("{map_id}").setView([data.center.lat, data.center.lon], data.center.zoom);

        L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
            maxZoom: 19,
            attribution: "OpenStreetMap"
        }}).addTo(map);

        function colour(v) {{
            if (v >= 80) return "#22c55e";
            if (v >= 60) return "#38bdf8";
            if (v >= 40) return "#eab308";
            return "#ef4444";
        }}

        data.postcodes.forEach(p => {{
            const radius = 2600 + Math.max(0, 100 - p.resilience_score) * 45;
            L.circleMarker([p.lat, p.lon], {{
                radius: 8 + Math.min(8, p.recommendation_score / 12),
                color: "white",
                weight: 2,
                fillColor: colour(p.resilience_score),
                fillOpacity: 0.93
            }}).addTo(map).bindPopup(
                "<b>Postcode: " + p.postcode + "</b><br>" +
                "Nearest place: " + p.nearest_place + "<br>" +
                "Resilience score: " + p.resilience_score + "<br>" +
                "Risk score: " + p.risk_score + "<br>" +
                "Priority: " + p.investment_priority + "<br>" +
                "Recommendation score: " + p.recommendation_score + "<br>" +
                "Financial loss: £" + Math.round(p.financial_loss_gbp).toLocaleString()
            );

            L.circle([p.lat, p.lon], {{
                radius: radius,
                color: colour(p.resilience_score),
                fillColor: colour(p.resilience_score),
                fillOpacity: 0.045,
                weight: 1,
                opacity: 0.55
            }}).addTo(map);
        }});

        setTimeout(() => map.invalidateSize(), 300);
    }});
    </script>
    """


def page_postcode_resilience(region, scenario, layer, places, outages, grid):
    pc = build_postcode_resilience(places, outages)

    if pc.empty:
        content = """
        <div class="card">
            <h2>Postcode-level Resilience Scores</h2>
            <div class="alert">No postcode resilience data could be generated.</div>
        </div>
        """
        return render_app("Postcode-level Resilience Scores", "Postcode-level resilience estimates.", content)

    rows = ""
    for _, r in pc.iterrows():
        rows += f"""
        <tr>
            <td><strong>{r['postcode']}</strong></td>
            <td>{r['nearest_place']}</td>
            <td><span class="badge {r['resilience_label']}">{r['resilience_label']}</span></td>
            <td>{round(r['resilience_score'],1)}</td>
            <td>{round(r['risk_score'],1)}</td>
            <td>{round(r['social_vulnerability'],1)}</td>
            <td>{int(r['outage_records'])}</td>
            <td>{round(r['energy_not_supplied_mw'],1)} MW</td>
            <td>£{round(r['financial_loss_gbp']/1_000_000,2)}m</td>
            <td>{r['investment_priority']}</td>
        </tr>
        """

    content = f"""
    <div class="grid">
        {metric_card("Postcode Areas", str(len(pc)), "generated from live outage labels + configured prefixes")}
        {metric_card("Lowest Resilience", str(round(pc['resilience_score'].min(),1)), "postcode-level")}
        {metric_card("Highest Priority Score", str(round(pc['recommendation_score'].max(),1)), "investment urgency")}
        {metric_card("Total Postcode Loss", f"£{round(pc['financial_loss_gbp'].sum()/1_000_000,2)}m", "estimated")}
    </div>

    <div class="card">
        <h2>Postcode-level resilience map</h2>
        <p style="color:#94a3b8;">
            Each postcode marker has a resilience score, risk score, financial-loss estimate and investment priority.
            For a production-grade map, this can be joined to a full UK postcode centroid or boundary dataset.
        </p>
        <div id="postcodeResilienceMap" class="map"></div>
    </div>

    <div class="card">
        <h2>Postcode-level resilience scores</h2>
        <table>
            <tr>
                <th>Postcode</th>
                <th>Nearest place</th>
                <th>Level</th>
                <th>Resilience</th>
                <th>Risk</th>
                <th>Social vuln.</th>
                <th>Outages</th>
                <th>ENS</th>
                <th>Financial loss</th>
                <th>Priority</th>
            </tr>
            {rows}
        </table>
    </div>
    """

    return render_app(
        "Postcode-level Resilience Scores",
        "Postcode-level resilience, risk, financial loss and priority scoring.",
        content,
        postcode_resilience_map_script("postcodeResilienceMap"),
    )


def page_investment(region, scenario, layer, places, outages, grid):
    rec = build_investment_recommendations(places, outages)

    if rec.empty:
        content = """
        <div class="card">
            <h2>Prioritised Investment Recommendations</h2>
            <div class="alert">No investment recommendations could be generated.</div>
        </div>
        """
        return render_app("Prioritised Investment Recommendations", "Investment ranking.", content)

    rows = ""
    for rank, (_, r) in enumerate(rec.iterrows(), start=1):
        rows += f"""
        <tr>
            <td><strong>{rank}</strong></td>
            <td><strong>{r['postcode']}</strong></td>
            <td>{r['nearest_place']}</td>
            <td><span class="badge {r['investment_priority'].replace(' ', '')}">{r['investment_priority']}</span></td>
            <td>{round(r['recommendation_score'],1)}</td>
            <td>{r['investment_category']}</td>
            <td>{r['recommended_action']}</td>
            <td>£{round(r['indicative_investment_cost_gbp']/1_000_000,2)}m</td>
            <td>£{round(r['financial_loss_gbp']/1_000_000,2)}m</td>
        </tr>
        """

    p1 = int((rec["investment_priority"] == "Priority 1").sum())
    total_cost = round(rec["indicative_investment_cost_gbp"].sum() / 1_000_000, 2)
    avoided_loss = round(rec["financial_loss_gbp"].sum() / 1_000_000, 2)

    content = f"""
    <div class="grid">
        {metric_card("Priority 1 Areas", str(p1), "highest urgency")}
        {metric_card("Top Score", str(round(rec['recommendation_score'].max(),1)), "postcode investment score")}
        {metric_card("Indicative Programme Cost", f"£{total_cost}m", "transparent prototype estimate")}
        {metric_card("Exposed Financial Loss", f"£{avoided_loss}m", "potential avoided-loss pool")}
    </div>

    <div class="card">
        <h2>Prioritised investment recommendations</h2>
        <p style="color:#94a3b8;">
            Ranking combines risk, social vulnerability, low resilience, ENS, outage concentration and financial-loss exposure.
            The action column translates the score into practical resilience investments.
        </p>
        <table>
            <tr>
                <th>Rank</th>
                <th>Postcode</th>
                <th>Nearest place</th>
                <th>Priority</th>
                <th>Score</th>
                <th>Category</th>
                <th>Recommended action</th>
                <th>Indicative cost</th>
                <th>Loss exposure</th>
            </tr>
            {rows}
        </table>
    </div>
    """

    return render_app(
        "Prioritised Investment Recommendations",
        "Ranked postcode-level investment actions based on resilience, risk, social vulnerability, ENS and financial-loss exposure.",
        content,
    )


def page_finance(region, scenario, layer, places, outages, grid):
    rows = ""
    for _, r in places.sort_values("total_financial_loss_gbp", ascending=False).iterrows():
        rows += f"""
        <tr>
            <td><strong>{r['place']}</strong></td>
            <td>{round(r['energy_not_supplied_mw'],1)} MW</td>
            <td>{round(r['ens_mwh'],1)} MWh</td>
            <td>{round(r['estimated_duration_hours'],1)} h</td>
            <td>£{round(r['voll_loss_gbp']/1_000_000,2)}m</td>
            <td>£{round(r['customer_interruption_loss_gbp']/1_000_000,2)}m</td>
            <td>£{round(r['business_disruption_loss_gbp']/1_000_000,2)}m</td>
            <td>£{round(r['restoration_loss_gbp']/1_000_000,2)}m</td>
            <td>£{round(r['critical_services_loss_gbp']/1_000_000,2)}m</td>
            <td><strong>£{round(r['total_financial_loss_gbp']/1_000_000,2)}m</strong></td>
        </tr>
        """

    total_loss = round(places["total_financial_loss_gbp"].sum() / 1_000_000, 2)
    total_ens = round(places["ens_mwh"].sum(), 1)

    content = f"""
    <div class="grid">
        {metric_card("Total Financial Loss", f"£{total_loss}m", "regional scenario estimate")}
        {metric_card("ENS", f"{total_ens} MWh", "energy not supplied over duration")}
        {metric_card("Largest Local Loss", f"£{round(places['total_financial_loss_gbp'].max()/1_000_000,2)}m", "highest local impact")}
        {metric_card("MC P95 Loss", f"£{round(places['mc_financial_loss_p95'].max()/1_000_000,2)}m", "uncertainty worst case")}
    </div>

    <div class="card">
        <h2>Financial loss breakdown</h2>
        <table>
            <tr>
                <th>Place</th>
                <th>ENS MW</th>
                <th>ENS MWh</th>
                <th>Duration</th>
                <th>VoLL loss</th>
                <th>Customer loss</th>
                <th>Business loss</th>
                <th>Restoration</th>
                <th>Critical services</th>
                <th>Total</th>
            </tr>
            {rows}
        </table>
    </div>
    """

    return render_app(
        "Financial Loss Model",
        "Estimated loss includes energy not supplied, value of lost load, customer interruption, business disruption, critical services and restoration cost.",
        content,
    )


def page_montecarlo(region, scenario, layer, places, outages, grid):
    worst = places.sort_values("mc_p95", ascending=False).iloc[0]
    hist = histogram(worst["mc_histogram"])

    rows = ""
    for _, r in places.sort_values("mc_p95", ascending=False).iterrows():
        rows += f"""
        <tr>
            <td><strong>{r['place']}</strong></td>
            <td>{r['mc_mean']}</td>
            <td>{r['mc_std']}</td>
            <td>{r['mc_p05']}</td>
            <td>{r['mc_p50']}</td>
            <td>{r['mc_p95']}</td>
            <td>{round(r['mc_extreme_probability']*100,1)}%</td>
            <td>£{round(r['mc_financial_loss_p95']/1_000_000,2)}m</td>
        </tr>
        """

    content = f"""
    <div class="grid">
        {metric_card("Worst P95 Risk", f"{worst['mc_p95']}", worst["place"])}
        {metric_card("Extreme Probability", f"{round(worst['mc_extreme_probability']*100,1)}%", "P(risk ≥ 80)")}
        {metric_card("P95 Financial Loss", f"£{round(worst['mc_financial_loss_p95']/1_000_000,2)}m", "uncertainty loss")}
        {metric_card("Mean MC Resilience", f"{worst['mc_resilience_mean']}", "uncertainty resilience")}
    </div>

    <div class="grid-2">
        <div class="card">
            <h2>Monte Carlo histogram — {worst['place']}</h2>
            {hist}
        </div>

        <div class="card">
            <h2>Improved Monte Carlo logic</h2>
            <p>Randomly perturbs wind, rain, temperature, AQI, solar radiation, cloud cover and ENS.</p>
            <div class="alert">P95 represents high-end risk under uncertainty, not a single deterministic forecast.</div>
        </div>
    </div>

    <div class="card">
        <h2>Monte Carlo table</h2>
        <table>
            <tr>
                <th>Place</th>
                <th>Mean</th>
                <th>Std</th>
                <th>P05</th>
                <th>P50</th>
                <th>P95</th>
                <th>Extreme prob.</th>
                <th>P95 loss</th>
            </tr>
            {rows}
        </table>
    </div>
    """

    return render_app("Monte Carlo", "Advanced uncertainty simulation with risk, resilience and financial loss.", content)


def page_satellite(region, scenario, layer, places, outages, grid):
    content = """
    <div class="card">
        <h2>NASA Satellite Storyboard</h2>
        <p style="color:#94a3b8;">
            Four-panel satellite visual narrative: pre-event, disturbance, peak failure and recovery.
        </p>
    </div>

    <div class="grid-2">
        <div class="card"><h3>(a) Pre-event</h3><div id="story0" class="small-map"></div></div>
        <div class="card"><h3>(b) Disturbance</h3><div id="story1" class="small-map"></div></div>
        <div class="card"><h3>(c) Peak failure</h3><div id="story2" class="small-map"></div></div>
        <div class="card"><h3>(d) Recovery</h3><div id="story3" class="small-map"></div></div>
    </div>

    <div class="card">
        <h2>Suggested caption</h2>
        <p>
            Spatiotemporal evolution of grid risk over the selected region using NASA GIBS imagery,
            Northern Powergrid outage markers, postcode information and scenario-adjusted hazard overlays.
        </p>
    </div>
    """

    return render_app("NASA Storyboard", "NASA GIBS layers with outage and risk overlays.", content, nasa_storyboard_script("story"))


def page_data(region, scenario, layer, places, outages, grid):
    place_rows = ""
    for _, r in places.iterrows():
        place_rows += f"""
        <tr>
            <td>{r['place']}</td>
            <td>{round(r['wind_speed_10m'],2)}</td>
            <td>{round(r['precipitation'],2)}</td>
            <td>{round(r['european_aqi'],2)}</td>
            <td>{round(r['imd_score'],2)}</td>
            <td>{r['imd_source']}</td>
            <td>{r['imd_match']}</td>
            <td>{round(r['social_vulnerability'],2)}</td>
            <td>{round(r['energy_not_supplied_mw'],2)}</td>
            <td>{round(r['total_financial_loss_gbp'],2)}</td>
            <td>{round(r['final_risk_score'],2)}</td>
            <td>{round(r['resilience_index'],2)}</td>
        </tr>
        """

    outage_rows = ""
    for _, r in outages.head(100).iterrows():
        outage_rows += f"""
        <tr>
            <td>{r['outage_reference']}</td>
            <td>{r['outage_status']}</td>
            <td>{r['postcode_label']}</td>
            <td>{int(r['affected_customers'])}</td>
            <td>{round(r['latitude'],4)}</td>
            <td>{round(r['longitude'],4)}</td>
        </tr>
        """

    content = f"""
    <div class="card">
        <h2>Place-level data</h2>
        <table>
            <tr>
                <th>Place</th>
                <th>Wind</th>
                <th>Rain</th>
                <th>AQI</th>
                <th>IoD score</th>
                <th>IoD source</th>
                <th>IoD match</th>
                <th>Social vuln.</th>
                <th>ENS MW</th>
                <th>Financial loss</th>
                <th>Risk</th>
                <th>Resilience</th>
            </tr>
            {place_rows}
        </table>
    </div>

    <div class="card">
        <h2>Northern Powergrid outage data</h2>
        <table>
            <tr>
                <th>Reference</th>
                <th>Status</th>
                <th>Postcode</th>
                <th>Customers</th>
                <th>Lat</th>
                <th>Lon</th>
            </tr>
            {outage_rows}
        </table>
    </div>
    """

    return render_app("Data", "Model inputs and outputs.", content)


@app.route("/download")
def download_csv():
    region, scenario, layer, page, mc = get_controls()
    places, outages, grid = get_data(region, scenario, mc)
    csv = places.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=grid_digital_twin_places.csv"},
    )


if __name__ == "__main__":
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    print("Website is running at: http://localhost:5000")
    app.run(debug=True)
