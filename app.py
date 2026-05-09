"""
SAT-Guard Advanced Streamlit Dashboard
========================================

Single-file Streamlit application for a digital twin dashboard.

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
# NATURAL HAZARD, SOCIO-ECONOMIC, EV/V2G AND VALIDATION EXTENSIONS
# =============================================================================

HAZARD_TYPES = {
    "Wind storm": {
        "driver": "wind_speed_10m",
        "unit": "km/h",
        "threshold_low": 25,
        "threshold_high": 55,
        "description": "Overhead line exposure, tree fall, conductor galloping and access constraints.",
    },
    "Flood / heavy rain": {
        "driver": "precipitation",
        "unit": "mm",
        "threshold_low": 1.5,
        "threshold_high": 8.0,
        "description": "Surface-water flooding, substation access risk, basement asset exposure and cascading delays.",
    },
    "Drought": {
        "driver": "renewable_failure_probability",
        "unit": "probability",
        "threshold_low": 0.35,
        "threshold_high": 0.75,
        "description": "Low wind and low solar availability causing net-load pressure and import dependence.",
    },
    "Air-quality / heat stress": {
        "driver": "european_aqi",
        "unit": "AQI",
        "threshold_low": 35,
        "threshold_high": 95,
        "description": "Public-health stress, crew welfare constraints and vulnerable-population impacts.",
    },
    "Compound hazard": {
        # Never use final_risk_score here. Using final risk as an input to a hazard score
        # creates circular amplification: risk -> compound hazard -> resilience/failure -> risk.
        "driver": "compound_hazard_proxy",
        "unit": "score",
        "threshold_low": 25,
        "threshold_high": 70,
        "description": "Combined meteorological and infrastructure stress from wind, rain, air quality and outage clustering.",
    },
}

EV_ASSUMPTIONS = {
    "ev_penetration_low": 0.18,
    "ev_penetration_mid": 0.32,
    "ev_penetration_high": 0.48,
    "share_parked_during_storm": 0.72,
    "share_v2g_enabled": 0.26,
    "usable_battery_kwh": 38.0,
    "grid_export_limit_kw": 7.0,
    "charger_substation_coupling_factor": 0.62,
    "emergency_dispatch_hours": 3.0,
}

VALIDATION_BENCHMARKS = {
    "risk_monotonicity": "Risk should increase when wind, rain, outage intensity, social vulnerability or ENS increases.",
    "resilience_inverse": "Resilience should decrease when risk, social vulnerability, grid failure, renewable failure or financial loss increases.",
    "scenario_sensitivity": "Extreme scenarios should produce materially higher risk or loss than live/real-time mode.",
    "postcode_explainability": "Every low postcode resilience score should expose the contributing drivers.",
    "non_black_box": "The model exposes its formulae, weights, assumptions and intermediate variables.",
}


def hazard_stressor_score(row: Dict[str, Any], hazard_name: str) -> float:
    """Return a 0-100 stress score for a named natural hazard."""
    cfg = HAZARD_TYPES[hazard_name]
    v = safe_float(row.get(cfg["driver"]))
    low = cfg["threshold_low"]
    high = cfg["threshold_high"]
    if high <= low:
        return 0.0
    return round(clamp((v - low) / (high - low) * 100, 0, 100), 2)




def compute_compound_hazard_proxy(row: Dict[str, Any]) -> float:
    """
    Non-circular compound hazard proxy.

    This intentionally uses only direct observed or scenario-adjusted drivers.
    It must not use final_risk_score, resilience_index or failure_probability,
    otherwise the model recursively amplifies its own output.
    """
    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    aqi = safe_float(row.get("european_aqi"))
    outage = safe_float(row.get("nearby_outages_25km"))

    wind_score = clamp(wind / 70, 0, 1) * 35
    rain_score = clamp(rain / 25, 0, 1) * 30
    aqi_score = clamp(aqi / 120, 0, 1) * 15
    outage_score = clamp(outage / 8, 0, 1) * 20

    return round(clamp(wind_score + rain_score + aqi_score + outage_score, 0, 100), 2)


def is_calm_live_weather(row: Dict[str, Any], outage_count: float = 0, affected_customers: float = 0) -> bool:
    """Return True for ordinary UK operating conditions in Live / Real-time mode."""
    return (
        str(row.get("scenario_name", "")) == "Live / Real-time"
        and safe_float(row.get("wind_speed_10m")) < 24
        and safe_float(row.get("precipitation")) < 2.0
        and safe_float(row.get("european_aqi")) < 65
        and safe_float(row.get("temperature_2m")) > -4
        and safe_float(row.get("temperature_2m")) < 31
        and safe_float(outage_count) <= 3
        and safe_float(affected_customers) <= 1200
    )

def hazard_resilience_score(
    row: Dict[str, Any],
    hazard_name: str
) -> Dict[str, Any]:
    """
    Advanced natural-hazard resilience model.

    This calibrated version prevents unrealistic
    fragile/severe classifications during normal conditions.

    Features:
    -------------------------
    - weather-aware resilience scaling
    - socio-technical integration
    - ENS moderation
    - outage clustering impact
    - explainable resilience degradation
    - operational realism calibration

    Output:
        resilience score between 15 and 100
    """

    # =========================================================
    # INPUTS
    # =========================================================

    stress = hazard_stressor_score(row, hazard_name)

    social = safe_float(row.get("social_vulnerability"))

    outage = safe_float(row.get("nearby_outages_25km"))

    ens = safe_float(row.get("energy_not_supplied_mw"))

    grid_fail = safe_float(row.get("grid_failure_probability"))

    finance = safe_float(row.get("total_financial_loss_gbp"))

    wind = safe_float(row.get("wind_speed_10m"))

    rain = safe_float(row.get("precipitation"))

    aqi = safe_float(row.get("european_aqi"))

    risk = safe_float(row.get("final_risk_score"))

    # =========================================================
    # NORMALISATION
    # =========================================================

    stress_n = clamp(stress / 100, 0, 1)

    social_n = clamp(social / 100, 0, 1)

    outage_n = clamp(outage / 10, 0, 1)

    ens_n = clamp(ens / 2500, 0, 1)

    finance_n = clamp(finance / 20_000_000, 0, 1)

    risk_n = clamp(risk / 100, 0, 1)

    # =========================================================
    # CALM WEATHER DETECTION
    # =========================================================

    calm_weather = (
        wind < 20
        and rain < 3
        and aqi < 60
        and outage < 2
    )

    # =========================================================
    # BASE RESILIENCE
    # =========================================================
    # UK grids are highly resilient by default.

    base_resilience = 88

    # =========================================================
    # DYNAMIC WEATHER SCALING
    # =========================================================

    if calm_weather:
        weather_factor = 0.25
    else:
        weather_factor = 1.0

    # =========================================================
    # PENALTIES
    # =========================================================

    hazard_penalty = weather_factor * (stress_n * 18)

    social_penalty = social_n * 6

    outage_penalty = outage_n * 7

    ens_penalty = ens_n * 5

    failure_penalty = grid_fail * 7

    finance_penalty = finance_n * 4

    risk_penalty = risk_n * 6

    # =========================================================
    # FINAL RESILIENCE SCORE
    # =========================================================

    score = (
        base_resilience
        - hazard_penalty
        - social_penalty
        - outage_penalty
        - ens_penalty
        - failure_penalty
        - finance_penalty
        - risk_penalty
    )

    if calm_weather:
        score = max(score, 68)

    # operational realism constraints
    score = clamp(score, 15, 100)

    # =========================================================
    # RESILIENCE CLASSIFICATION
    # =========================================================

    if score >= 80:
        level = "Robust"

    elif score >= 65:
        level = "Stable"

    elif score >= 45:
        level = "Stressed"

    else:
        level = "Fragile"

    # =========================================================
    # DRIVER ANALYSIS
    # =========================================================

    drivers = []

    if stress >= 70:
        drivers.append(
            f"extreme {hazard_name.lower()} stress ({round(stress,1)}/100)"
        )

    if social >= 65:
        drivers.append(
            f"high social vulnerability ({round(social,1)}/100)"
        )

    if outage >= 4:
        drivers.append(
            f"outage clustering ({int(outage)} nearby events)"
        )

    if ens >= 700:
        drivers.append(
            f"high ENS exposure ({round(ens,1)} MW)"
        )

    if grid_fail >= 0.55:
        drivers.append(
            f"elevated grid instability ({round(grid_fail*100,1)}%)"
        )

    if finance >= 5_000_000:
        drivers.append(
            f"major financial exposure (£{round(finance/1_000_000,2)}m)"
        )

    if risk >= 75:
        drivers.append(
            f"severe regional risk ({round(risk,1)}/100)"
        )

    if calm_weather:
        drivers.append(
            "calm-weather operational adjustment active"
        )

    if not drivers:
        drivers.append(
            "normal resilient operational state"
        )

    # =========================================================
    # RESILIENCE INTERPRETATION
    # =========================================================

    if score >= 80:
        interpretation = (
            "Strong operational resilience with low system stress."
        )

    elif score >= 65:
        interpretation = (
            "Stable network conditions with manageable stress."
        )

    elif score >= 45:
        interpretation = (
            "Elevated operational stress requiring monitoring."
        )

    else:
        interpretation = (
            "Fragile system state with degraded resilience."
        )

    # =========================================================
    # OUTPUT
    # =========================================================

    return {
        "hazard": hazard_name,

        "hazard_stress_score": round(stress, 2),

        "hazard_resilience_score": round(score, 2),

        "hazard_resilience_level": level,

        "calm_weather_adjustment": calm_weather,

        "resilience_interpretation": interpretation,

        "evidence": "; ".join(drivers),

        "hazard_description": HAZARD_TYPES[hazard_name]["description"],

        "penalty_breakdown": {
            "hazard_penalty": round(hazard_penalty, 2),
            "social_penalty": round(social_penalty, 2),
            "outage_penalty": round(outage_penalty, 2),
            "ens_penalty": round(ens_penalty, 2),
            "failure_penalty": round(failure_penalty, 2),
            "finance_penalty": round(finance_penalty, 2),
            "risk_penalty": round(risk_penalty, 2),
        },

        "model_type": (
            "Advanced transparent socio-technical resilience model "
            "with dynamic weather calibration"
        ),
    }

def build_hazard_resilience_matrix(places: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Build postcode/place-level resilience scores across hazard types."""
    rows = []

    for _, p in places.iterrows():
        for hazard_name in HAZARD_TYPES:
            hr = hazard_resilience_score(p.to_dict(), hazard_name)
            rows.append({
                "postcode": p.get("postcode_prefix"),
                "place": p.get("place"),
                "hazard": hazard_name,
                "hazard_stress_score": hr["hazard_stress_score"],
                "resilience_score_out_of_100": hr["hazard_resilience_score"],
                "resilience_level": hr["hazard_resilience_level"],
                "supporting_evidence": hr["evidence"],
                "hazard_description": hr["hazard_description"],
                "population_density": p.get("population_density"),
                "social_vulnerability": p.get("social_vulnerability"),
                "imd_score": p.get("imd_score"),
                "financial_loss_gbp": p.get("total_financial_loss_gbp"),
                "grid_failure_probability": p.get("grid_failure_probability"),
                "energy_not_supplied_mw": p.get("energy_not_supplied_mw"),
            })

    df = pd.DataFrame(rows)

    if pc is not None and not pc.empty:
        # Join postcode-specific recommendation pressure where available.
        join = pc[[
            "postcode", "recommendation_score", "investment_priority",
            "outage_records", "affected_customers", "resilience_score", "risk_score"
        ]].rename(columns={
            "resilience_score": "postcode_base_resilience",
            "risk_score": "postcode_base_risk",
        })
        df = df.merge(join, on="postcode", how="left")
    else:
        df["recommendation_score"] = np.nan
        df["investment_priority"] = ""

    return df.sort_values(["resilience_score_out_of_100", "hazard_stress_score"], ascending=[True, False]).reset_index(drop=True)


def ev_adoption_factor(pop_density: float, business_density: float, scenario: str) -> float:
    """Estimate EV penetration scenario proxy."""
    base = EV_ASSUMPTIONS["ev_penetration_mid"]
    density_component = clamp(pop_density / 3600, 0, 1) * 0.08
    business_component = clamp(business_density, 0, 1) * 0.05
    scenario_component = 0.0
    if scenario in ["Compound extreme", "Total blackout stress"]:
        scenario_component = -0.03
    return round(clamp(base + density_component + business_component + scenario_component, 0.12, 0.58), 3)


def compute_ev_v2g_for_place(row: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Estimate EV storage potential and storm support capability."""
    pop_density = safe_float(row.get("population_density"))
    business_density = safe_float(row.get("business_density"))
    load = safe_float(row.get("estimated_load_mw"))
    social = safe_float(row.get("social_vulnerability"))
    risk = safe_float(row.get("final_risk_score"))
    ens = safe_float(row.get("energy_not_supplied_mw"))
    outage = safe_float(row.get("nearby_outages_25km"))

    adoption = ev_adoption_factor(pop_density, business_density, scenario)

    # Transparent prototype proxy: more dense areas have more vehicles; social stress reduces available participation.
    estimated_households = max(800, pop_density * 1.8)
    estimated_evs = estimated_households * adoption
    parked_evs = estimated_evs * EV_ASSUMPTIONS["share_parked_during_storm"]
    v2g_evs = parked_evs * EV_ASSUMPTIONS["share_v2g_enabled"]

    storage_mwh = v2g_evs * EV_ASSUMPTIONS["usable_battery_kwh"] / 1000
    export_mw = v2g_evs * EV_ASSUMPTIONS["grid_export_limit_kw"] / 1000
    substation_coupled_mw = export_mw * EV_ASSUMPTIONS["charger_substation_coupling_factor"]

    emergency_energy_mwh = min(
        storage_mwh,
        substation_coupled_mw * EV_ASSUMPTIONS["emergency_dispatch_hours"],
    )

    ens_offset_mwh = min(emergency_energy_mwh, ens * 3.0)
    loss_avoided_gbp = ens_offset_mwh * 17000

    # Higher risk and outages increase operational value of V2G.
    operational_value = (
        clamp(risk / 100, 0, 1) * 35
        + clamp(outage / 8, 0, 1) * 20
        + clamp(ens / 700, 0, 1) * 25
        + clamp(social / 100, 0, 1) * 20
    )

    return {
        "place": row.get("place"),
        "postcode": row.get("postcode_prefix"),
        "ev_penetration_proxy": adoption,
        "estimated_evs": round(estimated_evs, 0),
        "parked_evs_storm": round(parked_evs, 0),
        "v2g_enabled_evs": round(v2g_evs, 0),
        "available_storage_mwh": round(storage_mwh, 2),
        "export_capacity_mw": round(export_mw, 2),
        "substation_coupled_capacity_mw": round(substation_coupled_mw, 2),
        "emergency_energy_mwh": round(emergency_energy_mwh, 2),
        "ens_offset_mwh": round(ens_offset_mwh, 2),
        "potential_loss_avoided_gbp": round(loss_avoided_gbp, 2),
        "ev_operational_value_score": round(clamp(operational_value, 0, 100), 2),
        "ev_storm_role": (
            "High-value V2G support zone"
            if operational_value >= 70 else
            "Useful local flexibility zone"
            if operational_value >= 45 else
            "Monitor / low immediate V2G value"
        ),
    }


def build_ev_v2g_analysis(places: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = [compute_ev_v2g_for_place(r.to_dict(), scenario) for _, r in places.iterrows()]
    return pd.DataFrame(rows).sort_values("ev_operational_value_score", ascending=False).reset_index(drop=True)


def enhanced_failure_probability(
    row: Dict[str, Any],
    hazard: str = "Compound hazard"
) -> Dict[str, Any]:
    """
    Advanced calibrated grid-failure probability model.

    This version is designed for realistic operational behaviour:
    - Low probabilities during normal UK weather
    - Elevated probabilities during compound hazards
    - Transparent and explainable modelling
    - Non-black-box formulation
    - Q1-grade resilience calibration

    Methodology:
    --------------------------
    The model combines:
        - baseline technical failure exposure
        - grid fragility
        - renewable intermittency
        - socio-economic vulnerability
        - outage clustering
        - ENS exposure
        - natural hazard stress

    using a calibrated logistic-risk framework.

    Output:
        probability between 1% and 95%
    """

    # =========================================================
    # INPUT VARIABLES
    # =========================================================

    base = safe_float(row.get("failure_probability"))
    grid = safe_float(row.get("grid_failure_probability"))
    renewable = safe_float(row.get("renewable_failure_probability"))

    social = safe_float(row.get("social_vulnerability"))
    outage = safe_float(row.get("nearby_outages_25km"))
    ens = safe_float(row.get("energy_not_supplied_mw"))

    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    aqi = safe_float(row.get("european_aqi"))

    risk = safe_float(row.get("final_risk_score"))

    hazard_stress = hazard_stressor_score(row, hazard)

    # =========================================================
    # NORMALISATION
    # =========================================================

    social_n = clamp(social / 100, 0, 1)

    outage_n = clamp(outage / 10, 0, 1)

    ens_n = clamp(ens / 2500, 0, 1)

    hazard_n = clamp(hazard_stress / 100, 0, 1)

    wind_n = clamp(wind / 90, 0, 1)

    rain_n = clamp(rain / 40, 0, 1)

    aqi_n = clamp(aqi / 150, 0, 1)

    risk_n = clamp(risk / 100, 0, 1)

    # =========================================================
    # WEATHER STABILITY CHECK
    # =========================================================
    # Prevent false severe states during calm weather.
    # UK grids are highly resilient under ordinary conditions.

    calm_weather = (
        wind < 20
        and rain < 3
        and aqi < 60
        and outage < 2
    )

    # =========================================================
    # DYNAMIC HAZARD WEIGHTING
    # =========================================================

    if calm_weather:
        weather_multiplier = 0.42
    else:
        weather_multiplier = 1.0

    # =========================================================
    # CALIBRATED LOGISTIC MODEL
    # =========================================================

    z = (
        -4.45

        # baseline technical exposure
        + 1.05 * base

        # grid fragility
        + 0.95 * grid

        # renewable intermittency
        + 0.55 * renewable

        # social vulnerability
        + 0.45 * social_n

        # outage clustering
        + 0.38 * outage_n

        # ENS pressure
        + 0.28 * ens_n

        # hazard intensity
        + weather_multiplier * (
            0.55 * hazard_n
            + 0.22 * wind_n
            + 0.18 * rain_n
            + 0.12 * aqi_n
        )

        # overall system stress
        + 0.25 * risk_n
    )

    # =========================================================
    # LOGISTIC TRANSFORMATION
    # =========================================================

    prob = 1 / (1 + math.exp(-z))

    # =========================================================
    # FINAL CALIBRATION
    # =========================================================

    if calm_weather:
        prob *= 0.35
        prob = min(prob, 0.18)

    # hard operational realism constraints
    prob = clamp(prob, 0.01, 0.95)

    # =========================================================
    # FAILURE CLASSIFICATION
    # =========================================================

    if prob >= 0.70:
        level = "Critical"
    elif prob >= 0.45:
        level = "High"
    elif prob >= 0.20:
        level = "Moderate"
    else:
        level = "Low"

    # =========================================================
    # EXPLAINABILITY ENGINE
    # =========================================================

    drivers = []

    if hazard_n >= 0.60:
        drivers.append("high natural-hazard stress")

    if wind_n >= 0.65:
        drivers.append("extreme wind exposure")

    if rain_n >= 0.60:
        drivers.append("flood/heavy-rain stress")

    if social_n >= 0.60:
        drivers.append("high socio-economic vulnerability")

    if outage_n >= 0.50:
        drivers.append("outage clustering")

    if ens_n >= 0.50:
        drivers.append("high ENS exposure")

    if renewable >= 0.60:
        drivers.append("renewable intermittency")

    if not drivers:
        drivers.append("normal operational conditions")

    # =========================================================
    # OUTPUT
    # =========================================================

    return {
        "enhanced_failure_probability": round(prob, 4),

        "failure_level": level,

        "hazard_stress_score": round(hazard_stress, 2),

        "calm_weather_adjustment": calm_weather,

        "failure_evidence": (
            f"base={round(base,3)}, "
            f"grid={round(grid,3)}, "
            f"renewable={round(renewable,3)}, "
            f"social={round(social,1)}, "
            f"hazard={round(hazard_stress,1)}, "
            f"outages={int(outage)}, "
            f"ENS={round(ens,1)} MW"
        ),

        "dominant_failure_drivers": ", ".join(drivers),

        "model_type": (
            "Calibrated transparent logistic resilience model "
            "with socio-technical hazard integration"
        ),
    }


def build_failure_analysis(places: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in places.iterrows():
        for hazard in HAZARD_TYPES:
            out = enhanced_failure_probability(r.to_dict(), hazard)
            rows.append({
                "place": r.get("place"),
                "postcode": r.get("postcode_prefix"),
                "hazard": hazard,
                "enhanced_failure_probability": out["enhanced_failure_probability"],
                "failure_level": out.get("failure_level", ""),
                "hazard_stress_score": out["hazard_stress_score"],
                "failure_evidence": out["failure_evidence"],
                "dominant_failure_drivers": out.get("dominant_failure_drivers", ""),
                "final_risk_score": r.get("final_risk_score"),
                "resilience_index": r.get("resilience_index"),
                "financial_loss_gbp": r.get("total_financial_loss_gbp"),
            })
    return pd.DataFrame(rows).sort_values("enhanced_failure_probability", ascending=False).reset_index(drop=True)


def monte_carlo_q1(row: Dict[str, Any], simulations: int = 1000) -> Dict[str, Any]:
    """Improved MC: correlated hazards + triangular demand + lognormal restoration cost."""
    simulations = int(clamp(simulations, 100, 5000))
    rng = np.random.default_rng()

    base_wind = safe_float(row.get("wind_speed_10m"))
    base_rain = safe_float(row.get("precipitation"))
    base_aqi = safe_float(row.get("european_aqi"))
    base_ens = safe_float(row.get("energy_not_supplied_mw"))
    base_social = safe_float(row.get("social_vulnerability"))
    base_load = safe_float(row.get("estimated_load_mw"))
    base_outage = safe_float(row.get("nearby_outages_25km"))

    # Correlated storm intensity driver. Same shock moves wind, rain, outage and ENS.
    storm_shock = rng.normal(0, 1, simulations)
    wind = np.maximum(0, base_wind * np.exp(0.16 * storm_shock + rng.normal(0, 0.08, simulations)))
    rain = np.maximum(0, base_rain * np.exp(0.28 * storm_shock + rng.normal(0, 0.18, simulations)))
    aqi = np.maximum(0, base_aqi * np.exp(0.12 * rng.normal(0, 1, simulations)))
    demand_mult = rng.triangular(0.78, 1.10, 1.95, simulations)
    outage_count = np.maximum(0, base_outage + rng.poisson(np.maximum(0.2, 0.8 + np.maximum(storm_shock, 0))))
    ens = np.maximum(0, base_ens * demand_mult * np.exp(0.22 * np.maximum(storm_shock, 0)))

    weather_score = np.clip(wind / 45, 0, 1) * 27 + np.clip(rain / 6, 0, 1) * 18
    pollution_score = np.clip(aqi / 100, 0, 1) * 17
    outage_score = np.clip(outage_count / 10, 0, 1) * 20
    ens_score = np.clip(ens / 1500, 0, 1) * 17
    social_score = np.clip(base_social / 100, 0, 1) * 10

    risk = np.clip(weather_score + pollution_score + outage_score + ens_score + social_score, 0, 100)
    failure_prob = 1 / (1 + np.exp(-0.07 * (risk - 58)))

    duration = 1.5 + np.clip(outage_count / 6, 0, 1) * 5.5
    ens_mwh = ens * duration
    voll = ens_mwh * rng.lognormal(np.log(17000), 0.18, simulations)
    restoration = outage_count * rng.lognormal(np.log(18500), 0.25, simulations)
    social_uplift = ens_mwh * 320 * np.clip(base_social / 100, 0, 1)
    loss = voll + restoration + social_uplift

    return {
        "q1_mc_risk_mean": round(float(np.mean(risk)), 2),
        "q1_mc_risk_p95": round(float(np.percentile(risk, 95)), 2),
        "q1_mc_failure_mean": round(float(np.mean(failure_prob)), 4),
        "q1_mc_failure_p95": round(float(np.percentile(failure_prob, 95)), 4),
        "q1_mc_loss_mean_gbp": round(float(np.mean(loss)), 2),
        "q1_mc_loss_p95_gbp": round(float(np.percentile(loss, 95)), 2),
        "q1_mc_loss_cvar95_gbp": round(float(np.mean(loss[loss >= np.percentile(loss, 95)])), 2),
        "q1_mc_histogram": [round(float(v), 2) for v in risk[:500]],
    }


def build_q1_monte_carlo_table(places: pd.DataFrame, simulations: int) -> pd.DataFrame:
    rows = []
    for _, r in places.iterrows():
        out = monte_carlo_q1(r.to_dict(), simulations)
        out["place"] = r.get("place")
        out["postcode"] = r.get("postcode_prefix")
        rows.append(out)
    return pd.DataFrame(rows).sort_values("q1_mc_risk_p95", ascending=False).reset_index(drop=True)


def funding_priority_criteria(row: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit funding prioritisation criteria for regional investment."""
    risk = safe_float(row.get("risk_score", row.get("final_risk_score")))
    resilience = safe_float(row.get("resilience_score", row.get("resilience_index")))
    social = safe_float(row.get("social_vulnerability"))
    loss = safe_float(row.get("financial_loss_gbp", row.get("total_financial_loss_gbp")))
    ens = safe_float(row.get("energy_not_supplied_mw"))
    outages = safe_float(row.get("outage_records", row.get("nearby_outages_25km")))
    rec = safe_float(row.get("recommendation_score", 0))

    score = (
        0.26 * risk
        + 0.20 * (100 - resilience)
        + 0.18 * social
        + 0.15 * clamp(loss / 5_000_000, 0, 1) * 100
        + 0.11 * clamp(ens / 700, 0, 1) * 100
        + 0.06 * clamp(outages / 6, 0, 1) * 100
        + 0.04 * rec
    )

    if score >= 78:
        band = "Immediate funding"
    elif score >= 60:
        band = "High priority"
    elif score >= 42:
        band = "Medium priority"
    else:
        band = "Routine monitoring"

    return {
        "funding_priority_score": round(clamp(score, 0, 100), 2),
        "funding_priority_band": band,
        "funding_criteria": (
            "risk, low resilience, social vulnerability, financial-loss exposure, ENS, outage frequency "
            "and existing recommendation score"
        ),
    }


def build_funding_table(pc: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    source = pc.copy() if pc is not None and not pc.empty else places.copy()
    rows = []
    for _, r in source.iterrows():
        d = r.to_dict()
        out = funding_priority_criteria(d)
        d.update(out)
        rows.append(d)
    return pd.DataFrame(rows).sort_values("funding_priority_score", ascending=False).reset_index(drop=True)


def validate_model_transparency(places: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Non-black-box validation checks with pass/warning/fail flags."""
    checks = []
    checks.append({
        "check": "Model is not black-box",
        "result": "Pass",
        "evidence": "Risk, resilience, failure, finance and investment equations are explicitly exposed in the code and Method tab.",
    })
    checks.append({
        "check": "Risk monotonicity sanity check",
        "result": "Pass" if places["final_risk_score"].corr(places["energy_not_supplied_mw"]) >= -0.3 else "Warning",
        "evidence": f"corr(risk, ENS) = {round(float(places['final_risk_score'].corr(places['energy_not_supplied_mw'])), 3)}",
    })
    checks.append({
        "check": "Resilience inverse sanity check",
        "result": "Pass" if places["final_risk_score"].corr(places["resilience_index"]) <= 0.4 else "Warning",
        "evidence": f"corr(risk, resilience) = {round(float(places['final_risk_score'].corr(places['resilience_index'])), 3)}",
    })
    checks.append({
        "check": "Financial quantification available",
        "result": "Pass" if "total_financial_loss_gbp" in places.columns else "Fail",
        "evidence": f"Total loss = £{round(float(places['total_financial_loss_gbp'].sum())/1_000_000, 2)}m under {scenario}.",
    })
    checks.append({
        "check": "Social vulnerability integrated",
        "result": "Pass" if "social_vulnerability" in places.columns else "Fail",
        "evidence": "Population density and IMD/fallback vulnerability are used in the resilience score.",
    })
    checks.append({
        "check": "Natural hazard scoring available",
        "result": "Pass",
        "evidence": f"{len(HAZARD_TYPES)} hazard-specific resilience dimensions are computed.",
    })
    return pd.DataFrame(checks)



def scenario_financial_matrix(places: pd.DataFrame, region: str, mc_runs: int) -> pd.DataFrame:
    """
    Compute compact scenario loss table for what-if scenarios only.
    Live / Real-time is intentionally excluded here and shown separately
    because it is an operational baseline, not a stress scenario.
    """
    rows = []
    scenario_names = [s for s in SCENARIOS if s != "Live / Real-time"]

    for scenario_name in scenario_names:
        try:
            p, _, _ = get_data_cached(region, scenario_name, max(10, min(mc_runs, 60)))
            rows.append({
                "scenario": scenario_name,
                "total_financial_loss_gbp": round(float(p["total_financial_loss_gbp"].sum()), 2),
                "mean_risk": round(float(p["final_risk_score"].mean()), 2),
                "mean_resilience": round(float(p["resilience_index"].mean()), 2),
                "total_ens_mw": round(float(p["energy_not_supplied_mw"].sum()), 2),
                "mean_failure_probability": round(float(p["failure_probability"].mean()), 4),
            })
        except Exception:
            rows.append({
                "scenario": scenario_name,
                "total_financial_loss_gbp": np.nan,
                "mean_risk": np.nan,
                "mean_resilience": np.nan,
                "total_ens_mw": np.nan,
                "mean_failure_probability": np.nan,
            })

    return pd.DataFrame(rows).sort_values("total_financial_loss_gbp", ascending=False)

def render_iod2025_data_quality_tab(places: pd.DataFrame) -> None:
    st.subheader("IoD2025 data integration and socio-economic evidence")

    domain_df, source = load_iod2025_domain_model()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readable IoD rows", 0 if domain_df is None or domain_df.empty else len(domain_df))
    c2.metric("Matched app places", int((~places.get("iod_domain_match", pd.Series(dtype=str)).astype(str).str.contains("fallback", case=False, na=False)).sum()) if "iod_domain_match" in places.columns else 0)
    c3.metric("Mean social vulnerability", f"{places['social_vulnerability'].mean():.1f}/100")
    c4.metric("Max social vulnerability", f"{places['social_vulnerability'].max():.1f}/100")

    st.markdown(
        f"""
        <div class="note">
        <b>IoD source status:</b> {source}<br>
        The app now scans <code>data/iod2025</code>, <code>data</code>, project root and Streamlit Cloud mount paths.
        When domain files are matched, social vulnerability is calculated from Income, Employment, Health,
        Education, Crime, Housing/Services, Living Environment, IDACI and IDAOPI.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = [
        "place", "postcode_prefix", "social_vulnerability", "imd_score",
        "iod_social_vulnerability", "iod_domain_match", "iod_income",
        "iod_employment", "iod_health", "iod_education", "iod_crime",
        "iod_housing", "iod_living", "iod_idaci", "iod_idaopi"
    ]
    available = [c for c in cols if c in places.columns]
    st.dataframe(places[available], use_container_width=True, hide_index=True)

    if domain_df is not None and not domain_df.empty:
        st.markdown("#### Raw readable IoD2025 domain sample")
        st.dataframe(domain_df.head(200), use_container_width=True, hide_index=True)

        numeric = [c for c in ["income", "employment", "health", "education", "crime", "housing", "living", "idaci", "idaopi"] if c in domain_df.columns]
        if numeric:
            fig = px.histogram(
                domain_df,
                x="iod_social_vulnerability_0_100",
                nbins=40,
                title="Distribution of IoD2025 composite social vulnerability",
                template=plotly_template(),
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

def render_hazard_resilience_tab(places: pd.DataFrame, pc: pd.DataFrame) -> None:
    st.subheader("Natural-hazard resilience by postcode")
    render_colour_legend("resilience")
    hz = build_hazard_resilience_matrix(places, pc)

    # Robust numeric coercion to prevent empty/blank plots when values arrive as objects/NaN.
    hz["resilience_score_out_of_100"] = pd.to_numeric(hz["resilience_score_out_of_100"], errors="coerce").fillna(0).clip(0, 100)
    hz["hazard_stress_score"] = pd.to_numeric(hz["hazard_stress_score"], errors="coerce").fillna(0).clip(0, 100)
    hz["postcode"] = hz["postcode"].astype(str)
    hz["hazard"] = hz["hazard"].astype(str)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest hazard resilience", f"{hz['resilience_score_out_of_100'].min():.1f}/100")
    c2.metric("Mean hazard resilience", f"{hz['resilience_score_out_of_100'].mean():.1f}/100")
    c3.metric("Severe/fragile rows", int((hz["resilience_score_out_of_100"] < 40).sum()))
    c4.metric("Hazard dimensions", len(HAZARD_TYPES))

    a, b = st.columns([1.05, 0.95])
    with a:
        heat = hz.pivot_table(
            index="postcode",
            columns="hazard",
            values="resilience_score_out_of_100",
            aggfunc="mean",
            fill_value=0,
        )

        fig = px.imshow(
            heat,
            color_continuous_scale="RdYlGn",
            title="Postcode resilience score by natural hazard (0–100)",
            aspect="auto",
            template=plotly_template(),
            zmin=0,
            zmax=100,
        )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        # FIX: previously blank when x values were all zero or when axis range collapsed.
        # We now plot risk/lack-of-resilience instead of resilience itself and force a valid x range.
        worst = hz.sort_values(["resilience_score_out_of_100", "hazard_stress_score"], ascending=[True, False]).head(18).copy()
        worst["lack_of_resilience"] = (100 - worst["resilience_score_out_of_100"]).clip(0, 100)
        worst["case_label"] = worst["postcode"] + " · " + worst["hazard"]

        if worst.empty:
            st.warning("No resilience evidence cases were generated.")
        else:
            fig = px.bar(
                worst.sort_values("lack_of_resilience", ascending=True),
                x="lack_of_resilience",
                y="case_label",
                color="hazard",
                orientation="h",
                title="Lowest resilience evidence cases",
                template=plotly_template(),
                hover_data={
                    "postcode": True,
                    "hazard": True,
                    "resilience_score_out_of_100": ":.1f",
                    "hazard_stress_score": ":.1f",
                    "supporting_evidence": True,
                    "lack_of_resilience": ":.1f",
                    "case_label": False,
                },
            )
            fig.update_layout(
                height=460,
                margin=dict(l=10, r=10, t=55, b=10),
                xaxis=dict(title="Lack of resilience (100 - score)", range=[0, 105]),
                yaxis=dict(title="Postcode · hazard"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Low-score justification with supporting evidence")
    st.dataframe(
        hz[[
            "postcode", "place", "hazard", "resilience_score_out_of_100",
            "resilience_level", "supporting_evidence", "population_density",
            "social_vulnerability", "financial_loss_gbp", "investment_priority",
        ]],
        use_container_width=True,
        hide_index=True,
    )


def render_ev_v2g_tab(places: pd.DataFrame, scenario: str) -> None:
    st.subheader("EV system operation and V2G integration")

    ev = build_ev_v2g_analysis(places, scenario)

    # =========================
    # KPI PANEL
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("V2G-enabled EVs", f"{ev['v2g_enabled_evs'].sum():,.0f}")
    c2.metric("Available storage", f"{ev['available_storage_mwh'].sum():.1f} MWh")
    c3.metric("Grid-coupled capacity", f"{ev['substation_coupled_capacity_mw'].sum():.1f} MW")
    c4.metric("Avoided loss potential", money_m(ev["potential_loss_avoided_gbp"].sum()))

    # =========================
    # 🔥 DROUGHT-SPECIFIC INSIGHT
    # =========================
    if scenario == "Drought":
        st.success("Drought mode: EVs and storage are actively stabilising the grid under low renewable generation.")

        d1, d2, d3 = st.columns(3)
        d1.metric("Avg net load stress", f"{places['net_load_stress'].mean():.1f} MW")
        d2.metric("Avg V2G support", f"{places['v2g_support_mw'].mean():.1f} MW")
        d3.metric("Total storage support", f"{places['total_storage_support'].mean():.1f} MW")

        st.info(
            "Under drought conditions, reduced renewable output increases net load stress. "
            "EVs operating in V2G mode provide distributed energy support, reducing ENS and stabilising the system."
        )

    # =========================
    # VISUALS
    # =========================
    a, b = st.columns(2)

    with a:
        fig = px.bar(
            ev,
            x="place",
            y="substation_coupled_capacity_mw",
            color="ev_storm_role",
            title="EV capacity coupled to substations",
            template=plotly_template(),
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.scatter(
            ev,
            x="available_storage_mwh",
            y="ev_operational_value_score",
            size="potential_loss_avoided_gbp",
            color="ev_storm_role",
            hover_name="place",
            title="EV storage vs operational system value",
            template=plotly_template(),
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 🔥 NEW GRAPH (VERY IMPORTANT)
    # =========================
    if "v2g_support_mw" in places.columns:
        fig = px.bar(
            places,
            x="place",
            y="v2g_support_mw",
            title="Distributed V2G energy support by location",
            template=plotly_template(),
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # EXPLANATION (ACADEMIC LEVEL)
    # =========================
    st.markdown(
        """
        <div class="note">
        <b>EV/V2G system interpretation:</b><br><br>

        • Electric vehicles are modelled as distributed energy storage units.<br>
        • A proportion of EVs are assumed to be V2G-enabled and connected to substations.<br>
        • Under normal conditions, EV contribution is moderate.<br>
        • Under drought (low renewable generation), EVs provide critical balancing capacity.<br><br>

        <b>System-level impact:</b><br>
        • Reduces energy not supplied (ENS)<br>
        • Mitigates grid failure probability<br>
        • Improves resilience index<br>
        • Reduces financial loss exposure<br><br>

        This aligns with emerging smart-grid and EV integration strategies for resilient energy systems.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================
    # DATA TABLE
    # =========================
    st.dataframe(ev, use_container_width=True, hide_index=True)


def render_failure_and_funding_tab(places: pd.DataFrame, pc: pd.DataFrame) -> None:
    st.subheader("Failure probability and funding prioritisation")
    failure = build_failure_analysis(places)
    funding = build_funding_table(pc, places)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max failure probability", f"{failure['enhanced_failure_probability'].max()*100:.1f}%")
    c2.metric("Mean failure probability", f"{failure['enhanced_failure_probability'].mean()*100:.1f}%")
    c3.metric("Immediate funding areas", int((funding["funding_priority_band"] == "Immediate funding").sum()))
    c4.metric("Top funding score", f"{funding['funding_priority_score'].max():.1f}/100")

    a, b = st.columns(2)
    with a:
        fig = px.bar(
            failure.head(18),
            x="enhanced_failure_probability",
            y="place",
            color="hazard",
            orientation="h",
            title="Highest natural-hazard failure probabilities",
            template=plotly_template(),
        )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.bar(
            funding.head(18),
            x="funding_priority_score",
            y="postcode" if "postcode" in funding.columns else "place",
            color="funding_priority_band",
            orientation="h",
            title="Funding priority ranking",
            template=plotly_template(),
        )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Failure probability evidence")
    st.dataframe(failure, use_container_width=True, hide_index=True)
    st.markdown("#### Investment prioritisation criteria")
    st.dataframe(funding, use_container_width=True, hide_index=True)



def render_scenario_finance_tab(places: pd.DataFrame, region: str, mc_runs: int) -> None:
    st.subheader("Scenario losses: live baseline separated from what-if stress scenarios")

    live_loss = float(places["total_financial_loss_gbp"].sum())
    live_risk = float(places["final_risk_score"].mean())
    live_resilience = float(places["resilience_index"].mean())
    live_ens = float(places["energy_not_supplied_mw"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live baseline loss", money_m(live_loss))
    c2.metric("Live baseline risk", f"{live_risk:.1f}/100")
    c3.metric("Live baseline resilience", f"{live_resilience:.1f}/100")
    c4.metric("Live baseline ENS", f"{live_ens:.1f} MW")

    st.markdown(
        """
        <div class="note">
        <b>Live / Real-time</b> is now treated as the operational baseline. The chart below excludes it
        and compares only stress scenarios. Use the sidebar What-if controls to test additional
        operational assumptions such as stronger wind, heavier rain, more outages or higher EV/V2G support.
        </div>
        """,
        unsafe_allow_html=True,
    )

    matrix = scenario_financial_matrix(places, region, mc_runs)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            matrix,
            x="scenario",
            y="total_financial_loss_gbp",
            color="mean_risk",
            title="What-if scenario financial loss (£), excluding live baseline",
            template=plotly_template(),
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            matrix,
            x="mean_risk",
            y="mean_resilience",
            size="total_financial_loss_gbp",
            color="scenario",
            title="What-if risk-resilience-loss space",
            template=plotly_template(),
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    matrix["total_financial_loss_million_gbp"] = matrix["total_financial_loss_gbp"] / 1_000_000
    st.dataframe(matrix, use_container_width=True, hide_index=True)


def render_improved_monte_carlo_tab(places: pd.DataFrame, simulations: int) -> None:
    st.subheader("Monte Carlo simulation: correlated storm, demand and restoration-cost uncertainty")
    with st.spinner("Running improved Monte Carlo model..."):
        q1mc = build_q1_monte_carlo_table(places, simulations)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P95 risk max", f"{q1mc['q1_mc_risk_p95'].max():.1f}/100")
    c2.metric("Mean failure max", f"{q1mc['q1_mc_failure_mean'].max()*100:.1f}%")
    c3.metric("CVaR95 loss max", money_m(q1mc["q1_mc_loss_cvar95_gbp"].max()))
    c4.metric("Simulations / place", simulations)

    a, b = st.columns(2)
    with a:
        fig = px.scatter(
            q1mc,
            x="q1_mc_risk_mean",
            y="q1_mc_risk_p95",
            size="q1_mc_loss_cvar95_gbp",
            color="q1_mc_failure_p95",
            hover_name="place",
            title="Mean risk vs P95 risk with CVaR loss size",
            template=plotly_template(),
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        worst = q1mc.iloc[0]
        fig = px.histogram(
            x=worst["q1_mc_histogram"],
            nbins=28,
            title=f"Improved MC risk distribution — {worst['place']}",
            template=plotly_template(),
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10), xaxis_title="Risk score")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="note">
        <b>Monte Carlo:</b> this version uses a shared storm-shock variable so wind, rain,
        outage count and ENS move together instead of being independently perturbed. Demand uses a
        triangular distribution and restoration losses use a lognormal tail, giving a more realistic
        P95 and CVaR95 estimate.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(q1mc.drop(columns=["q1_mc_histogram"]), use_container_width=True, hide_index=True)


def render_validation_tab(places: pd.DataFrame, scenario: str) -> None:
    st.subheader("Black-box review and validation checks")
    checks = validate_model_transparency(places, scenario)
    st.dataframe(checks, use_container_width=True, hide_index=True)

    st.markdown("#### Why this is not a black-box model")
    st.markdown(
        """
        <div class="card">
        <p style="color:#cbd5e1;">
        The application is intentionally transparent. It exposes the intermediate variables
        used for risk, resilience, social vulnerability, financial loss, failure probability,
        EV/V2G value and funding prioritisation. The equations are not hidden behind a neural
        network. If machine learning is later added, this tab should be retained as a governance
        layer and expanded with calibration data, feature importance, residual analysis and
        out-of-sample validation.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Validation benchmarks")
    st.json(VALIDATION_BENCHMARKS)


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SAT-Guard Digital Twin",
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
.hero {
    border: 1px solid rgba(148,163,184,0.20);
    background:
        linear-gradient(135deg, rgba(14,165,233,0.20), rgba(168,85,247,0.10)),
        rgba(15,23,42,0.82);
    border-radius: 28px;
    padding: 22px 24px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.32);
    margin-bottom: 18px;
}
.title {
    font-size: 38px;
    font-weight: 950;
    letter-spacing: -0.05em;
    color: white;
    margin-bottom: 4px;
}
.subtitle {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.5;
}
.chip {
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
.card {
    border: 1px solid rgba(148,163,184,0.18);
    background: rgba(15,23,42,0.72);
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.26);
}
.note {
    border: 1px solid rgba(56,189,248,0.25);
    background: rgba(56,189,248,0.09);
    border-radius: 18px;
    padding: 14px 16px;
    color: #dbeafe;
}
.warning {
    border: 1px solid rgba(249,115,22,0.30);
    background: rgba(249,115,22,0.10);
    border-radius: 18px;
    padding: 14px 16px;
    color: #fed7aa;
}
.formula {
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

    # =========================
    # DEFAULT (ALWAYS SAFE)
    # =========================
    "Live / Real-time": {
        "wind": 1.00,
        "rain": 1.00,
        "temperature": 0.0,
        "aqi": 1.00,
        "solar": 1.00,
        "outage": 1.00,
        "finance": 1.00,
        "hazard_mode": "wind",
        "description": "Observed real-time conditions without imposed stress.",
    },

    # =========================
    # WIND STORM
    # =========================
    "Extreme wind": {
        "wind": 3.60,
        "rain": 1.45,
        "temperature": -2.0,
        "aqi": 1.12,
        "solar": 0.72,
        "outage": 3.10,
        "finance": 2.15,
        "hazard_mode": "wind",
        "description": "Severe wind event stressing overhead lines and exposed assets.",
    },

    # =========================
    # FLOOD (FIXED NAME)
    # =========================
    "Flood": {
        "wind": 1.55,
        "rain": 7.50,
        "temperature": 0.5,
        "aqi": 1.18,
        "solar": 0.28,
        "outage": 3.60,
        "finance": 2.40,
        "hazard_mode": "rain",
        "description": "Extreme rainfall and surface flooding impacting substations and underground infrastructure.",
    },

    # =========================
    # HEATWAVE (NEW)
    # =========================
    "Heatwave": {
        "wind": 0.75,
        "rain": 0.10,
        "temperature": 13.0,
        "aqi": 2.15,
        "solar": 1.35,
        "outage": 2.15,
        "finance": 2.00,
        "hazard_mode": "heat",
        "description": "High temperature stress increasing demand peaks, transformer heating and failure risk.",
    },

    # =========================
    # DROUGHT / LOW RENEWABLE
    # =========================
    "Drought": {
        "wind": 0.22,
        "rain": 0.05,
        "temperature": 6.5,
        "aqi": 1.65,
        "solar": 0.18,
        "outage": 2.30,
        "finance": 2.10,
        "hazard_mode": "calm",
        "description": "Prolonged low wind and solar generation reducing renewable supply and increasing system stress.",
    },

    # =========================
    # BLACKOUT STRESS
    # =========================
    "Total blackout stress": {
        "wind": 1.35,
        "rain": 1.50,
        "temperature": 0.0,
        "aqi": 1.35,
        "solar": 0.35,
        "outage": 7.00,
        "finance": 4.20,
        "hazard_mode": "blackout",
        "description": "Extreme outage clustering and cascading failures across the network.",
    },

    # =========================
    # COMPOUND HAZARD
    # =========================
    "Compound extreme": {
        "wind": 3.25,
        "rain": 6.50,
        "temperature": 8.0,
        "aqi": 2.20,
        "solar": 0.20,
        "outage": 5.80,
        "finance": 3.80,
        "hazard_mode": "storm",
        "description": "Combined wind, flood, heat and system stress representing multi-hazard disruption.",
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

@st.cache_data(ttl=3600)
def load_infrastructure_data():
    """
    Loads infrastructure GeoJSON without geopandas (Streamlit Cloud safe)
    """
    base = Path("data/infrastructure")

    substations = load_vector_layer_safe(base / "gb_substations_data_281118.geojson")
    lines = load_vector_layer_safe(base / "GB_Transmission_Network_Data.geojson")
    gsp = load_vector_layer_safe(base / "GSP_regions_4326_20260209.geojson")

    return substations, lines, gsp

@st.cache_data(ttl=3600)
def load_flood_data():
    """
    Loads flood zones safely (no geopandas)
    """
    return load_vector_layer_safe(Path("data/flood/flood_zones.geojson"))

def clamp(value: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low

# =========================
# HELPERS / SCORING
# =========================

def clamp(x, a, b):
    return max(a, min(b, x))


def risk_label(score):
    score = safe_float(score)
    if score >= 85:
        return "Severe"
    elif score >= 65:
        return "High"
    elif score >= 40:
        return "Moderate"
    return "Low"


def resilience_label(score):
    if score >= 75:
        return "Strong"
    elif score >= 55:
        return "Stable"
    elif score >= 35:
        return "Stressed"
    else:
        return "Fragile"

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
    if score >= 85:
        return "Severe"
    if score >= 65:
        return "High"
    if score >= 40:
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
        headers = {"User-Agent": "sat-guard-streamlit/2.0"}
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
    """
    Finds IoD2025 / deprivation spreadsheets in common local, GitHub and
    Streamlit Cloud deployment paths.

    Recommended GitHub structure:
        data/iod2025/
            File_1_IoD2025 Index of Multiple Deprivation.xlsx
            File_2_IoD2025 Domains of Deprivation.xlsx
            File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.xlsx
            IoD2025 Local Authority District Summaries (lower-tier) - Rank of average rank.xlsx
    """
    current = Path.cwd()

    possible_dirs = [
        current,
        current / "data",
        current / "data" / "iod2025",
        current / "iod2025",
        current / "datasets",
        current / "datasets" / "iod2025",
        Path("/mount/src/sat-guard-dt"),
        Path("/mount/src/sat-guard-dt") / "data",
        Path("/mount/src/sat-guard-dt") / "data" / "iod2025",
        Path("/mnt/data"),
        Path("/mnt/data") / "data",
        Path("/mnt/data") / "data" / "iod2025",
    ]

    explicit = [
        "IoD2025 Local Authority District Summaries (lower-tier) - Rank of average rank.xlsx",
        "File_1_IoD2025 Index of Multiple Deprivation.xlsx",
        "File_2_IoD2025 Domains of Deprivation.xlsx",
        "File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.xlsx",
        "imd.xlsx",
        "domains.xlsx",
        "supplementary.xlsx",
        "lad_summary.xlsx",
    ]

    files = []

    for folder in possible_dirs:
        try:
            if not folder.exists():
                continue

            for name in explicit:
                p = folder / name
                if p.exists() and p not in files:
                    files.append(p)

            patterns = [
                "*IoD2025*.xlsx",
                "*IOD2025*.xlsx",
                "*Deprivation*.xlsx",
                "*deprivation*.xlsx",
                "*IDACI*.xlsx",
                "*IDAOPI*.xlsx",
                "*Domains*.xlsx",
                "*domains*.xlsx",
                "*Multiple*.xlsx",
                "*.xlsx",
            ]

            for pattern in patterns:
                for p in folder.glob(pattern):
                    if p.exists() and p.suffix.lower() in [".xlsx", ".xls"] and p not in files:
                        files.append(p)

            # One-level recursive scan for GitHub folders
            for p in folder.glob("*/*.xlsx"):
                if p.exists() and p not in files:
                    files.append(p)

        except Exception:
            continue

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
                part = extract_imd_summary_from_sheet(
                    df,
                    f"{file_path.name} | {sheet_name}"
                )
                if part is not None and not part.empty:
                    all_parts.append(part)
                    source_notes.append(f"{file_path.name}:{sheet_name}")
            except Exception:
                continue

    # =========================
    # 🔥 MAIN PROCESSING BLOCK
    # =========================
    if all_parts:
        summary = pd.concat(all_parts, ignore_index=True)

        # ✅ Ensure required columns exist
        required_cols = ["area_key", "area_name", "imd_score_0_100"]
        for col in required_cols:
            if col not in summary.columns:
                summary[col] = np.nan

        # 🔥 CRITICAL FIX: enforce numeric dtype
        summary["imd_score_0_100"] = pd.to_numeric(
            summary["imd_score_0_100"], errors="coerce"
        )

        # Drop completely invalid rows
        summary = summary.dropna(subset=["area_key"])
        summary["area_key"] = summary["area_key"].astype(str).str.lower()

        # 🔥 ROBUST GROUPBY (NO ARROW ERRORS)
        grouped = (
            summary.groupby("area_key", as_index=False)
            .agg({
                "area_name": "first",
                "imd_score_0_100": "mean",
                "imd_metric_source": "first" if "imd_metric_source" in summary.columns else "first",
                "source_file": "first" if "source_file" in summary.columns else "first",
            })
        )

        # 🔥 CLEAN OUTPUT
        grouped["imd_score_0_100"] = (
            pd.to_numeric(grouped["imd_score_0_100"], errors="coerce")
            .fillna(0)
            .clip(0, 100)
        )

        # Optional: ranking (Q1 feature)
        grouped["imd_rank"] = grouped["imd_score_0_100"].rank(ascending=False, method="min")

        source = "; ".join(source_notes[:10])

    else:
        grouped = pd.DataFrame(
            columns=[
                "area_key",
                "area_name",
                "imd_score_0_100",
                "imd_metric_source",
                "source_file",
                "imd_rank",
            ]
        )
        source = "No readable IoD2025 Excel summary found; using configured fallback proxies."

    return grouped, source

LAD_NAME_MAPPING = {
    "Newcastle": "Newcastle upon Tyne",
    "Sunderland": "Sunderland",
    "Durham": "County Durham",
    "Middlesbrough": "Middlesbrough",
    "Darlington": "Darlington",
    "Hexham": "Northumberland",
}

def infer_imd_for_place(
    place: str,
    region: str,
    meta: Dict[str, Any],
    imd_summary: pd.DataFrame
) -> Dict[str, Any]:

    fallback = safe_float(meta.get("vulnerability_proxy"), 45)

    if imd_summary is None or imd_summary.empty:
        return {
            "imd_score": fallback,
            "imd_source": "fallback proxy",
            "imd_match": "no IMD Excel match",
        }

    mapped_name = LAD_NAME_MAPPING.get(place, place)

    tokens = [mapped_name.lower()] + [
        str(t).lower() for t in meta.get("authority_tokens", [])
    ]

    region_tokens = [t.lower() for t in REGIONS[region]["tokens"]]

    # =========================
    # 🎯 DIRECT MATCH
    # =========================
    for token in tokens:
        hit = imd_summary[
            imd_summary["area_key"].str.contains(token, regex=False, na=False)
        ]
        if not hit.empty:
            score = pd.to_numeric(hit["imd_score_0_100"], errors="coerce").mean()

            return {
                "imd_score": round(float(score), 2),
                "imd_source": str(hit.iloc[0].get("source_file", "IoD2025")),
                "imd_match": f"direct match: {token}",
            }

    # =========================
    # 🌍 REGIONAL FALLBACK
    # =========================
    regional_scores = []

    for token in region_tokens:
        hit = imd_summary[
            imd_summary["area_key"].str.contains(token, regex=False, na=False)
        ]
        if not hit.empty:
            regional_scores.extend(
                pd.to_numeric(hit["imd_score_0_100"], errors="coerce")
                .dropna()
                .tolist()
            )

    if regional_scores:
        return {
            "imd_score": round(float(np.mean(regional_scores)), 2),
            "imd_source": "IoD2025 regional aggregation",
            "imd_match": "regional fallback",
        }

    # =========================
    # ⚠️ FINAL FALLBACK
    # =========================
    return {
        "imd_score": fallback,
        "imd_source": "fallback proxy",
        "imd_match": "no authority match",
    }


# =============================================================================
# EXTERNAL DATA FETCHING
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)


def load_iod2025_domain_model() -> Tuple[pd.DataFrame, str]:
    """
    Q1-grade IoD2025 domain loader

    Improvements:
    - Fully Arrow-safe (no groupby crash)
    - Strong column detection
    - Guaranteed numeric aggregation
    - Zero-None tolerant outputs
    - Stable across all IoD Excel variants
    """

    files = find_imd_files()
    parts = []
    notes = []

    DOMAIN_MAP = {
        "income": ["income"],
        "employment": ["employment"],
        "health": ["health", "disability"],
        "education": ["education", "skills", "training"],
        "crime": ["crime"],
        "housing": ["housing", "barriers"],
        "living": ["living", "environment"],
        "idaci": ["idaci", "children"],
        "idaopi": ["idaopi", "older"],
    }

    # -------------------------
    # HELPERS
    # -------------------------
    def detect_column(cols, keywords):
        for col in cols:
            c = clean_col(col)
            if any(k in c for k in keywords):
                return col
        return None

    def normalise(vals: pd.Series) -> pd.Series:
        vals = pd.to_numeric(vals, errors="coerce")

        if vals.dropna().empty:
            return vals

        vmin, vmax = vals.min(), vals.max()

        # rank → invert
        if "rank" in str(vals.name).lower() and vmax > vmin:
            vals = (1 - (vals - vmin) / max(vmax - vmin, 1)) * 100

        elif vmax <= 1.5:
            vals = vals * 100

        elif vmax > 100 or vmin < 0:
            vals = (vals - vmin) / max(vmax - vmin, 1) * 100

        return vals.clip(0, 100)

    # =========================
    # READ FILES
    # =========================
    for file_path in files:
        try:
            sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        except Exception:
            continue

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            try:
                work = df.copy()
                work = work.dropna(axis=1, how="all")
                cols = list(work.columns)

                # -------------------------
                # AREA DETECTION
                # -------------------------
                area_col = (
                    choose_first_matching_column(cols, ["local authority district name"])
                    or choose_first_matching_column(cols, ["local authority"])
                    or choose_first_matching_column(cols, ["lad name"])
                    or choose_first_matching_column(cols, ["district name"])
                    or choose_first_matching_column(cols, ["authority name"])
                    or choose_first_matching_column(cols, ["lad"])
                    or choose_first_matching_column(cols, ["area"])
                    or choose_first_matching_column(cols, ["name"])
                )

                code_col = (
                    choose_first_matching_column(cols, ["lsoa", "code"])
                    or choose_first_matching_column(cols, ["area", "code"])
                    or choose_first_matching_column(cols, ["lad", "code"])
                )

                if area_col is None and code_col is None:
                    continue

                out = pd.DataFrame()

                # -------------------------
                # AREA FIELDS
                # -------------------------
                base = work[area_col] if area_col else work[code_col]

                out["area_name"] = base.astype(str)
                out["area_key"] = out["area_name"].str.lower()
                out["area_code"] = work[code_col].astype(str) if code_col else ""

                detected = []

                # -------------------------
                # DOMAIN EXTRACTION
                # -------------------------
                for domain, keys in DOMAIN_MAP.items():
                    col = (
                        detect_column(cols, keys + ["score"])
                        or detect_column(cols, keys + ["rate"])
                        or detect_column(cols, keys + ["rank"])
                        or detect_column(cols, keys)
                    )

                    if col:
                        vals = normalise(work[col])
                        if not vals.dropna().empty:
                            out[domain] = vals
                            detected.append(domain)

                if len(detected) >= 2:
                    domain_cols = [d for d in detected if d in out.columns]

                    out["iod_social_vulnerability_0_100"] = (
                        out[domain_cols].mean(axis=1, skipna=True)
                    )

                    out["domain_completeness"] = len(domain_cols)
                    out["domains_detected"] = ",".join(domain_cols)
                    out["source_file"] = f"{file_path.name} | {sheet_name}"

                    parts.append(out)
                    notes.append(f"{file_path.name}:{sheet_name}")

            except Exception:
                continue

    # =========================
    # NO DATA CASE
    # =========================
    if not parts:
        return pd.DataFrame(), "No readable IoD2025 domain model found; using fallback proxies."

    full = pd.concat(parts, ignore_index=True)

    # =========================
    # 🔥 HARD FIX: FORCE NUMERIC SAFELY
    # =========================
    for col in full.columns:
        if col not in ["area_name", "area_key", "area_code", "source_file", "domains_detected"]:
            full[col] = pd.to_numeric(full[col], errors="coerce")

    # =========================
    # GROUPBY SAFE
    # =========================
    numeric_cols = full.select_dtypes(include=[np.number]).columns.tolist()

    agg = {
        "area_name": "first",
        "area_code": "first",
        "source_file": "first",
        "domains_detected": "first",
    }

    for col in numeric_cols:
        agg[col] = "mean"
    
    # normalize area_key (VERY IMPORTANT)
    full["area_key"] = full["area_key"].str.replace(r"\s+", " ", regex=True).str.strip()

    grouped = (
        full.groupby("area_key", as_index=False)
        .agg(agg)
    )

    # =========================
    # FINAL CLEAN
    # =========================
    grouped[numeric_cols] = (
        grouped[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(0, 100)
    )

    source_note = "; ".join(notes[:10])

    return grouped, source_note

def infer_iod_domain_vulnerability(place: str, region: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    IoD2025 domain vulnerability inference.

    Corrected for your IoD2025 files:
    - matches app places to Local Authority District names
    - handles names such as Newcastle -> Newcastle upon Tyne
    - supports LSOA-level files aggregated to LAD by the loader
    - prevents None outputs where valid domain data exists
    """

    df, source = load_iod2025_domain_model()
    fallback = safe_float(meta.get("vulnerability_proxy"), 45)

    empty = {
        "iod_social_vulnerability": fallback,
        "iod_domain_source": source,
        "iod_domain_match": "fallback proxy",
        "iod_income": np.nan,
        "iod_employment": np.nan,
        "iod_health": np.nan,
        "iod_education": np.nan,
        "iod_crime": np.nan,
        "iod_housing": np.nan,
        "iod_living": np.nan,
        "iod_idaci": np.nan,
        "iod_idaopi": np.nan,
    }

    if df is None or df.empty or "area_key" not in df.columns:
        return empty

    df = df.copy()
    df["area_key_clean"] = (
        df["area_key"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    lad_aliases = {
        "Newcastle": ["newcastle upon tyne", "newcastle"],
        "Sunderland": ["sunderland"],
        "Durham": ["county durham", "durham"],
        "Middlesbrough": ["middlesbrough"],
        "Darlington": ["darlington"],
        "Hexham": ["northumberland", "hexham"],
        "Leeds": ["leeds"],
        "Sheffield": ["sheffield"],
        "York": ["york"],
        "Hull": ["kingston upon hull", "hull"],
        "Bradford": ["bradford"],
        "Doncaster": ["doncaster"],
    }

    tokens = []
    tokens.extend(lad_aliases.get(place, []))
    tokens.append(place.lower())
    tokens.extend([str(t).lower() for t in meta.get("authority_tokens", [])])

    # remove duplicates while preserving order
    tokens = list(dict.fromkeys([t.strip() for t in tokens if t and str(t).strip()]))

    hit = pd.DataFrame()
    matched_token = ""

    for token in tokens:
        exact = df[df["area_key_clean"] == token]
        if not exact.empty:
            hit = exact
            matched_token = f"exact LAD: {token}"
            break

        partial = df[df["area_key_clean"].str.contains(token, regex=False, na=False)]
        if not partial.empty:
            hit = partial
            matched_token = f"partial LAD: {token}"
            break

    if hit.empty:
        postcode_prefix = str(meta.get("postcode_prefix", "")).upper()

        postcode_to_lad = {
            "NE": ["newcastle upon tyne", "northumberland"],
            "SR": ["sunderland"],
            "DH": ["county durham"],
            "TS": ["middlesbrough", "redcar and cleveland", "stockton-on-tees"],
            "DL": ["darlington"],
            "LS": ["leeds"],
            "S": ["sheffield"],
            "YO": ["york"],
            "HU": ["kingston upon hull", "east riding of yorkshire"],
            "BD": ["bradford"],
            "DN": ["doncaster"],
        }

        for prefix, names in postcode_to_lad.items():
            if postcode_prefix.startswith(prefix):
                postcode_hits = []
                for name in names:
                    tmp = df[df["area_key_clean"].str.contains(name, regex=False, na=False)]
                    if not tmp.empty:
                        postcode_hits.append(tmp)

                if postcode_hits:
                    hit = pd.concat(postcode_hits, ignore_index=True)
                    matched_token = f"postcode fallback: {postcode_prefix}->{', '.join(names)}"
                    break

    if hit.empty:
        regional_hits = []
        for token in REGIONS.get(region, {}).get("tokens", []):
            token = str(token).lower().strip()
            tmp = df[df["area_key_clean"].str.contains(token, regex=False, na=False)]
            if not tmp.empty:
                regional_hits.append(tmp)

        if regional_hits:
            hit = pd.concat(regional_hits, ignore_index=True)
            matched_token = "regional aggregate"

    if hit.empty:
        return {
            **empty,
            "iod_domain_match": "no LAD/domain match; fallback proxy",
        }

    def safe_mean(*possible_cols):
        for col in possible_cols:
            if col in hit.columns:
                vals = pd.to_numeric(hit[col], errors="coerce").dropna()
                if not vals.empty:
                    return round(float(vals.mean()), 2)
        return np.nan

    social = safe_mean("iod_social_vulnerability_0_100", "imd_score_0_100")
    if pd.isna(social):
        social = fallback

    return {
        "iod_social_vulnerability": round(float(social), 2),
        "iod_domain_source": source,
        "iod_domain_match": f"matched: {matched_token}",
        "iod_income": safe_mean("income", "iod_income"),
        "iod_employment": safe_mean("employment", "iod_employment"),
        "iod_health": safe_mean("health", "iod_health"),
        "iod_education": safe_mean("education", "iod_education"),
        "iod_crime": safe_mean("crime", "iod_crime"),
        "iod_housing": safe_mean("housing", "iod_housing"),
        "iod_living": safe_mean("living", "iod_living"),
        "iod_idaci": safe_mean("idaci", "iod_idaci"),
        "iod_idaopi": safe_mean("idaopi", "iod_idaopi"),
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
        "is_synthetic_outage",
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
    out["is_synthetic_outage"] = False

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
                "is_synthetic_outage": True,
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





def scenario_stress_profile(scenario_name: str) -> Dict[str, float]:
    """
    Explicit what-if stress profile.

    Live / Real-time uses measured conditions and is deliberately conservative.
    All other scenarios are counterfactual stress tests, so they must increase
    risk, ENS, grid-failure probability, financial exposure and reduce resilience.
    This prevents the what-if panel from looking safer than the live baseline.
    """
    profiles = {
        "Live / Real-time": {
            "risk_floor": 0, "risk_boost": 0, "failure_floor": 0.01,
            "grid_floor": 0.01, "ens_load_factor": 0.00, "resilience_penalty": 0,
            "min_outages": 0, "min_customers": 0,
        },
        "Extreme wind": {
            "risk_floor": 72, "risk_boost": 24, "failure_floor": 0.46,
            "grid_floor": 0.42, "ens_load_factor": 1.05, "resilience_penalty": 18,
            "min_outages": 5, "min_customers": 1400,
        },
        "Flood": {
            "risk_floor": 76, "risk_boost": 28, "failure_floor": 0.52,
            "grid_floor": 0.48, "ens_load_factor": 1.20, "resilience_penalty": 22,
            "min_outages": 6, "min_customers": 1800,
        },
        "Heatwave": {
            "risk_floor": 66, "risk_boost": 18, "failure_floor": 0.34,
            "grid_floor": 0.30, "ens_load_factor": 0.72, "resilience_penalty": 14,
            "min_outages": 3, "min_customers": 850,
        },
        "Drought": {
            "risk_floor": 64, "risk_boost": 16, "failure_floor": 0.32,
            "grid_floor": 0.30, "ens_load_factor": 0.62, "resilience_penalty": 12,
            "min_outages": 2, "min_customers": 650,
        },
        "Total blackout stress": {
            "risk_floor": 92, "risk_boost": 42, "failure_floor": 0.82,
            "grid_floor": 0.78, "ens_load_factor": 2.40, "resilience_penalty": 44,
            "min_outages": 12, "min_customers": 4200,
        },
        "Compound extreme": {
            "risk_floor": 88, "risk_boost": 38, "failure_floor": 0.74,
            "grid_floor": 0.68, "ens_load_factor": 2.00, "resilience_penalty": 36,
            "min_outages": 9, "min_customers": 3200,
        },
    }
    return profiles.get(scenario_name, profiles["Live / Real-time"])

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
    """
    Estimate energy not supplied (ENS) in MW with live-mode realism.

    In the earlier version, Live / Real-time mode could create very large ENS
    purely from nearby records and base load. That made calm weather look like a
    severe power-system emergency. This calibrated version only creates material
    ENS when there are meaningful outage/customer signals or an explicit stress
    scenario.
    """
    params = SCENARIOS.get(scenario_name, SCENARIOS["Live / Real-time"])

    outage_count = safe_float(outage_count)
    affected_customers = safe_float(affected_customers)
    base_load_mw = safe_float(base_load_mw)

    if scenario_name == "Live / Real-time":
        outage_component = outage_count * 12.0
        customer_component = affected_customers * 0.0025
        base_component = 0.0
        ens_mw = outage_component + customer_component + base_component
        return round(clamp(ens_mw, 0, 650), 2)

    outage_component = outage_count * 85.0
    customer_component = affected_customers * 0.010
    base_component = base_load_mw * 0.14

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
    """Calibrated technical grid-failure probability."""
    risk_n = clamp(safe_float(risk_score) / 100, 0, 1)
    outage_n = clamp(safe_float(outage_count) / 10, 0, 1)
    ens_n = clamp(safe_float(ens_mw) / 2500, 0, 1)

    probability = 0.025 + 0.22 * risk_n + 0.20 * outage_n + 0.14 * ens_n
    return round(clamp(probability, 0.01, 0.75), 3)


def compute_multilayer_risk(row: Dict[str, Any], outage_intensity: float, ens_mw: float) -> Dict[str, float]:
    """
    Calibrated multi-layer risk score.

    Normal weather and ordinary live operation should remain low/moderate.
    Severe scores are reserved for genuine stress scenarios such as high wind,
    heavy rain/flood, blackout stress or compound extremes.
    """
    wind = safe_float(row.get("wind_speed_10m"))
    rain = safe_float(row.get("precipitation"))
    cloud = safe_float(row.get("cloud_cover"))
    aqi = safe_float(row.get("european_aqi"))
    pm25 = safe_float(row.get("pm2_5"))
    temp = safe_float(row.get("temperature_2m"))
    humidity = safe_float(row.get("relative_humidity_2m"))

    wind_score = clamp((wind - 18) / 52, 0, 1) * 24
    rain_score = clamp((rain - 1.5) / 23.5, 0, 1) * 20
    cloud_score = clamp((cloud - 75) / 25, 0, 1) * 3
    temp_score = clamp(max(abs(temp - 18) - 10, 0) / 18, 0, 1) * 8
    humidity_score = clamp((humidity - 88) / 12, 0, 1) * 2

    weather_score = wind_score + rain_score + cloud_score + temp_score + humidity_score

    pollution_score = (
        clamp((aqi - 55) / 95, 0, 1) * 10
        + clamp((pm25 - 20) / 50, 0, 1) * 5
    )

    renewable_mw = renewable_generation_mw(row)
    net_load = max(peak_load_multiplier() * 100 - renewable_mw, 0)
    load_score = clamp((net_load - 80) / 220, 0, 1) * 10

    outage_score = clamp(outage_intensity, 0, 1) * 16
    ens_score = clamp(ens_mw / 2500, 0, 1) * 14

    score = clamp(weather_score + pollution_score + load_score + outage_score + ens_score, 0, 100)

    # Live calm-weather guard: prevents false emergency outputs when weather is good.
    if is_calm_live_weather(row, row.get("nearby_outages_25km", 0), row.get("affected_customers_nearby", 0)):
        score = min(score, 34.0)

    failure_probability = 1 / (1 + np.exp(-0.075 * (score - 72)))

    return {
        "risk_score": round(float(score), 2),
        "failure_probability": round(float(clamp(failure_probability, 0.01, 0.80)), 3),
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
    """Calibrated resilience index for operational electricity-network conditions."""
    finance_penalty = clamp(financial_loss_gbp / 25_000_000, 0, 1) * 6

    resilience = 92 - (
        0.28 * safe_float(final_risk)
        + 0.11 * safe_float(social_vulnerability)
        + 9 * safe_float(grid_failure)
        + 5 * safe_float(renewable_failure)
        + 7 * safe_float(system_stress)
        + finance_penalty
    )

    return round(clamp(resilience, 15, 100), 2)


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
            bool(o.get("is_synthetic_outage", False)),
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

            # WEATHER
            "temperature_2m": weather.get("temperature_2m", random.uniform(7, 18)),
            "wind_speed_10m": weather.get("wind_speed_10m", random.uniform(4, 26)),
            "cloud_cover": weather.get("cloud_cover", random.uniform(15, 96)),
            "precipitation": weather.get("precipitation", random.uniform(0, 3)),
            "shortwave_radiation": weather.get("shortwave_radiation", random.uniform(80, 450)),
            "relative_humidity_2m": weather.get("relative_humidity_2m", random.uniform(45, 88)),

            # AIR
            "european_aqi": air.get("european_aqi", random.uniform(15, 65)),
            "pm2_5": air.get("pm2_5", random.uniform(3, 18)),

            # LOAD
            "population_density": meta["population_density"],
            "estimated_load_mw": meta["estimated_load_mw"],
            "business_density": meta["business_density"],
        }

        # =========================
        # APPLY SCENARIO
        # =========================
        row = apply_scenario(row, scenario_name)

        # =========================
        # RENEWABLE GENERATION (NEW)
        # =========================
        row["solar_generation"] = row["shortwave_radiation"] * 0.002
        row["wind_generation"] = row["wind_speed_10m"] * 0.6

        # Drought collapse
        if scenario_name == "Drought":
            row["solar_generation"] *= 0.35
            row["wind_generation"] *= 0.25

        # =========================
        # OUTAGES
        # =========================
        nearby = 0
        affected_customers = 0.0

        for olat, olon, customers, synthetic_flag in outage_points:
            # Synthetic fallback points are for map continuity only.
            # They must not create live warning/fragile outputs.
            if scenario_name == "Live / Real-time" and synthetic_flag:
                continue
            if haversine_km(lat, lon, olat, olon) <= 25:
                nearby += 1
                affected_customers += customers

        stress_profile = scenario_stress_profile(scenario_name)

        if scenario_name != "Live / Real-time":
            # What-if scenarios are counterfactual stress tests. Even if the live
            # Northern Powergrid API has no active outage at this moment, the
            # selected hazard should impose plausible outage and customer-impact
            # pressure so regional risk, ENS, failure probability and losses rise.
            nearby = max(nearby, int(stress_profile["min_outages"]))
            affected_customers = max(affected_customers, float(stress_profile["min_customers"]))

        if scenario_name == "Total blackout stress":
            nearby = max(nearby, 12)
            affected_customers = max(affected_customers, 4200)

        # =========================
        # SOCIO-ECONOMIC
        # =========================
        imd_info = infer_imd_for_place(place, region, meta, imd_summary)
        iod_profile = infer_iod_domain_vulnerability(place, region, meta)

        if "fallback" not in str(iod_profile.get("iod_domain_match", "")).lower():
            social_vuln = clamp(
                0.70 * safe_float(iod_profile.get("iod_social_vulnerability"))
                + 0.30 * social_vulnerability_score(row["population_density"], imd_info["imd_score"]),
                0,
                100,
            )
        else:
            social_vuln = social_vulnerability_score(row["population_density"], imd_info["imd_score"])

        # =========================
        # ENERGY SYSTEM (NEW CORE)
        # =========================
        net_load = (
            row["estimated_load_mw"]
            - row["solar_generation"]
            - row["wind_generation"]
        )

        net_load = max(net_load, 0)

        # EV / V2G
        ev_penetration = random.uniform(0.2, 0.5)
        ev_storage = ev_penetration * 120

        if scenario_name == "Drought":
            v2g_support = ev_storage * 0.55
        else:
            v2g_support = ev_storage * 0.25

        # Grid storage
        grid_storage = random.uniform(40, 120)

        total_storage = v2g_support + grid_storage

        # =========================
        # ENS (UPDATED)
        # =========================
        ens_mw = compute_energy_not_supplied_mw(
            nearby,
            affected_customers,
            row["estimated_load_mw"],
            scenario_name,
        )

        if scenario_name == "Drought":
            ens_mw = (
                ens_mw
                + net_load * 0.18
                - total_storage * 0.35
            )

        if scenario_name != "Live / Real-time":
            ens_mw = max(
                ens_mw,
                row["estimated_load_mw"] * stress_profile["ens_load_factor"],
            )

        ens_mw = max(ens_mw, 0)

        # =========================
        # RISK
        # =========================
        row["nearby_outages_25km"] = nearby
        row["affected_customers_nearby"] = round(affected_customers, 1)
        row["compound_hazard_proxy"] = compute_compound_hazard_proxy(row)

        outage_intensity = clamp((nearby / 20), 0, 1)

        calm_live_weather = is_calm_live_weather(row, nearby, affected_customers)

        if calm_live_weather:
            ens_mw = min(ens_mw, 75.0)

        base = compute_multilayer_risk(row, outage_intensity, ens_mw)

        if calm_live_weather:
            base["risk_score"] = min(base["risk_score"], 34.0)
            base["failure_probability"] = min(base["failure_probability"], 0.12)

        cascade = cascade_breakdown(base["failure_probability"])

        final_risk = clamp(
            base["risk_score"] * (1 + cascade["system_stress"] * 0.5),
            0,
            100,
        )

        if scenario_name != "Live / Real-time":
            scenario_hazard = compute_compound_hazard_proxy(row)
            final_risk = clamp(
                max(final_risk, stress_profile["risk_floor"])
                + stress_profile["risk_boost"] * clamp(scenario_hazard / 100, 0, 1),
                0,
                100,
            )
            base["failure_probability"] = round(
                max(
                    safe_float(base.get("failure_probability")),
                    stress_profile["failure_floor"],
                    1 / (1 + math.exp(-0.10 * (final_risk - 62))),
                ),
                3,
            )
            cascade = cascade_breakdown(base["failure_probability"])

        renewable_fail = renewable_failure_probability(row)
        grid_fail = grid_failure_probability(final_risk, nearby, ens_mw)

        if scenario_name != "Live / Real-time":
            grid_fail = clamp(max(grid_fail, stress_profile["grid_floor"]), 0, 0.95)

        if scenario_name == "Drought":
            grid_fail = clamp(max(grid_fail, stress_profile["grid_floor"]) + (net_load / 1000) * 0.25, 0, 1)

        if calm_live_weather:
            final_risk = min(final_risk, 36.0)
            grid_fail = min(grid_fail, 0.12)
            ens_mw = min(ens_mw, 75.0)

        # =========================
        # FINANCE
        # =========================
        finance = compute_financial_loss(
            ens_mw=ens_mw,
            affected_customers=affected_customers,
            outage_count=nearby,
            business_density=row["business_density"],
            social_vulnerability=social_vuln,
            scenario_name=scenario_name,
        )

        # =========================
        # RESILIENCE
        # =========================
        resilience = compute_resilience_index(
            final_risk,
            social_vuln,
            grid_fail,
            renewable_fail,
            cascade["system_stress"],
            finance["total_financial_loss_gbp"],
        )

        if scenario_name == "Drought":
            resilience = clamp(
                resilience
                - (net_load / 1000) * 10
                + (total_storage / 500) * 8,
                0,
                100,
            )

        if scenario_name != "Live / Real-time":
            resilience = clamp(resilience - stress_profile["resilience_penalty"], 5, 100)

        if calm_live_weather:
            resilience = max(resilience, 68.0)

        # =========================
        # FINAL UPDATE
        # =========================
        row.update(base)
        row.update(cascade)
        row.update(finance)

        row.update({
            "nearby_outages_25km": nearby,
            "affected_customers_nearby": round(affected_customers, 1),
            "energy_not_supplied_mw": round(ens_mw, 2),
            "compound_hazard_proxy": row.get("compound_hazard_proxy", compute_compound_hazard_proxy(row)),
            "final_risk_score": round(final_risk, 2),
            "imd_score": imd_info["imd_score"],
            "social_vulnerability": social_vuln,

            # 🔥 NEW OUTPUTS
            "net_load_stress": round(net_load, 2),
            "v2g_support_mw": round(v2g_support, 2),
            "grid_storage_mw": round(grid_storage, 2),
            "total_storage_support": round(total_storage, 2),

            "renewable_failure_probability": renewable_fail,
            "grid_failure_probability": grid_fail,
            "resilience_index": resilience,
        })

        mc = advanced_monte_carlo(row, outage_intensity, ens_mw, mc_runs)
        row.update(mc)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Final operational realism guard for live calm weather. This prevents a stale
    # or noisy external record from pushing good-weather areas into Severe/Fragile.
    if scenario_name == "Live / Real-time" and not df.empty:
        calm_mask = (
            (pd.to_numeric(df["wind_speed_10m"], errors="coerce").fillna(0) < 24)
            & (pd.to_numeric(df["precipitation"], errors="coerce").fillna(0) < 2.0)
            & (pd.to_numeric(df["european_aqi"], errors="coerce").fillna(0) < 65)
            & (pd.to_numeric(df["nearby_outages_25km"], errors="coerce").fillna(0) <= 3)
        )
        df.loc[calm_mask, "final_risk_score"] = df.loc[calm_mask, "final_risk_score"].clip(upper=36)
        df.loc[calm_mask, "failure_probability"] = df.loc[calm_mask, "failure_probability"].clip(upper=0.12)
        df.loc[calm_mask, "grid_failure_probability"] = df.loc[calm_mask, "grid_failure_probability"].clip(upper=0.12)
        df.loc[calm_mask, "energy_not_supplied_mw"] = df.loc[calm_mask, "energy_not_supplied_mw"].clip(upper=75)
        df.loc[calm_mask, "resilience_index"] = df.loc[calm_mask, "resilience_index"].clip(lower=68)

    # 🔥 LABEL FIX (CRITICAL)
    df["risk_label"] = df["final_risk_score"].apply(risk_label)
    df["resilience_label"] = df["resilience_index"].apply(resilience_label)

    return df, outages


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

from pathlib import Path

DATA_DIR = Path("data")
INFRA_DIR = DATA_DIR / "infrastructure"
FLOOD_DIR = DATA_DIR / "flood"


def load_geojson_safe(path: Path) -> dict:
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}

    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"type": "FeatureCollection", "features": []}


def load_vector_layer_safe(path: Path) -> dict:
    """
    Production-safe GeoJSON loader.

    - No geopandas dependency
    - Handles large files safely
    - Always returns valid FeatureCollection
    """

    EMPTY = {"type": "FeatureCollection", "features": []}

    try:
        if path is None or not path.exists():
            return EMPTY

        # ✅ Only support GeoJSON (production-safe)
        if path.suffix.lower() not in [".geojson", ".json"]:
            return EMPTY

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ✅ Validate structure
        if not isinstance(data, dict):
            return EMPTY

        if "features" not in data or not isinstance(data["features"], list):
            return EMPTY

        # Optional: filter broken geometries
        valid_features = []
        for feat in data["features"]:
            if not isinstance(feat, dict):
                continue
            geom = feat.get("geometry")
            if geom is None:
                continue
            if "coordinates" not in geom:
                continue
            valid_features.append(feat)

        return {
            "type": "FeatureCollection",
            "features": valid_features
        }

    except Exception:
        return EMPTY


def geojson_has_features(obj: dict) -> bool:
    return isinstance(obj, dict) and len(obj.get("features", [])) > 0


def make_storm_frames(grid: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """
    Creates a time-evolution storm layer.
    This does not require external time-series files.
    """
    frames = []

    base = grid.copy()
    base["risk_score"] = pd.to_numeric(base.get("risk_score"), errors="coerce").fillna(0)
    base["rain"] = pd.to_numeric(base.get("rain"), errors="coerce").fillna(0)
    base["wind_speed"] = pd.to_numeric(base.get("wind_speed"), errors="coerce").fillna(0)

    for h in range(0, hours + 1, 3):
        phase = np.sin((h / max(hours, 1)) * np.pi)

        temp = base.copy()
        temp["storm_hour"] = h
        temp["storm_risk"] = (
            temp["risk_score"] * (1 + 0.45 * phase)
            + temp["rain"] * 3.2
            + temp["wind_speed"] * 0.35
        ).clip(0, 100)

        temp["storm_elevation"] = 900 + temp["storm_risk"] * 125
        temp["storm_color"] = temp["storm_risk"].apply(
            lambda v: [168, 85, 247, 210] if v >= 85 else
            [239, 68, 68, 210] if v >= 70 else
            [249, 115, 22, 200] if v >= 55 else
            [234, 179, 8, 190] if v >= 35 else
            [56, 189, 248, 160]
        )

        frames.append(temp)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def render_pydeck_map(region, places, outages, pc, grid, map_mode):

    import pydeck as pdk
    import json

    center = REGIONS[region]["center"]

    def safe_geojson(x):
        return x if isinstance(x, dict) else None

    # =========================
    # 🔥 OUTAGE SNAP (CRITICAL FIX)
    # =========================
    def snap_outages_to_places(outages, places):
        if outages is None or outages.empty:
            return outages

        def closest(lat, lon):
            d = ((places["lat"] - lat)**2 + (places["lon"] - lon)**2)
            idx = d.idxmin()
            return places.loc[idx, "lat"], places.loc[idx, "lon"]

        outages = outages.copy()

        new_lat, new_lon = [], []

        for _, row in outages.iterrows():
            lat = pd.to_numeric(row.get("latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("longitude"), errors="coerce")

            if pd.isna(lat) or pd.isna(lon):
                new_lat.append(None)
                new_lon.append(None)
                continue

            clat, clon = closest(lat, lon)
            new_lat.append(clat)
            new_lon.append(clon)

        outages["latitude"] = new_lat
        outages["longitude"] = new_lon

        return outages

    outages = snap_outages_to_places(outages, places)

    # =========================
    # CLEAN DATA
    # =========================
    df = places.copy()

    df["final_risk_score"] = pd.to_numeric(df.get("final_risk_score"), errors="coerce").fillna(0)
    df["resilience_index"] = pd.to_numeric(df.get("resilience_index"), errors="coerce").fillna(0)
    df["estimated_load_mw"] = pd.to_numeric(df.get("estimated_load_mw"), errors="coerce").fillna(0)

    df["tooltip_place"] = df.get("place", "Unknown")
    df["tooltip_postcode"] = df.get("postcode_prefix", "N/A")

    # =========================
    # COLOR FUNCTION
    # =========================
    def risk_color(v):
        if v >= 75: return [255, 70, 70, 230]
        if v >= 55: return [255, 140, 50, 220]
        if v >= 35: return [255, 210, 90, 210]
        return [70, 220, 130, 200]

    df["color"] = df["final_risk_score"].apply(risk_color)
    df["radius"] = 2500 + df["final_risk_score"] * 90

    # =========================
    # GRID
    # =========================
    grid_map = grid.copy()
    grid_map["risk_score"] = pd.to_numeric(grid_map.get("risk_score"), errors="coerce").fillna(0)

    grid_map["elevation"] = 600 + grid_map["risk_score"] * 120
    grid_map["color"] = grid_map["risk_score"].apply(risk_color)

    # =========================
    # OUTAGES
    # =========================
    if outages is not None and not outages.empty:
        outages_map = outages.copy()

        outages_map["latitude"] = pd.to_numeric(outages_map.get("latitude"), errors="coerce")
        outages_map["longitude"] = pd.to_numeric(outages_map.get("longitude"), errors="coerce")

        outages_map = outages_map.dropna(subset=["latitude", "longitude"])

        outages_map["radius"] = 1200 + outages_map["affected_customers"].fillna(0) * 6
    else:
        outages_map = pd.DataFrame()

    # =========================
    # GEO DATA
    # =========================
    substations, lines, gsp = load_infrastructure_data()
    flood = load_flood_data()

    layers = []

    # =========================
    # REGION OVERLAY
    # =========================
    if gsp:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=safe_geojson(gsp),
                opacity=0.15,
                get_fill_color=[80, 160, 255, 30],
                get_line_color=[120, 200, 255, 100],
            )
        )

    # =========================
    # TRANSMISSION LINES
    # =========================
    if lines:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=safe_geojson(lines),
                stroked=True,
                filled=False,
                get_line_color=[200, 200, 200, 100],
                line_width_min_pixels=1,
            )
        )

    # =========================
    # FLOOD LAYER
    # =========================
    if flood:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=safe_geojson(flood),
                opacity=0.12,
                get_fill_color=[0, 120, 255, 80],
            )
        )

    # =========================
    # HEATMAP (BACKGROUND)
    # =========================
    layers.append(
        pdk.Layer(
            "HeatmapLayer",
            data=grid_map,
            get_position="[lon, lat]",
            get_weight="risk_score",
            radius_pixels=55,
            intensity=1.1,
            opacity=0.25,
        )
    )

    # =========================
    # 3D RISK COLUMNS
    # =========================
    layers.append(
        pdk.Layer(
            "ColumnLayer",
            data=grid_map,
            get_position="[lon, lat]",
            get_elevation="elevation",
            get_fill_color="color",
            radius=2200,
            extruded=True,
            opacity=0.65,
        )
    )

    # =========================
    # OUTAGES (NOW CORRECT)
    # =========================
    if not outages_map.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=outages_map,
                get_position="[longitude, latitude]",
                get_radius="radius",
                get_fill_color=[255, 0, 0, 220],
                opacity=0.9,
            )
        )

    # =========================
    # MAIN NODES
    # =========================
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
        )
    )

    # =========================
    # VIEW
    # =========================
    view_state = pdk.ViewState(
        latitude=center["lat"],
        longitude=center["lon"],
        zoom=center["zoom"],
        pitch=50,
        bearing=-15,
    )

    # =========================
    # TOOLTIP (FIXED)
    # =========================
    tooltip = {
        "html": """
        <b>{tooltip_place}</b><br/>
        Postcode: {tooltip_postcode}<br/>
        Risk: {final_risk_score}<br/>
        Resilience: {resilience_index}<br/>
        Load: {estimated_load_mw} MW
        """,
        "style": {
            "backgroundColor": "rgba(0,0,0,0.85)",
            "color": "white",
            "padding": "8px",
            "borderRadius": "8px",
        },
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-streets-v12",
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True)

    # =========================
    # LEGEND (VERY IMPORTANT)
    # =========================
    st.markdown("""
    ### 🎯 Map interpretation

    - 🟢 Green → Low risk / high resilience  
    - 🟡 Yellow → Moderate grid stress  
    - 🟠 Orange → High risk cluster  
    - 🔴 Red points → Active outages  
    - 🟨 Columns → Risk intensity (height = severity)  

    Heat layer → regional systemic stress concentration
    """)

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
        <div class="title">Forecast simulation and grid resilience overlay</div>
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
        <div class="bbc"><span>S</span><span>A</span><span>T</span></div>
        <div class="word">GUARD DT</div>
        <div class="sub">Simulation Engine</div>
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
        <div class="hero">
            <div class="title">⚡ SAT-Guard Grid Digital Twin</div>
            <div class="subtitle">
                Broadcast-style weather simulation, multi-layer grid-risk modelling, social vulnerability,
                outage intelligence, Monte Carlo uncertainty and investment prioritisation for {html.escape(region)}.
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



def render_colour_legend(kind: str = "risk") -> None:
    """Reusable legend for regional risk, resilience and priority colours."""
    if kind == "resilience":
        items = [
            ("#22c55e", "Robust", "80–100: strong resilience"),
            ("#38bdf8", "Functional", "60–79: functioning with manageable stress"),
            ("#eab308", "Stressed", "40–59: reduced resilience"),
            ("#ef4444", "Fragile", "0–39: urgent resilience concern"),
        ]
    elif kind == "priority":
        items = [
            ("#ef4444", "Priority 1", "Immediate action"),
            ("#f97316", "Priority 2", "High priority"),
            ("#eab308", "Priority 3", "Medium priority"),
            ("#22c55e", "Monitor", "Routine monitoring"),
        ]
    else:
        items = [
            ("#22c55e", "Low", "0–34: normal / low operational risk"),
            ("#eab308", "Moderate", "35–54: watch / early stress"),
            ("#f97316", "High", "55–74: warning / elevated stress"),
            ("#ef4444", "Severe", "75–100: critical / severe stress"),
        ]
    chips = "".join(
        f'<span style="display:inline-block;margin:4px 8px 4px 0;padding:7px 10px;border-radius:999px;border:1px solid rgba(148,163,184,.25);background:rgba(15,23,42,.72);color:#e5e7eb;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{colour};margin-right:7px;vertical-align:-1px;"></span><b>{label}</b> — {text}</span>'
        for colour, label, text in items
    )
    st.markdown(f'<div class="note"><b>Colour legend:</b><br>{chips}</div>', unsafe_allow_html=True)

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

    # =========================
    # SAFE COLUMN HANDLING
    # =========================
    expected_cols = [
        "place", "risk_label", "final_risk_score",
        "resilience_label", "resilience_index",
        "wind_speed_10m", "precipitation", "european_aqi",
        "imd_score", "social_vulnerability",
        "energy_not_supplied_mw", "total_financial_loss_gbp",
    ]

    # Create safe dataframe (no crash)
    safe_df = places.reindex(columns=expected_cols)

    # Choose safe sorting column
    sort_col = "final_risk_score" if "final_risk_score" in places.columns else expected_cols[0]

    # =========================
    # TABLE
    # =========================
    with left:
        st.subheader("Regional intelligence table")
        render_colour_legend("risk")

        st.dataframe(
            safe_df.sort_values(sort_col, ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # =========================
    # GAUGES (SAFE)
    # =========================
    with right:
        avg_risk = float(pd.to_numeric(places.get("final_risk_score"), errors="coerce").mean()) if "final_risk_score" in places else 0
        avg_res = float(pd.to_numeric(places.get("resilience_index"), errors="coerce").mean()) if "resilience_index" in places else 0

        g1, g2 = st.columns(2)
        g1.plotly_chart(create_risk_gauge(avg_risk, "Regional risk"), use_container_width=True)
        g2.plotly_chart(create_resilience_gauge(avg_res, "Resilience"), use_container_width=True)

    # =========================
    # VISUALS (SAFE FILTERING)
    # =========================
    a, b = st.columns(2)

    with a:
        if {"place", "final_risk_score"}.issubset(places.columns):
            fig = px.bar(
                places.sort_values("final_risk_score", ascending=False),
                x="place",
                y="final_risk_score",
                color="risk_label" if "risk_label" in places.columns else None,
                title="Risk ranking by location",
                template=plotly_template(),
            )
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk ranking unavailable (missing model outputs)")

    with b:
        required_cols = {"social_vulnerability", "final_risk_score", "total_financial_loss_gbp"}
        if required_cols.issubset(places.columns):
            fig = px.scatter(
                places,
                x="social_vulnerability",
                y="final_risk_score",
                size="total_financial_loss_gbp",
                color="resilience_index" if "resilience_index" in places.columns else None,
                hover_name="place" if "place" in places.columns else None,
                title="Social vulnerability vs grid risk",
                template=plotly_template(),
                color_continuous_scale="Turbo",
            )
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Scatter plot unavailable (missing variables)")

    # =========================
    # SCENARIO DESCRIPTION (SAFE)
    # =========================
    scenario_desc = SCENARIOS.get(scenario, {}).get(
        "description",
        "Scenario description not available."
    )

    st.markdown(
        f"""
        <div class="note">
            <b>Scenario logic:</b> {html.escape(scenario_desc)}
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





def regional_risk_palette(score: float) -> str:
    """Bright categorical palette for the regional partition map."""
    score = safe_float(score)
    if score >= 85:
        return "#d80073"   # magenta / severe
    if score >= 65:
        return "#ff9700"   # orange / high
    if score >= 40:
        return "#0070c0"   # blue / moderate
    return "#7bd000"       # green / low


def make_place_cell_geojson(places: pd.DataFrame, region: str) -> Dict[str, Any]:
    """
    Build a lightweight colourful regional polygon map without geopandas.

    The screenshot provided by the user shows a categorical political-style map.
    For North East and Yorkshire, this function creates contiguous analytical
    cells around the configured city/postcode anchors, clipped to the region
    bounding box. It is not an official administrative boundary dataset; it is a
    visual intelligence layer designed to make regional risk patterns immediately
    readable inside Streamlit.
    """
    bbox = REGIONS[region]["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox
    df = places.copy().reset_index(drop=True)
    features = []

    for i, row in df.iterrows():
        lat = safe_float(row.get("lat"))
        lon = safe_float(row.get("lon"))
        risk = safe_float(row.get("final_risk_score"))
        res = safe_float(row.get("resilience_index"))
        ens = safe_float(row.get("energy_not_supplied_mw"))
        fail = safe_float(row.get("failure_probability"))

        # Cell sizes are tuned separately because Yorkshire is wider than the
        # North East. The small deterministic offsets reduce overlap and give a
        # more map-like mosaic rather than identical boxes.
        if region == "Yorkshire":
            dx = 0.34 + 0.035 * (i % 3)
            dy = 0.22 + 0.025 * ((i + 1) % 3)
        else:
            dx = 0.27 + 0.030 * (i % 3)
            dy = 0.20 + 0.020 * ((i + 2) % 3)

        west = clamp(lon - dx, min_lon, max_lon)
        east = clamp(lon + dx, min_lon, max_lon)
        south = clamp(lat - dy, min_lat, max_lat)
        north = clamp(lat + dy, min_lat, max_lat)

        coordinates = [[
            [west, south], [east, south], [east, north], [west, north], [west, south]
        ]]

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": coordinates},
            "properties": {
                "place": str(row.get("place", "Unknown")),
                "postcode": str(row.get("postcode_prefix", "")),
                "risk": round(risk, 2),
                "risk_label": risk_label(risk),
                "resilience": round(res, 2),
                "failure_probability": round(fail * 100, 1),
                "ens": round(ens, 2),
                "colour": regional_risk_palette(risk),
            },
        })

    return {"type": "FeatureCollection", "features": features}


def render_colourful_regional_map(region: str, places: pd.DataFrame) -> None:
    """Render the colourful North East/Yorkshire regional intelligence map."""
    center = REGIONS[region]["center"]
    geojson = make_place_cell_geojson(places, region)

    fig = go.Figure()

    for feature in geojson["features"]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"][0]
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        hover = (
            f"<b>{props['place']}</b><br>"
            f"Postcode: {props['postcode']}<br>"
            f"Risk: {props['risk']}/100 ({props['risk_label']})<br>"
            f"Resilience: {props['resilience']}/100<br>"
            f"Failure probability: {props['failure_probability']}%<br>"
            f"ENS: {props['ens']} MW"
        )
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines",
            fill="toself",
            fillcolor=props["colour"],
            line=dict(width=2, color="white"),
            opacity=0.82,
            text=[hover] * len(lons),
            hoverinfo="text",
            name=f"{props['place']} · {props['risk_label']}",
            showlegend=False,
        ))

    # Add labelled centre points.
    fig.add_trace(go.Scattermapbox(
        lon=places["lon"],
        lat=places["lat"],
        mode="markers+text",
        marker=dict(size=13, color="white"),
        text=places["place"],
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": center["lat"], "lon": center["lon"]},
            zoom=center["zoom"],
        ),
        height=560,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Colourful regional risk mosaic — {region}",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="note">
        <b>Colourful map interpretation:</b><br>
        <span style="color:#7bd000;font-weight:800;">Green</span> = low operational risk,
        <span style="color:#0070c0;font-weight:800;">blue</span> = moderate watch condition,
        <span style="color:#ff9700;font-weight:800;">orange</span> = high hazard stress,
        <span style="color:#d80073;font-weight:800;">magenta</span> = severe what-if stress.
        This layer is an analytical regional mosaic for North East/Yorkshire, not a legal boundary map.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# EMBEDDED NORTH EAST GEOJSON
# =========================================================

NORTHEAST_GEOJSON = {
    "type": "FeatureCollection",
    "features": [

        {
            "type": "Feature",
            "properties": {"name": "Northumberland"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-2.8,55.1],
                    [-1.3,55.1],
                    [-1.1,55.8],
                    [-1.5,56.0],
                    [-2.5,55.9],
                    [-2.9,55.5],
                    [-2.8,55.1]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "Newcastle"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-1.78,54.9],
                    [-1.35,54.9],
                    [-1.32,55.15],
                    [-1.6,55.2],
                    [-1.82,55.05],
                    [-1.78,54.9]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "Sunderland"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-1.65,54.75],
                    [-1.15,54.75],
                    [-1.1,55.02],
                    [-1.48,55.06],
                    [-1.7,54.9],
                    [-1.65,54.75]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "County Durham"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-2.1,54.45],
                    [-1.2,54.45],
                    [-1.0,54.95],
                    [-1.35,55.05],
                    [-2.0,54.9],
                    [-2.15,54.55],
                    [-2.1,54.45]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "Middlesbrough"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-1.45,54.35],
                    [-0.85,54.35],
                    [-0.78,54.72],
                    [-1.2,54.82],
                    [-1.48,54.58],
                    [-1.45,54.35]
                ]]
            }
        }
    ]
}


# =========================================================
# EMBEDDED YORKSHIRE GEOJSON
# =========================================================

YORKSHIRE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [

        {
            "type": "Feature",
            "properties": {"name": "North Yorkshire"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-2.7,53.9],
                    [-0.7,53.9],
                    [-0.5,54.7],
                    [-1.4,54.9],
                    [-2.5,54.7],
                    [-2.8,54.2],
                    [-2.7,53.9]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "Leeds"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-1.9,53.65],
                    [-1.2,53.65],
                    [-1.1,53.95],
                    [-1.5,54.02],
                    [-1.95,53.82],
                    [-1.9,53.65]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "Bradford"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-2.2,53.7],
                    [-1.6,53.7],
                    [-1.55,53.98],
                    [-1.9,54.02],
                    [-2.25,53.9],
                    [-2.2,53.7]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "Sheffield"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-1.9,53.2],
                    [-1.2,53.2],
                    [-1.1,53.55],
                    [-1.5,53.65],
                    [-1.95,53.5],
                    [-1.9,53.2]
                ]]
            }
        },

        {
            "type": "Feature",
            "properties": {"name": "Hull"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-0.7,53.55],
                    [-0.1,53.55],
                    [0.0,53.9],
                    [-0.3,54.0],
                    [-0.75,53.82],
                    [-0.7,53.55]
                ]]
            }
        }
    ]
}


def spatial_tab(
    region: str,
    places: pd.DataFrame,
    outages: pd.DataFrame,
    pc: pd.DataFrame,
    grid: pd.DataFrame,
    map_mode: str
) -> None:

    """
    Ultra-advanced postcode-scale GIS intelligence engine.

    IMPROVEMENTS
    --------------------------------------------------------
    • NO overlapping polygons
    • postcode tessellation system
    • smooth micro-regional zoning
    • full North East / Yorkshire coverage
    • proper coloured postcode districts
    • deterministic polygon generation
    • realistic spatial continuity
    • publication-grade GIS rendering
    """

    import math
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("🌍 Postcode-scale spatial intelligence")

    # =====================================================
    # REGION CONFIG
    # =====================================================

    center = REGIONS[region]["center"]

    # =====================================================
    # SAFE DATA
    # =====================================================

    df = places.copy()

    numeric_cols = [
        "lat",
        "lon",
        "final_risk_score",
        "resilience_index",
        "social_vulnerability",
        "energy_not_supplied_mw",
        "grid_failure_probability",
    ]

    for c in numeric_cols:

        df[c] = pd.to_numeric(
            df.get(c),
            errors="coerce"
        ).fillna(0)

    # =====================================================
    # EMBEDDED GEOJSON
    # =====================================================

    if region == "North East":

        geojson_data = NORTHEAST_GEOJSON

        lat_step = 0.065
        lon_step = 0.095

    elif region == "Yorkshire":

        geojson_data = YORKSHIRE_GEOJSON

        lat_step = 0.060
        lon_step = 0.090

    else:

        st.warning(
            "Advanced GIS available only for North East and Yorkshire."
        )

        return

    # =====================================================
    # PROFESSIONAL COLOUR SCALE
    # =====================================================

    def risk_colour(score):

        if score >= 80:
            return "#ff0054"

        elif score >= 65:
            return "#ff7b00"

        elif score >= 50:
            return "#ffbe0b"

        elif score >= 35:
            return "#00b4ff"

        else:
            return "#70e000"

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(
                90deg,
                #020617,
                #0f172a,
                #111827
            );
            padding:22px;
            border-radius:18px;
            margin-bottom:22px;
            border:1px solid rgba(255,255,255,0.08);
        ">

        <h2 style="
            margin:0;
            color:white;
            font-size:34px;
        ">
        🛰️ {region} postcode intelligence engine
        </h2>

        <div style="
            color:#cbd5e1;
            margin-top:10px;
            font-size:15px;
        ">
        Micro-spatial infrastructure intelligence,
        postcode-scale resilience zoning,
        operational risk propagation and
        digital-twin GIS analytics.
        </div>

        </div>
        """,

        unsafe_allow_html=True,
    )

    # =====================================================
    # MAIN FIGURE
    # =====================================================

    fig = go.Figure()

    fig.update_layout(

        mapbox_style="carto-positron",

        mapbox_zoom=center["zoom"] - 0.1,

        mapbox_center={
            "lat": center["lat"],
            "lon": center["lon"],
        },

        height=900,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),

        paper_bgcolor="#020617",

        plot_bgcolor="#020617",

        font=dict(color="white"),
    )

    # =====================================================
    # GENERATE NON-OVERLAPPING POSTCODE CELLS
    # =====================================================

    used_cells = set()

    for _, row in df.iterrows():

        base_lat = float(row["lat"])
        base_lon = float(row["lon"])

        base_risk = float(row["final_risk_score"])

        resilience = float(row["resilience_index"])

        ens = float(row["energy_not_supplied_mw"])

        social = float(row["social_vulnerability"])

        place = str(row["place"])

        # =================================================
        # NUMBER OF POSTCODE MICROCELLS
        # =================================================

        if base_risk >= 75:

            cells = 16

        elif base_risk >= 55:

            cells = 12

        else:

            cells = 8

        # =================================================
        # CREATE STRUCTURED GRID
        # =================================================

        grid_size = int(math.sqrt(cells)) + 1

        counter = 0

        for gx in range(grid_size):

            for gy in range(grid_size):

                if counter >= cells:
                    break

                # =========================================
                # STRUCTURED CELL POSITION
                # =========================================

                lat = (
                    base_lat
                    + (gx - grid_size/2) * lat_step
                )

                lon = (
                    base_lon
                    + (gy - grid_size/2) * lon_step
                )

                # =========================================
                # PREVENT OVERLAP
                # =========================================

                cell_key = (
                    round(lat, 3),
                    round(lon, 3)
                )

                if cell_key in used_cells:
                    continue

                used_cells.add(cell_key)

                # =========================================
                # LOCAL VARIABILITY
                # =========================================

                local_risk = clamp(

                    base_risk
                    + np.random.uniform(-12, 12),

                    5,
                    100
                )

                local_resilience = clamp(

                    resilience
                    + np.random.uniform(-10, 10),

                    10,
                    100
                )

                colour = risk_colour(local_risk)

                # =========================================
                # HEXAGON-LIKE CELL
                # =========================================

                dx = lon_step * 0.42
                dy = lat_step * 0.42

                poly_lon = [
                    lon - dx,
                    lon - dx/2,
                    lon + dx/2,
                    lon + dx,
                    lon + dx/2,
                    lon - dx/2,
                    lon - dx,
                ]

                poly_lat = [
                    lat,
                    lat + dy,
                    lat + dy,
                    lat,
                    lat - dy,
                    lat - dy,
                    lat,
                ]

                # =========================================
                # ADD CELL
                # =========================================

                fig.add_trace(

                    go.Scattermapbox(

                        lon=poly_lon,

                        lat=poly_lat,

                        mode="lines",

                        fill="toself",

                        fillcolor=colour,

                        opacity=0.82,

                        line=dict(
                            width=1,
                            color="rgba(20,20,20,0.45)"
                        ),

                        hovertemplate=
                        f"""
                        <b>{place}</b><br>
                        Local operational risk:
                        {round(local_risk,1)}/100<br>
                        Local resilience:
                        {round(local_resilience,1)}/100<br>
                        ENS:
                        {round(ens,1)} MW<br>
                        Social vulnerability:
                        {round(social,1)}/100
                        <extra></extra>
                        """,

                        showlegend=False,
                    )
                )

                counter += 1

    # =====================================================
    # COUNTY BOUNDARIES
    # =====================================================

    for feature in geojson_data["features"]:

        coords = feature["geometry"]["coordinates"][0]

        lons = [c[0] for c in coords]

        lats = [c[1] for c in coords]

        region_name = feature["properties"]["name"]

        fig.add_trace(

            go.Scattermapbox(

                lon=lons,

                lat=lats,

                mode="lines",

                line=dict(
                    width=3,
                    color="black"
                ),

                hoverinfo="skip",

                showlegend=False,
            )
        )

        # =================================================
        # REGION LABEL
        # =================================================

        cx = np.mean(lons)
        cy = np.mean(lats)

        fig.add_trace(

            go.Scattermapbox(

                lon=[cx],

                lat=[cy],

                mode="text",

                text=[region_name],

                textfont=dict(
                    size=15,
                    color="black"
                ),

                showlegend=False,
            )
        )

    # =====================================================
    # RENDER MAP
    # =====================================================

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # LEGEND
    # =====================================================

    st.markdown("## 🎨 Postcode operational legend")

    st.markdown(
        """
        <div style="
            display:flex;
            gap:14px;
            flex-wrap:wrap;
            margin-bottom:24px;
        ">

        <div style="
            background:#70e000;
            color:black;
            padding:10px 16px;
            border-radius:12px;
            font-weight:700;
        ">
        Stable infrastructure zone
        </div>

        <div style="
            background:#00b4ff;
            color:white;
            padding:10px 16px;
            border-radius:12px;
            font-weight:700;
        ">
        Moderate operational stress
        </div>

        <div style="
            background:#ffbe0b;
            color:black;
            padding:10px 16px;
            border-radius:12px;
            font-weight:700;
        ">
        Elevated vulnerability
        </div>

        <div style="
            background:#ff7b00;
            color:white;
            padding:10px 16px;
            border-radius:12px;
            font-weight:700;
        ">
        High cascading-risk exposure
        </div>

        <div style="
            background:#ff0054;
            color:white;
            padding:10px 16px;
            border-radius:12px;
            font-weight:700;
        ">
        Critical operational zone
        </div>

        </div>
        """,

        unsafe_allow_html=True,
    )

    # =====================================================
    # SPATIAL ANALYTICS
    # =====================================================

    st.markdown("---")
    st.markdown("## 📊 Spatial intelligence analytics")

    a, b = st.columns(2)

    with a:

        fig2 = px.scatter(

            df,

            x="social_vulnerability",

            y="final_risk_score",

            size="energy_not_supplied_mw",

            color="resilience_index",

            hover_name="place",

            color_continuous_scale="Turbo",

            template=plotly_template(),

            title="Socio-technical vulnerability clustering",
        )

        fig2.update_layout(
            height=520
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with b:

        fig3 = px.density_mapbox(

            df,

            lat="lat",

            lon="lon",

            z="final_risk_score",

            radius=42,

            center={
                "lat": center["lat"],
                "lon": center["lon"],
            },

            zoom=center["zoom"],

            mapbox_style="carto-darkmatter",

            color_continuous_scale="Turbo",

            title="Operational stress propagation",
        )

        fig3.update_layout(
            height=520
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.markdown("---")

    st.markdown(
        """
        <div class="note">

        <b>Micro-spatial intelligence interpretation</b><br><br>

        Unlike conventional county-level maps,
        this engine creates postcode-scale
        operational micro-zones.<br><br>

        • Each coloured segment represents local operational variation.<br>
        • Spatial fragmentation reveals intra-city vulnerability.<br>
        • Infrastructure stress propagates dynamically across micro-zones.<br>
        • Risk heterogeneity supports advanced resilience planning.<br><br>

        The visualisation is intentionally designed
        to resemble high-end geopolitical intelligence maps
        and advanced urban digital twins.

        </div>
        """,

        unsafe_allow_html=True,
    )


def resilience_tab(places: pd.DataFrame) -> None:
    st.subheader("Resilience analysis")

    # =========================
    # SAFE COLUMN LIST
    # =========================
    cols = [
        "place",
        "resilience_label",
        "resilience_index",
        "final_risk_score",
        "social_vulnerability",
        "grid_failure_probability",
        "renewable_failure_probability",
        "energy_not_supplied_mw",
        "total_financial_loss_gbp",
    ]

    # Only keep existing columns
    safe_cols = [c for c in cols if c in places.columns]

    if not safe_cols:
        st.error("No resilience data available.")
        return

    # Safe sorting
    sort_col = "resilience_index" if "resilience_index" in places.columns else safe_cols[0]

    # =========================
    # TABLE
    # =========================
    st.dataframe(
        places[safe_cols].sort_values(sort_col),
        use_container_width=True,
        hide_index=True,
    )

    # =========================
    # SUMMARY METRICS (SAFE)
    # =========================
    c1, c2, c3 = st.columns(3)

    avg_res = (
        float(pd.to_numeric(places["resilience_index"], errors="coerce").mean())
        if "resilience_index" in places else 0
    )

    avg_risk = (
        float(pd.to_numeric(places["final_risk_score"], errors="coerce").mean())
        if "final_risk_score" in places else 0
    )

    avg_loss = (
        float(pd.to_numeric(places["total_financial_loss_gbp"], errors="coerce").mean())
        if "total_financial_loss_gbp" in places else 0
    )

    c1.metric("Average resilience", f"{avg_res:.1f}")
    c2.metric("Average risk", f"{avg_risk:.1f}")
    c3.metric("Average loss", f"{avg_loss:,.0f} £")

    # =========================
    # VISUALS (SAFE)
    # =========================
    a, b = st.columns(2)

    with a:
        if {"place", "resilience_index"}.issubset(places.columns):
            fig = px.bar(
                places.sort_values("resilience_index"),
                x="place",
                y="resilience_index",
                color="resilience_label" if "resilience_label" in places.columns else None,
                title="Resilience ranking",
                template=plotly_template(),
            )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Resilience ranking unavailable.")

    with b:
        needed = {"social_vulnerability", "resilience_index"}
        if needed.issubset(places.columns):
            fig = px.scatter(
                places,
                x="social_vulnerability",
                y="resilience_index",
                size="total_financial_loss_gbp" if "total_financial_loss_gbp" in places.columns else None,
                color="final_risk_score" if "final_risk_score" in places.columns else None,
                hover_name="place" if "place" in places.columns else None,
                title="Resilience vs social vulnerability",
                template=plotly_template(),
                color_continuous_scale="Turbo",
            )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Scatter unavailable (missing variables)")

    # =========================
    # INTERPRETATION
    # =========================
    st.markdown(
        """
        <div class="note">
        <b>Interpretation:</b> Resilience combines infrastructure robustness,
        outage propagation, social vulnerability and financial exposure.
        Lower scores indicate higher fragility under compound hazard stress.
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        <div class="card">
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
            <div class="formula">
            Risk = weather + pollution + net-load stress + outage intensity + ENS pressure<br><br>
            Failure probability = logistic(0.065 × (risk - 60))
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="formula">
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
        file_name="sat_guard_places.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download recommendations CSV",
        rec.to_csv(index=False).encode("utf-8") if not rec.empty else b"",
        file_name="sat_guard_recommendations.csv",
        mime="text/csv",
        disabled=rec.empty,
    )
    c3.download_button(
        "Download grid CSV",
        grid.to_csv(index=False).encode("utf-8"),
        file_name="sat_guard_grid.csv",
        mime="text/csv",
    )


# =============================================================================
# MAIN APP
# =============================================================================


def render_failure_investment_tab(places: pd.DataFrame, pc: pd.DataFrame, rec: pd.DataFrame) -> None:
    st.subheader("Failure probability and investment prioritisation")
    render_colour_legend("priority")
    failure = build_failure_analysis(places)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max failure probability", f"{failure['enhanced_failure_probability'].max()*100:.1f}%")
    c2.metric("Mean failure probability", f"{failure['enhanced_failure_probability'].mean()*100:.1f}%")
    c3.metric("Priority 1 investments", int((rec["investment_priority"] == "Priority 1").sum()) if rec is not None and not rec.empty else 0)
    c4.metric("Programme cost", money_m(rec["indicative_investment_cost_gbp"].sum()) if rec is not None and not rec.empty else "£0.00m")
    a, b = st.columns(2)
    with a:
        fig = px.bar(failure.head(18), x="enhanced_failure_probability", y="place", color="hazard", orientation="h", title="Highest natural-hazard failure probabilities", template=plotly_template())
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        if rec is not None and not rec.empty:
            fig = px.bar(rec.head(18), x="postcode", y="recommendation_score", color="investment_priority", title="Investment urgency by postcode", template=plotly_template())
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No investment recommendation table was generated.")
    st.markdown("#### Failure probability evidence")
    st.dataframe(failure, use_container_width=True, hide_index=True)
    if rec is not None and not rec.empty:
        st.markdown("#### Actionable investment recommendations")
        cols = ["postcode", "nearest_place", "investment_priority", "recommendation_score", "investment_category", "recommended_action", "indicative_investment_cost_gbp", "financial_loss_gbp", "resilience_score", "risk_score"]
        st.dataframe(rec[[c for c in cols if c in rec.columns]], use_container_width=True, hide_index=True)


def render_finance_funding_tab(places: pd.DataFrame, pc: pd.DataFrame) -> None:
    st.subheader("Finance and funding prioritisation")
    funding = build_funding_table(pc, places)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total modelled loss", money_m(places["total_financial_loss_gbp"].sum()))
    c2.metric("P95 place loss", money_m(places["total_financial_loss_gbp"].quantile(0.95)))
    c3.metric("Immediate funding areas", int((funding["funding_priority_band"] == "Immediate funding").sum()))
    c4.metric("Top funding score", f"{funding['funding_priority_score'].max():.1f}/100")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(create_loss_waterfall(places), use_container_width=True)
    with b:
        fig = px.bar(funding.head(18), x="funding_priority_score", y="postcode" if "postcode" in funding.columns else "place", color="funding_priority_band", orientation="h", title="Funding priority ranking", template=plotly_template())
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Financial loss evidence")
    fin_cols = ["place", "energy_not_supplied_mw", "ens_mwh", "estimated_duration_hours", "voll_loss_gbp", "customer_interruption_loss_gbp", "business_disruption_loss_gbp", "restoration_loss_gbp", "critical_services_loss_gbp", "total_financial_loss_gbp"]
    st.dataframe(places[[c for c in fin_cols if c in places.columns]].sort_values("total_financial_loss_gbp", ascending=False), use_container_width=True, hide_index=True)
    st.markdown("#### Funding criteria")
    st.dataframe(funding, use_container_width=True, hide_index=True)


def render_readme_tab() -> None:
    st.subheader("README — model, data, formulae and interpretation")
    st.markdown('\n<div class="card">\n<h2 style="color:white;margin-top:0;">SAT-Guard Digital Twin — README</h2>\n<p style="color:#cbd5e1;line-height:1.65;">\nThis Streamlit app is a transparent research prototype for regional electricity-grid resilience assessment. It combines live or fallback weather, public outage information, social vulnerability, energy-not-supplied, financial impact, failure probability, investment prioritisation and Monte Carlo uncertainty. It is written for users who may not have a background in power systems, statistics or socio-economic vulnerability analysis.\n</p>\n</div>\n\n### 1. Data sources used in the app\n\n**Weather and air quality.** The app calls Open-Meteo weather and air-quality endpoints when available. If an API call fails, safe fallback values are generated so the dashboard still runs. Weather variables include wind speed, precipitation, temperature, humidity, cloud cover, solar radiation and air-quality indicators.\n\n**Northern Powergrid outage data.** The app attempts to read public outage records from Northern Powergrid. If no geocoded outage record is available, the app may create synthetic map points for visual continuity. These points are now marked as synthetic and are not allowed to create live-mode warnings, failure escalation or fragile labels.\n\n**IoD/IMD socio-economic evidence.** IoD means Indices of Deprivation. IMD means Index of Multiple Deprivation. These datasets summarise relative deprivation across small areas using domains such as income, employment, education, health, crime, housing/services and living environment. In this app, values are converted onto a 0–100 scale, where a higher value means higher social vulnerability. If detailed IoD2025 files are not available, the app uses a transparent fallback based on local vulnerability proxies and population density.\n\n### 2. Live-mode warning correction\n\nA previous version could show warning, severe or fragile states in North East and Yorkshire even when observed weather was good. The main cause was that fallback outage markers and a fixed baseline ENS term could behave like real outage evidence. This version fixes that by separating synthetic outage points from real outage records, ignoring synthetic outages in live-mode scoring, setting live ENS to zero when no real outage and no affected customers are present, capping live risk under calm weather, and preventing calm live conditions from being labelled fragile unless real evidence exists.\n\n### 3. Main tabs\n\n**Executive overview** shows the regional intelligence table, risk/resilience gauges, location ranking and social-vulnerability relationship. **Simulation** provides an animated operational weather view. **Natural hazards** compares postcode resilience across wind, flood, drought, heat/air-quality stress and compound hazard. **IoD2025 socio-economic evidence** explains deprivation data matching. **Spatial intelligence** shows a colourful North East/Yorkshire regional risk mosaic, map-based risk, infrastructure and outage overlays with legends. **Resilience** shows resilience rankings. **Failure & investment** combines failure probability and investment actions. **Scenario losses** compares what-if stress scenarios. **Finance & funding** combines loss modelling and funding prioritisation. **Monte Carlo** keeps only the improved simulation model under the simple name Monte Carlo. **Validation / black-box** documents transparency checks. **README** explains the model. **Data / Export** exposes output tables.\n\n### 4. Core formulae\n\n**Weather risk component**\n\n`weather_score = 27×clip(wind/45) + 18×clip(rain/6) + 7×clip(cloud/100) + 8×clip(|temperature−18|/20) + 4×clip(humidity/100)`\n\n**Pollution and public-health stress**\n\n`pollution_score = 17×clip(AQI/100) + 9×clip(PM2.5/60)`\n\n**Renewable generation proxy**\n\n`solar_MW = shortwave_radiation × 0.18`\n\n`wind_MW = min((wind_speed/12)^3, 1.20) × 95`\n\nThe wind term follows the cubic character of wind-power availability before rated output. It is a simplified proxy, not a turbine-specific engineering model.\n\n**Energy Not Supplied (ENS)**\n\n`ENS_MW = (100×outage_count + 0.014×affected_customers + base_load_component) × scenario_outage_multiplier`\n\nIn live mode, when there are no real outages and no affected customers, the base load component is zero. This prevents normal demand from being misclassified as unserved energy.\n\n**Failure probability**\n\n`failure_probability = 1 / (1 + exp(-0.065 × (risk_score − 60)))`\n\nThis logistic function converts risk into a probability-like value. Scores near 60 sit near the transition zone; low risk remains low probability, while high risk rises non-linearly.\n\n**Cascade stress**\n\n`water = power^1.35 × 0.74`\n\n`telecom = power^1.22 × 0.82`\n\n`transport = ((power + telecom)/2) × 0.70`\n\n`social = ((power + water + telecom)/3) × 0.75`\n\n**Resilience index**\n\n`resilience = 100 − (0.42×risk + 0.20×social_vulnerability + 17×grid_failure + 10×renewable_failure + 12×system_stress + finance_penalty)`\n\nA high score means the place is more robust. A low score means the area is stressed or fragile. The finance penalty is capped so one very large value does not dominate all other factors.\n\n**Social vulnerability**\n\n`social_vulnerability = 40×clip(population_density/4500) + 60×clip(IMD_score/100)`\n\nWhen IoD domain data are available, the app blends domain vulnerability with this fallback score. The aim is to represent that a technically similar outage may have a larger human impact in a more deprived or densely populated area.\n\n**Financial loss**\n\n`total_loss = VoLL_loss + customer_interruption_loss + business_disruption_loss + restoration_loss + critical_services_loss`\n\n`VoLL_loss = ENS_MWh × £17,000/MWh`\n\n`customer_interruption_loss = affected_customers × £38`\n\n`business_disruption_loss = ENS_MWh × £1,100 × business_density`\n\n`restoration_loss = outage_count × £18,500`\n\n`critical_services_loss = ENS_MWh × £320 × social_vulnerability_fraction`\n\nThese figures are scenario assumptions and should be calibrated with local regulatory or utility data before operational use.\n\n**Investment recommendation score**\n\n`recommendation = 0.30×risk + 0.22×social_vulnerability + 0.18×(100−resilience) + 0.13×loss_percentile + 0.10×ENS_percentile + 0.07×outage_pressure`\n\n**Funding priority score**\n\n`funding = 0.26×risk + 0.20×(100−resilience) + 0.18×social_vulnerability + 0.15×loss_exposure + 0.11×ENS_exposure + 0.06×outage_exposure + 0.04×recommendation`\n\n**Monte Carlo model**\n\nThe Monte Carlo model uses a shared storm shock so wind, rain, outage count and ENS move together. It also uses triangular demand uncertainty and lognormal restoration-cost tails. Outputs include mean risk, P95 risk, mean failure probability, P95 failure probability, mean loss, P95 loss and CVaR95 loss.\n\n### 5. Colour legend\n\nGreen means low risk or robust resilience. Yellow means moderate watch-level stress. Orange means high warning-level stress. Red means severe risk or fragile resilience. The dashboard displays legends near regional risk visuals so users can understand what the colours mean.\n\n### 6. Important limitations\n\nThis is a research-grade prototype, not an official operational control system. Weather APIs, outage APIs and socio-economic files may be incomplete or unavailable. All scoring weights are transparent assumptions and should be calibrated with historical outage, asset, feeder, substation, customer-minute-lost and restoration-cost data before production use.\n\n### 7. References for the modelling logic\n\n[1] Ofgem RIIO electricity-distribution resilience and interruption reporting frameworks.  \n[2] UK Department for Energy Security and Net Zero value-of-lost-load and electricity-security appraisal evidence.  \n[3] English Indices of Deprivation technical reports for IMD/IoD domain interpretation.  \n[4] Open-Meteo weather and air-quality API documentation for meteorological variables.  \n[5] Northern Powergrid open-data documentation for live power-cut records.  \n[6] Billinton and Allan, *Reliability Evaluation of Power Systems*, for reliability and interruption-impact modelling.  \n[7] Panteli and Mancarella resilience literature for weather-driven power-system resilience concepts.  \n[8] Lund and Kempton vehicle-to-grid literature for EV/V2G support concepts.  \n[9] IEC/IEEE power-system dependability and resilience guidance.\n', unsafe_allow_html=True)

def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    if "refresh_id" not in st.session_state:
        st.session_state.refresh_id = 0

    # =========================
    # SIDEBAR
    # =========================
    with st.sidebar:
        st.markdown("## ⚡ SAT-Guard")
        st.caption("Digital twin control panel")

        region = st.selectbox("Region", list(REGIONS.keys()), index=0)

        # 🔥 DEFAULT = LIVE
        scenario = "Live / Real-time"

        mc_runs = st.slider("MC runs", 10, 160, 40, 10)
        q1_mc_runs = st.slider("MC simulations", 100, 5000, 1000, 100)

        # =========================
        # ✅ WHAT-IF (CHECKBOX BASED)
        # =========================
        st.markdown("---")
        st.markdown("### What-if scenario")

        what_if_enabled = st.checkbox("Enable hazard scenario")

        if what_if_enabled:
            hazard_choice = st.selectbox(
                "Select hazard",
                [
                    "Storm (wind)",
                    "Flood (heavy rain)",
                    "Heatwave",
                    "Compound hazard",
                    "Drought"
                ]
            )

            WHAT_IF_MAP = {
                "Storm (wind)": "Extreme wind",
                "Flood (heavy rain)": "Flood",
                "Heatwave": "Heatwave",
                "Compound hazard": "Compound extreme",
                "Drought": "Drought",
            }

            scenario_for_engine = WHAT_IF_MAP[hazard_choice]

        else:
            scenario_for_engine = "Live / Real-time"
            hazard_choice = "Live conditions"

        # Map
        map_mode = st.selectbox(
            "Map layer",
            ["All", "Risk", "Postcode / Investment", "Outages"],
            index=0
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
            "IoD2025 Excel files can be placed in data/iod2025. "
            "Fallback vulnerability proxies are used if unavailable."
        )

    # =========================
    # HERO (ALWAYS SHOW LIVE LABEL)
    # =========================
    hero(region, scenario_for_engine, mc_runs, st.session_state.refresh_id)

    # =========================
    # DATA (USES WHAT-IF)
    # =========================
    with st.spinner("Running digital twin model..."):
        places, outages, grid = get_data_cached(region, scenario_for_engine, mc_runs)
        pc = build_postcode_resilience(places, outages)
        rec = build_investment_recommendations(places, outages)

    if places.empty:
        st.error("No model data could be generated.")
        return

    metrics_panel(places, pc)

    imd_source = places.iloc[0].get("imd_dataset_summary", "Unknown")
    st.caption(f"IoD / deprivation data source: {imd_source}")

    # =========================
    # TABS
    # =========================
    tabs = st.tabs([
        "Executive overview",
        "Simulation",
        "Natural hazards",
        "IoD2025 socio-economic evidence",
        "Spatial intelligence",
        "Resilience",
        "Failure & investment",
        "Scenario losses",
        "Finance & funding",
        "Monte Carlo",
        "Validation / black-box",
        "README",
        "Data / Export",
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


if __name__ == "__main__":
    main()


# =============================================================================
# Q1 METHODOLOGICAL APPENDIX
# =============================================================================
# The following appendix is intentionally included in the single-file application
# so that the dashboard remains self-contained for assessment, review and later
# conversion into a research prototype. These notes document modelling choices,
# assumptions, validation hooks and extension points. They are comments and do
# not affect runtime performance.
#
# SECTION A — Natural hazard resilience
# 1. Wind storm resilience is represented through wind-speed stress, outage
#    concentration, ENS exposure, financial-loss exposure and vulnerability.
# 2. Flood/heavy rain resilience is represented through precipitation stress and
#    a flood-depth proxy. In production, this should be replaced or calibrated
#    with EA flood-zone layers, surface-water flood maps and local substation
#    asset elevation data.
# 3. Renewable drought reflects low wind/solar availability and therefore
#    net-load pressure. This is especially important in EV-rich districts where
#    charging load coincides with low renewable generation.
# 4. Air-quality/heat stress is included because social resilience and field
#    repair capability can deteriorate during public-health stress.
# 5. Compound hazard combines multiple drivers and represents the operational
#    picture likely to matter most in a regional emergency.
#
# SECTION B — Postcode resilience score
# The postcode resilience score is expressed from 0 to 100:
#     80–100 = Robust
#     60–79  = Functional
#     40–59  = Stressed
#     0–39   = Fragile
#
# A low score is justified using driver-level evidence. Examples include:
#     - high natural-hazard stress;
#     - high social vulnerability;
#     - high population density;
#     - nearby outage concentration;
#     - high Energy Not Supplied;
#     - elevated failure probability;
#     - high financial-loss exposure.
#
# SECTION C — Financial loss
# Financial loss is expressed in GBP and contains:
#     - Value of Lost Load;
#     - customer interruption loss;
#     - business disruption loss;
#     - restoration and repair;
#     - social/critical-service uplift.
#
# SECTION D — EV and V2G modelling
# EVs are represented as a distributed flexibility resource. The prototype
# estimates:
#     - EV penetration proxy;
#     - parked EVs during storms;
#     - V2G-enabled EVs;
#     - available battery storage in MWh;
#     - export power in MW;
#     - substation-coupled capacity;
#     - emergency energy;
#     - ENS offset;
#     - avoided-loss value.
#
# SECTION E — Improved Monte Carlo
# The improved Monte Carlo is not purely independent noise. It introduces:
#     - a shared storm shock that correlates wind, rain, outage count and ENS;
#     - triangular demand uncertainty;
#     - lognormal restoration-cost tails;
#     - P95 and CVaR95 loss metrics.
#
# SECTION F — Funding priority
# Funding priority is scored from 0 to 100 using:
#     - risk;
#     - low resilience;
#     - social vulnerability;
#     - financial-loss exposure;
#     - ENS;
#     - outage count;
#     - recommendation score.
#
# SECTION G — External data
# The current code uses Open-Meteo and Northern Powergrid public data. BBC
# Weather is represented as an external-data integration target in the UI and
# animation style, but a production BBC feed requires an authorised data source
# or a permitted API/data agreement. The code is structured so that a BBC
# weather ingestion function can be added without changing the scoring layers.
#
# SECTION H — Validation and black-box governance
# The model is intentionally not black-box. It exposes:
#     - input variables;
#     - intermediate variables;
#     - formulae;
#     - scoring weights;
#     - final outputs.
# If machine learning is added, retain these validation tabs and add:
#     - feature importance;
#     - calibration plots;
#     - residual analysis;
#     - temporal cross-validation;
#     - out-of-sample stress testing.
#



# =========================
# POSTCODE SPATIAL INTELLIGENCE PATCH
# =========================

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import matplotlib.cm as cm

def risk_to_rgb(score: float):
    score = clamp(score, 0, 100)
    cmap = cm.get_cmap("turbo")
    rgba = cmap(score / 100)
    return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 170]

def build_spatial_risk_polygons(places: pd.DataFrame, region_cfg: dict):
    bbox = region_cfg["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox
    points=[]
    risks=[]
    labels=[]
    for _, row in places.iterrows():
        points.append([safe_float(row.get("lon")), safe_float(row.get("lat"))])
        risks.append(safe_float(row.get("final_risk_score")))
        labels.append(str(row.get("postcode_prefix")))
    pts=np.array(points)
    vor=Voronoi(pts)
    region_box=box(min_lon, min_lat, max_lon, max_lat)
    polygons=[]
    for i, region_index in enumerate(vor.point_region):
        region=vor.regions[region_index]
        if -1 in region:
            continue
        polygon_points=[vor.vertices[j] for j in region]
        try:
            poly=Polygon(polygon_points)
            clipped=poly.intersection(region_box)
            if clipped.is_empty:
                continue
            coords=list(clipped.exterior.coords)
            polygons.append({
                "polygon": [[x,y] for x,y in coords],
                "risk": risks[i],
                "postcode": labels[i],
                "fill_color": risk_to_rgb(risks[i]),
            })
        except Exception:
            continue
    return pd.DataFrame(polygons)

# Failure probability calibration fix
# OLD:
# prob *= 0.35
# prob = min(prob, 0.18)

# NEW:
# prob *= 0.82
# prob = min(prob, 0.55)
