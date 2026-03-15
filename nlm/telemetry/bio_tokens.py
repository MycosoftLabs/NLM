"""
NLM Bio-Token Vocabulary - Micro-Speak for Nature Signals

Bio-tokens are the semantic vocabulary for translating raw sensor data
into nature-oriented micro-speak tokens consumed by NLM.

Created: February 17, 2026
"""

from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Bio-Token Vocabulary - Micro-speak for nature signals
# ---------------------------------------------------------------------------

BIO_TOKEN_VOCABULARY: Dict[str, str] = {
    # ---- Temperature -------------------------------------------------
    "TEMP_OPT": "temperature_optimal",
    "TEMP_STRESS_HI": "temperature_stress_high",
    "TEMP_STRESS_LO": "temperature_stress_low",
    "TEMP_COOL": "temperature_cool",
    "TEMP_WARM": "temperature_warm",
    # ---- Humidity ----------------------------------------------------
    "HUM_HIGH": "humidity_high",
    "HUM_LOW": "humidity_low",
    "HUM_OPT": "humidity_optimal",
    # ---- Light -------------------------------------------------------
    "LIGHT_LOW": "light_low",
    "LIGHT_OPT": "light_optimal",
    "LIGHT_HIGH": "light_high",
    "LIGHT_DAWN": "light_dawn",
    "LIGHT_DUSK": "light_dusk",
    # ---- Biological / FCI --------------------------------------------
    "GROWTH_ACT": "growth_active",
    "GROWTH_DORM": "growth_dormant",
    "GROWTH_SPIKE": "growth_spike",
    "STIM_RESP": "stimulus_response",
    "BIO_QUIESC": "bioelectric_quiescent",
    "BIO_ACTIVE": "bioelectric_active",
    "BIO_MIGRATE": "migration_detected",
    "BIO_BREED": "breeding_season",
    "BIO_BLOOM": "bloom_event",
    # ---- Species / Taxonomy ------------------------------------------
    "SPX_OBSERVE": "species_observation",
    "SPX_THREAT": "species_threatened",
    "SPX_INVASIVE": "species_invasive",
    "SPX_ENDEMIC": "species_endemic",
    "SPX_EXTINCT": "species_extinction_risk",
    # ---- Seismic / Geological ----------------------------------------
    "SEISMIC_PRE": "seismic_precursor",
    "SEISMIC_ACT": "seismic_active",
    "SEISMIC_QUIET": "seismic_quiet",
    "SEISMIC_MAJOR": "seismic_major_event",
    "ULF_DETECT": "ulf_detected",
    "VOLC_ACT": "volcanic_activity",
    "VOLC_ERUPT": "volcanic_eruption",
    "VOLC_QUIET": "volcanic_quiet",
    # ---- Atmospheric -------------------------------------------------
    "CO2_HIGH": "co2_high",
    "CO2_OPT": "co2_optimal",
    "CO2_CRITICAL": "co2_critical",
    "CH4_ELEV": "methane_elevated",
    "VOC_ELEV": "voc_elevated",
    "PRESS_DROP": "pressure_dropping",
    "PRESS_STABLE": "pressure_stable",
    "PRESS_RISE": "pressure_rising",
    "WIND_CALM": "wind_calm",
    "WIND_STRONG": "wind_strong",
    "WIND_SEVERE": "wind_severe",
    # ---- Air Quality -------------------------------------------------
    "AQI_GOOD": "air_quality_good",
    "AQI_MOD": "air_quality_moderate",
    "AQI_UNHEALTHY": "air_quality_unhealthy",
    "AQI_HAZ": "air_quality_hazardous",
    "PM25_ELEV": "pm25_elevated",
    "PM10_ELEV": "pm10_elevated",
    "OZONE_HIGH": "ozone_high",
    # ---- Weather Events ----------------------------------------------
    "WX_STORM": "storm_detected",
    "WX_HURRICANE": "hurricane_detected",
    "WX_TORNADO": "tornado_detected",
    "WX_FLOOD": "flood_detected",
    "WX_LIGHTNING": "lightning_detected",
    "WX_DROUGHT": "drought_condition",
    "WX_HEATWAVE": "heatwave_condition",
    "WX_FREEZE": "freeze_condition",
    # ---- Fire --------------------------------------------------------
    "FIRE_ACT": "wildfire_active",
    "FIRE_CONTAIN": "wildfire_contained",
    "FIRE_OUT": "wildfire_extinguished",
    "FIRE_RISK_HI": "fire_risk_high",
    # ---- Water / Ocean -----------------------------------------------
    "WATER_CLEAN": "water_quality_good",
    "WATER_CONTAM": "water_contaminated",
    "WATER_FLOOD": "water_flood_stage",
    "OCEAN_WARM": "ocean_warming",
    "OCEAN_ACID": "ocean_acidification",
    "TIDE_HIGH": "tide_high",
    "TIDE_LOW": "tide_low",
    "WAVE_SEVERE": "wave_severe",
    # ---- Substrate / Resource ----------------------------------------
    "MOIST_OPT": "moisture_optimal",
    "MOIST_DRY": "moisture_dry",
    "MOIST_SAT": "moisture_saturated",
    "PH_ACID": "ph_acidic",
    "PH_NEUTRAL": "ph_neutral",
    "PH_ALK": "ph_alkaline",
    "NUTRIENT_FLOW": "nutrient_flow_detected",
    "SUBSTRATE_STABLE": "substrate_stable",
    "SOIL_CONTAM": "soil_contaminated",
    # ---- Infrastructure / Pollution ----------------------------------
    "INFRA_EMIT": "infrastructure_emission",
    "INFRA_SPILL": "infrastructure_spill",
    "INFRA_OFFLINE": "infrastructure_offline",
    "INFRA_ONLINE": "infrastructure_online",
    "POLLUT_DETECT": "pollution_detected",
    "POLLUT_CLEAR": "pollution_cleared",
    "RADI_ELEV": "radiation_elevated",
    "RADI_NORM": "radiation_normal",
    # ---- Signal / RF -------------------------------------------------
    "SIG_STRONG": "signal_strong",
    "SIG_WEAK": "signal_weak",
    "SIG_NONE": "signal_absent",
    "SIG_INTERF": "signal_interference",
    "RF_DETECT": "rf_emission_detected",
    "EMF_ELEV": "emf_elevated",
    # ---- Space / Solar -----------------------------------------------
    "SOLAR_QUIET": "solar_quiet",
    "SOLAR_FLARE": "solar_flare",
    "SOLAR_CME": "coronal_mass_ejection",
    "GEO_STORM": "geomagnetic_storm",
    "SAT_PASS": "satellite_pass",
    "SAT_LAUNCH": "satellite_launch",
    "SPACE_DEBRIS": "space_debris_alert",
    # ---- Transportation / Tracking -----------------------------------
    "TRNS_VESSEL": "vessel_detected",
    "TRNS_AIRCRAFT": "aircraft_detected",
    "TRNS_LAUNCH": "launch_event",
    "AIS_ALERT": "ais_anomaly",
    "ADSB_ALERT": "adsb_anomaly",
    # ---- Telemetry / Device ------------------------------------------
    "TELEM_OK": "telemetry_nominal",
    "TELEM_WARN": "telemetry_warning",
    "TELEM_FAIL": "telemetry_failure",
    "DEVICE_ONLINE": "device_online",
    "DEVICE_OFFLINE": "device_offline",
    "BRAIN_BOARD": "myco_brain_board_signal",
    # ---- Early warning / Prediction ----------------------------------
    "FRUITING_LIKELY": "fruiting_likely",
    "FRUITING_IMM": "fruiting_imminent",
    "STRESS_ALERT": "stress_alert",
    "ANOMALY": "anomaly_detected",
    "TREND_UP": "trend_increasing",
    "TREND_DOWN": "trend_decreasing",
    "TREND_STABLE": "trend_stable",
    "EVENT_CLUSTER": "event_cluster_detected",
}

BIO_TOKEN_REVERSE: Dict[str, str] = {v: k for k, v in BIO_TOKEN_VOCABULARY.items()}


def get_token_code(semantic: str) -> str:
    """Get token code (e.g. TEMP_OPT) from semantic label."""
    return BIO_TOKEN_REVERSE.get(semantic, semantic)


def get_semantic(token_code: str) -> str:
    """Get semantic label from token code."""
    return BIO_TOKEN_VOCABULARY.get(token_code, token_code)


def all_tokens() -> List[str]:
    """Return list of all token codes."""
    return list(BIO_TOKEN_VOCABULARY.keys())


def all_semantics() -> Set[str]:
    """Return set of all semantic labels."""
    return set(BIO_TOKEN_VOCABULARY.values())
