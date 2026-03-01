"""
NLM Translation Layer

Translates raw sensor data through:
1. Raw -> Normalized (scaling, calibration)
2. Normalized -> Bio-Tokens (vocabulary mapping)
3. Bio-Tokens -> NMF (frame assembly)

Created: February 17, 2026
"""

from typing import Any, Dict, List, Optional

from nlm.telemetry.bio_tokens import get_semantic
from nlm.telemetry.nmf import NatureMessageFrame

# ---------------------------------------------------------------------------
# Normalization thresholds
# ---------------------------------------------------------------------------

TEMP_OPT_LO = 20.0
TEMP_OPT_HI = 28.0
TEMP_STRESS_LO = 15.0
TEMP_STRESS_HI = 35.0
HUM_LOW = 40.0
HUM_HIGH = 80.0
LIGHT_LOW = 100.0
LIGHT_OPT = 500.0
LIGHT_HIGH = 2000.0
CO2_HIGH = 1500.0
PH_ACID = 6.0
PH_ALK = 8.0
MOISTURE_DRY = 30.0
MOISTURE_SAT = 80.0


def normalize_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Step 1: Raw -> Normalized. Scale and calibrate raw sensor values."""
    out: Dict[str, Any] = dict(raw)
    numeric_keys = {"temperature_c", "humidity_pct", "light_lux", "co2_ppm", "ph", "moisture_pct", "magnitude"}
    for key_map in [
        (("temperature_c", "temp_c", "temperature"), "temperature_c"),
        (("humidity_pct", "humidity", "rh"), "humidity_pct"),
        (("light_lux", "lux", "light"), "light_lux"),
        (("co2_ppm", "co2"), "co2_ppm"),
        (("ph",), "ph"),
        (("moisture_pct", "moisture"), "moisture_pct"),
        (("magnitude", "mag"), "magnitude"),
        (("pattern_name", "pattern"), "pattern_name"),
    ]:
        keys, out_key = key_map[0], key_map[1]
        val = None
        for k in keys:
            if raw.get(k) is not None:
                val = raw[k]
                break
        if val is not None:
            if out_key in numeric_keys:
                try:
                    out[out_key] = float(val)
                except (TypeError, ValueError):
                    out[out_key] = val
            else:
                out[out_key] = val
    return out


def raw_to_bio_tokens(raw: Dict[str, Any]) -> List[str]:
    """Step 2: Raw/Normalized -> Bio-Tokens. Map sensor values to token codes."""
    tokens: List[str] = []
    norm = normalize_raw(raw)

    t = norm.get("temperature_c")
    if t is not None:
        if TEMP_OPT_LO <= t <= TEMP_OPT_HI:
            tokens.append("TEMP_OPT")
        elif t > TEMP_STRESS_HI:
            tokens.append("TEMP_STRESS_HI")
        elif t < TEMP_STRESS_LO:
            tokens.append("TEMP_STRESS_LO")
        elif t < TEMP_OPT_LO:
            tokens.append("TEMP_COOL")
        else:
            tokens.append("TEMP_WARM")

    h = norm.get("humidity_pct")
    if h is not None:
        if h > HUM_HIGH:
            tokens.append("HUM_HIGH")
        elif h < HUM_LOW:
            tokens.append("HUM_LOW")
        else:
            tokens.append("HUM_OPT")

    lx = norm.get("light_lux")
    if lx is not None:
        if lx < LIGHT_LOW:
            tokens.append("LIGHT_LOW")
        elif LIGHT_OPT <= lx <= LIGHT_HIGH:
            tokens.append("LIGHT_OPT")
        elif lx > LIGHT_HIGH:
            tokens.append("LIGHT_HIGH")

    co2 = norm.get("co2_ppm")
    if co2 is not None:
        tokens.append("CO2_HIGH" if co2 > CO2_HIGH else "CO2_OPT")

    ph = norm.get("ph")
    if ph is not None:
        if ph < PH_ACID:
            tokens.append("PH_ACID")
        elif ph > PH_ALK:
            tokens.append("PH_ALK")
        else:
            tokens.append("PH_NEUTRAL")

    m = norm.get("moisture_pct")
    if m is not None:
        if m < MOISTURE_DRY:
            tokens.append("MOIST_DRY")
        elif m > MOISTURE_SAT:
            tokens.append("MOIST_SAT")
        else:
            tokens.append("MOIST_OPT")

    mag = norm.get("magnitude")
    if mag is not None:
        tokens.append("SEISMIC_ACT" if float(mag) >= 2.0 else "SEISMIC_QUIET")

    pattern = norm.get("pattern_name")
    if pattern:
        s = str(pattern).lower()
        if "growth" in s or "active" in s:
            tokens.append("GROWTH_ACT")
        elif "dormant" in s:
            tokens.append("GROWTH_DORM")
        else:
            tokens.append("STIM_RESP")

    return tokens


def build_nmf(
    raw: Dict[str, Any],
    envelopes: Optional[List[Dict[str, Any]]] = None,
    source_id: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> NatureMessageFrame:
    """Step 3: Build Nature Message Frame from raw data and optional envelopes."""
    tokens = raw_to_bio_tokens(raw)
    norm = normalize_raw(raw)
    structured = {
        "normalized": norm,
        "bio_tokens": [get_semantic(t) for t in tokens],
        "token_codes": tokens,
    }
    return NatureMessageFrame(
        bio_tokens=tokens,
        structured_output=structured,
        envelopes=envelopes or [],
        context=context or {},
        confidence=1.0,
        source_id=source_id,
    )


def translate(
    raw: Dict[str, Any],
    envelopes: Optional[List[Dict[str, Any]]] = None,
    source_id: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> NatureMessageFrame:
    """Full translation: Raw -> Normalized -> Bio-Tokens -> NMF."""
    return build_nmf(raw, envelopes=envelopes, source_id=source_id, context=context)
