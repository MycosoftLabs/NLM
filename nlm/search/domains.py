"""
NLM Earth Domain Taxonomy — aligned with MINDEX v3
====================================================

Exact mirror of mindex's ``ALL_DOMAINS`` and ``DOMAIN_GROUPS`` so that
NLM search requests, CREP layers, and Myca queries map 1-to-1 onto the
mindex unified-search and earth routers.

Schemas in mindex:
  earth.*   — earthquakes, volcanoes, wildfires, storms, lightning, tornadoes, floods
  species.* — all-kingdom organisms + sightings
  infra.*   — facilities, power_grid, water_systems, internet_cables
  signals.* — antennas, wifi_hotspots, signal_measurements
  transport.* — aircraft, vessels, airports, ports, spaceports, launches
  space.*   — satellites, solar_events
  atmos.*   — air_quality, greenhouse_gas, weather_observations, remote_sensing
  hydro.*   — buoys, stream_gauges
  monitor.* — cameras
  military.* — installations
  core.*    — taxon, compounds, dna_sequences, publications, observation
  telemetry.* — device, stream, sample
  crep.*    — unified_entities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ======================================================================
# Exact domain list from mindex unified_search.py
# ======================================================================

ALL_DOMAINS: List[str] = [
    # Biological
    "taxa", "species", "compounds", "genetics", "observations",
    # Earth events
    "earthquakes", "volcanoes", "wildfires", "storms", "lightning",
    "tornadoes", "floods",
    # Atmosphere
    "air_quality", "greenhouse_gas", "weather", "remote_sensing",
    # Water
    "buoys", "stream_gauges",
    # Infrastructure
    "facilities", "power_grid", "water_systems", "internet_cables",
    # Signals
    "antennas", "wifi_hotspots", "signal_measurements",
    # Transport
    "aircraft", "vessels", "airports", "ports", "spaceports", "launches",
    # Space
    "satellites", "solar_events",
    # Monitoring
    "cameras",
    # Military
    "military_installations",
    # Telemetry
    "devices", "telemetry",
    # Knowledge
    "research", "crep_entities",
]

# Exact group aliases from mindex unified_search.py
DOMAIN_GROUPS: Dict[str, List[str]] = {
    "all": ALL_DOMAINS,
    "biological": ["taxa", "species", "compounds", "genetics", "observations"],
    "life": ["taxa", "species", "observations"],
    "earth_events": [
        "earthquakes", "volcanoes", "wildfires", "storms",
        "lightning", "tornadoes", "floods",
    ],
    "hazards": [
        "earthquakes", "volcanoes", "wildfires", "storms",
        "tornadoes", "floods",
    ],
    "atmosphere": ["air_quality", "greenhouse_gas", "weather", "remote_sensing"],
    "water": ["buoys", "stream_gauges", "water_systems"],
    "infrastructure": ["facilities", "power_grid", "water_systems", "internet_cables"],
    "pollution": ["facilities", "air_quality", "greenhouse_gas"],
    "signals": ["antennas", "wifi_hotspots", "signal_measurements"],
    "transport": ["aircraft", "vessels", "airports", "ports", "spaceports", "launches"],
    "aviation": ["aircraft", "airports"],
    "maritime": ["vessels", "ports", "buoys"],
    "space": ["satellites", "solar_events", "launches"],
    "monitoring": ["cameras"],
    "military": ["military_installations"],
    "telemetry": ["devices", "telemetry"],
}

# Map each domain to its mindex DB schema.table
DOMAIN_TABLES: Dict[str, str] = {
    "taxa": "core.taxon",
    "species": "species.organisms",
    "compounds": "core.compounds",
    "genetics": "core.dna_sequences",
    "observations": "core.observation",  # + species.sightings
    "earthquakes": "earth.earthquakes",
    "volcanoes": "earth.volcanoes",
    "wildfires": "earth.wildfires",
    "storms": "earth.storms",
    "lightning": "earth.lightning",
    "tornadoes": "earth.tornadoes",
    "floods": "earth.floods",
    "air_quality": "atmos.air_quality",
    "greenhouse_gas": "atmos.greenhouse_gas",
    "weather": "atmos.weather_observations",
    "remote_sensing": "atmos.remote_sensing",
    "buoys": "hydro.buoys",
    "stream_gauges": "hydro.stream_gauges",
    "facilities": "infra.facilities",
    "power_grid": "infra.power_grid",
    "water_systems": "infra.water_systems",
    "internet_cables": "infra.internet_cables",
    "antennas": "signals.antennas",
    "wifi_hotspots": "signals.wifi_hotspots",
    "signal_measurements": "signals.signal_measurements",
    "aircraft": "transport.aircraft",
    "vessels": "transport.vessels",
    "airports": "transport.airports",
    "ports": "transport.ports",
    "spaceports": "transport.spaceports",
    "launches": "transport.launches",
    "satellites": "space.satellites",
    "solar_events": "space.solar_events",
    "cameras": "monitor.cameras",
    "military_installations": "military.installations",
    "devices": "telemetry.device",
    "telemetry": "telemetry.stream",
    "research": "core.publications",
    "crep_entities": "crep.unified_entities",
}

# Domain → parent group (for CREP layer assignment)
DOMAIN_TO_GROUP: Dict[str, str] = {
    "taxa": "biological", "species": "biological", "compounds": "biological",
    "genetics": "biological", "observations": "biological",
    "earthquakes": "earth_events", "volcanoes": "earth_events",
    "wildfires": "earth_events", "storms": "earth_events",
    "lightning": "earth_events", "tornadoes": "earth_events",
    "floods": "earth_events",
    "air_quality": "atmosphere", "greenhouse_gas": "atmosphere",
    "weather": "atmosphere", "remote_sensing": "atmosphere",
    "buoys": "water", "stream_gauges": "water",
    "facilities": "infrastructure", "power_grid": "infrastructure",
    "water_systems": "infrastructure", "internet_cables": "infrastructure",
    "antennas": "signals", "wifi_hotspots": "signals",
    "signal_measurements": "signals",
    "aircraft": "transport", "vessels": "transport",
    "airports": "transport", "ports": "transport",
    "spaceports": "transport", "launches": "transport",
    "satellites": "space", "solar_events": "space",
    "cameras": "monitoring",
    "military_installations": "military",
    "devices": "telemetry", "telemetry": "telemetry",
    "research": "knowledge", "crep_entities": "knowledge",
}


@dataclass
class SearchDomain:
    """One searchable domain aligned with mindex."""
    key: str
    label: str
    description: str
    group: str
    table: str
    crep_layer: Optional[str] = None
    bio_token_prefixes: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    @property
    def schema(self) -> str:
        return self.table.split(".")[0] if "." in self.table else ""


# ======================================================================
# Domain metadata (labels, descriptions, CREP layers, tags)
# ======================================================================

_DOMAIN_META: Dict[str, Dict[str, Any]] = {
    # Biological
    "taxa": {"label": "Taxa (Fungi)", "desc": "Fungal taxonomy (core.taxon)", "crep": "species", "tokens": ["BIO", "GROWTH"], "tags": {"fungi", "mushroom", "mycology"}},
    "species": {"label": "All Species", "desc": "All-kingdom organisms", "crep": "species", "tokens": ["BIO"], "tags": {"species", "organism", "biodiversity", "plant", "bird", "mammal", "insect", "marine", "reptile"}},
    "compounds": {"label": "Compounds", "desc": "Bioactive chemical compounds", "crep": None, "tokens": [], "tags": {"chemistry", "compound", "molecule"}},
    "genetics": {"label": "Genetics", "desc": "DNA/RNA sequences", "crep": None, "tokens": [], "tags": {"genetics", "genome", "dna", "sequence"}},
    "observations": {"label": "Observations", "desc": "Field observations and sightings", "crep": "species", "tokens": ["BIO", "SPX"], "tags": {"observation", "sighting"}},
    # Earth events
    "earthquakes": {"label": "Earthquakes", "desc": "Seismic events (USGS)", "crep": "earthquakes", "tokens": ["SEISMIC"], "tags": {"earthquake", "seismic", "quake"}},
    "volcanoes": {"label": "Volcanoes", "desc": "Active and dormant volcanoes", "crep": "volcanoes", "tokens": ["VOLC"], "tags": {"volcano", "eruption", "lava"}},
    "wildfires": {"label": "Wildfires", "desc": "Active fires (FIRMS/NIFC)", "crep": "wildfires", "tokens": ["FIRE"], "tags": {"wildfire", "fire", "burn"}},
    "storms": {"label": "Storms", "desc": "Hurricanes, cyclones, tropical storms", "crep": None, "tokens": ["WX"], "tags": {"storm", "hurricane", "cyclone"}},
    "lightning": {"label": "Lightning", "desc": "Lightning strike data", "crep": None, "tokens": ["WX"], "tags": {"lightning"}},
    "tornadoes": {"label": "Tornadoes", "desc": "Tornado events", "crep": None, "tokens": ["WX"], "tags": {"tornado"}},
    "floods": {"label": "Floods", "desc": "Flood events and river levels", "crep": None, "tokens": ["WATER"], "tags": {"flood", "river"}},
    # Atmosphere
    "air_quality": {"label": "Air Quality", "desc": "AQI, PM2.5, ozone stations", "crep": "air_quality", "tokens": ["AQI", "PM25"], "tags": {"air_quality", "aqi", "pm25", "pollution"}},
    "greenhouse_gas": {"label": "Greenhouse Gases", "desc": "CO2, methane, N2O measurements", "crep": None, "tokens": ["CO2", "CH4"], "tags": {"co2", "methane", "emissions", "carbon"}},
    "weather": {"label": "Weather", "desc": "Weather observation stations", "crep": "weather", "tokens": ["TEMP", "HUM", "PRESS", "WIND"], "tags": {"weather", "temperature", "wind", "forecast"}},
    "remote_sensing": {"label": "Remote Sensing", "desc": "MODIS, Landsat, Sentinel, AIRS imagery", "crep": None, "tokens": [], "tags": {"satellite", "modis", "landsat", "ndvi"}},
    # Water
    "buoys": {"label": "Ocean Buoys", "desc": "NDBC buoys and ocean sensors", "crep": "buoys", "tokens": ["OCEAN", "WAVE"], "tags": {"buoy", "ocean", "wave"}},
    "stream_gauges": {"label": "Stream Gauges", "desc": "USGS river/stream gauges", "crep": None, "tokens": ["WATER"], "tags": {"river", "streamflow", "gauge"}},
    # Infrastructure
    "facilities": {"label": "Facilities", "desc": "Factories, power plants, mining, dams", "crep": "facilities", "tokens": ["INFRA"], "tags": {"factory", "power_plant", "mining", "oil", "gas", "dam", "water_treatment"}},
    "power_grid": {"label": "Power Grid", "desc": "Substations, transmission lines", "crep": None, "tokens": ["INFRA"], "tags": {"grid", "transmission", "substation"}},
    "water_systems": {"label": "Water Systems", "desc": "Rivers, dams, reservoirs, treatment", "crep": None, "tokens": ["WATER"], "tags": {"dam", "reservoir", "treatment_plant"}},
    "internet_cables": {"label": "Internet Cables", "desc": "Submarine and terrestrial fiber", "crep": None, "tokens": ["SIG"], "tags": {"internet", "cable", "fiber", "submarine"}},
    # Signals
    "antennas": {"label": "Antennas", "desc": "Cell towers, AM/FM, broadcast", "crep": "antennas", "tokens": ["SIG", "RF"], "tags": {"cell_tower", "antenna", "radio", "5g"}},
    "wifi_hotspots": {"label": "WiFi Hotspots", "desc": "Known WiFi/Bluetooth networks", "crep": "wifi_hotspots", "tokens": ["SIG"], "tags": {"wifi", "bluetooth", "hotspot"}},
    "signal_measurements": {"label": "Signal Measurements", "desc": "RF field strength measurements", "crep": None, "tokens": ["SIG", "RF", "EMF"], "tags": {"signal", "measurement", "rf"}},
    # Transport
    "aircraft": {"label": "Aircraft", "desc": "Live aircraft positions (ADS-B)", "crep": "aircraft", "tokens": ["TRNS"], "tags": {"airplane", "flight", "aircraft", "adsb"}},
    "vessels": {"label": "Vessels", "desc": "Ship positions (AIS)", "crep": "vessels", "tokens": ["TRNS"], "tags": {"ship", "boat", "vessel", "ais"}},
    "airports": {"label": "Airports", "desc": "Airports and airfields", "crep": "airports", "tokens": [], "tags": {"airport"}},
    "ports": {"label": "Ports", "desc": "Seaports and harbors", "crep": "ports", "tokens": [], "tags": {"port", "harbor"}},
    "spaceports": {"label": "Spaceports", "desc": "Launch facilities worldwide", "crep": None, "tokens": [], "tags": {"spaceport", "launch_pad"}},
    "launches": {"label": "Launches", "desc": "Rocket launches from Earth", "crep": None, "tokens": ["SAT"], "tags": {"launch", "rocket"}},
    # Space
    "satellites": {"label": "Satellites", "desc": "All tracked satellites in orbit", "crep": None, "tokens": ["SAT"], "tags": {"satellite", "orbit", "iss", "starlink"}},
    "solar_events": {"label": "Solar Events", "desc": "Solar flares, CMEs, geomagnetic storms", "crep": None, "tokens": ["SOLAR", "GEO"], "tags": {"solar_flare", "cme", "geomagnetic_storm"}},
    # Monitoring
    "cameras": {"label": "Cameras", "desc": "Public webcams and CCTV", "crep": "cameras", "tokens": [], "tags": {"webcam", "cctv", "camera"}},
    # Military
    "military_installations": {"label": "Military Sites", "desc": "Known military installations", "crep": "military", "tokens": [], "tags": {"military", "base", "defense"}},
    # Telemetry
    "devices": {"label": "Devices", "desc": "MycoBrain devices and sensors", "crep": None, "tokens": ["TELEM", "BRAIN_BOARD"], "tags": {"device", "sensor", "mycobrain"}},
    "telemetry": {"label": "Telemetry Streams", "desc": "Sensor data streams", "crep": None, "tokens": ["TELEM"], "tags": {"telemetry", "stream", "reading"}},
    # Knowledge
    "research": {"label": "Research", "desc": "Scientific publications", "crep": None, "tokens": [], "tags": {"paper", "publication", "research"}},
    "crep_entities": {"label": "CREP Entities", "desc": "Unified map entities", "crep": None, "tokens": [], "tags": {"crep", "map", "entity"}},
}


class DomainRegistry:
    """Central registry of all searchable Earth domains — mirrors mindex exactly."""

    def __init__(self) -> None:
        self._domains: Dict[str, SearchDomain] = {}
        for key in ALL_DOMAINS:
            meta = _DOMAIN_META.get(key, {})
            self._domains[key] = SearchDomain(
                key=key,
                label=meta.get("label", key.replace("_", " ").title()),
                description=meta.get("desc", ""),
                group=DOMAIN_TO_GROUP.get(key, "other"),
                table=DOMAIN_TABLES.get(key, ""),
                crep_layer=meta.get("crep"),
                bio_token_prefixes=meta.get("tokens", []),
                tags=meta.get("tags", set()),
            )

    def get(self, key: str) -> Optional[SearchDomain]:
        return self._domains.get(key)

    def list_domains(self) -> List[SearchDomain]:
        return list(self._domains.values())

    def list_keys(self) -> List[str]:
        return list(ALL_DOMAINS)

    def list_groups(self) -> List[str]:
        return sorted(DOMAIN_GROUPS.keys())

    def resolve_types(self, types_str: Optional[str]) -> List[str]:
        """Resolve a types string (comma-separated domains or group alias) into domain keys."""
        if not types_str or types_str == "all":
            return list(ALL_DOMAINS)
        parts = [t.strip() for t in types_str.split(",") if t.strip()]
        resolved: List[str] = []
        for p in parts:
            if p in DOMAIN_GROUPS:
                resolved.extend(DOMAIN_GROUPS[p])
            elif p in self._domains:
                resolved.append(p)
        return list(dict.fromkeys(resolved))  # dedupe preserving order

    def search_domains(self, query: str) -> List[SearchDomain]:
        q = query.lower()
        return [
            d for d in self._domains.values()
            if q in d.key or q in d.label.lower()
            or q in d.description.lower()
            or any(q in t for t in d.tags)
        ]

    def domains_with_crep_layer(self) -> List[SearchDomain]:
        return [d for d in self._domains.values() if d.crep_layer]

    def register(self, domain: SearchDomain) -> None:
        self._domains[domain.key] = domain
