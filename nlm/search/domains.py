"""
NLM Earth Domain Taxonomy
==========================

Every searchable category of Earth data.  Hierarchical keys like
``life.fungi.species`` or ``environment.earthquakes`` let the search
engine, ingestion pipeline, CREP, and Myca know what exists and which
data sources serve each domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class SearchDomain:
    key: str
    label: str
    description: str
    parent_key: Optional[str] = None
    source_keys: List[str] = field(default_factory=list)
    crep_layer: Optional[str] = None
    bio_token_prefixes: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return self.key.count(".")

    @property
    def root(self) -> str:
        return self.key.split(".")[0]


class DomainRegistry:
    """Central registry of all searchable Earth domains."""

    def __init__(self) -> None:
        self._domains: Dict[str, SearchDomain] = {}
        self._build()

    def get(self, key: str) -> Optional[SearchDomain]:
        return self._domains.get(key)

    def list_domains(self, root: Optional[str] = None) -> List[SearchDomain]:
        if root:
            return [d for d in self._domains.values() if d.root == root]
        return list(self._domains.values())

    def list_roots(self) -> List[str]:
        return sorted({d.root for d in self._domains.values()})

    def search_domains(self, query: str) -> List[SearchDomain]:
        q = query.lower()
        return [
            d for d in self._domains.values()
            if q in d.key.lower() or q in d.label.lower()
            or q in d.description.lower() or any(q in t for t in d.tags)
        ]

    def register(self, domain: SearchDomain) -> None:
        self._domains[domain.key] = domain

    def all_source_keys(self) -> Set[str]:
        keys: Set[str] = set()
        for d in self._domains.values():
            keys.update(d.source_keys)
        return keys

    def domains_for_source(self, source_key: str) -> List[SearchDomain]:
        return [d for d in self._domains.values() if source_key in d.source_keys]

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _r(self, key, label, desc, *, parent=None, sources=None,
           crep=None, tokens=None, tags=None):
        self._domains[key] = SearchDomain(
            key=key, label=label, description=desc, parent_key=parent,
            source_keys=sources or [], crep_layer=crep,
            bio_token_prefixes=tokens or [], tags=tags or set(),
        )

    def _build(self) -> None:
        self._build_life()
        self._build_environment()
        self._build_infrastructure()
        self._build_signals()
        self._build_space()
        self._build_transportation()
        self._build_science()
        self._build_intelligence()

    # ---- LIFE --------------------------------------------------------

    def _build_life(self) -> None:
        self._r("life", "All Life", "Every living organism on Earth",
                sources=["gbif", "inaturalist", "ncbi", "bold"],
                crep="life_layer", tokens=["BIO"],
                tags={"species", "organism", "biodiversity"})

        # Fungi
        self._r("life.fungi", "Fungi", "All fungal species",
                parent="life",
                sources=["gbif", "inaturalist", "mycobank", "fungidb",
                         "fungorum", "mushroom_observer"],
                crep="fungi_layer", tokens=["BIO", "GROWTH"],
                tags={"fungi", "mushroom", "mycology", "yeast", "mold"})
        self._r("life.fungi.species", "Fungal Species",
                "Fungal taxonomy records", parent="life.fungi",
                sources=["gbif", "mycobank", "fungidb", "fungorum"])
        self._r("life.fungi.observations", "Fungal Observations",
                "Field observations of fungi", parent="life.fungi",
                sources=["inaturalist", "gbif", "mushroom_observer"])
        self._r("life.fungi.compounds", "Fungal Compounds",
                "Bioactive compounds from fungi", parent="life.fungi",
                sources=["pubchem", "fungidb", "mindex"])
        self._r("life.fungi.genetics", "Fungal Genetics",
                "Genomic data for fungi", parent="life.fungi",
                sources=["ncbi", "bold", "fungidb"])

        # Plants
        self._r("life.plants", "Plants", "All plant species",
                parent="life",
                sources=["gbif", "inaturalist", "tropicos", "ipni", "powo"],
                crep="plants_layer", tokens=["BIO"],
                tags={"plant", "flora", "botany", "tree", "flower"})

        # Animal groups
        for key, label, extra_src, extra_tags in [
            ("life.birds", "Birds",
             ["ebird", "birdlife"],
             {"bird", "avian", "ornithology"}),
            ("life.mammals", "Mammals",
             ["iucn"],
             {"mammal", "wildlife"}),
            ("life.reptiles", "Reptiles",
             ["reptile_database"],
             {"reptile", "herpetology"}),
            ("life.amphibians", "Amphibians",
             ["amphibiaweb"],
             {"amphibian", "frog", "salamander"}),
            ("life.insects", "Insects",
             ["bold"],
             {"insect", "entomology", "arthropod", "butterfly", "bee"}),
            ("life.marine", "Marine Life",
             ["obis", "worms", "fishbase"],
             {"marine", "ocean", "fish", "coral", "whale", "shark"}),
        ]:
            self._r(key, label, f"All {label.lower()}",
                    parent="life",
                    sources=["gbif", "inaturalist", "ncbi"] + extra_src,
                    crep=f"{key.split('.')[-1]}_layer", tokens=["BIO"],
                    tags=extra_tags)

        # Microbes
        self._r("life.bacteria", "Bacteria", "All bacterial species",
                parent="life", sources=["ncbi", "silva"],
                crep="bacteria_layer", tags={"bacteria", "microbe"})
        self._r("life.archaea", "Archaea", "All archaeal species",
                parent="life", sources=["ncbi", "silva"],
                tags={"archaea", "extremophile"})

    # ---- ENVIRONMENT -------------------------------------------------

    def _build_environment(self) -> None:
        self._r("environment", "Environment",
                "All environmental and climate data",
                crep="env_layer",
                tokens=["TEMP", "HUM", "CO2", "PRESS"],
                tags={"environment", "climate", "weather"})

        self._r("environment.weather", "Weather",
                "Real-time and historical weather",
                parent="environment",
                sources=["noaa_weather", "openweather", "airs", "era5"],
                crep="weather_layer",
                tags={"weather", "forecast", "temperature", "wind"})
        self._r("environment.weather.storms", "Storms",
                "Hurricanes, cyclones, tropical storms",
                parent="environment.weather",
                sources=["noaa_nhc", "jtwc", "ibtracs"],
                crep="storms_layer",
                tags={"storm", "hurricane", "cyclone", "typhoon"})
        self._r("environment.weather.tornadoes", "Tornadoes",
                "Tornado events and watches",
                parent="environment.weather",
                sources=["noaa_spc"],
                crep="tornadoes_layer", tags={"tornado"})
        self._r("environment.weather.lightning", "Lightning",
                "Lightning strike data",
                parent="environment.weather",
                sources=["vaisala_gld360", "wwlln"],
                crep="lightning_layer", tags={"lightning"})
        self._r("environment.weather.floods", "Floods",
                "Flood events and river levels",
                parent="environment.weather",
                sources=["noaa_nwm", "glofas", "usgs_waterwatch"],
                crep="floods_layer", tags={"flood", "river"})

        # Satellite & Remote Sensing
        self._r("environment.satellite", "Satellite Imagery",
                "MODIS, Landsat, Sentinel, VIIRS",
                parent="environment",
                sources=["modis", "landsat", "sentinel", "viirs"],
                crep="satellite_layer",
                tags={"satellite", "remote_sensing", "ndvi"})

        # Air Quality & Emissions
        self._r("environment.air_quality", "Air Quality",
                "AQI, particulate matter, pollutants",
                parent="environment",
                sources=["epa_aqs", "openaq", "airnow", "copernicus_cams"],
                crep="air_quality_layer",
                tags={"air_quality", "aqi", "pm25", "ozone", "pollution"})
        self._r("environment.emissions", "Emissions",
                "CO2, methane, greenhouse gases",
                parent="environment",
                sources=["noaa_gml", "copernicus_ghg", "oco2", "tropomi"],
                crep="emissions_layer", tokens=["CO2"],
                tags={"co2", "methane", "emissions", "carbon"})

        # Water & Oceans
        self._r("environment.water", "Water",
                "Water quality, rivers, aquifers",
                parent="environment",
                sources=["usgs_water", "epa_waters"],
                crep="water_layer", tags={"water", "river", "lake"})
        self._r("environment.oceans", "Oceans",
                "SST, currents, salinity, buoys",
                parent="environment",
                sources=["noaa_ocean", "copernicus_marine", "argo", "ndbc"],
                crep="ocean_layer",
                tags={"ocean", "sea", "buoy", "tide", "sst"})

        # Soil
        self._r("environment.soil", "Soil",
                "Soil quality, composition, contamination",
                parent="environment",
                sources=["soilgrids", "usda_nrcs"],
                crep="soil_layer", tokens=["MOIST", "PH"],
                tags={"soil", "ground", "contamination"})

        # Geological events
        self._r("environment.earthquakes", "Earthquakes",
                "Seismic events worldwide",
                parent="environment",
                sources=["usgs_earthquake", "emsc", "iris"],
                crep="earthquakes_layer", tokens=["SEISMIC"],
                tags={"earthquake", "seismic", "quake"})
        self._r("environment.volcanoes", "Volcanoes",
                "Volcanic activity and eruptions",
                parent="environment",
                sources=["smithsonian_gvp", "mirova"],
                crep="volcanoes_layer",
                tags={"volcano", "eruption", "lava"})
        self._r("environment.wildfires", "Wildfires",
                "Active fires and burn areas",
                parent="environment",
                sources=["firms", "nifc", "copernicus_effis"],
                crep="wildfires_layer",
                tags={"wildfire", "fire", "burn"})

    # ---- INFRASTRUCTURE ----------------------------------------------

    def _build_infrastructure(self) -> None:
        self._r("infrastructure", "Infrastructure",
                "All human-built infrastructure",
                crep="infra_layer",
                tags={"infrastructure", "facility", "industrial"})

        self._r("infrastructure.energy", "Energy",
                "Power plants, grid, oil & gas",
                parent="infrastructure",
                sources=["wri_powerplants", "eia", "osm",
                         "global_energy_monitor"],
                crep="energy_layer",
                tags={"energy", "power", "electricity", "coal",
                      "nuclear", "solar", "wind", "hydro",
                      "oil", "gas", "pipeline", "refinery"})
        self._r("infrastructure.mining", "Mining",
                "Mining operations and mineral sites",
                parent="infrastructure",
                sources=["usgs_mrds", "osm"],
                crep="mining_layer",
                tags={"mining", "mine", "quarry", "ore"})
        self._r("infrastructure.factories", "Factories",
                "Manufacturing and industrial facilities",
                parent="infrastructure",
                sources=["epa_tri", "epa_frs", "osm"],
                crep="factories_layer",
                tags={"factory", "manufacturing", "industrial"})
        self._r("infrastructure.water", "Water Infrastructure",
                "Dams, treatment plants, reservoirs",
                parent="infrastructure",
                sources=["nid", "epa_sdwis", "osm"],
                crep="water_infra_layer",
                tags={"dam", "reservoir", "treatment_plant"})
        self._r("infrastructure.waste", "Waste & Pollution",
                "Landfills, superfund, toxic sites",
                parent="infrastructure",
                sources=["epa_superfund", "epa_rcra", "osm"],
                crep="waste_layer",
                tags={"waste", "landfill", "superfund", "toxic"})

    # ---- SIGNALS & COMMUNICATION -------------------------------------

    def _build_signals(self) -> None:
        self._r("signals", "Signals & Communication",
                "All electromagnetic and communication infrastructure",
                crep="signals_layer",
                tags={"signal", "antenna", "rf", "electromagnetic"})

        self._r("signals.cell_towers", "Cell Towers",
                "Cellular antenna locations",
                parent="signals",
                sources=["opencellid", "fcc_antenna", "osm"],
                crep="cell_towers_layer",
                tags={"cell_tower", "4g", "5g", "lte"})
        self._r("signals.radio", "Radio Antennas",
                "AM/FM broadcast antennas",
                parent="signals",
                sources=["fcc_antenna", "fcc_fm", "fcc_am"],
                crep="radio_layer",
                tags={"radio", "am", "fm", "broadcast"})
        self._r("signals.wifi_bluetooth", "WiFi & Bluetooth",
                "Known hotspots and device clusters",
                parent="signals",
                sources=["wigle", "osm"],
                crep="wifi_layer",
                tags={"wifi", "bluetooth", "hotspot"})
        self._r("signals.internet_cables", "Internet Cables",
                "Submarine and terrestrial fiber",
                parent="signals",
                sources=["submarinecablemap", "telegeography"],
                crep="cables_layer",
                tags={"internet", "cable", "fiber", "submarine"})
        self._r("signals.power_grid", "Power Grid",
                "Transmission lines and substations",
                parent="signals",
                sources=["eia", "osm", "hifld"],
                crep="power_grid_layer",
                tags={"grid", "transmission", "substation"})

    # ---- SPACE -------------------------------------------------------

    def _build_space(self) -> None:
        self._r("space", "Space", "All space-related data",
                crep="space_layer",
                tags={"space", "satellite", "orbit"})

        self._r("space.satellites", "Satellites",
                "All tracked satellites in orbit",
                parent="space",
                sources=["celestrak", "space_track", "n2yo"],
                crep="satellites_layer",
                tags={"satellite", "iss", "starlink", "gps_sat"})
        self._r("space.weather", "Space Weather",
                "Solar flares, CMEs, geomagnetic storms",
                parent="space",
                sources=["noaa_swpc", "nasa_donki", "soho", "stereo"],
                crep="space_weather_layer",
                tags={"solar_flare", "cme", "geomagnetic_storm", "aurora"})
        self._r("space.launches", "Space Launches",
                "Rocket launches from Earth",
                parent="space",
                sources=["launch_library", "spacex_api"],
                crep="launches_layer",
                tags={"launch", "rocket"})
        self._r("space.nasa_feeds", "NASA Data Feeds",
                "SOHO, STEREO A/B, SDO imagery and telemetry",
                parent="space",
                sources=["soho", "stereo", "sdo", "nasa_eyes"],
                crep="nasa_feeds_layer",
                tags={"soho", "stereo", "sdo", "nasa"})

    # ---- TRANSPORTATION ----------------------------------------------

    def _build_transportation(self) -> None:
        self._r("transportation", "Transportation",
                "All moving vehicles and transport infrastructure",
                crep="transport_layer",
                tags={"transportation", "vehicle"})

        self._r("transportation.aviation", "Aviation",
                "All aircraft in flight and airports",
                parent="transportation",
                sources=["adsb_exchange", "opensky", "faa_airports"],
                crep="aviation_layer",
                tags={"airplane", "flight", "airport"})
        self._r("transportation.maritime", "Maritime",
                "Ships, boats, AIS transponders",
                parent="transportation",
                sources=["ais", "marinetraffic", "vesselfinder"],
                crep="maritime_layer",
                tags={"ship", "boat", "vessel", "ais", "port"})
        self._r("transportation.ports", "Ports",
                "Shipping ports and harbors",
                parent="transportation",
                sources=["wpi", "osm"],
                crep="ports_layer",
                tags={"port", "harbor", "dock"})
        self._r("transportation.spaceports", "Spaceports",
                "Launch facilities worldwide",
                parent="transportation",
                sources=["launch_library", "osm"],
                crep="spaceports_layer",
                tags={"spaceport", "launch_pad"})

    # ---- SCIENCE DATABASES -------------------------------------------

    def _build_science(self) -> None:
        self._r("science", "Science",
                "Scientific databases, literature, research",
                tags={"science", "research", "academic"})

        self._r("science.pubchem", "PubChem",
                "Chemical compound database",
                parent="science", sources=["pubchem"],
                tags={"chemistry", "compound", "molecule"})
        self._r("science.genbank", "GenBank",
                "Genetic sequence database",
                parent="science", sources=["ncbi"],
                tags={"genetics", "genome", "dna", "sequence"})
        self._r("science.literature", "Scientific Literature",
                "Research papers and citations",
                parent="science",
                sources=["semantic_scholar", "crossref", "pubmed"],
                tags={"paper", "publication", "journal"})
        self._r("science.mathematics", "Mathematics",
                "Computations and constants",
                parent="science", sources=["wolfram", "oeis"],
                tags={"math", "equation"})
        self._r("science.physics", "Physics",
                "Physics datasets and constants",
                parent="science", sources=["nist", "cern_opendata"],
                tags={"physics", "particle", "quantum"})

    # ---- INTELLIGENCE / MONITORING -----------------------------------

    def _build_intelligence(self) -> None:
        self._r("intelligence", "Intelligence & Monitoring",
                "Publicly accessible monitoring feeds",
                crep="intel_layer",
                tags={"monitoring", "surveillance", "camera"})

        self._r("intelligence.webcams", "Public Webcams",
                "Publicly accessible webcams and CCTV",
                parent="intelligence",
                sources=["insecam", "webcamtaxi", "windy_webcams"],
                crep="webcams_layer",
                tags={"webcam", "cctv", "camera"})
        self._r("intelligence.military", "Military Sites",
                "Publicly known military installations",
                parent="intelligence",
                sources=["osm", "milbases_osint"],
                crep="military_layer",
                tags={"military", "base", "defense"})
