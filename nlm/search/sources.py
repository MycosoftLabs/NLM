"""
NLM Data Source Registry
=========================

Every external API, database, or feed that NLM can pull from.  Each source
has a key, base URL, auth method, rate-limit info, and the response format
so the ingestion pipeline knows how to normalise the data before storing
it in the local mindex database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AuthMethod(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    TOKEN = "token"


class ResponseFormat(str, Enum):
    JSON = "json"
    GEOJSON = "geojson"
    CSV = "csv"
    XML = "xml"
    BINARY = "binary"


@dataclass
class DataSource:
    key: str
    name: str
    description: str
    base_url: str
    auth_method: AuthMethod = AuthMethod.NONE
    response_format: ResponseFormat = ResponseFormat.JSON
    rate_limit_rpm: int = 60
    supports_geospatial: bool = False
    supports_temporal: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSourceRegistry:
    """Catalogue of every external data source NLM can ingest from."""

    def __init__(self) -> None:
        self._sources: Dict[str, DataSource] = {}
        self._build()

    def get(self, key: str) -> Optional[DataSource]:
        return self._sources.get(key)

    def list_sources(self) -> List[DataSource]:
        return list(self._sources.values())

    def search(self, query: str) -> List[DataSource]:
        q = query.lower()
        return [
            s for s in self._sources.values()
            if q in s.key or q in s.name.lower()
            or q in s.description.lower()
            or any(q in t for t in s.tags)
        ]

    def register(self, source: DataSource) -> None:
        self._sources[source.key] = source

    def _s(self, key, name, desc, url, *, auth=AuthMethod.NONE,
           fmt=ResponseFormat.JSON, rpm=60, geo=False, temporal=False,
           tags=None):
        self._sources[key] = DataSource(
            key=key, name=name, description=desc, base_url=url,
            auth_method=auth, response_format=fmt, rate_limit_rpm=rpm,
            supports_geospatial=geo, supports_temporal=temporal,
            tags=tags or [],
        )

    # ------------------------------------------------------------------
    # Build all sources
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # ---- Biodiversity --------------------------------------------
        self._s("gbif", "GBIF", "Global Biodiversity Information Facility",
                "https://api.gbif.org/v1", geo=True, temporal=True,
                tags=["species", "occurrence", "biodiversity"])
        self._s("inaturalist", "iNaturalist", "Citizen science observations",
                "https://api.inaturalist.org/v1", geo=True, temporal=True,
                tags=["species", "observation", "citizen_science"])
        self._s("ncbi", "NCBI", "National Center for Biotechnology Information",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
                auth=AuthMethod.API_KEY,
                tags=["genetics", "genome", "sequence", "taxonomy"])
        self._s("bold", "BOLD Systems", "Barcode of Life",
                "https://v3.boldsystems.org/index.php/API_Public",
                tags=["barcode", "dna", "species"])
        self._s("iucn", "IUCN Red List", "Threatened species assessments",
                "https://apiv3.iucnredlist.org/api/v3",
                auth=AuthMethod.TOKEN,
                tags=["conservation", "endangered", "threatened"])

        # ---- Mycology ------------------------------------------------
        self._s("mycobank", "MycoBank", "Fungal nomenclature database",
                "https://www.mycobank.org/Services/SearchService.asmx",
                fmt=ResponseFormat.XML,
                tags=["fungi", "nomenclature", "taxonomy"])
        self._s("fungidb", "FungiDB", "Fungal and oomycete genomics",
                "https://fungidb.org/fungidb/service",
                tags=["fungi", "genomics"])
        self._s("fungorum", "Index Fungorum", "Global fungal names index",
                "http://www.indexfungorum.org/ixfwebservice/fungus.asmx",
                fmt=ResponseFormat.XML,
                tags=["fungi", "nomenclature"])
        self._s("mushroom_observer", "Mushroom Observer",
                "Community mushroom observations",
                "https://mushroomobserver.org/api2",
                geo=True, tags=["fungi", "observation"])

        # ---- Botany --------------------------------------------------
        self._s("tropicos", "Tropicos", "Missouri Botanical Garden plant data",
                "https://services.tropicos.org/Name",
                auth=AuthMethod.API_KEY, tags=["plant", "taxonomy"])
        self._s("ipni", "IPNI", "International Plant Names Index",
                "https://www.ipni.org/api/1", tags=["plant", "nomenclature"])
        self._s("powo", "POWO", "Plants of the World Online",
                "https://powo.science.kew.org/api/2",
                tags=["plant", "distribution"])

        # ---- Ornithology ---------------------------------------------
        self._s("ebird", "eBird", "Cornell Lab bird observations",
                "https://api.ebird.org/v2",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["bird", "observation"])
        self._s("birdlife", "BirdLife International",
                "Global bird conservation data",
                "https://datazone.birdlife.org/api",
                tags=["bird", "conservation"])

        # ---- Marine --------------------------------------------------
        self._s("obis", "OBIS", "Ocean Biodiversity Information System",
                "https://api.obis.org/v3", geo=True, temporal=True,
                tags=["marine", "ocean", "species"])
        self._s("worms", "WoRMS", "World Register of Marine Species",
                "https://www.marinespecies.org/rest",
                tags=["marine", "taxonomy"])
        self._s("fishbase", "FishBase", "Global fish species database",
                "https://fishbase.ropensci.org",
                tags=["fish", "marine"])

        # ---- Other taxonomic -----------------------------------------
        self._s("reptile_database", "Reptile Database",
                "Reptile species database",
                "http://reptile-database.reptarium.cz/api",
                tags=["reptile"])
        self._s("amphibiaweb", "AmphibiaWeb", "Amphibian species database",
                "https://amphibiaweb.org/api",
                tags=["amphibian"])
        self._s("silva", "SILVA", "Ribosomal RNA gene database",
                "https://www.arb-silva.de/no_cache/download",
                tags=["bacteria", "archaea", "rrna"])

        # ---- Weather & Atmosphere ------------------------------------
        self._s("noaa_weather", "NOAA Weather", "National Weather Service",
                "https://api.weather.gov", fmt=ResponseFormat.GEOJSON,
                geo=True, temporal=True,
                tags=["weather", "forecast", "nws"])
        self._s("openweather", "OpenWeatherMap", "Global weather data",
                "https://api.openweathermap.org/data/3.0",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["weather", "forecast"])
        self._s("airs", "AIRS", "Atmospheric Infrared Sounder",
                "https://airs.jpl.nasa.gov/data/get-data",
                tags=["atmosphere", "temperature", "humidity"])
        self._s("era5", "ERA5", "ECMWF climate reanalysis",
                "https://cds.climate.copernicus.eu/api/v2",
                auth=AuthMethod.API_KEY, temporal=True,
                tags=["climate", "reanalysis", "historical"])
        self._s("noaa_nhc", "NOAA NHC", "National Hurricane Center",
                "https://www.nhc.noaa.gov/data",
                geo=True, tags=["hurricane", "storm", "tropical"])
        self._s("noaa_spc", "NOAA SPC", "Storm Prediction Center",
                "https://www.spc.noaa.gov/products",
                geo=True, tags=["tornado", "severe_weather"])
        self._s("jtwc", "JTWC", "Joint Typhoon Warning Center",
                "https://www.metoc.navy.mil/jtwc/jtwc.html",
                tags=["typhoon", "cyclone"])
        self._s("ibtracs", "IBTrACS", "Historical tropical cyclone tracks",
                "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs",
                geo=True, temporal=True, tags=["storm", "historical"])

        # ---- Lightning -----------------------------------------------
        self._s("vaisala_gld360", "Vaisala GLD360",
                "Global lightning detection",
                "https://www.vaisala.com/en/products/systems/lightning-detection",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["lightning"])
        self._s("wwlln", "WWLLN",
                "World Wide Lightning Location Network",
                "http://wwlln.net/data",
                geo=True, tags=["lightning"])

        # ---- Satellite & Remote Sensing ------------------------------
        self._s("modis", "MODIS", "Moderate Resolution Imaging Spectroradiometer",
                "https://modis.gsfc.nasa.gov/data",
                geo=True, temporal=True,
                tags=["satellite", "imagery", "ndvi", "fire"])
        self._s("landsat", "Landsat", "USGS Landsat imagery",
                "https://landsatlook.usgs.gov/stac-server",
                geo=True, temporal=True,
                tags=["satellite", "imagery", "land_cover"])
        self._s("sentinel", "Sentinel", "Copernicus Sentinel satellites",
                "https://scihub.copernicus.eu/dhus/odata/v1",
                auth=AuthMethod.TOKEN, geo=True, temporal=True,
                tags=["satellite", "imagery", "sar"])
        self._s("viirs", "VIIRS", "Visible Infrared Imaging Radiometer Suite",
                "https://firms.modaps.eosdis.nasa.gov/api",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["satellite", "fire", "nightlight"])

        # ---- Air Quality & Emissions ---------------------------------
        self._s("epa_aqs", "EPA AQS", "EPA Air Quality System",
                "https://aqs.epa.gov/data/api",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["air_quality", "pollutant"])
        self._s("openaq", "OpenAQ", "Open air quality data",
                "https://api.openaq.org/v2", geo=True, temporal=True,
                tags=["air_quality", "pm25"])
        self._s("airnow", "AirNow", "US/Canada air quality",
                "https://www.airnowapi.org/aq/data",
                auth=AuthMethod.API_KEY, geo=True,
                tags=["air_quality", "aqi"])
        self._s("copernicus_cams", "CAMS", "Copernicus Atmosphere Monitoring",
                "https://ads.atmosphere.copernicus.eu/api/v2",
                auth=AuthMethod.API_KEY,
                tags=["air_quality", "atmosphere"])
        self._s("noaa_gml", "NOAA GML", "Global Monitoring Laboratory",
                "https://gml.noaa.gov/webdata",
                temporal=True,
                tags=["co2", "methane", "greenhouse"])
        self._s("copernicus_ghg", "Copernicus GHG",
                "Greenhouse gas monitoring",
                "https://cds.climate.copernicus.eu/api/v2",
                auth=AuthMethod.API_KEY,
                tags=["co2", "methane"])
        self._s("oco2", "OCO-2", "Orbiting Carbon Observatory",
                "https://oco2.gesdisc.eosdis.nasa.gov",
                auth=AuthMethod.TOKEN, geo=True,
                tags=["co2", "carbon"])
        self._s("tropomi", "TROPOMI", "Tropospheric Monitoring Instrument",
                "https://s5phub.copernicus.eu/dhus",
                auth=AuthMethod.TOKEN, geo=True,
                tags=["no2", "methane", "ozone"])

        # ---- Seismic & Geological ------------------------------------
        self._s("usgs_earthquake", "USGS Earthquake",
                "Real-time earthquake data",
                "https://earthquake.usgs.gov/fdsnws/event/1",
                geo=True, temporal=True,
                tags=["earthquake", "seismic"])
        self._s("emsc", "EMSC", "European-Mediterranean Seismological Centre",
                "https://www.seismicportal.eu/fdsnws/event/1",
                geo=True, temporal=True, tags=["earthquake"])
        self._s("iris", "IRIS", "Seismological data",
                "https://service.iris.edu/fdsnws/event/1",
                geo=True, temporal=True, tags=["seismic", "waveform"])
        self._s("smithsonian_gvp", "GVP",
                "Smithsonian Global Volcanism Program",
                "https://volcano.si.edu/database",
                geo=True, tags=["volcano", "eruption"])
        self._s("mirova", "MIROVA", "Middle InfraRed Observation of Volcanic Activity",
                "https://www.mirovaweb.it",
                geo=True, tags=["volcano", "thermal"])

        # ---- Wildfire ------------------------------------------------
        self._s("firms", "FIRMS", "Fire Information for Resource Management",
                "https://firms.modaps.eosdis.nasa.gov/api",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["fire", "wildfire"])
        self._s("nifc", "NIFC", "National Interagency Fire Center",
                "https://data-nifc.opendata.arcgis.com/api",
                geo=True, tags=["fire", "wildfire"])
        self._s("copernicus_effis", "EFFIS",
                "European Forest Fire Information System",
                "https://effis.jrc.ec.europa.eu/api",
                geo=True, tags=["fire", "wildfire"])

        # ---- Water & Oceans ------------------------------------------
        self._s("usgs_water", "USGS Water", "USGS water data",
                "https://waterservices.usgs.gov/nwis",
                geo=True, temporal=True,
                tags=["water", "river", "streamflow"])
        self._s("epa_waters", "EPA WATERS", "EPA water quality",
                "https://watersgeo.epa.gov/arcgis/rest/services",
                geo=True, tags=["water", "quality"])
        self._s("usgs_waterwatch", "USGS WaterWatch",
                "Real-time streamflow",
                "https://waterwatch.usgs.gov",
                geo=True, temporal=True, tags=["flood", "streamflow"])
        self._s("glofas", "GloFAS", "Global Flood Awareness System",
                "https://cds.climate.copernicus.eu/api/v2",
                auth=AuthMethod.API_KEY, geo=True,
                tags=["flood", "river"])
        self._s("noaa_nwm", "NOAA NWM", "National Water Model",
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm",
                geo=True, tags=["flood", "hydrology"])
        self._s("noaa_ocean", "NOAA Ocean", "Ocean data",
                "https://www.ncei.noaa.gov/erddap",
                geo=True, temporal=True,
                tags=["ocean", "sst", "current"])
        self._s("copernicus_marine", "CMEMS",
                "Copernicus Marine Environment Monitoring",
                "https://nrt.cmems-du.eu/motu-web/Motu",
                auth=AuthMethod.TOKEN, geo=True, temporal=True,
                tags=["ocean", "marine"])
        self._s("argo", "Argo", "Global ocean profiling floats",
                "https://api-argo.ifremer.fr",
                geo=True, temporal=True,
                tags=["ocean", "temperature", "salinity"])
        self._s("ndbc", "NDBC", "National Data Buoy Center",
                "https://www.ndbc.noaa.gov/data",
                geo=True, temporal=True,
                tags=["buoy", "wave", "ocean"])
        self._s("noaa_tides", "NOAA Tides",
                "Tides and currents",
                "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
                geo=True, temporal=True, tags=["tide", "current"])

        # ---- Soil ----------------------------------------------------
        self._s("soilgrids", "SoilGrids", "Global soil data",
                "https://rest.isric.org/soilgrids/v2.0",
                geo=True, tags=["soil", "composition"])
        self._s("usda_nrcs", "USDA NRCS", "Soil Survey",
                "https://sdmdataaccess.sc.egov.usda.gov",
                geo=True, tags=["soil", "survey"])

        # ---- Infrastructure ------------------------------------------
        self._s("wri_powerplants", "WRI Power Plants",
                "Global Power Plant Database",
                "https://datasets.wri.org/api/3",
                geo=True, tags=["power_plant", "energy"])
        self._s("eia", "EIA", "Energy Information Administration",
                "https://api.eia.gov/v2",
                auth=AuthMethod.API_KEY,
                tags=["energy", "oil", "gas", "electricity"])
        self._s("global_energy_monitor", "GEM",
                "Global Energy Monitor",
                "https://globalenergymonitor.org/api",
                geo=True, tags=["energy", "coal", "gas"])
        self._s("osm", "OpenStreetMap", "Overpass API for infrastructure",
                "https://overpass-api.de/api/interpreter",
                geo=True,
                tags=["infrastructure", "road", "building"])
        self._s("hifld", "HIFLD", "Homeland Infrastructure Foundation",
                "https://hifld-geoplatform.opendata.arcgis.com/api",
                geo=True,
                tags=["infrastructure", "critical"])
        self._s("usgs_mrds", "USGS MRDS",
                "Mineral Resources Data System",
                "https://mrdata.usgs.gov/mrds/api",
                geo=True, tags=["mining", "mineral"])
        self._s("epa_tri", "EPA TRI",
                "Toxics Release Inventory",
                "https://enviro.epa.gov/triexplorer",
                geo=True, tags=["pollution", "toxic", "factory"])
        self._s("epa_frs", "EPA FRS",
                "Facility Registry Service",
                "https://ofmpub.epa.gov/frs_public2/frs_rest_services.get_facilities",
                geo=True, tags=["facility", "regulated"])
        self._s("nid", "NID", "National Inventory of Dams",
                "https://nid.sec.usace.army.mil/api",
                auth=AuthMethod.API_KEY, geo=True, tags=["dam"])
        self._s("epa_sdwis", "EPA SDWIS",
                "Safe Drinking Water Information System",
                "https://enviro.epa.gov/enviro/efservice",
                geo=True, tags=["water", "drinking"])
        self._s("epa_superfund", "EPA Superfund",
                "Superfund site locations",
                "https://enviro.epa.gov/enviro/efservice",
                geo=True, tags=["superfund", "contamination"])
        self._s("epa_rcra", "EPA RCRA",
                "Hazardous waste sites",
                "https://enviro.epa.gov/enviro/efservice",
                geo=True, tags=["hazardous", "waste"])

        # ---- Signals & Communication ---------------------------------
        self._s("opencellid", "OpenCelliD",
                "Open cell tower database",
                "https://opencellid.org/ajax/searchCell.php",
                geo=True, tags=["cell_tower", "cellular"])
        self._s("fcc_antenna", "FCC Antenna Structure",
                "FCC registered antenna structures",
                "https://www.fcc.gov/api",
                geo=True, tags=["antenna", "tower"])
        self._s("fcc_fm", "FCC FM", "FM radio stations",
                "https://www.fcc.gov/api", geo=True,
                tags=["radio", "fm"])
        self._s("fcc_am", "FCC AM", "AM radio stations",
                "https://www.fcc.gov/api", geo=True,
                tags=["radio", "am"])
        self._s("wigle", "WiGLE", "Wireless network mapping",
                "https://api.wigle.net/api/v2",
                auth=AuthMethod.API_KEY, geo=True,
                tags=["wifi", "bluetooth", "cell"])
        self._s("submarinecablemap", "Submarine Cable Map",
                "Undersea internet cable routes",
                "https://www.submarinecablemap.com/api/v3",
                geo=True, tags=["cable", "internet", "submarine"])
        self._s("telegeography", "TeleGeography",
                "Telecom infrastructure data",
                "https://www.telegeography.com/api",
                geo=True, tags=["cable", "internet"])
        self._s("opensignal", "OpenSignal",
                "Mobile signal strength maps",
                "https://opensignal.com/api",
                auth=AuthMethod.API_KEY, geo=True,
                tags=["signal", "coverage"])
        self._s("fcc_broadband", "FCC Broadband Map",
                "US broadband coverage",
                "https://broadbandmap.fcc.gov/api",
                geo=True, tags=["broadband", "internet"])

        # ---- Space ---------------------------------------------------
        self._s("celestrak", "CelesTrak",
                "Satellite catalog and TLE data",
                "https://celestrak.org/NORAD/elements/gp.php",
                tags=["satellite", "tle", "orbit"])
        self._s("space_track", "Space-Track",
                "DoD satellite tracking",
                "https://www.space-track.org/basicspacedata/query",
                auth=AuthMethod.TOKEN,
                tags=["satellite", "debris"])
        self._s("n2yo", "N2YO", "Satellite tracking",
                "https://api.n2yo.com/rest/v1/satellite",
                auth=AuthMethod.API_KEY,
                tags=["satellite", "tracking"])
        self._s("noaa_swpc", "NOAA SWPC",
                "Space Weather Prediction Center",
                "https://services.swpc.noaa.gov/json",
                temporal=True,
                tags=["solar_flare", "geomagnetic", "space_weather"])
        self._s("nasa_donki", "NASA DONKI",
                "Space Weather Database of Notifications",
                "https://api.nasa.gov/DONKI",
                auth=AuthMethod.API_KEY, temporal=True,
                tags=["cme", "solar_flare"])
        self._s("soho", "SOHO",
                "Solar and Heliospheric Observatory",
                "https://soho.nascom.nasa.gov/data",
                temporal=True,
                tags=["solar", "soho", "sun"])
        self._s("stereo", "STEREO",
                "Solar Terrestrial Relations Observatory A & B",
                "https://stereo.gsfc.nasa.gov/data",
                temporal=True,
                tags=["solar", "stereo", "sun"])
        self._s("sdo", "SDO",
                "Solar Dynamics Observatory",
                "https://sdo.gsfc.nasa.gov/data",
                temporal=True,
                tags=["solar", "sdo", "sun"])
        self._s("nasa_eyes", "NASA Eyes",
                "NASA visualization platform",
                "https://eyes.nasa.gov/apps",
                tags=["nasa", "visualization"])
        self._s("launch_library", "Launch Library 2",
                "Rocket launch database",
                "https://ll.thespacedevs.com/2.2.0/launch",
                temporal=True,
                tags=["launch", "rocket"])
        self._s("spacex_api", "SpaceX API",
                "SpaceX launch data",
                "https://api.spacexdata.com/v4",
                tags=["spacex", "launch"])

        # ---- Transportation ------------------------------------------
        self._s("adsb_exchange", "ADS-B Exchange",
                "Real-time aircraft tracking",
                "https://adsbexchange.com/api",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["aircraft", "flight", "adsb"])
        self._s("opensky", "OpenSky Network",
                "Open aircraft tracking",
                "https://opensky-network.org/api",
                geo=True, temporal=True,
                tags=["aircraft", "flight"])
        self._s("faa_airports", "FAA Airports",
                "US airport data",
                "https://services.faa.gov/airport/status",
                geo=True, tags=["airport"])
        self._s("ais", "AIS",
                "Automatic Identification System for vessels",
                "https://data.aishub.net/ws.php",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["ship", "vessel", "ais"])
        self._s("marinetraffic", "MarineTraffic",
                "Global ship tracking",
                "https://services.marinetraffic.com/api",
                auth=AuthMethod.API_KEY, geo=True, temporal=True,
                tags=["ship", "vessel"])
        self._s("vesselfinder", "VesselFinder",
                "Ship tracking",
                "https://api.vesselfinder.com/vessels",
                auth=AuthMethod.API_KEY, geo=True,
                tags=["ship", "vessel"])
        self._s("wpi", "World Port Index",
                "Global port locations",
                "https://msi.nga.mil/api/publications/download",
                geo=True, tags=["port", "harbor"])

        # ---- Science -------------------------------------------------
        self._s("pubchem", "PubChem", "Chemical compound database",
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
                tags=["chemistry", "compound"])
        self._s("semantic_scholar", "Semantic Scholar",
                "AI-powered academic search",
                "https://api.semanticscholar.org/graph/v1",
                tags=["paper", "citation"])
        self._s("crossref", "Crossref", "DOI metadata",
                "https://api.crossref.org/works",
                tags=["doi", "publication"])
        self._s("pubmed", "PubMed", "Biomedical literature",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
                tags=["medical", "biology"])
        self._s("wolfram", "Wolfram Alpha", "Computational knowledge",
                "https://api.wolframalpha.com/v2/query",
                auth=AuthMethod.API_KEY,
                tags=["math", "computation"])
        self._s("oeis", "OEIS", "Online Encyclopedia of Integer Sequences",
                "https://oeis.org/search",
                tags=["math", "sequence"])
        self._s("nist", "NIST", "Physical constants and data",
                "https://physics.nist.gov/cgi-bin/cuu",
                tags=["physics", "constants"])
        self._s("cern_opendata", "CERN Open Data",
                "Particle physics data",
                "https://opendata.cern.ch/api",
                tags=["physics", "particle"])

        # ---- Monitoring/Intelligence ---------------------------------
        self._s("insecam", "Insecam", "Public webcam directory",
                "http://www.insecam.org/en/jsoncountries",
                geo=True, tags=["webcam", "camera"])
        self._s("webcamtaxi", "Webcam.travel",
                "Worldwide webcams",
                "https://api.webcams.travel/rest/v1",
                auth=AuthMethod.API_KEY, geo=True,
                tags=["webcam", "camera"])
        self._s("windy_webcams", "Windy Webcams",
                "Weather webcams",
                "https://api.windy.com/webcams/api/v3",
                auth=AuthMethod.API_KEY, geo=True,
                tags=["webcam", "weather"])
        self._s("milbases_osint", "Military OSINT",
                "Publicly known military installations",
                "https://overpass-api.de/api/interpreter",
                geo=True, tags=["military", "base"])

        # ---- Mycosoft Internal ---------------------------------------
        self._s("mindex", "MINDEX", "Mycosoft local indexed database",
                "http://localhost:8003/api",
                geo=True, temporal=True,
                tags=["local", "mindex", "internal"])
