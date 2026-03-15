"""
NLM Universal Earth Search
===========================

All-inclusive search across every domain of Earth data: all species, all
environments, all infrastructure, all signals, all events, all science.
Feeds into mindex for local storage/training and CREP for map visualization.
"""

from nlm.search.domains import DomainRegistry, SearchDomain
from nlm.search.sources import DataSourceRegistry, DataSource
from nlm.search.engine import UniversalSearchEngine, SearchRequest, SearchResult
from nlm.search.pipeline import IngestionPipeline
from nlm.search.crep import CREPMapBridge
from nlm.search.myca import MycaQueryInterface

__all__ = [
    "DomainRegistry",
    "SearchDomain",
    "DataSourceRegistry",
    "DataSource",
    "UniversalSearchEngine",
    "SearchRequest",
    "SearchResult",
    "IngestionPipeline",
    "CREPMapBridge",
    "MycaQueryInterface",
]
