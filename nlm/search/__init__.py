"""
NLM Universal Earth Search — aligned with MINDEX v3
=====================================================

All-inclusive search across every domain of Earth data via mindex's
unified-search/earth endpoint.  35 domains, 16 CREP map layers,
ETL sync pipeline, and Myca AI query interface.
"""

from nlm.search.domains import (
    ALL_DOMAINS,
    DOMAIN_GROUPS,
    DOMAIN_TABLES,
    DOMAIN_TO_GROUP,
    DomainRegistry,
    SearchDomain,
)
from nlm.search.engine import (
    SearchHit,
    SearchRequest,
    SearchResult,
    UniversalSearchEngine,
)
from nlm.search.crep import CREPMapBridge
from nlm.search.myca import MycaQueryInterface
from nlm.search.pipeline import IngestionPipeline
from nlm.search.sources import DataSource, DataSourceRegistry

__all__ = [
    "ALL_DOMAINS",
    "DOMAIN_GROUPS",
    "DOMAIN_TABLES",
    "DOMAIN_TO_GROUP",
    "DomainRegistry",
    "SearchDomain",
    "SearchHit",
    "SearchRequest",
    "SearchResult",
    "UniversalSearchEngine",
    "CREPMapBridge",
    "MycaQueryInterface",
    "IngestionPipeline",
    "DataSource",
    "DataSourceRegistry",
]
