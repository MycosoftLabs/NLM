"""
NLM CREP Map Bridge — aligned with MINDEX v3
==============================================

Translates search results and ingested data into CREP map layers so that
humans and machines see the same data simultaneously:

- Humans: visual map layers on the CREP globe (species, events, infra, …)
- Machines: GeoJSON API endpoints that agents call to get spatial context

Matches mindex's exact 16 CREP layers and MapEntity format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nlm.search.domains import DOMAIN_TO_GROUP
from nlm.search.engine import SearchHit, SearchResult

logger = logging.getLogger(__name__)


# ======================================================================
# CREP layer config — matches mindex earth.py's 16 map layers exactly
# ======================================================================

CREP_LAYERS: Dict[str, Dict[str, Any]] = {
    "earthquakes":  {"label": "Earthquakes",   "color": "#ef4444", "icon": "seismic",    "cluster": False},
    "volcanoes":    {"label": "Volcanoes",      "color": "#f97316", "icon": "volcano",    "cluster": False},
    "wildfires":    {"label": "Wildfires",      "color": "#dc2626", "icon": "fire",       "cluster": True},
    "facilities":   {"label": "Facilities",     "color": "#f59e0b", "icon": "building",   "cluster": True},
    "antennas":     {"label": "Antennas",       "color": "#8b5cf6", "icon": "radio",      "cluster": True},
    "aircraft":     {"label": "Aircraft",       "color": "#ec4899", "icon": "plane",      "cluster": False},
    "vessels":      {"label": "Vessels",        "color": "#3b82f6", "icon": "ship",       "cluster": False},
    "airports":     {"label": "Airports",       "color": "#f472b6", "icon": "airport",    "cluster": True},
    "ports":        {"label": "Ports",          "color": "#60a5fa", "icon": "anchor",     "cluster": True},
    "cameras":      {"label": "Cameras",        "color": "#6b7280", "icon": "camera",     "cluster": True},
    "military":     {"label": "Military Sites", "color": "#991b1b", "icon": "shield",     "cluster": True},
    "buoys":        {"label": "Ocean Buoys",    "color": "#06b6d4", "icon": "buoy",       "cluster": True},
    "weather":      {"label": "Weather",        "color": "#a855f7", "icon": "cloud",      "cluster": True},
    "air_quality":  {"label": "Air Quality",    "color": "#84cc16", "icon": "wind",       "cluster": True},
    "wifi_hotspots": {"label": "WiFi Hotspots", "color": "#7c3aed", "icon": "wifi",       "cluster": True},
    "species":      {"label": "Species",        "color": "#22c55e", "icon": "leaf",       "cluster": True},
}

# Map domain keys to their CREP layer (mirrors _DOMAIN_META crep values in domains.py)
_DOMAIN_TO_CREP: Dict[str, str] = {
    "taxa": "species", "species": "species", "observations": "species",
    "earthquakes": "earthquakes", "volcanoes": "volcanoes", "wildfires": "wildfires",
    "facilities": "facilities",
    "antennas": "antennas", "wifi_hotspots": "wifi_hotspots",
    "aircraft": "aircraft", "vessels": "vessels",
    "airports": "airports", "ports": "ports",
    "cameras": "cameras",
    "military_installations": "military",
    "buoys": "buoys",
    "weather": "weather",
    "air_quality": "air_quality",
}


@dataclass
class CREPLayer:
    """One map layer for the CREP globe — matches mindex MapEntity grouping."""
    layer_id: str
    label: str
    visible: bool = True
    style: Dict[str, Any] = field(default_factory=dict)
    features: List[Dict[str, Any]] = field(default_factory=list)

    def to_geojson(self) -> Dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "metadata": {
                "layer_id": self.layer_id,
                "label": self.label,
                "style": self.style,
                "visible": self.visible,
            },
            "features": self.features,
        }


class CREPMapBridge:
    """
    Converts search results into CREP map layers.

    Usage::

        bridge = CREPMapBridge()
        layers = bridge.build_layers(search_result)
        geojson = bridge.to_geojson_collection(layers)
    """

    def build_layers(self, result: SearchResult) -> List[CREPLayer]:
        """Group search hits into CREP layers by domain."""
        layer_map: Dict[str, CREPLayer] = {}

        for hit in result.universal_results:
            lid = _DOMAIN_TO_CREP.get(hit.domain)
            if not lid:
                continue  # domain has no CREP layer

            if lid not in layer_map:
                cfg = CREP_LAYERS.get(lid, {"label": lid.title(), "color": "#6b7280", "icon": "pin"})
                layer_map[lid] = CREPLayer(
                    layer_id=lid,
                    label=cfg.get("label", lid),
                    style={
                        "color": cfg.get("color", "#6b7280"),
                        "icon": cfg.get("icon", "pin"),
                        "cluster": cfg.get("cluster", True),
                    },
                )

            feature = self._hit_to_feature(hit, lid)
            if feature:
                layer_map[lid].features.append(feature)

        return list(layer_map.values())

    def to_geojson_collection(
        self, layers: List[CREPLayer],
    ) -> Dict[str, Any]:
        """Merge all layers into one GeoJSON collection with layer metadata."""
        features: List[Dict[str, Any]] = []
        layer_meta: List[Dict[str, Any]] = []

        for layer in layers:
            layer_meta.append({
                "layer_id": layer.layer_id,
                "label": layer.label,
                "style": layer.style,
                "feature_count": len(layer.features),
            })
            features.extend(layer.features)

        return {
            "type": "FeatureCollection",
            "metadata": {
                "layers": layer_meta,
                "total_features": len(features),
            },
            "features": features,
        }

    def build_layer_config(self) -> List[Dict[str, Any]]:
        """Return the full set of available CREP layer definitions (16 layers)."""
        return [
            {
                "layer_id": lid,
                "label": cfg["label"],
                "color": cfg["color"],
                "icon": cfg["icon"],
                "cluster": cfg.get("cluster", True),
            }
            for lid, cfg in CREP_LAYERS.items()
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hit_to_feature(self, hit: SearchHit, layer_id: str) -> Optional[Dict[str, Any]]:
        """Convert a SearchHit to a GeoJSON Feature matching mindex MapEntity format."""
        if hit.lat is None or hit.lng is None:
            return None

        style = CREP_LAYERS.get(layer_id, {})

        return {
            "type": "Feature",
            "geometry": {
                "type": hit.geometry_type or "Point",
                "coordinates": [float(hit.lng), float(hit.lat)],
            },
            "properties": {
                "id": hit.id,
                "entity_type": hit.entity_type,
                "domain": hit.domain,
                "name": hit.name,
                "description": hit.description,
                "occurred_at": hit.occurred_at,
                "source": hit.source,
                "image_url": hit.image_url,
                "crep_layer": layer_id,
                "color": style.get("color", "#6b7280"),
                "icon": style.get("icon", "pin"),
                **hit.properties,
            },
        }
