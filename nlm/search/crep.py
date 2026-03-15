"""
NLM CREP Map Bridge
====================

Translates search results and ingested data into CREP map layers so that
humans and machines see the same data simultaneously:

- Humans: visual map layers on the CREP globe (species, events, infra, …)
- Machines: GeoJSON API endpoints that agents call to get spatial context

Every search domain has a ``crep_layer`` id.  This module builds the
GeoJSON FeatureCollections and layer configs that CREP consumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nlm.search.engine import SearchHit, SearchResult

logger = logging.getLogger(__name__)


# Layer visual config defaults keyed by domain root
LAYER_STYLES: Dict[str, Dict[str, Any]] = {
    "life": {
        "color": "#22c55e",
        "icon": "leaf",
        "cluster": True,
    },
    "environment": {
        "color": "#3b82f6",
        "icon": "cloud",
        "cluster": False,
    },
    "infrastructure": {
        "color": "#f59e0b",
        "icon": "building",
        "cluster": True,
    },
    "signals": {
        "color": "#8b5cf6",
        "icon": "radio",
        "cluster": True,
    },
    "space": {
        "color": "#06b6d4",
        "icon": "satellite",
        "cluster": False,
    },
    "transportation": {
        "color": "#ec4899",
        "icon": "plane",
        "cluster": False,
    },
    "science": {
        "color": "#14b8a6",
        "icon": "flask",
        "cluster": True,
    },
    "intelligence": {
        "color": "#ef4444",
        "icon": "camera",
        "cluster": True,
    },
}


@dataclass
class CREPLayer:
    """One map layer for the CREP globe."""
    layer_id: str
    label: str
    domain_root: str
    visible: bool = True
    style: Dict[str, Any] = field(default_factory=dict)
    features: List[Dict[str, Any]] = field(default_factory=list)

    def to_geojson(self) -> Dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "metadata": {
                "layer_id": self.layer_id,
                "label": self.label,
                "domain_root": self.domain_root,
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

        for hit in result.hits:
            lid = hit.crep_layer or hit.domain_key.split(".")[0] + "_layer"

            if lid not in layer_map:
                root = hit.domain_key.split(".")[0]
                style = LAYER_STYLES.get(root, {"color": "#6b7280", "icon": "pin"})
                layer_map[lid] = CREPLayer(
                    layer_id=lid,
                    label=lid.replace("_", " ").title(),
                    domain_root=root,
                    style=style,
                )

            feature = self._hit_to_feature(hit)
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
                "domain_root": layer.domain_root,
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
        """Return the full set of available CREP layer definitions."""
        from nlm.search.domains import DomainRegistry

        registry = DomainRegistry()
        configs: List[Dict[str, Any]] = []
        seen: set = set()

        for domain in registry.list_domains():
            lid = domain.crep_layer
            if not lid or lid in seen:
                continue
            seen.add(lid)
            root = domain.root
            style = LAYER_STYLES.get(root, {"color": "#6b7280", "icon": "pin"})
            configs.append({
                "layer_id": lid,
                "label": domain.label,
                "domain_key": domain.key,
                "domain_root": root,
                "style": style,
            })

        return configs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hit_to_feature(self, hit: SearchHit) -> Optional[Dict[str, Any]]:
        if hit.geojson:
            # Enrich existing geojson
            feat = dict(hit.geojson)
            props = feat.get("properties", {})
            props.update({
                "hit_id": hit.id,
                "domain": hit.domain_key,
                "source": hit.source_key,
                "title": hit.title,
                "score": hit.score,
                "crep_layer": hit.crep_layer,
            })
            feat["properties"] = props
            return feat

        if not hit.location:
            return None

        lat = hit.location.get("lat")
        lon = hit.location.get("lon")
        if lat is None or lon is None:
            return None

        root = hit.domain_key.split(".")[0]
        style = LAYER_STYLES.get(root, {})

        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            },
            "properties": {
                "hit_id": hit.id,
                "domain": hit.domain_key,
                "source": hit.source_key,
                "title": hit.title,
                "description": hit.description,
                "score": hit.score,
                "timestamp": hit.timestamp,
                "crep_layer": hit.crep_layer,
                "color": style.get("color", "#6b7280"),
                "icon": style.get("icon", "pin"),
            },
        }
