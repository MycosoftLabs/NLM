"""
NLM Client - Main interface for interacting with NLM.

Nature Learning Model client for environmental data processing,
predictions, and recommendations. Uses physics, chemistry, and
biology modules for biosphere-grounded intelligence.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NLMClient:
    """
    Main client for interacting with the Nature Learning Model.
    
    This client provides a high-level interface for:
    - Processing multi-modal data
    - Querying the knowledge graph
    - Generating predictions
    - Getting recommendations
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        natureos_api_url: Optional[str] = None,
        mindex_api_url: Optional[str] = None,
        mas_api_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the NLM client.
        
        Args:
            database_url: PostgreSQL database URL
            natureos_api_url: NatureOS API endpoint
            mindex_api_url: MINDEX API endpoint
            mas_api_url: MAS API endpoint
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Database
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://user:pass@localhost:5432/nlm"
        )
        
        # Integration URLs
        self.natureos_api_url = natureos_api_url or os.getenv(
            "NATUREOS_API_URL",
            "http://localhost:8002"
        )
        self.mindex_api_url = mindex_api_url or os.getenv(
            "MINDEX_API_URL",
            "http://localhost:8003"
        )
        self.mas_api_url = mas_api_url or os.getenv(
            "MAS_API_URL",
            "http://localhost:8001"
        )
        
        logger.info("NLM client initialized")

    def _get_location_tuple(self, location: Optional[Dict[str, float]]) -> Tuple[float, float, float]:
        """Convert location dict to (lat, lon, alt) tuple."""
        if not location:
            return (37.7749, -122.4194, 0.0)  # Default SF
        lat = float(location.get("lat", location.get("latitude", 37.7749)))
        lon = float(location.get("lon", location.get("longitude", -122.4194)))
        alt = float(location.get("alt", location.get("altitude", 0.0)))
        return (lat, lon, alt)

    async def process_environmental_data(
        self,
        temperature: float,
        humidity: float,
        co2: Optional[float] = None,
        location: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process environmental data through NLM.

        Uses FieldPhysicsModel for geomagnetic, lunar, and atmospheric
        context. Derives fungal-relevant insights from sensor readings.
        """
        logger.info(f"Processing environmental data: temp={temperature}, humidity={humidity}")
        ts = timestamp or datetime.utcnow()
        unix_ts = time.mktime(ts.timetuple()) if hasattr(ts, "timetuple") else time.time()
        loc = self._get_location_tuple(location)
        pressure = kwargs.get("pressure")
        insights: List[str] = []
        predictions: List[Dict[str, Any]] = []

        try:
            from nlm.physics.field_physics import FieldPhysicsModel

            fpm = FieldPhysicsModel()
            atmo = fpm.get_atmospheric_conditions(loc, unix_ts)
            lunar = fpm.get_lunar_gravitational_influence(loc, unix_ts)

            if 18 <= temperature <= 26 and 50 <= humidity <= 85:
                insights.append("Conditions within optimal fungal growth range")
            elif temperature < 10 or temperature > 32:
                insights.append("Temperature outside typical fungal activity range")
            if humidity < 40:
                insights.append("Low humidity may limit fungal metabolism")
            if co2 and co2 > 1500:
                insights.append("Elevated CO2 may indicate confined space or fermentation")
            if pressure and pressure < 1000:
                insights.append("Low pressure may correlate with moisture and fruiting")
            if lunar.get("phase") in ("full_moon", "new_moon"):
                insights.append("Lunar phase often correlates with fungal fruiting flushes")

            fruiting = fpm.predict_fruiting_conditions(loc, unix_ts)
            predictions.append({
                "type": "fruiting_probability",
                "value": fruiting.get("fruiting_probability", 0),
                "species": "generic",
                "optimal_conditions": fruiting.get("optimal_conditions", {}),
            })
        except ImportError as e:
            logger.warning(f"FieldPhysicsModel not available: {e}")
            if 18 <= temperature <= 28 and 40 <= humidity <= 70:
                insights.append("Environmental conditions within typical range")
            predictions.append({"type": "baseline", "value": 0.5, "confidence": 0.3})

        return {
            "status": "processed",
            "temperature": temperature,
            "humidity": humidity,
            "co2": co2,
            "pressure": pressure,
            "location": location,
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "insights": insights,
            "predictions": predictions,
            "confidence": 0.7 if insights else 0.5,
        }
    
    async def query_knowledge_graph(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph via MINDEX.

        Delegates to MINDEX API when available for species, compounds,
        and research data. Returns structured entities and relations.
        """
        logger.info(f"Querying knowledge graph: {query}")
        try:
            import httpx

            base = self.mindex_api_url.rstrip("/")
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{base}/api/search/unified",
                    params={"q": query, "limit": limit},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    entities = [r for r in results if isinstance(r, dict)]
                    return {
                        "query": query,
                        "context": context or {},
                        "results": entities[:limit],
                        "entities": entities[:limit],
                        "relations": [],
                    }
        except Exception as e:
            logger.debug(f"MINDEX query failed: {e}")
        return {
            "query": query,
            "context": context or {},
            "results": [],
            "entities": [],
            "relations": [],
        }
    
    async def predict(
        self,
        entity_type: str,
        entity_id: str,
        time_horizon: str = "30d",
        conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a prediction using NLM physics/biology models.

        Supports species_growth, fruiting_conditions, environmental_trend.
        """
        logger.info(f"Generating prediction: {entity_type}/{entity_id}")
        conditions = conditions or {}
        loc = self._get_location_tuple(conditions.get("location"))
        unix_ts = time.time()

        prediction: Dict[str, Any] = {}
        confidence = 0.5

        try:
            from nlm.physics.field_physics import FieldPhysicsModel

            fpm = FieldPhysicsModel()
            if entity_type in ("species_growth", "fruiting_conditions", "generic"):
                result = fpm.predict_fruiting_conditions(loc, unix_ts, entity_id)
                prediction = {
                    "fruiting_probability": result.get("fruiting_probability"),
                    "optimal_conditions": result.get("optimal_conditions", {}),
                    "optimal_date": result.get("optimal_timestamp"),
                }
                confidence = 0.7
            elif entity_type == "environmental_trend":
                atmo = fpm.get_atmospheric_conditions(loc, unix_ts)
                prediction = {
                    "current": atmo,
                    "trend": "stable",
                    "time_horizon": time_horizon,
                }
                confidence = 0.6
        except ImportError:
            prediction = {"fallback": True, "message": "NLM modules not loaded"}

        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "time_horizon": time_horizon,
            "conditions": conditions,
            "prediction": prediction,
            "confidence": confidence,
        }
    
    async def recommend(
        self,
        scenario: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get recommendations based on scenario.

        Uses NLM physics and biology for optimal_growth_conditions,
        cultivation_timing, and environmental_optimization.
        """
        logger.info(f"Getting recommendations: {scenario}")
        constraints = constraints or {}
        loc = self._get_location_tuple(constraints.get("location"))
        unix_ts = time.time()
        recommendations: List[Dict[str, Any]] = []
        reasoning = ""

        try:
            from nlm.physics.field_physics import FieldPhysicsModel

            fpm = FieldPhysicsModel()
            fruiting = fpm.predict_fruiting_conditions(loc, unix_ts)
            atmo = fpm.get_atmospheric_conditions(loc, unix_ts)
            lunar = fpm.get_lunar_gravitational_influence(loc, unix_ts)

            if scenario in ("optimal_growth_conditions", "cultivation_timing", "generic"):
                recommendations.append({
                    "action": "Monitor temperature and humidity",
                    "target_temp_c": "18-24",
                    "target_humidity_pct": "75-90",
                    "source": "FieldPhysicsModel",
                })
                recommendations.append({
                    "action": "Consider lunar phase for fruiting",
                    "current_phase": lunar.get("phase"),
                    "optimal_phases": ["full_moon", "new_moon"],
                    "source": "Lunar influence model",
                })
                recommendations.append({
                    "action": "Maintain stable atmospheric pressure",
                    "current_pressure_hpa": atmo.get("pressure_hpa"),
                    "optimal_range": "1005-1015",
                    "source": "Atmospheric conditions",
                })
                reasoning = "Based on geomagnetic, lunar, and atmospheric models for fungal cultivation."
            elif scenario == "environmental_optimization":
                recommendations.append({
                    "action": "Adjust environmental parameters",
                    "optimal_conditions": fruiting.get("optimal_conditions", {}),
                    "source": "Fruiting prediction model",
                })
                reasoning = "Derived from fruiting probability model and field physics."
        except ImportError as e:
            logger.warning(f"NLM modules not available: {e}")
            recommendations.append({
                "action": "Ensure temperature 18-26°C and humidity 60-85%",
                "source": "Generic fungal growth guidelines",
            })
            reasoning = "Fallback recommendations; NLM physics modules not loaded."

        return {
            "scenario": scenario,
            "constraints": constraints,
            "recommendations": recommendations,
            "reasoning": reasoning,
        }

