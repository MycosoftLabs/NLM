"""
Merkle-Rooted Cognition
=======================

Every NLM event is geotagged, timestamped, and Merkle-rooted.
Sensory data is coupled to spacetime. Self-state, world-state,
and event-state are each independently Merkleized.

The canonical frame identity:
    frame_root = merkle(self_root || world_root || event_root || parent_frame_root)

Merkle roots are for identity, replay, integrity, and lineage.
The model consumes decoded structured state.
MINDEX stores roots, lineage, provenance, and replay trail.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


def merkle_hash(*parts: str) -> str:
    """Compute a SHA-256 Merkle hash from concatenated string parts."""
    combined = "||".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def hash_dict(data: Dict[str, Any]) -> str:
    """Deterministic hash of a dictionary (sorted keys, JSON serialized)."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    """SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_event_root(
    timestamp: str,
    geolocation: str,
    sensor_hashes: Sequence[str],
    bio_tokens: Sequence[str],
) -> str:
    """Compute the event root from observation data.

    The event root captures the identity of what was observed,
    when, and where — independent of self-state or world-state.
    """
    sensor_combined = merkle_hash(*sensor_hashes) if sensor_hashes else "empty"
    tokens_combined = merkle_hash(*bio_tokens) if bio_tokens else "empty"
    return merkle_hash(timestamp, geolocation, sensor_combined, tokens_combined)


def compute_self_root(self_state: Dict[str, Any]) -> str:
    """Compute the self root from MYCA/MAS internal state."""
    return hash_dict(self_state)


def compute_world_root(world_state: Dict[str, Any]) -> str:
    """Compute the world root from external world state."""
    return hash_dict(world_state)


def compute_frame_root(
    self_root: str,
    world_root: str,
    event_root: str,
    parent_frame_root: str,
) -> str:
    """Compute the canonical frame root.

    frame_root = merkle(self_root || world_root || event_root || parent_frame_root)

    This provides:
    - Continuity: linked to parent frame
    - Integrity: tamper-evident via hash chain
    - Replay: deterministic recomputation from state
    - Provenance: full audit trail via lineage
    """
    return merkle_hash(self_root, world_root, event_root, parent_frame_root)


def verify_frame_root(
    frame_root: str,
    self_root: str,
    world_root: str,
    event_root: str,
    parent_frame_root: str,
) -> bool:
    """Verify that a frame root matches its constituent roots."""
    expected = compute_frame_root(self_root, world_root, event_root, parent_frame_root)
    return frame_root == expected


@dataclass
class LineageRecord:
    """A single entry in the Merkle lineage DAG.

    Stored in MINDEX for provenance, audit, and replay.
    """

    frame_root: str
    parent_frame_root: str
    self_root: str
    world_root: str
    event_root: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    producer: str = ""  # which service/agent created this frame
    content_hash: str = ""  # hash of the full frame content
    source_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_root": self.frame_root,
            "parent_frame_root": self.parent_frame_root,
            "self_root": self.self_root,
            "world_root": self.world_root,
            "event_root": self.event_root,
            "timestamp": self.timestamp.isoformat(),
            "producer": self.producer,
            "content_hash": self.content_hash,
            "source_refs": self.source_refs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineageRecord:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(timezone.utc)
        return cls(
            frame_root=data["frame_root"],
            parent_frame_root=data.get("parent_frame_root", ""),
            self_root=data.get("self_root", ""),
            world_root=data.get("world_root", ""),
            event_root=data.get("event_root", ""),
            timestamp=ts,
            producer=data.get("producer", ""),
            content_hash=data.get("content_hash", ""),
            source_refs=data.get("source_refs", []),
            metadata=data.get("metadata", {}),
        )


def verify_lineage(records: List[LineageRecord]) -> bool:
    """Verify a chain of lineage records.

    Each record's parent_frame_root must match the previous record's frame_root.
    Returns True if the chain is valid.
    """
    if not records:
        return True

    for i in range(1, len(records)):
        if records[i].parent_frame_root != records[i - 1].frame_root:
            return False
        if not verify_frame_root(
            records[i].frame_root,
            records[i].self_root,
            records[i].world_root,
            records[i].event_root,
            records[i].parent_frame_root,
        ):
            return False
    return True


def build_lineage_dag(records: List[LineageRecord]) -> Dict[str, List[str]]:
    """Build an adjacency list from lineage records.

    Returns mapping from frame_root to list of child frame_roots.
    """
    dag: Dict[str, List[str]] = {}
    for record in records:
        parent = record.parent_frame_root
        if parent not in dag:
            dag[parent] = []
        dag[parent].append(record.frame_root)
    return dag
