"""
NLM Merkle-Rooted Cognition

SHA-256 Merkle tree implementation for hash-rooted cognitive state.
Every RootedNatureFrame is Merkle-rooted: frame_root commits to
self_root, world_root, event_root, and parent_frame_root.

This provides:
- Tamper-evident state lineage (hash-chain of frames)
- Verifiable provenance (Merkle proofs for any sub-field)
- Replay verification (recompute roots from stored data)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# Genesis root used when there is no parent frame
GENESIS_ROOT = b"\x00" * 32


def sha256(data: bytes) -> bytes:
    """Compute SHA-256 digest."""
    return hashlib.sha256(data).digest()


def merkle_hash(left: bytes, right: bytes) -> bytes:
    """Hash two child nodes into a parent node."""
    return sha256(left + right)


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """
    Build a Merkle root from an ordered list of leaf hashes.

    If no leaves, returns GENESIS_ROOT.
    If one leaf, returns the leaf hash itself.
    Otherwise builds a balanced binary tree (padding the last leaf if odd).
    """
    if not leaves:
        return GENESIS_ROOT
    layer: list[bytes] = list(leaves)
    if len(layer) == 1:
        return layer[0]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last for balance
        next_layer: list[bytes] = []
        for i in range(0, len(layer), 2):
            next_layer.append(merkle_hash(layer[i], layer[i + 1]))
        layer = next_layer
    return layer[0]


def content_hash(data: bytes) -> bytes:
    """SHA-256 content hash for arbitrary payloads."""
    return sha256(data)


def field_hash(name: str, value: bytes) -> bytes:
    """Hash a named field: H(name || value)."""
    return sha256(name.encode("utf-8") + value)


def sorted_field_root(fields: dict[str, bytes]) -> bytes:
    """Compute Merkle root over sorted field hashes."""
    leaves = [field_hash(k, v) for k, v in sorted(fields.items())]
    return merkle_root(leaves)


def compute_frame_root(
    self_root: bytes,
    world_root: bytes,
    event_root: bytes,
    parent_frame_root: Optional[bytes] = None,
) -> bytes:
    """
    Compute the top-level frame root.

    frame_root = H(H(self_root, world_root), H(event_root, parent_frame_root))
    """
    parent = parent_frame_root or GENESIS_ROOT
    return merkle_hash(
        merkle_hash(self_root, world_root),
        merkle_hash(event_root, parent),
    )


@dataclass(frozen=True)
class MerkleProof:
    """Proof that a leaf is included in a Merkle root."""

    root: bytes
    leaf: bytes
    index: int
    siblings: List[bytes] = field(default_factory=list)
    directions: List[int] = field(default_factory=list)  # 0=left, 1=right

    def verify(self) -> bool:
        """Verify this proof against the stored root."""
        current = self.leaf
        for sibling, direction in zip(self.siblings, self.directions):
            if direction == 0:
                current = merkle_hash(sibling, current)
            else:
                current = merkle_hash(current, sibling)
        return current == self.root


class MerkleTree:
    """
    Full Merkle tree with proof generation and verification.

    Build from leaves, then generate proofs for any leaf index.
    """

    def __init__(self, leaves: Sequence[bytes]):
        self._leaves = list(leaves)
        self._layers: list[list[bytes]] = []
        self._root = self._build()

    @property
    def root(self) -> bytes:
        return self._root

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    def _build(self) -> bytes:
        if not self._leaves:
            self._layers = [[GENESIS_ROOT]]
            return GENESIS_ROOT
        layer = list(self._leaves)
        self._layers = [layer[:]]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            next_layer: list[bytes] = []
            for i in range(0, len(layer), 2):
                next_layer.append(merkle_hash(layer[i], layer[i + 1]))
            self._layers.append(next_layer)
            layer = next_layer
        return layer[0]

    def proof(self, index: int) -> MerkleProof:
        """Generate a Merkle proof for the leaf at the given index."""
        if index < 0 or index >= len(self._leaves):
            raise IndexError(f"Leaf index {index} out of range [0, {len(self._leaves)})")
        siblings: list[bytes] = []
        directions: list[int] = []
        idx = index
        for layer in self._layers[:-1]:
            padded = list(layer)
            if len(padded) % 2 == 1:
                padded.append(padded[-1])
            if idx % 2 == 0:
                siblings.append(padded[idx + 1])
                directions.append(1)  # sibling is to the right
            else:
                siblings.append(padded[idx - 1])
                directions.append(0)  # sibling is to the left
            idx //= 2
        return MerkleProof(
            root=self._root,
            leaf=self._leaves[index],
            index=index,
            siblings=siblings,
            directions=directions,
        )

    def verify(self, proof: MerkleProof) -> bool:
        """Verify a proof against this tree's root."""
        return proof.root == self._root and proof.verify()
