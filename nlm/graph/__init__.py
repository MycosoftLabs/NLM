"""
NLM Graph — Multi-Resolution Merkle HyperDAG and GraphRAG retrieval.

The world model backbone: structured graph with integrity, not a flat vector store.
"""

from nlm.graph.hyperdag import HyperDAG, HyperNode, HyperEdge, CausalEdge, DAGLayer
from nlm.graph.retrieval import GraphRetriever, HybridRetriever

__all__ = [
    "HyperDAG",
    "HyperNode",
    "HyperEdge",
    "CausalEdge",
    "DAGLayer",
    "GraphRetriever",
    "HybridRetriever",
]
