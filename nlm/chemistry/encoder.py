"""
Chemistry Encoder
=================

Encodes chemical compounds into fixed-size vector representations
for machine learning and similarity computations.

Encoding Methods:
- Molecular fingerprints (ECFP-like)
- SMILES-based hashing
- Property-based encoding
- Graph neural network embeddings (future)

Usage:
    encoder = ChemistryEncoder(embedding_dim=128)
    embedding = encoder.encode(compound_data)
    similarity = encoder.cosine_similarity(vec1, vec2)
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ChemistryEncoder:
    """
    Encodes chemical compounds into vector representations.
    
    Supports multiple encoding strategies for different use cases
    including similarity search and property prediction.
    """
    
    # Atom properties for encoding
    ATOM_MASSES = {
        "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
        "S": 32.065, "P": 30.974, "F": 18.998, "Cl": 35.453,
        "Br": 79.904, "I": 126.904, "Se": 78.971,
    }
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize the encoder.
        
        Args:
            embedding_dim: Dimension of output vectors
        """
        self.embedding_dim = embedding_dim
        print(f"Initialized ChemistryEncoder with dim={embedding_dim}")
    
    def encode(self, compound_data: Dict[str, Any]) -> np.ndarray:
        """
        Encode a compound into a vector representation.
        
        Args:
            compound_data: Dictionary with name, formula, smiles, etc.
        
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        # Start with zeros
        embedding = np.zeros(self.embedding_dim)
        
        # SMILES-based encoding (if available)
        smiles = compound_data.get("smiles", "")
        if smiles:
            smiles_embed = self._encode_smiles(smiles)
            embedding[:len(smiles_embed)] = smiles_embed
        
        # Formula-based encoding
        formula = compound_data.get("formula", "")
        if formula:
            formula_embed = self._encode_formula(formula)
            idx = self.embedding_dim // 3
            embedding[idx:idx + len(formula_embed)] = formula_embed
        
        # Property-based encoding
        properties_embed = self._encode_properties(compound_data)
        idx = 2 * self.embedding_dim // 3
        embedding[idx:idx + len(properties_embed)] = properties_embed
        
        # Activity-based encoding
        activities = compound_data.get("activities", [])
        if activities:
            activity_embed = self._encode_activities(activities)
            # Add to existing embedding (blend)
            embedding += activity_embed * 0.1
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _encode_smiles(self, smiles: str) -> np.ndarray:
        """Generate embedding from SMILES string."""
        dim = self.embedding_dim // 3
        embedding = np.zeros(dim)
        
        # Hash-based fingerprint simulation
        for i, char in enumerate(smiles):
            hash_val = hashlib.md5(f"{char}{i}".encode()).hexdigest()
            idx = int(hash_val, 16) % dim
            embedding[idx] += 1
        
        # Add structural features
        # Ring count approximation
        ring_count = smiles.count("1") + smiles.count("2") + smiles.count("3")
        if ring_count > 0 and dim > 5:
            embedding[5] += ring_count * 0.5
        
        # Heteroatom count
        heteroatoms = sum(1 for c in smiles if c in "NOSPFClBrI")
        if heteroatoms > 0 and dim > 10:
            embedding[10] += heteroatoms * 0.3
        
        # Branch count
        branches = smiles.count("(")
        if branches > 0 and dim > 15:
            embedding[15] += branches * 0.2
        
        return embedding
    
    def _encode_formula(self, formula: str) -> np.ndarray:
        """Generate embedding from molecular formula."""
        dim = self.embedding_dim // 3
        embedding = np.zeros(dim)
        
        # Parse formula and encode atom counts
        atoms = self._parse_formula(formula)
        
        # Encode each atom type
        atom_order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
        for i, atom in enumerate(atom_order):
            if i < dim:
                count = atoms.get(atom, 0)
                embedding[i] = math.log1p(count)  # Log-transform counts
        
        # Calculate summary statistics
        total_atoms = sum(atoms.values())
        total_mass = sum(count * self.ATOM_MASSES.get(atom, 0) for atom, count in atoms.items())
        
        if dim > 15:
            embedding[12] = total_atoms / 100  # Normalize
            embedding[13] = total_mass / 500   # Normalize
            embedding[14] = len(atoms) / 10    # Atom diversity
        
        return embedding
    
    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """Parse molecular formula into atom counts."""
        atoms: Dict[str, int] = {}
        i = 0
        
        while i < len(formula):
            if formula[i].isupper():
                # Start of element symbol
                element = formula[i]
                i += 1
                
                # Check for lowercase (second character)
                if i < len(formula) and formula[i].islower():
                    element += formula[i]
                    i += 1
                
                # Get count
                count_str = ""
                while i < len(formula) and formula[i].isdigit():
                    count_str += formula[i]
                    i += 1
                
                count = int(count_str) if count_str else 1
                atoms[element] = atoms.get(element, 0) + count
            else:
                i += 1
        
        return atoms
    
    def _encode_properties(self, compound_data: Dict[str, Any]) -> np.ndarray:
        """Encode compound properties."""
        dim = self.embedding_dim // 3
        embedding = np.zeros(dim)
        
        # Molecular weight
        mw = compound_data.get("molecular_weight", compound_data.get("weight", 0))
        if mw and dim > 0:
            embedding[0] = mw / 1000  # Normalize to 0-1 range
        
        # LogP (lipophilicity) - if available
        logp = compound_data.get("logp", compound_data.get("logP", None))
        if logp is not None and dim > 1:
            embedding[1] = (logp + 5) / 10  # Normalize -5 to 5 range
        
        # Number of H-bond donors/acceptors - estimate from formula
        formula = compound_data.get("formula", "")
        if formula:
            atoms = self._parse_formula(formula)
            n_count = atoms.get("N", 0)
            o_count = atoms.get("O", 0)
            
            if dim > 2:
                embedding[2] = n_count / 10
            if dim > 3:
                embedding[3] = o_count / 10
        
        # Rotatable bonds estimate
        smiles = compound_data.get("smiles", "")
        if smiles and dim > 4:
            single_bonds = smiles.count("-") + len(smiles) // 5  # Rough estimate
            embedding[4] = single_bonds / 20
        
        return embedding
    
    def _encode_activities(self, activities: List[str]) -> np.ndarray:
        """Encode biological activities."""
        embedding = np.zeros(self.embedding_dim)
        
        # Hash each activity to a position
        for activity in activities:
            hash_val = hashlib.md5(activity.lower().encode()).hexdigest()
            idx = int(hash_val, 16) % self.embedding_dim
            embedding[idx] += 1.0
        
        return embedding
    
    def cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def batch_encode(self, compounds: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode multiple compounds.
        
        Args:
            compounds: List of compound dictionaries
        
        Returns:
            Numpy array of shape (n_compounds, embedding_dim)
        """
        embeddings = np.zeros((len(compounds), self.embedding_dim))
        
        for i, compound in enumerate(compounds):
            embeddings[i] = self.encode(compound)
        
        return embeddings
    
    def find_similar(
        self,
        query: Dict[str, Any],
        database: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Find similar compounds in a database.
        
        Args:
            query: Query compound
            database: List of compounds to search
            top_k: Number of results to return
        
        Returns:
            List of (index, similarity, compound) tuples
        """
        query_vec = self.encode(query)
        database_vecs = self.batch_encode(database)
        
        similarities = []
        for i, vec in enumerate(database_vecs):
            sim = self.cosine_similarity(query_vec, vec)
            similarities.append((i, sim, database[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


def encode_compound(
    compound_data: Dict[str, Any],
    dim: int = 128,
) -> np.ndarray:
    """
    Convenience function to encode a single compound.
    
    Args:
        compound_data: Compound information
        dim: Embedding dimension
    
    Returns:
        Embedding vector
    """
    encoder = ChemistryEncoder(embedding_dim=dim)
    return encoder.encode(compound_data)
