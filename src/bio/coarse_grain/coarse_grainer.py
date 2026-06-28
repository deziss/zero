"""src/bio/coarse_grain/coarse_grainer.py — Hierarchical coarse-graining

Maps atomistic detail → coarse-grained representation.

Coarse-graining levels:
  LG: Local (per-atom)
  MG: Mesoscopic (per-molecule cluster)
  HG: Hierarchical (per-cell/tissue)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum


class CoarseningLevel(Enum):
    ATOMISTIC = 0  # per-atom
    MOLECULAR = 1  # per-molecule
    CLUSTER = 2    # per-molecule cluster
    CELLULAR = 3   # per-cell aggregate
    TISSUE = 4     # per-tissue field


class CoarseGrainer:
    """
    Performs hierarchical coarse-graining for multiscale simulation.
    
    Converts fine-scale data to coarse-scale representation
    for efficient computation where high resolution isn't needed.
    """
    
    def __init__(
        self,
        cluster_radius: float = 5.0,
        min_cluster_size: int = 5,
        max_levels: int = 5,
    ):
        self.cluster_radius = cluster_radius
        self.min_cluster_size = min_cluster_size
        self.max_levels = max_levels
    
    def compute_coarse_representation(
        self,
        positions: np.ndarray,
        properties: Dict[str, np.ndarray],
        level: CoarseningLevel = CoarseningLevel.CLUSTER,
    ) -> Dict:
        """
        Compute coarse representation at specified level.
        
        Args:
            positions: (n_atoms, 3) atomic positions
            properties: dict of property arrays per atom
            level: target coarse-graining level
            
        Returns:
            coarse_dict with aggregated properties per cluster
        """
        n_atoms = len(positions)
        
        if level == CoarseningLevel.ATOMISTIC:
            return {
                'positions': positions.copy(),
                'properties': {k: v.copy() for k, v in properties.items()},
                'n_clusters': n_atoms,
                'level': level.name,
            }
        
        # Find clusters via distance-based grouping
        clusters = self._find_clusters(positions)
        
        coarse_props = {}
        coarse_positions = []
        
        for cluster_id, member_indices in enumerate(clusters):
            if len(member_indices) < self.min_cluster_size:
                continue
            
            # Aggregate properties
            cluster_pos = positions[member_indices]
            coarse_positions.append(cluster_pos.mean(axis=0))
            
            for key, arr in properties.items():
                member_vals = arr[member_indices]
                
                # Weighted averaging (mass-weighted if mass data available)
                coarse_props[f'{key}_mean'] = member_vals.mean()
                coarse_props[f'{key}_var'] = member_vals.var() if len(member_vals) > 1 else 0
                coarse_props[f'{key}_max'] = member_vals.max()
                coarse_props[f'{key}_min'] = member_vals.min()
                coarse_props[f'{key}_sum'] = member_vals.sum()
        
        return {
            'positions': np.array(coarse_positions),
            'properties': coarse_props,
            'n_clusters': len(clusters),
            'level': level.name,
            'cluster_sizes': [len(c) for c in clusters],
        }
    
    def _find_clusters(self, positions: np.ndarray) -> List[List[int]]:
        """Find clusters using radius-based grouping."""
        n = len(positions)
        assigned = [False] * n
        clusters = []
        
        for i in range(n):
            if assigned[i]:
                continue
            
            cluster = [i]
            assigned[i] = True
            
            for j in range(n):
                if assigned[j]:
                    continue
                
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= self.cluster_radius:
                    cluster.append(j)
                    assigned[j] = True
            
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
        
        return clusters
    
    def multi_scale_decomposition(
        self,
        positions: np.ndarray,
        properties: Dict[str, np.ndarray],
    ) -> Dict[CoarseningLevel, Dict]:
        """
        Decompose system into all coarse-graining levels.
        
        Returns:
            dict mapping each level to its coarse representation
        """
        result = {}
        
        for level in CoarseningLevel:
            result[level] = self.compute_coarse_representation(
                positions, properties, level
            )
        
        return result


class AdaptiveCoarsening:
    """
    Dynamically adjusts coarse-graining level per region.
    
    Uses the importance scoring from the adaptive depth
    engine to determine where coarse-graining is appropriate.
    """
    
    def __init__(
        self,
        coarse_grainer: CoarseGrainer,
        importance_threshold: float = 0.6,
    ):
        self.coarse_grainer = coarse_grainer
        self.importance_threshold = importance_threshold
    
    def adaptive_coarsening(
        self,
        positions: np.ndarray,
        importance_scores: np.ndarray,
        properties: Dict[str, np.ndarray],
    ) -> Tuple[Dict, List[int]]:
        """
        Perform adaptive coarse-graining based on importance.
        
        Args:
            positions: atomic positions
            importance_scores: per-atom importance [0, 1]
            properties: per-atom properties
            
        Returns:
            coarse_dict, fine_atom_indices (those NOT coarse-grained)
        """
        # Determine which atoms to keep fine (high importance)
        fine_mask = importance_scores > self.importance_threshold
        fine_indices = np.where(fine_mask)[0].tolist()
        coarse_indices = np.where(~fine_mask)[0]
        
        if len(coarse_indices) == 0:
            # All atoms are important → no coarse-graining
            return self.coarse_grainer.compute_coarse_representation(
                positions, properties, CoarseningLevel.ATOMISTIC
            ), fine_indices
        
        # Compute coarse representation for low-importance regions
        coarse_pos = positions[coarse_indices]
        coarse_props = {
            k: properties[k][coarse_indices]
            for k in properties
        }
        
        coarse = self.coarse_grainer.compute_coarse_representation(
            coarse_pos, coarse_props, CoarseningLevel.CELLULAR
        )
        
        # Add fine atoms to coarse dict
        coarse['fine_indices'] = fine_indices
        coarse['fine_positions'] = positions[fine_indices]
        
        return coarse, fine_indices
