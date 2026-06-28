"""src/qm/mapping/qm_mm_mapper.py — QM/MM region mapping

Maps atoms between quantum and molecular mechanics regions.
Decides which atoms need QM treatment vs MM approximation.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional


class QM_MMMapper:
    """
    Maps atoms between QM and MM regions for hybrid simulation.
    
    Strategy:
    1. Identify reactive center(s) -- active sites
    2. Include nearest-neighbor shell in QM region
    3. Rest treated as MM (force field)
    4. Link atoms (boundaries) use constraint potentials
    """
    
    def __init__(
        self,
        n_qm_atoms: int = 50,
        n_layer_atoms: int = 10,
        cutoff_angstrom: float = 5.0,
    ):
        self.n_qm_atoms = n_qm_atoms
        self.n_layer_atoms = n_layer_atoms
        self.cutoff = cutoff_angstrom
    
    def identify_qm_region(
        self,
        centers: np.ndarray,
        all_positions: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Identify which atoms belong in the QM region."""
        n_total = len(all_positions)
        qm_idx = set()
        
        for center in centers:
            dists = np.linalg.norm(
                all_positions - center, axis=1
            )
            in_cutoff = np.where(dists <= self.cutoff)[0]
            qm_idx.update(in_cutoff.tolist())
        
        if len(qm_idx) < self.n_qm_atoms:
            dists_all = np.array([
                min(np.linalg.norm(pos - c) for c in centers)
                for pos in all_positions
            ])
            remaining = set(
                np.argsort(dists_all)[:self.n_qm_atoms]
            ) - qm_idx
            qm_idx.update(remaining)
        
        return list(qm_idx), [i for i in range(n_total) if i not in qm_idx]
    
    def get_link_atoms(
        self,
        qm_indices: List[int],
        mm_indices: List[int],
    ) -> List[int]:
        """Identify link atoms (boundary between QM and MM)."""
        link = []
        for mm_idx in mm_indices:
            for qm_idx in qm_indices:
                if abs(mm_idx - qm_idx) <= 5:
                    link.append(mm_idx)
                    break
        return list(set(link))
    
    def split_system(
        self,
        all_positions: np.ndarray,
        all_charges: np.ndarray,
        centers: np.ndarray,
    ) -> Dict:
        """Split full system into QM/MM regions."""
        qm_idx, mm_idx = self.identify_qm_region(
            centers, all_positions
        )
        
        link = self.get_link_atoms(qm_idx, mm_idx)
        link = list(set(link) & set(mm_idx))
        
        return {
            'qm_indices': qm_idx,
            'mm_indices': mm_idx,
            'link_indices': link,
            'qm_positions': all_positions[qm_idx],
            'qm_charges': all_charges[qm_idx],
            'mm_positions': all_positions[mm_idx],
            'mm_charges': all_charges[mm_idx],
        }
