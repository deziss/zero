"""src/bio/cellular/population.py — Cell population simulation

Manages populations of cells with:
  - Division, death, signaling, differentiation
  - Spatial layout
  - Emergent tissue properties
"""

import numpy as np
from typing import List, Optional
from .cell import Cell, CellType


class CellPopulation:
    """
    Population of interacting cells.
    
    Simulates tissue-like cell populations with emergent behavior.
    """
    
    def __init__(self, initial_size: int = 100):
        self.cells: List[Cell] = []
        self.sim_step = 0
        self._next_id = 0
        
        # Spawn initial cells
        for _ in range(initial_size):
            cell = Cell(
                cell_id=self._next_id,
                position=np.array([np.random.uniform(0, 50),
                                  np.random.uniform(0, 50),
                                  np.random.uniform(0, 50)]),
                cell_type=CellType.STEM,
                health=1.0,
            )
            self.cells.append(cell)
            self._next_id += 1
        
        self.signaling_field = {}  # molecule_key -> (grid, shape)
    
    def step(
        self,
        dt: float = 1.0,
        signaling_range: float = 10.0,
    ) -> Dict[str, float]:
        """
        Advance all cells by one step.
        
        Returns:
            stats dict with population metrics.
        """
        self.sim_step += 1
        new_cells = []
        dead_cells = []
        
        # Update signaling field
        for mol in self.signaling_field:
            for cell in self.cells:
                # Receive signals within range
                pass
        
        for cell in self.cells:
            result = cell.step(dt)
            
            if isinstance(result, list):
                # Division
                for child in result:
                    child.cell_id = self._next_id
                    self._next_id += 1
                    new_cells.append(child)
            
            if result is None:
                # Death
                dead_cells.append(cell.cell_id)
        
        # Add new cells, remove dead cells
        self.cells.extend(new_cells)
        self.cells = [c for c in self.cells if c.cell_id not in dead_cells]
        
        # Compute stats
        stats = self.compute_statistics()
        
        return stats
    
    def compute_statistics(self) -> Dict[str, float]:
        """Compute population statistics."""
        if not self.cells:
            return {
                'n_cells': 0,
                'n_stem': 0,
                'n_prolif': 0,
                'n_diff': 0,
                'avg_health': 0.0,
                'avg_age': 0.0,
            }
        
        types = np.bincount(
            [c.cell_type.value for c in self.cells],
            minlength=5
        )
        
        return {
            'n_cells': len(self.cells),
            'n_stem': types[0],
            'n_prolif': types[2],
            'n_diff': types[1],
            'n_apopt': types[3],
            'n_necrot': types[4],
            'avg_health': np.mean([c.health for c in self.cells]),
            'avg_age': np.mean([c.age for c in self.cells]),
        }
    
    def add_cells(
        self,
        n_new: int,
        cell_type: CellType = CellType.STEM,
    ):
        """Add new cells to the population."""
        for _ in range(n_new):
            cell = Cell(
                cell_id=self._next_id,
                position=np.random.uniform(0, 50, 3),
                cell_type=cell_type,
                health=1.0,
            )
            self.cells.append(cell)
            self._next_id += 1
