"""src/bio/cellular/cell.py — Cellular abstractions

Models cells as objects with:
  - State: position, type, age, health
  - Behavior: division, migration, death
  - Phenomena: signaling, differentiation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import enum


class CellState:
    """Cell behavioral state."""
    QUIESCENT = 0
    RESTING = 1
    ACTIVE = 2
    DIFFERENTIATING = 3


class Cell:
    """
    Single cell simulation entity.
    
    Attributes:
        cell_id: unique identifier
        position: (x, y, z) in simulation space
        cell_type: enum CellType
        age: time steps since creation
        health: [0, 1] vitality
        velocity: (vx, vy, vz)
        signaling_signals: list of signaling molecules produced
    """
    
    def __init__(
        self,
        cell_id: int,
        position: np.ndarray,
        cell_type: CellType = CellType.STEM,
        health: float = 1.0,
    ):
        self.cell_id = cell_id
        self.position = position.astype(np.float64)
        self.cell_type = cell_type
        self.age = 0
        self.health = health
        self.velocity = np.zeros(3, dtype=np.float64)
        self.signaling = {}
        self.receptors = {}
        self.division_counter = 0
        self.differentiation_prob = 0.0
        self.death_prob = 0.001
        self.born_at = 0  # simulation step born
    
    def step(self, dt: float = 1.0):
        """Advance cell state by one time step."""
        self.age += 1
        self.division_counter += 1
        
        # Update health
        self.health -= self.death_prob * dt
        self.health = max(0.0, min(1.0, self.health))
        
        # Update position with velocity
        self.position += self.velocity * dt
        
        # Random walk (diffusion)
        self.velocity += np.random.normal(0, 0.1, 3) * dt
        self.velocity *= 0.9  # friction
        self.position += np.random.normal(0, 0.05, 3)
        
        # Check division eligibility
        if self._should_divide():
            return self._divide()
        
        # Check death
        if self._should_die():
            self.cell_type = CellType.NECROTIC
            self.health = 0.0
            return None  # removed
        
        return self
    
    def _should_divide(self) -> bool:
        """Check if the cell should divide."""
        if self.division_counter < 10:
            return False
        if self.health < 0.5:
            return False
        prob = {
            CellType.STEM: 0.3,
            CellType.PROLIFERATING: 0.5,
            CellType.DIFFERENTIATING: 0.0,
            CellType.APOPTOTIC: 0.0,
            CellType.NECROTIC: 0.0,
        }.get(self.cell_type, 0.1)
        return np.random.random() < (prob * self.health)
    
    def _divide(self) -> Optional['Cell']:
        """Divide the cell into two."""
        half_health = self.health / 2
        child1 = Cell(
            cell_id=-1,  # assigned later
            position=self.position + np.random.normal(0, 0.05, 3),
            cell_type=self.cell_type,
            health=half_health,
        )
        child2 = Cell(
            cell_id=-1,
            position=self.position + np.random.normal(0, 0.05, 3),
            cell_type=self.cell_type,
            health=half_health,
        )
        self.health = half_health
        self.division_counter = 0
        return [child1, child2]
    
    def _should_die(self) -> bool:
        """Check if the cell should undergo apoptosis."""
        return np.random.random() < self.death_prob * (1.0 - self.health)
    
    def secrete_signaling_molecule(self, molecule: str, concentration: float):
        """Release signaling molecule into environment."""
        self.signaling[molecule] = concentration
    
    def receive_signal(self, molecule: str, concentration: float):
        """Receive signaling molecule from environment."""
        if molecule not in self.receptors:
            self.receptors[molecule] = 0
        self.receptors[molecule] += concentration
    
    def differentiate(self) -> 'Cell':
        """Differentiate into a new cell type."""
        new_type = {
            CellType.STEM: CellType.PROLIFERATING,
            CellType.PROLIFERATING: CellType.DIFFERENTIATING,
            CellType.DIFFERENTIATING: None,
            CellType.APOPTOTIC: None,
            CellType.NECROTIC: None,
        }.get(self.cell_type)
        
        if new_type is None:
            return None
        
        self.cell_type = new_type
        self.diff_prob = 0.5  # higher diff prob after initial
        self.division_prob = 0.0  # stop dividing after diff
    
    def __repr__(self):
        return (f"Cell(id={self.cell_id}, type={self.cell_type.name}, "
                f"pos=({self.position[0]:.2f}, {self.position[1]:.2f}, "
                f"{self.position[2]:.2f}), health={self.health:.2f})")
