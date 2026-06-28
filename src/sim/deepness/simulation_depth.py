"""src/sim/deepness/simulation_depth.py

Simulation depth levels for the adaptive engine.

Each level represents a different simulation resolution:
  0: Statistical (coarse-grained ensemble)
  1: Cellular (mesoscopic, molecular field)
  2: Molecular (atomistic simulation, MD)
  3: Atomistic fine (high-res MD, tight cutoffs)
  4: Quantum (DFT region)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimulationDepth:
    """Enumeration of simulation depth levels."""
    STATISTICAL = 0
    CELLULAR = 1
    MOLECULAR = 2
    ATOMISTIC_FINE = 3
    QUANTUM = 4
    
    NAMES = {
        0: "Statistical",
        1: "Cellular",
        2: "Molecular",
        3: "Atomistic Fine",  
        4: "Quantum",
    }
    
    @classmethod
    def description(cls, depth):
        return cls.NAMES.get(depth, f"Unknown ({depth})")


class DepthSelector(nn.Module):
    """
    Selects simulation depth per atom/region based on importance scoring.
    """
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.importance_net = ImportanceNet(hidden_dim)
    
    def forward(self, node_features):
        """
        Args:
            node_features: (n_atoms, hidden_dim) node features
            
        Returns:
            depths: (n_atoms,) depth scores [0, 1, 2, 3, 4]
        """
        importance = self.importance_net(node_features)  # (n, 1)
        # Convert to discrete depth
        thresholds = torch.tensor([0.2, 0.4, 0.6, 0.8], device=node_features.device)
        depths = torch.zeros(len(node_features), dtype=torch.long, device=node_features.device)
        
        for i, threshold in enumerate(thresholds):
            if importance.any() > threshold:
                depths[importance.squeeze() > threshold] = i + 1
        
        return depths


class ImportanceNet(nn.Module):
    """Network that predicts importance from node features."""
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.silu(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.silu(),
            nn.Linear(hidden_dim // 2, 1),
            nn.sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    s = SimulationDepth()
    print(f"Molecular depth codes:")
    for k, v in s.NAMES.items():
        print(f"  {k}: {v}")
    print(f"\nDescription of depth 3: {DepthSelector.description(3)}")
