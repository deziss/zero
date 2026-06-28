"""src/sim/layers/uncertainty_estimator.py

Uncertainty estimator for adaptive simulation depth.

Predicts the confidence of the current simulation depth at each atom
location. High uncertainty → need deeper simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyEstimator(nn.Module):
    """
    Estimates per-atom simulation uncertainty.
    
    Uses an ensemble of shallow networks to predict uncertainty via
    prediction variance.
    """
    
    def __init__(self, n_heads=4, hidden_dim=32):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.silu(),
            nn.Linear(hidden_dim, 1)
        ) for _ in range(n_heads)])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for head in self.heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=0.5)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Args:
            x: (n_atoms, hidden_dim) node features
            
        Returns:
            uncertainty: (n_atoms,) uncertainty score
            predictions: (n_atoms, n_heads) ensemble predictions
        """
        predictions = torch.stack([head(x)[:, 0] for head in self.heads], dim=1)
        uncertainty = predictions.std(dim=1)
        # Normalize
        uncertainty = F.normalize(uncertainty, p=2, dim=0)
        
        return uncertainty, predictions


class ImportanceScorer(nn.Module):
    """
    Score each atom's importance for deeper simulation.
    
    Combines multiple signals:
    - Force magnitude (direct indicator of activity)
    - Energy gradient (indication of chemical bonds, reactions)
    - Uncertainty from prediction
    - Local density (high density → need accuracy)
    """
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.combiner = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 5 signal types
            nn.silu(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.combiner:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, force_mag, energy_grad, uncertainty, local_density, charge_mag):
        """
        Args:
            force_mag: (n_atoms,) force magnitude
            energy_grad: (n_atoms,) energy gradient
            uncertainty: (n_atoms,) prediction uncertainty
            local_density: (n_atoms,) local atom density
            charge_mag: (n_atoms,) magnitude of atomic charge
            
        Returns:
            importance: (n_atoms,) importance scores in [0, 1]
        """
        signals = torch.stack([force_mag, energy_grad, uncertainty, local_density, charge_mag], dim=-1)
        importance_raw = self.combiner(signs)
        importance = torch.sigmoid(importance_raw)[:, 0]
        
        return importance


if __name__ == "__main__":
    # Test
    est = UncertaintyEstimator(n_heads=4, hidden_dim=16)
    x = torch.randn(10, 16)
    unc, preds = est(x)
    print(f"Uncertainty shape: {unc.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Uncertainty: {unc.tolist()}")
