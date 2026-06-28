"""src/sim/layers/depth_selector.py

Depth selector layer: maps per-atom features to simulation depth scores.

Architecture:
  Input: atom features (node embeddings from GNN/MACE)
  Output: depth_scores (per-atom, continuous [0, 1])
  
  depth 0 = statistical (coarsest)
  depth 1 = cellular (mesoscopic)
  depth 2 = molecular (atomistic simulation)
  depth 3 = atomistic fine (high-res MD)
  depth 4 = quantum (DFT region)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthSelectorLayer(nn.Module):
    """
    Neural layer that predicts simulation depth for each atom.
    
    Takes equivariant node features as input and outputs a depth
    score between 0 and 1. Higher = more precision needed.
    """
    
    def __init__(self, hidden_dim=64, n_depths=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_depths = n_depths
        
        # Feature extraction from equivariant node embeddings
        self.feat_extract = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.silu(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.silu(),
        )
        
        # Depth scoring head
        self.score_head = nn.Linear(hidden_dim // 2, n_depths)
        
        # Uncertainty gates (learned thresholds)
        self.thresholds = nn.Parameter(torch.tensor([0.3, 0.5, 0.7, 0.9]))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in [self.feat_extract[-2], self.feat_extract[-1], self.score_head]:
            nn.init.kaiming_uniform_(layer.weight, a=0.5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.constant_(self.thresholds, [0.3, 0.5, 0.7, 0.9])
    
    def forward(self, x):
        """
        Args:
            x: (n_atoms, hidden_dim) equivariant node features
            
        Returns:
            depth_scores: (n_atoms,) scores in [0, 4]
            depth_probs: (n_atoms, n_depths) soft depth assignment
        """
        feats = self.feat_extract(x)  # (n, hidden//2)
        scores_raw = self.score_head(feats)  # (n, n_depths)
        depth_probs = F.softmax(scores_raw, dim=-1)
        
        # Convert to discrete depth scores
        depth_scores = (depth_probs * torch.arange(self.n_depths, device=x.device)).sum(dim=-1)
        
        return depth_scores, depth_probs


class DepthSelector(nn.Module):
    """
    Simpler depth selector using deterministic feature thresholds.
    
    Uses fixed rules instead of learned scores:
  - high_force_norm → depth 3-4
  - high_gradient_mag → depth 2-3
  - low_activity → depth 0-1
"""
    
    def __init__(self, force_threshold=0.5, gradient_threshold=1.0):
        super().__init__()
        self.force_threshold = force_threshold
        self.gradient_threshold = gradient_threshold
    
    def forward(self, force_norm, gradient_norm):
        """
        Args:
            force_norm: (n_atoms,) magnitude of local forces
            gradient_norm: (n_atoms,) gradient of energy surface
            
        Returns:
            depths: (n_atoms,) discrete depth [0, 1, 2, 3, 4]
        """
        if force_norm is not None and force_norm.max() > self.force_threshold:
            if gradient_norm is not None and gradient_norm.max() > self.gradient_threshold:
                return torch.minimum(torch.tensor(4), ((force_norm / self.force_threshold) + (gradient_norm / self.gradient_threshold) / 2).round().long())
            else:
                return torch.minimum(torch.tensor(3), (force_norm / self.force_threshold * 2).round().long())
        elif gradient_norm is not None and gradient_norm.any() > self.gradient_threshold:
            return torch.minimum(torch.tensor(2), (gradient_norm / self.gradient_threshold * 2).round().long() + 1)
        else:
            # Low activity → coarse
            return torch.zeros_like(force_norm, dtype=torch.long)
    
    @staticmethod
    def from_depth_scores(depth_scores, threshold=0.5):
        """Convert continuous depth scores to discrete."""
        depths = torch.floor(depth_scores).long()
        depths = torch.clamp(depths, 0, 4)
        return depths
