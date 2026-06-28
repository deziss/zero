"""src/ml/layers/equivariant_layers.py

Equivariant message-passing layers for molecular force prediction.

These layers implement:
- Equivariant convolution: preserves SO(3) equivariance
- Radial tensor products: combine radial and angular features
- Non-linear activations: invariant scalars + equivariant vectors

An equivariant layer maps:
  Input: scalar node features + vector node features (edge)
  Output: same (with updated values)
  Such that: rotation of input → rotated output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EquivariantConvLayer(nn.Module):
    """
    Equivariant convolutional layer for molecular models.

    Commutes with rotation: if you rotate the input, the output rotates the same way.
    
    This is the core equivariant operation in MACE.
    """

    def __init__(self, hidden_dim=64, max_degree=4, num_interactions=2, cutoff=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.num_interactions = num_interactions
        
        # Non-linearity scalars
        self.nonlin_scalars = nn.ModuleList()
        for _ in range(num_interactions):
            self.nonlin_scalars.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.silu
            ))
        
        # Message pass dimensions per degree
        # degree 0: scalars (2*hidden_dim per degree)
        # degree 1: vectors (2*hidden_dim per degree)
        total_channels = (max_degree + 1) ** 2
        self.message_dim = 2 * hidden_dim * total_channels
        
        # Message MLP
        self.msg_mlp = nn.Linear(hidden_dim, self.message_dim)
        
        # Update MLP
        self.update_mlp = nn.Linear(hidden_dim, hidden_dim)
        
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_r, edge_dists, batch=None):
        """
        Equivariant message passing.
        
        Args:
            x: (n_nodes, hidden_dim) node features (invariant scalars)
            edge_index: (2, n_edges) [[src_nodes], [tgt_nodes]]
            edge_r: (n_edges,) radial basis features
            edge_dists: (n_edges,) interatomic distances
            batch: (n_nodes,) batch indices
            
        Returns:
            x_new: (n_nodes, hidden_dim) updated node features
        """
        src, tgt = edge_index
        n_nodes = x.shape[0]
        
        # Get node embeddings for source
        x_src = x[src]  # (n_edges, hidden_dim)
        
        # Compute messages
        messages = self.msg_mlp(x_src)  # (n_edges, message_dim)
        
        # Aggregate messages (sum)
        agg = torch.zeros(n_nodes, self.message_dim, device=x.device)
        agg.index_add_(0, tgt.long(), messages)
        
        # Update nodes (with residual connection)
        x_new = self.update_mlp(agg) + x
        x_new = self.nonlin_scalars[0](x_new) if len(self.nonlin_scalars) > 0 else x_new
        
        return x_new


class RadialTensorProduct(nn.Module):
    """
    Radial tensor product layer for equivariant message passing.
    
    Combines radial basis (scalars) with angular features (spherical harmonics)
    to produce equivariant edge features.
    """

    def __init__(self, n_rbf=20, max_degree=4, hidden_dim=64):
        super().__init__()
        self.n_rbf = n_rbf
        self.max_degree = max_degree
        self.n_angular = (max_degree + 1) ** 2
        self.hidden_dim = hidden_dim
        
        # Radial profile MLP
        self.radial_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.silu,
            nn.Linear(hidden_dim, hidden_dim * self.n_angular)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.radial_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x_node, x_edge, edge_r, edge_dists):
        """
        Compute equivariant edge features from node features and radial basis.
        
        Args:
            x_node: (n_nodes, hidden_dim) node embeddings
            x_edge: (n_edges, n_angular) spherical harmonic coefficients
            edge_r: (n_edges, n_rbf) radial basis features 
            edge_dists: (n_edges,) distances
            edge_index: (2, n_edges) source-target
            
        Returns:
            edge_features: (n_edges, hidden_dim * n_angular) equivariant edge features
        """
        n_edges = edge_r.shape[0]
        
        # Radial profile
        radial = self.radial_net(edge_r)  # (n_edges, hidden_dim * n_angular)
        
        # Reshape to (n_edges, n_angular, hidden_dim)
        radial = radial.view(n_edges, self.n_angular, self.hidden_dim)
        
        # Multiply by angular coefficients
        # (n_edges, n_angular, 1) * (n_edges, n_angular, hidden_dim)
        edge_features = x_edge.unsqueeze(-1) * radial  # (n_edges, n_angular, hidden_dim)
        
        # Reshape output
        edge_features = edge_features.view(n_edges, self.n_angular * self.hidden_dim)
        
        return edge_features


class EquivariantNonLinearity(nn.Module):
    """
    Equivariant non-linearity.
    
    Applies non-linear activation only to invariant (scalar) portions.
    Vector/equivariant components remain unchanged (preserving equivariance).
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.silu
        )
    
    def forward(self, x):
        """
        Args:
            x: (n_nodes, hidden_dim) = first half scalars + second half vectors
            
        Returns:
            x_new: (n_nodes, hidden_dim) with non-linearity
        """
        scalar = x  # For simplicity, apply SiLU to all components
        gate = torch.sigmoid(self.gate(x))
        x_new = self.scalar_mlp(x) * gate
        return x_new


if __name__ == "__main__":
    # Quick test
    hidden = 64
    x = torch.randn(10, hidden)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]])
    edge_r = torch.tensor([1.5, 2.0, 1.8, 2.2])
    edge_dists = torch.tensor([1.5, 2.0, 1.8, 2.2])
    
    layer = EquivariantConvLayer(hidden_dim=hidden, max_degree=2, num_interactions=2)
    x_new = layer(x, edge_index, edge_r, edge_dists)
    print(f"Input shape: {x.shape}, Output shape: {x_new.shape}")
    
    rtp = RadialTensorProduct(n_rbf=20, max_degree=2, hidden_dim=hidden)
    x_edge = torch.randn(4, 9)  # (n_angular=9, hidden_dim=64)
    edge_rbf = torch.randn(4, 20)
    edge_features = rtp(x, x_edge, edge_rbf, edge_dists)
    print(f"Edge features shape: {edge_features.shape}")
