"""src/ml/model/mace.py

MACE-inspired equivariant molecular force predictor.

Architecture:
1. Input encoding: atomic numbers → node features, positions → radial + angular features
2. Message passing: equivariant convolutions with radial tensor products
3. Readout: invariant scalars → energy, equivariant vectors → forces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.radial_basis import bessel_basis
from ..layers.spherical_harmonics import real_spherical_harmonics
from ..layers.equivariant_layers import EquivariantConvLayer, RadialTensorProduct


class MACE(nn.Module):
    """
    MACE (Molecular Attention Equivariant) inspired architecture.
    
    This model:
    1. Encodes atomic types and positions
    2. Uses equivariant message passing with radial/angular features
    3. Outputs per-atom energy and forces
    """

    def __init__(self, 
                 max_degree=4,
                 max_l=4,
                 num_interactions=2,
                 hidden_dim=128,
                 n_rbf=20,
                 r_cut=5.0,
                 cutoff_mode="bessel"):
        """
        Args:
            max_degree: maximum degree of spherical harmonics
            max_l: maximum angular momentum (for tensor products)
            num_interactions: number of equivariant conv layers
            hidden_dim: hidden dimension
            n_rbf: number of radial basis functions
            r_cut: cutoff radius
            cutoff_mode: "bessel" or "airy"
        """
        super().__init__()
        
        self.max_degree = max_degree
        self.n_rbf = n_rbf
        self.r_cut = r_cut
        
        # Atomic number encoding
        n_atomic_numbers = 100  # support up to Z=100
        self.atom_embedding = nn.Linear(n_atomic_numbers, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList()
        for i in range(num_interactions):
            layer = EquivariantConvLayer(
                hidden_dim=hidden_dim,
                max_degree=max_degree,
                num_interactions=1
            )
            self.layers.append(layer)
        
        # Edge feature network
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.silu,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Radial tensor product
        n_angular = (max_degree + 1) ** 2
        self.rtp = RadialTensorProduct(n_rbf, max_degree, hidden_dim)
        
        # Energy readout
        self.energy_readout = nn.Linear(hidden_dim, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.energy_readout.weight, a=0.5)
        if self.energy_readout.bias is not None:
            nn.init.zeros_(self.energy_readout.bias)

    def encode(self, atomic_numbers, batch_indices=None):
        """Encode input atoms to node features + edge features."""
        n_atoms = atomic_numbers.shape[0]
        n_edges = n_atoms * (n_atoms - 1)
        
        # Create edge list (all-to-all within molecule)
        src_idx = torch.arange(n_atoms).unsqueeze(1).expand(n_atoms, -1).reshape(-1)
        tgt_idx = torch.arange(n_atoms).unsqueeze(0).expand(n_atoms, -1).reshape(-1)
        
        # Mask: i != j and within batch
        mask = src_idx != tgt_idx
        src, tgt = src_idx[mask], tgt_idx[mask]
        edge_index = torch.stack([src, tgt])
        
        # Encode atomic numbers
        one_hot = F.one_hot(atomic_numbers.long(), n_atomic_numbers).float()
        node_features = self.atom_embedding(one_hot)  # (n_atoms, hidden_dim)
        
        # Compute distances and radial basis
        n_edges = edge_index.shape[1]
        edge_rbf = torch.zeros(n_edges, self.n_rbf, device=node_features.device)
        
        # Compute distances
        src_pos = positions[src]  # (n_edges, 3)
        tgt_pos = positions[tgt]
        edge_dists = torch.norm(src_pos - tgt_pos, dim=-1)
        edge_vec = (src_pos - tgt_pos) / (edge_dists + 1e-10).unsqueeze(-1)
        
        # Radial basis
        edge_rbf = bessel_basis(edge_dists, n_rbf=n_rbf, r_min=0.0, r_cut=self.r_cut)
        
        # Spherical harmonics
        edge_sh = real_spherical_harmonics(edge_vec, degrees=self.max_degree)
        
        return node_features, edge_index, edge_rbf, edge_dists, edge_vec, edge_sh

    def forward(self, atomic_numbers, positions):
        """
        Args:
            atomic_numbers: (n_atoms,) atomic numbers
            positions: (n_atoms, 3) atomic positions
            
        Returns:
            energy: (batch_size,) total energy
            forces: (n_atoms, 3) total forces
        """
        n_atoms = atomic_numbers.shape[0]
        
        # Edge list (all-to-all, excluding self)
        src_idx = torch.arange(n_atoms)
        tgt_idx = torch.arange(n_atoms)
        src_idx = src_idx.unsqueeze(1).expand(n_atoms, n_atoms).reshape(-1)
        tgt_idx = tgt_idx.unsqueeze(0).expand(n_atoms, n_atoms).reshape(-1)
        mask = src_idx != tgt_idx
        src, tgt = src_idx[mask], tgt_idx[mask]
        edge_index = torch.stack([src, tgt])
        
        # Encode atomic numbers
        one_hot = F.one_hot(atomic_numbers.long(), 100).float()
        x = self.atom_embedding(one_hot)  # (n_atoms, hidden_dim)
        
        # Edge features
        src_pos = positions[src]
        tgt_pos = positions[tgt]
        edge_dists = torch.norm(src_pos - tgt_pos, dim=-1)
        edge_vec = (src_pos - tgt_pos) / (edge_dists + 1e-10).unsqueeze(-1)
        
        edge_rbf = bessel_basis(edge_dists, n_rbf=self.n_rbf, r_min=0.0, r_cut=self.r_cut)
        edge_sh = real_spherical_harmonics(edge_vec, degrees=self.max_degree)
        
        # Message passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_rbf, edge_dists, edge_sh)
        
        # Energy readout
        energy = self.energy_readout(x)[:, 0].sum()
        energy = energy.squeeze()
        
        # Forces via backpropagation of energy
        forces = torch.autograd.grad(
            energy, positions, grad_outputs=torch.ones_like(energy), 
            create_graph=True
        )[0]
        
        return energy, forces


if __name__ == "__main__":
    # Quick test
    n_atoms = 5
    model = MACE(max_degree=2, max_l=2, num_interactions=2, hidden_dim=32, n_rbf=10, r_cut=5.0)
    
    # Dummy input: 5 H atoms in a small molecule
    atomic_numbers = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
    positions = torch.randn(n_atoms, 3) * 0.5 + torch.arange(n_atoms).unsqueeze(-1) * 1.5
    
    energy, forces = model(atomic_numbers, positions)
    print(f"Energy: {energy.item():.4f}")
    print(f"Forces shape: {forces.shape}")
    print(f"Force norm: {torch.norm(forces).item():.4f}")
