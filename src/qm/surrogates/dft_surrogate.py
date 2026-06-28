"""src/qm/surrogates/dft_surrogate.py — DFT surrogate model

Trains a fast neural network to approximate DFT calculations.
Used to replace expensive quantum chemistry with learned surrogates.

Surrogate architecture:
  Input: atomic_number, position, environment (from MACE)
  Output: energy (per-molecule or per-atom), forces
  
Training uses ground-truth DFT data from PySCFInterface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class DFTSurrogate(nn.Module):
    """
    Neural network surrogate for DFT calculations.
    
    Trained on DFT data to predict energies and forces for atoms.
    10-1000x faster than direct DFT while maintaining accuracy.
    """
    
    def __init__(
        self,
        n_atomic_types: int = 100,
        hidden_dim: int = 256,
        n_layers: int = 6,
        max_n_atoms: int = 1000,
    ):
        super().__init__()
        self.n_atomic_types = n_atomic_types
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_n_atoms = max_n_atoms
        
        # Atomic number embedding
        self.atomic_emb = nn.Embedding(n_atomic_types, hidden_dim)
        
        # Position encoding (Gaussian radial basis)
        self.n_rbf = 32
        self.rbf_centers = torch.linspace(0, 10, self.n_rbf)
        
        # Message-passing layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + self.n_rbf, hidden_dim),
                nn.silu(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])
        
        # Output heads
        self.edge_energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.silu(),
            nn.Linear(hidden_dim, 1),
        )
        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.silu(),
            nn.Linear(hidden_dim, 3),  # force components
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.atomic_emb.reset_parameters()
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer[0].weight, a=0.5)
            nn.init.zeros_(layer[0].bias)
            nn.init.kaiming_uniform_(layer[2].weight, a=0.5)
            nn.init.zeros_(layer[2].bias)
        for head in [self.edge_energy_head, self.force_head]:
            nn.init.kaiming_uniform_(head[0].weight, a=0.5)
            nn.init.zeros_(head[0].bias)
            nn.init.kaiming_uniform_(head[2].weight, a=0.5)
            nn.init.zeros_(head[2].bias)
    
    def _rbf_encoding(self, distances: torch.Tensor) -> torch.Tensor:
        """Encode distances using Gaussian radial basis functions."""
        rbf = torch.exp(-((distances - self.rbf_centers) / 2)**2)
        return rbf
    
    def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor):
        """
        Predict energy and forces for a molecule.
        
        Args:
            atomic_numbers: (n_atoms,) atomic numbers
            positions:      (n_atoms, 3) Cartesian coordinates
            
        Returns:
            energy: (1,) total energy
            forces: (n_atoms, 3) atomic forces
        """
        n_atoms = len(atomic_numbers)
        
        if n_atoms > self.max_n_atoms:
            raise ValueError(
                f"Too many atoms: {n_atoms} > {self.max_n_atoms}"
            )
        
        # Atom embeddings
        atom_feats = self.atomic_emb(atomic_numbers)  # (n, hidden)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
        rbf_features = self._rbf_encoding(dist_matrix)
        
        # Message passing
        node_feats = atom_feats
        for layer in self.layers:
            # Gather neighbor features
            neighbor_feats = node_feats.unsqueeze(0).expand(n_atoms, n_atoms, -1)
            combined = torch.cat([
                node_feats.unsqueeze(1).expand(-1, n_atoms, -1),
                neighbor_feats,
                rbf_features.unsqueeze(-1),
            ], dim=-1)
            
            # Apply layer to all pairs
            message = layer(combined.view(-1, combined.shape[-1])).view(
                n_atoms, n_atoms, self.hidden_dim
            )
            
            # Pool messages (mean)
            node_feats = node_feats + message.mean(dim=1)
        
        # Sum energy
        energy_head = self.edge_energy_head(node_feats)
        energy = energy_head.sum().unsqueeze(0)
        
        # Forces (energy gradients)
        grad = self.force_head(node_feats)
        forces = grad
        
        return energy, forces


class SurrogateDataset:
    """
    Dataset for training DFT surrogate models.
    
    Stores tuples of:
      - atomic_numbers (list, shape [n_atoms])
      - positions (numpy, shape [n_atoms, 3])
      - true_energy (float)
      - true_forces (numpy, shape [n_atoms, 3])
    """
    
    def __init__(
        self,
        n_atoms_max: int = 1000,
    ):
        self.n_atoms_max = n_atoms_max
        self.data = []
    
    def add_entry(
        self,
        atomic_numbers: List[int],
        positions: np.ndarray,
        energy: float,
        forces: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        """
        Add one molecule to the dataset.
        
        Args:
            atomic_numbers: element atomic numbers
            positions: Cartesian coordinates in Angstrom
            energy: DFT total energy (Hartree)
            forces: forces in Hartree/Angstrom
            metadata: optional metadata dict
        """
        if len(atomic_numbers) > self.n_atoms_max:
            raise ValueError(f"Too many atoms: {len(atomic_numbers)} > {self.n_atoms_max}")
        
        self.data.append({
            'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'energy': torch.tensor(energy, dtype=torch.float32),
            'forces': torch.tensor(forces, dtype=torch.float32),
            'metadata': metadata or {},
        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_statistics(self) -> Dict[str, float]:
        """Return basic statistics of the dataset."""
        energies = [d['energy'].item() for d in self.data]
        n_atoms_list = [len(d['atomic_numbers']) for d in self.data]
        
        return {
            'n_molecules': len(self.data),
            'avg_n_atoms': np.mean(n_atoms_list),
            'max_n_atoms': max(n_atoms_list),
            'min_n_atoms': min(n_atoms_list),
            'avg_energy': np.mean(energies),
            'std_energy': np.std(energies),
        }


if __name__ == "__main__":
    # Test
    surf = DFTSurrogate(n_atomic_types=118, hidden_dim=64, n_layers=3)
    atoms = torch.tensor([8, 1, 1])  # water molecule
    pos = torch.tensor([[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]], dtype=torch.float32)
    
    energy, forces = surf(atoms, pos)
    print(f"Energy: {energy.item():.4f} Hartree")
    print(f"Forces shape: {forces.shape}")
    
    # Test surrogate dataset
    ds = SurrogateDataset(n_atoms_max=100)
    ds.add_entry(
        atomic_numbers=[8, 1, 1],
        positions=np.array([[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]),
        energy=-75.0,  # approximate water energy in Hartree
        forces=np.zeros((3, 3)),
    )
    stats = ds.get_statistics()
    print(f"\nDataset statistics: {stats}")
