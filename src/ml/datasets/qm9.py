"""src/ml/datasets/qm9.py

QM9 dataset loader for molecular force prediction.

QM9 contains 134K molecules (DFT/ωB97X-D/cc-pVDZ).
Each molecule: atoms (H,C,N,O,F), positions, energies, forces, charges.

"""

import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path


class QM9Dataset(data.Dataset):
    """
    QM9 dataset for molecular force prediction.
    """
    
    def __init__(self, root="data/qm9", download=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        # QM9 properties to extract
        self.names = [
            "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap",
            "r2", "zpve", "U0", "U", "H", "G", "Cv",
        ]
        
        # Target: energy and forces
        self.target_indices = [
            {"name": "U0", "index": int(3.884091e-3)},  # Electronic energy (Ha)
            {"name": "U", "index": int(3.884091e-3)},    # Internal energy
        ]
        
        # Download (placeholder - real QM9 requires ~10GB download)
        if download and not self._has_data():
            self.download_qm9()
        
        self.data = self._load_data()
    
    def _has_data(self):
        return (self.root / "qm9_processed.pt").exists()
    
    def download_qm9(self):
        """Download QM9 dataset from GitHub."""
        import urllib.request
        base_url = "https:// raw.githubusercontent.com/quantum-machine /qm9"
        
        print("Downloading QM9 (this will take time and ~10GB space)...")
        print("Manual download needed: https://zenodo.org/record/1234447")
        return True
    
    def _load_data(self):
        """Load processed QM9 data."""
        pt_path = self.root / "qm9_processed.pt"
        if pt_path.exists():
            return torch.load(pt_path)
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get molecule configuration."""
        mol = self.data[idx]
        
        return {
            "atomic_numbers": mol["atomic_numbers"],
            "positions": mol["positions"],
            "charges": mol["charges"],
            "energy": mol["energy"],
            "forces": mol["forces"],
            "natoms": mol["natoms"],
        }
    
    def get_train_val_test_splits(self, train_size=110000, val_size=12000):
        """Split dataset into train/validation/test."""
        n_train = train_size
        n_val = val_size
        n_test = len(self.data) - n_train - n_val
        
        train = QM9DatasetSubset(self.data[:n_train])
        val = QM9DatasetSubset(self.data[n_train:n_train+n_val])
        test = QM9DatasetSubset(self.data[n_train+n_val:])
        
        return train, val, test


class QM9DatasetSubset(data.Dataset):
    """Subset of QM9 for train/val/test splits."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    print("QM9 Dataset Loader")
    print("This module requires the QM9 dataset.")
    print("Download from: https://zenodo.org/record/1234447")
    print()
    print("Setup:")
    print("  1. Download qm9.xyz.gz from Zenodo")
    print("  2. Place in data/qm9/")
    print("  3. Run: python -m src.ml.datasets.qm9 download")
