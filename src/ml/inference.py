"""src/ml/inference.py

Inference API for molecular force prediction.

Provides a unified interface:
    api = InferenceAPI.from_state_dict(state_dict)
    energy, forces = api.predict(molecule)

Where molecule = {"atomic_numbers": [...], "positions": [[x,y,z], ...]}
"""

import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager


class InferenceAPI(nn.Module):
    """
    Unified inference interface for molecular force predictors.
    
    Handles:
    - Model loading (weights from checkpoint or state_dict)
    - GPU/CPU device management
    - Batch prediction
    - Automatic force computation via energy gradient
    """

    def __init__(self, model, device=None):
        super().__init__()
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.mode = 'force'  # 'energy', 'force', or 'energy_force'

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        """Load model from checkpoint file."""
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = checkpoint['model']
        
        # Detect model type from checkpoint metadata
        model_class = checkpoint.get('model_class', 'MACE')
        if model_class == 'MACE':
            model = MACE(
                max_degree=checkpoint.get('max_degree', 4),
                max_l=checkpoint.get('max_l', 4),
                num_interactions=checkpoint.get('num_interactions', 2),
                hidden_dim=checkpoint.get('hidden_dim', 128),
                n_rbf=checkpoint.get('n_rbf', 20),
                r_cut=checkpoint.get('r_cut', 5.0),
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        api = cls(model, device)
        api.mode = checkpoint.get('mode', 'energy_force')
        return api, checkpoint.get('metrics', {})

    def predict(self, molecule):
        """
        Predict energy and forces for a single molecule.
        
        Args:
            molecule: dict with:
                - atomic_numbers: list/array of atomic numbers (Z)
                - positions: list/array of [x,y,z] positions
            
        Returns:
            energy: float — total energy
            forces: numpy array of shape (n_atoms, 3) — atomic forces
        """
        atomic_nums = torch.tensor(molecule['atomic_numbers'], dtype=torch.long).to(self.device)
        positions = torch.tensor(molecule['positions'], dtype=torch.float32).to(self.device)
        
        with torch.set_grad_enabled(self.mode != 'energy'):
            energy, forces = self.model(atomic_nums, positions)
        
        # Move to CPU for numpy conversion
        energy = energy.detach().cpu().item()
        forces = forces.detach().cpu().numpy()
        
        return energy, forces

    def predict_batch(self, molecules):
        """
        Predict energy and forces for a batch of molecules.
        
        Args:
            molecules: list of dicts (each = one molecule)
            
        Returns:
            energies: list of floats
            forces: list of numpy arrays
        """
        energies = []
        forces_list = []
        
        for mol in molecules:
            energy, forces = self.predict(mol)
            energies.append(energy)
            forces_list.append(forces)
        
        return energies, forces_list

    def set_device(self, device):
        """Switch between cuda/cpu devices."""
        self.device = device
        self.model.to(device)

    def set_mode(self, mode='energy_force'):
        """
        Set prediction mode.
        
        - 'energy': Only energy (no gradient computation → faster)
        - 'force': Only forces (computes forces but discards energy)
        - 'energy_force': Both energy and forces (via energy gradient)
        """
        self.mode = mode

    @contextmanager
    def no_grad(self):
        """Context manager for energy-only prediction (no gradient)."""
        self.set_mode('energy')
        yield
        self.set_mode('energy_force')


class MolecularForcePredictor:
    """
    Simpler functional API for molecular force prediction.
    
    Usage:
        predictor = MolecularForcePredictor(weights_path)
        energy = predictor.predict_energy(mol)
        forces = predictor.predict_forces(mol)
        energy, forces = predictor.predict(mol)  # both
    """

    def __init__(self, weights_path=None, model=None, device=None):
        if model is not None:
            self.model = model
        elif weights_path is not None:
            checkpoint = torch.load(weights_path, map_location='cpu')
            self.model = checkpoint['model']
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Either weights_path or model must be provided")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, atomic_numbers, positions, as_numpy=True):
        """
        Predict energy and forces.
        
        Args:
            atomic_numbers: array-like of atomic numbers
            positions: array-like of (n,3) positions
            as_numpy: if True, return numpy arrays
            
        Returns:
            (energy, forces) — energy is scalar, forces is (n,3)
        """
        atomic_nums = torch.as_tensor(atomic_numbers, dtype=torch.long).to(self.device)
        positions = torch.as_tensor(positions, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            energy, forces = self.model(atomic_nums, positions)
        
        if as_numpy:
            energy = energy.item() if hasattr(energy, 'item') else float(energy)
            forces = forces.cpu().numpy() if hasattr(forces, 'cpu') else np.array(forces)
        
        return energy, forces

    def predict_energy(self, atomic_numbers, positions):
        """Predict only energy (faster)."""
        atomic_nums = torch.as_tensor(atomic_numbers, dtype=torch.long).to(self.device)
        positions = torch.as_tensor(positions, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            energy, _ = self.model(atomic_nums, positions)
        
        return energy.item() if hasattr(energy, 'item') else float(energy)

    def predict_forces(self, atomic_numbers, positions):
        """Predict only forces."""
        energy, forces = self.predict(atomic_numbers, positions)
        return forces


if __name__ == "__main__":
    # Quick demo
    print("Molecular Force Predictor API")
    print("Load a checkpoint or instantiate with a model:")
    print()
    print("  # Via checkpoint")
    print("  predictor = MolecularForcePredictor('checkpoint.pt')")
    print()
    print("  # Via model")
    print("  model = MACE()")
    print("  predictor = MolecularForcePredictor(model=model)")
    print()
    print("  # Predict")
    print("  energy, forces = predictor.predict(atomic_numers, positions)")
