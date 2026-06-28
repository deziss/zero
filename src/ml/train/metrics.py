"""src/ml/train/metrics.py

Force and energy metrics for molecular force predictor evaluation.

All metrics are in eV/Angstrom units.
"""

import numpy as np
import torch


def compute_metrics(pred_energy, true_energy, pred_force, true_force, units="eV_A"):
    """
    Compute force and energy metrics.

    Args:
        pred_energy: predicted energy (N,) or tensor
        true_energy: true energy (N,) or tensor
        pred_force: predicted forces (N×n_atoms×3) or tensor
        true_force: true forces (N×n_atoms×3) or tensor
        units: "eV_A", "kJ_mol_nm", "hartree_bohr"

    Returns:
        metrics: dict with MAE, RMSE for energy and forces
    """
    # Convert to numpy
    if isinstance(pred_energy, torch.Tensor):
        pred_energy = pred_energy.detach().cpu().numpy()
    if isinstance(true_energy, torch.Tensor):
        true_energy = true_energy.detach().cpu().numpy()
    if isinstance(pred_force, torch.Tensor):
        pred_force = pred_force.detach().cpu().numpy()
    if isinstance(true_force, torch.Tensor):
        true_force = true_force.detach().cpu().numpy()
    
    # Force per atom
    pred_force_per_atom = np.sqrt(np.mean(pred_force ** 2, axis=-1))  # (N,)
    true_force_per_atom = np.sqrt(np.mean(true_force ** 2, axis=-1))  # (N,)
    
    diff_force_per_atom = pred_force_per_atom - true_force_per_atom  # (N,)
    diff_energy = pred_energy - true_energy  # (N,)
    
    metrics = {
        # Energy metrics
        "energy_mae": np.mean(np.abs(diff_energy)),
        "energy_rmse": np.sqrt(np.mean(diff_energy ** 2)),
        "energy_mse": np.mean(diff_energy ** 2),
        
        # Force metrics  
        "force_mae": np.mean(np.abs(diff_force_per_atom)),
        "force_rmse": np.sqrt(np.mean(diff_force_per_atom ** 2)),
        "force_mse": np.mean(diff_force_per_atom ** 2),
    }
    
    return metrics


def compute_force_angle_error(pred_force, true_force):
    """Compute force angle error (between prediction and ground truth)."""
    if isinstance(pred_force, torch.Tensor):
        pred_force = pred_force.detach().cpu().numpy()
    if isinstance(true_force, torch.Tensor):
        true_force = true_force.detach().cpu().numpy()
    
    # Angle between force vectors
    cos_sin = np.dot(pred_force, true_force) / (np.linalg.norm(pred_force) * np.linalg.norm(true_force))
    cos_sin = np.clip(cos_sin, -1.0, 1.0)
    
    angle_error = np.arccos(cos_sin) * 180.0 / np.pi  # Degrees
    
    return {
        "angle_mae": np.mean(angle_error),
        "angle_rmse": np.sqrt(np.mean(angle_error ** 2)),
    }


if __name__ == "__main__":
    # Test
    n_samples = 100
    n_atoms = 10
    pred_e = np.random.randn(n_samples)
    true_e = pred_e + np.random.randn(n_samples) * 0.01
    pred_f = np.random.randn(n_samples, n_atoms, 3)
    true_f = pred_f + np.random.randn(n_samples, n_atoms, 3) * 0.01
    
    metrics = compute_metrics(pred_e, true_e, pred_f, true_f)
    print(f"Energy MAE: {metrics['energy_mae']:.6f}")
    print(f"Force MAE: {metrics['force_mae']:.6f}")
