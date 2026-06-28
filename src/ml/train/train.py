"""src/ml/train/train.py

Training loop for molecular force prediction.

Loss: L = λ_E * MSE(E_pred, E_true) + λ_F * MSE(F_pred, F_true)
"""

import torch
import torch.nn as nn
import numpy as np


def compute_metrics(pred_energy, true_energy, pred_force, true_force):
    """Compute force and energy metrics."""
    n_atoms = true_force.shape[0]
    
    # Energy metrics
    energy_mae = np.mean(np.abs(pred_energy - true_energy))
    energy_mse = np.mean((pred_energy - true_energy) ** 2)
    energy_rmse = np.sqrt(energy_mse)
    
    # Force metrics per atom
    force_diff = pred_force - true_force
    
    # Per-atom force error
    force_per_atom = torch.norm(force_diff, dim=-1)  # (n_atoms,)
    force_mae = np.mean(force_per_atom.numpy())
    force_rmse = np.sqrt(np.mean(force_per_atom.numpy() ** 2))
    
    return {
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "force_mae": force_mae,
        "force_rmse": force_rmse,
    }


def train_loop(model, train_loader, val_loader, device, 
              lr=0.01, weight_decay=1e-4, epochs=100,
              log_every=10, clip_value=100.0):
    """
    Training loop for molecular force prediction.
    
    Args:
        model: PyTorch model (MACE or equivariant network)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device
        lr: learning rate
        weight_decay: L2 regularization
        epochs: number of epochs
        log_every: log every N epochs
        clip_value: gradient clipping threshold
        
    Returns:
        history: dict with training history
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = nn.MSELoss()
    
    history = {
        "train_loss": [], 
        "val_energy_mae": [], 
        "val_force_mae": []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            atomic_nums = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            targets_energy = batch["energy"].to(device)
            targets_force = batch["forces"].to(device)
            
            # Forward pass
            pred_energy, pred_force = model(atomic_nums, positions)
            
            # Loss: energy + force
            loss_energy = loss_fn(pred_energy, targets_energy)
            loss_force = loss_fn(pred_force, targets_force)
            loss = loss_energy + loss_force
            # Note: real training uses weighted loss for force (usually ~10x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        else:
            scheduler.step()
            break
        
        # Validation
        val_history = validation_loop(model, val_loader, device)
        # (val_history not available in this version)
        
        # Logging
        epoch_loss = epoch_loss / n_batches
        history["train_loss"].append(epoch_loss)
        
        if log_every > 0 and epoch % log_every == 0:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {epoch_loss:.6f} | "
                  f"Val Energy MAE: {val_history['energy_mae']:.6f} | "
                  f"Val Force MAE: {val_history['force_mae']:.6f}")
        else:
            if epoch < 5 or epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Loss: {epoch_loss:.6f}")
    
    return history


def validation_loop(model, val_loader, device):
    """Run validation and return metrics."""
    model.eval()
    all_energy = []
    all_force = []
    all_true_energy = []
    all_true_force = []
    
    with torch.no_grad():
        for batch in val_loader:
            atomic_nums = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            true_energy = batch["energy"].to(device)
            true_force = batch["forces"].to(device)
            
            pred_energy, pred_force = model(atomic_nums, positions)
            
            all_true_energy.append(true_energy.cpu())
            all_true_force.append(true_force.cpu())
            all_energy.append(pred_energy.cpu())
            all_force.append(pred_force.cpu())
    
    all_true_energy = torch.cat(all_true_energy) if all_true_energy else []
    all_true_force = torch.cat(all_true_force) if all_true_force else []
    all_energy = torch.cat(all_energy) if all_energy else []
    all_force = torch.cat(all_force) if all_force else []
    
    metrics = {
        "energy_mae": np.mean(np.abs(all_energy - all_true_energy)) if len(all_energy) > 0 else float("inf"),
        "force_mae": np.mean(torch.norm(all_force - all_true_force, dim=-1).numpy()) if len(all_force) > 0 else float("inf"),
    }
    
    return metrics


if __name__ == "__main__":
    print("Training loop for molecular force prediction.")
    print("Usage:")
    print("  from src.ml.train import train_loop")
    print("  history = train_loop(model, train_loader, val_loader, device, ...)")
