"""src/qm/train/surrogate_trainer.py — Training pipeline for DFT surrogates

Trains neural surrogates on DFT-generated training data.
Handles data loading, loss computation, checkpointing, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class SurrogateTrainer:
    """
    Training loop for DFT surrogate models.
    
    Handles:
    - Data loading from SurrogateDataset
    - Multi-task loss (energy + force)
    - Checkpoint save/load
    - Evaluation on test set
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints/qm/',
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-task loss weights
        w_energy = 1.0
        w_force = 10.0  # forces usually have larger scale
        
        self.criterion_energy = nn.L1Loss()
        self.criterion_force = nn.MSELoss()
        
        # Optimizer
        if hasattr(model, 'parameters'):
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                (p for p in model.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def train(
        self,
        dataset,
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train surrogate model.
        
        Args:
            dataset: SurrogateDataset with molecular data
            epochs: number of training epochs
            batch_size: mini-batch size
            val_split: fraction of data for validation
            
        Returns:
            history dict with 'train/energy', 'train/force', etc.
        """
        n_total = len(dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        # Train/val split
        train_ds = torch.utils.data.Subset(
            dataset, list(range(n_train))
        )
        val_ds = torch.utils.data.Subset(
            dataset, list(range(n_train, n_total))
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )
        
        history = {
            'train/energy': [],
            'train/force': [],
            'val/energy': [],
            'val/force': [],
        }
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_e_loss = 0.0
            train_f_loss = 0.0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                # Forward
                atoms = batch[0].to(self.device)
                pos = batch[1].to(self.device)
                true_e = batch[2].to(self.device)
                true_f = batch[3].to(self.device)
                
                pred_e, pred_f = self.model(atoms, pos)
                
                # Loss
                e_loss = self.criterion_energy(pred_e, true_e)
                f_loss = self.criterion_force(pred_f, true_f)
                total_loss = w_energy * e_loss + w_force * f_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                train_e_loss += e_loss.item() * len(batch)
                train_f_loss += f_loss.item() * len(batch)
            
            train_e_loss /= n_train
            train_f_loss /= n_train
            
            # Validation
            self.model.eval()
            val_e_loss = 0.0
            val_f_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    atoms = batch[0].to(self.device)
                    pos = batch[1].to(self.device)
                    true_e = batch[2].to(self.device)
                    true_f = batch[3].to(self.device)
                    
                    pred_e, pred_f = self.model(atoms, pos)
                    
                    val_e_loss += self.criterion_energy(pred_e, true_e)
                    val_f_loss += self.criterion_force(pred_f, true_f)
                    val_count += len(batch)
            
            val_e_loss /= max(val_count, 1)
            val_f_loss /= max(val_count, 1)
            
            history['train/energy'].append(train_e_loss)
            history['train/force'].append(train_f_loss)
            history['val/energy'].append(val_e_loss)
            history['val/force'].append(val_f_loss)
            
            self.scheduler.step(val_e_loss)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'  Epoch {epoch+1:3d}: '
                      f'E={train_e_loss:.4f}/{val_e_loss:.4f} '
                      f'F={train_f_loss:.6f}/{val_f_loss:.6f}')
            
            # Save best checkpoint
            if val_e_loss < self.best_loss:
                self.best_loss = val_e_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, history)
        
        print(f'  Best epoch: {self.best_epoch}, '
              f'Best val E: {self.best_loss:.4f}')
        
        return history
    
    def _save_checkpoint(self, epoch: int, history: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'history': history,
        }
        path = self.checkpoint_dir / f'checkpoint_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f'    Saved checkpoint: {path}')
    
    def evaluate(self, test_ds, batch_size=32) -> Dict[str, float]:
        """Evaluate model on test dataset."""
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size
        )
        
        self.model.eval()
        total_e_mae = 0.0
        total_f_mae = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                atoms = batch[0].to(self.device)
                pos = batch[1].to(self.device)
                true_e = batch[2].to(self.device)
                true_f = batch[3].to(self.device)
                
                pred_e, pred_f = self.model(atoms, pos)
                
                e_mae = torch.abs(pred_e - true_e).mean()
                f_mae = torch.abs(pred_f - true_f).mean()
                
                total_e_mae += e_mae * len(batch)
                total_f_mae += f_mae * len(batch)
                count += len(batch)
        
        return {
            'energy_mae_per_atom': (total_e_mae / max(count, 1)).item(),
            'force_mae_per_atom': (total_f_mae / max(count, 1)).item(),
        }
    
    def save_config(self, config_path: str):
        """Save training configuration."""
        config = {
            'model_type': type(self.model).__name__,
            'lr': self.optimizer.param_groups[0]['lr'],
            'device': self.device,
            'checkpoint_dir': str(self.checkpoint_dir),
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


# Example usage:
# trainer = SurrogateTrainer(model, lr=1e-3)
# history = trainer.train(dataset, epochs=100, batch_size=32)
# metrics = trainer.evaluate(test_dataset)
# trainer.save_config('configs/training/last_config.json')
