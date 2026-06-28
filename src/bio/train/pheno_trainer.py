"""src/bio/train/pheno_trainer.py — Training pipeline for neural operators

Trains FNO on biological simulation data.
Handles data preparation, training loop, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class PhenomenologicalTrainer:
    """
    Training loop for FNO-based biological simulation models.
    
    Tasks:
    - Cell population dynamics prediction
    - Signaling field evolution
    - Differentiation pattern prediction
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints/bio/',
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.MSELoss()
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.5
        )
        
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def train(
        self,
        dataloader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train the FNO model.
        
        Args:
            dataloader: training DataLoader
            val_loader: validation DataLoader
            epochs: number of training epochs
            
        Returns:
            history dict with training/validation losses
        """
        history = {'train/loss': [], 'val/loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= max(n_batches, 1)
            history['train/loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                history['val/loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                if epoch % 20 == 0 or epoch == epochs - 1:
                    print(f'  Epoch {epoch+1:3d}: '
                          f'train={train_loss:.4f}, val={val_loss:.4f}')
                
                # Save best
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, history)
            else:
                if epoch % 20 == 0 or epoch == epochs - 1:
                    print(f'  Epoch {epoch+1:3d}: train={train_loss:.4f}')
            
            if train_loss < 1e-6:
                print(f'  Converged at epoch {epoch+1}: {train_loss:.2e}')
                break
        
        print(f'  Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}')
        
        return history
    
    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """Evaluate on validation dataloader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _save_checkpoint(self, epoch: int, history: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'loss': self.best_loss,
            'history': history,
        }
        path = self.checkpoint_dir / f'checkpoint_{epoch}.pt'
        torch.save(checkpoint, path)
    
    def save_config(self, config_path: str):
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
# trainer = PhenomenologicalTrainer(fno_model, lr=1e-3)
# history = trainer.train(train_loader, val_loader=val_loader, epochs=100)
# trainer.save_config('configs/bio/training/last_config.json')
