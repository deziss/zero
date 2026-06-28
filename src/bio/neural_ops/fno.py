"""src/bio/neural_ops/fno.py — Fourier Neural Operator for field-level simulation

FNO learns a continuous operator mapping molecular fields → dynamics.
Used for cellular-scale simulation and pattern prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class SpectralConv2d(nn.Module):
    """2D spectral convolution for FNO."""
    
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = (1.0 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, 2)
        )
    
    def forward(self, x):
        batch, h, w = x.shape[0], x.shape[-2], x.shape[-1]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batch, self.out_channels, h, w // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        for i in range(self.modes):
            for j in range(self.modes):
                if i < x_ft.shape[-2] // 2 + 1 and j < x_ft.shape[-1] // 2 + 1:
                    out_ft[:, :, i, j] = torch.einsum(
                        'ki,ij->kj',
                        x_ft[:, :, i, j].reshape(-1, x_ft.shape[1]),
                        self.weights1[:, :, i, j].reshape(self.out_channels, self.in_channels)
                    )
        
        x_out = torch.fft.irfft2(out_ft, s=(h, w))
        return x_out


class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator for continuous field operators.
    
    Maps molecular field → field dynamics (e.g., concentration, density).
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_dim: int = 64,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        
        # Lift to latent dimension
        self.fc0 = nn.Linear(in_channels, latent_dim)
        
        # Spectral layers
        self.spec = nn.ModuleList([
            SpectralConv2d(latent_dim, latent_dim, modes)
            for _ in range(n_layers)
        ])
        
        # MLP layers
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, width),
                nn.GELU(),
                nn.Linear(width, latent_dim),
            ) for _ in range(n_layers)
        ])
        
        # Project to output
        self.fc_out = nn.Linear(latent_dim, out_channels)
        
        # Boundary correction
        self.boundary_conv1 = nn.Conv2d(in_channels, latent_dim, 1)
        self.boundary_conv2 = nn.Conv2d(latent_dim, out_channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, h, w, in_channels) input field
            
        Returns:
            output: (batch, h, w, out_channels) predicted field
        """
        batch, h, w = x.shape[0], x.shape[-2], x.shape[-1]
        
        # Lift
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, latent, h, w)
        
        # Boundary correction
        x1 = self.boundary_conv1(x)
        x2 = self.boundary_conv2(x1)
        x = x + x2
        
        # FNO layers
        for i in range(self.n_layers):
            x1 = self.spec[i](x)
            x2 = self.mlp[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x1 + x2
        
        # Decode
        x = x.permute(0, 2, 3, 1)
        x = self.fc_out(x)
        
        return x


class SpectralConv1d(SpectralConv2d):
    """1D spectral convolution for 1D field operators."""
    
    def __init__(self, in_channels, out_channels, modes):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1.0 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, 2)
        )
    
    def forward(self, x):
        batch, h = x.shape[0], x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=1)
        
        out_ft = torch.zeros(batch, self.out_channels, x.shape[-1] // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        for i in range(self.modes):
            if i < x_ft.shape[1]:
                out_ft[:, :, i] = torch.einsum(
                    'i,ij->j',
                    x_ft[:, i],
                    self.weights1[:, :, i].reshape(self.out_channels, self.in_channels)
                )
        
        x_out = torch.fft.irfft(out_ft, n=x.shape[1], dim=1)
        return x_out


if __name__ == "__main__":
    # Example
    fno = FourierNeuralOperator(in_channels=3, out_channels=3, latent_dim=32, modes=8)
    x = torch.randn(1, 64, 64, 3)
    y = fno(x)
    print(f"FNO input: {x.shape}")
    print(f"FNO output: {y.shape}")
