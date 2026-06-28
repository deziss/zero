"""src/ml/layers/radial_basis.py

Bessel and Airy radial basis functions for equivariant message passing.

Bessel:
    g_n(r) = sqrt(2/(r_cut^3)) * n*pi/(r*r_cut) * sin(n*pi*(r - r_min)/r_cut)

Airy (smooth extension):
    g_n(r) = sqrt(2/(r_cut)) * sin(n*pi*(r - r_min)/r_cut) * envelope(r)

Envelope ensures smooth cutoff at r_cut.
"""

import torch
import torch.nn as nn


def bessel_basis(distances, n_rbf, r_min=0.0, r_cut=5.0):
    """
    Compute Bessel radial basis functions for interatomic distances.

    Args:
        distances: (n_edges,) interatomic distances
        n_rbf: number of radial basis functions
        r_min: inner cutoff (usually 0)
        r_cut: outer cutoff (usually 5.0 Angstroms)

    Returns:
        basis: (n_edges, n_rbf) radial basis expansions
    """
    if distances.max() > r_cut:
        raise ValueError(f"Distance {distances.max():.3f} exceeds cutoff {r_cut}")

    r_min = torch.tensor(r_min, dtype=distances.dtype, device=distances.device)
    r_cut = torch.tensor(r_cut, dtype=distances.dtype, device=distances.device)

    n = torch.arange(1, n_rbf + 1, dtype=distances.dtype, device=distances.device)

    # Bessel function: j_n(r) = sqrt(2/r_cut) * sin(n*pi*(r-r_min)/r_cut) / (r - r_min)
    numerator = n * torch.pi * (distances - r_min) / r_cut
    denominator = (r_cut * (distances - r_min))

    basis = torch.sqrt(2.0 / r_cut) * torch.sin(numerator) / torch.where(
        denominator.abs() > 1e-10, denominator, torch.ones_like(denominator)
    )

    return basis.unsqueeze(-1)  # (n_edges, n_rbf, 1)


def airy_basis(distances, n_rbf, r_min=0.0, r_cut=5.0):
    """
    Airy radial basis with smooth envelope function.
    Enforce zero at r_cut for smooth cutoff.

    g_n(r) = sqrt(2/(r_cut - r_min)) * sin(n*pi*(r-r_min)/(r_cut-r_min)) * envelope(r)
    envelope(r) = (1 - (r-r_min)^p / (r_cut-r_min)^p) for p >= 3
    """
    envelope_power = 5
    env = (1.0 - ((distances - r_min) / (r_cut - r_min)) ** envelope_power)
    env = torch.clamp(env, min=0.0)

    normalization = torch.sqrt(2.0 / (r_cut - r_min))

    n = torch.arange(1, n_rbf + 1, dtype=distances.dtype, device=distances.device)
    phase = n * torch.pi * (distances - r_min) / (r_cut - r_min)

    basis = normalization * torch.sin(phase) * env
    return basis.unsqueeze(-1)  # (n_edges, n_rbf, 1)


class RadialBasisNetwork(nn.Module):
    """Learnable radial basis function network with a learned cutoff."""

    def __init__(self, n_rbf=20, r_min=0.0, r_cut=10.0, learned_cutoff=False):
        super().__init__()
        self.n_rbf = n_rbf
        self.r_min = torch.nn.Parameter(torch.tensor(r_min), requires_grad=False)
        self.r_cut = torch.nn.Parameter(torch.tensor(r_cut), requires_grad=learned_cutoff)
        self.learned_cutoff = learned_cutoff

    def forward(self, distances):
        return bessel_basis(distances, self.n_rbf, self.r_min, self.r_cut)


if __name__ == "__main__":
    # Quick test
    dists = torch.linspace(0.1, 4.9, 100)
    basis = bessel_basis(dists, n_rbf=10, r_min=0.0, r_cut=5.0)
    print(f"Bessel basis shape: {basis.shape}")
    print(f"First 5 values of basis[0]: {basis[0, :5, 0].tolist()}")
