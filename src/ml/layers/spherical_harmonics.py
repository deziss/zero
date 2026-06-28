"""src/ml/layers/spherical_harmonics.py

Real spherical harmonics for equivariant message passing.

We use the convention where real spherical harmonics Y_l^m are real-valued.
For degree l, there are 2l+1 harmonics (m = -l, ..., l).

These are used to expand the angular dependencies of atomic environments.

Note: We implement a simplified version that doesn't require external shtns library.
"""

import torch
import torch.nn as nn


def real_spherical_harmonics(pos, degrees=4):
    """
    Compute real spherical harmonics from position vectors.

    Args:
        pos: (n_points, 3) positions from origin
        degrees: maximum degree of harmonics

    Returns:
        yharm: (n_points, n_harmonics) where n_harmonics = (degrees+1)^2
    """
    r = torch.norm(pos, dim=-1, keepdim=True)  # (n, 1)
    r_safe = torch.where(r > 1e-10, r, torch.ones_like(r))

    # Spherical coordinates
    theta = torch.acos(torch.clamp(pos[..., 2:3] / r_safe, -1.0, 1.0))  # polar angle
    phi = torch.atan2(pos[..., 1:2], pos[..., 0:1])  # azimuthal angle

    results = []
    n_harmonics = 0

    for l in range(degrees + 1):
        for m in range(-l, l + 1):
            if m == 0:
                # m=0: Legendre polynomial P_l(cos(θ))
                val = _legendre(l, torch.cos(theta))
            elif m > 0:
                # m>0: cos(m*phi) * P_l^m(cos(θ))
                pm = _associated_legendre(l, m, torch.cos(theta))
                val = torch.sqrt(2.0) * torch.cos(m * phi) * pm
            else:
                # m<0: sin(|m|*phi) * P_l^|m|(cos(θ))
                pm = _associated_legendre(l, -m, torch.cos(theta))
                val = torch.sqrt(2.0) * torch.sin(-m * phi) * pm

            results.append(val)
            n_harmonics += 1

    yharm = torch.cat([r.expand_as(v) * v for r, v in zip([r_safe.expand(n_harmonics, -1) for _ in range(n_harmonics)], results)], dim=-1)
    yharm = torch.stack(results, dim=-1)  # (n, n_harmonics)

    return yharm


def _legendre(l, x):
    """Compute Legendre polynomial P_l(x)."""
    if l == 0:
        return torch.ones_like(x)
    elif l == 1:
        return x
    else:
        # Recursion: (n+1)*P_{n+1} = (2n+1)*x*P_n - n*P_{n-1}
        P_prev2 = torch.ones_like(x)  # P_0
        P_prev1 = x  # P_1
        for n in range(1, l):
            P_curr = ((2 * n + 1) * x * P_prev1 - n * P_prev2) / (n + 1)
            P_prev2 = P_prev1
            P_prev1 = P_curr
        return P_curr


def _associated_legendre(l, m, x):
    """Compute associated Legendre polynomial P_l^m(x)."""
    if m < 0:
        m = -m
    if l < m:
        return torch.zeros_like(x)

    # First compute (P_l^m where m=0)
    P = _legendre(l, x)

    for mm in range(1, m + 1):
        P = -(2 * l - 1) * x * P / (2 * l - 1)  # derivative factor
        if l != mm:
            P = -(2 * l - 1) * torch.sqrt(1 - x * x) * _legendre(l - 1, x) / (2 * l - 1)

    return P


def sph_harm(m, l, theta, phi):
    """Compute a single spherical harmonic Y_l^m."""
    return real_spherical_harmonics_theta_phi(torch.stack([theta, phi], dim=-1), l - m, l)


def real_spherical_harmonics_theta_phi(thetaphi, max_degree):
    """Compute real spherical harmonics from (theta, phi) pairs."""
    results = []
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            if m == 0:
                val = _legendre_at_degree(l, torch.cos(thetaphi[..., 0]))
            elif m > 0:
                val = torch.cos(m * thetaphi[..., 1]) * _assoc_legendre_at_degree(l, m, torch.cos(thetaphi[..., 0]))
            else:
                val = torch.sin(-m * thetaphi[..., 1]) * _assoc_legendre_at_degree(l, -m, torch.cos(thetaphi[..., 0]))
            results.append(val.unsqueeze(-1))
    return torch.cat(results, dim=-1)


def _legendre_at_degree(l, x):
    if l == 0:
        return torch.ones_like(x)
    elif l == 1:
        return x
    P0, P1 = torch.ones_like(x), x
    for n in range(1, l):
        P2 = ((2*n+1)*x*P1 - n*P0)/(n+1)
        P0, P1 = P1, P2
    return P1


def _assoc_legendre_at_degree(l, m, x):
    if m > l:
        return torch.zeros_like(x)
    P = _legendre_at_degree(l, x)
    for mm in range(1, m + 1):
        P = -(2 * l - 1) * x * P / (2 * l - 1)
    return P


if __name__ == "__main__":
    # Test
    pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    yharm = real_spherical_harmonics(pos, degrees=2)
    print(f"Spherical harmonics shape: {yharm.shape}")
    print(f"Y(1,0,0): {yharm[0].tolist()}")
    print(f"Y(0,1,0): {yharm[1].tolist()}")
