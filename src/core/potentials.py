import numpy as np

SIGMA = 1.0
EPSILON = 1.0
RCUT = 2.5 * SIGMA


def lj_potential(r2):
    r6 = (SIGMA * SIGMA / r2) ** 3
    r12 = r6 * r6
    return 4.0 * EPSILON * (r12 - r6)


def lj_force(r2):
    r6 = (SIGMA * SIGMA / r2) ** 3
    r12 = r6 * r6
    return 48.0 * EPSILON * (r12 - 0.5 * r6) / r2


def lj_potential_pbc(positions, box_length):
    n = positions.shape[0]
    total_energy = 0.0
    forces = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[i] - positions[j]
            dr -= box_length * np.round(dr / box_length)
            r2 = np.dot(dr, dr)
            if r2 < RCUT * RCUT:
                total_energy += lj_potential(r2)
                f = lj_force(r2)
                forces[i] += f * dr
                forces[j] -= f * dr
    return total_energy, forces


def lj_force_pbc(positions, box_length):
    n = positions.shape[0]
    dr = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dr -= box_length * np.round(dr / box_length)
    r2 = np.sum(dr * dr, axis=-1)
    mask = (r2 < RCUT * RCUT) & (r2 > 0)
    f = np.zeros_like(r2)
    s2 = SIGMA * SIGMA / r2[mask]
    s6 = s2 * s2 * s2
    f[mask] = 48.0 * EPSILON * (s6 * s6 - 0.5 * s6) / r2[mask]
    return np.sum(f[:, :, np.newaxis] * dr, axis=1)
