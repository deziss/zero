import numpy as np
from .potentials import lj_potential_pbc, RCUT, EPSILON, SIGMA


def compute_kinetic_energy(velocities, mass=1.0):
    return 0.5 * mass * np.sum(velocities * velocities)


def compute_potential_energy(positions, box_length):
    pe, _ = lj_potential_pbc(positions, box_length)
    return pe


def compute_temperature(kinetic_energy, n_particles):
    dof = 3 * n_particles - 3
    return 2.0 * kinetic_energy / dof


def compute_pressure(positions, forces, box_length, temperature, n_particles):
    volume = box_length ** 3
    ideal = n_particles * temperature / volume
    virial = np.sum(positions * forces) / 3.0
    return ideal + virial / volume


def compute_energy(positions, velocities, box_length, mass=1.0):
    ke = compute_kinetic_energy(velocities, mass)
    pe = compute_potential_energy(positions, box_length)
    return ke, pe, ke + pe


def compute_rdf(positions, box_length, n_bins=200, rmax=None):
    if rmax is None:
        rmax = 0.5 * box_length
    n = positions.shape[0]
    dr = rmax / n_bins
    hist = np.zeros(n_bins)
    for i in range(n):
        for j in range(i + 1, n):
            dr_vec = positions[i] - positions[j]
            dr_vec -= box_length * np.round(dr_vec / box_length)
            r = np.sqrt(np.dot(dr_vec, dr_vec))
            if r < rmax:
                idx = int(r / dr)
                if idx < n_bins:
                    hist[idx] += 2
    rho = n / (box_length ** 3)
    vol_shell = 4.0 * np.pi * rho / 3.0
    r_edges = np.linspace(0, rmax, n_bins + 1)
    for i in range(n_bins):
        r_low = r_edges[i]
        r_high = r_edges[i + 1]
        volume = (4.0 / 3.0) * np.pi * (r_high ** 3 - r_low ** 3)
        norm = volume * rho * n
        if norm > 0:
            hist[i] /= norm
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    return r_centers, hist


def compute_msd(positions_ref, positions_traj):
    dr = positions_traj - positions_ref[np.newaxis, :, :]
    dr -= np.round(dr / 1e8) * 1e8
    return np.mean(np.sum(dr * dr, axis=-1), axis=-1)
