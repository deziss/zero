import numpy as np


def fcc_lattice(n_cells, box_length):
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
    a = box_length / n_cells
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    pos = (np.array([ix, iy, iz]) + b) * a
                    positions.append(pos)
    return np.array(positions)


def assign_velocities(n_particles, temperature, mass=1.0):
    sigma = np.sqrt(temperature / mass)
    velocities = np.random.randn(n_particles, 3) * sigma
    v_com = np.mean(velocities, axis=0)
    velocities -= v_com
    v2 = np.sum(velocities * velocities)
    scale = np.sqrt(3.0 * (n_particles - 1) * temperature / (mass * v2))
    velocities *= scale
    return velocities
