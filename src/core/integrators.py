import numpy as np
from .potentials import lj_force_pbc


def velocity_verlet(positions, velocities, forces, box_length, dt, mass=1.0):
    positions_new = positions + velocities * dt + 0.5 * forces * dt * dt / mass
    forces_new = lj_force_pbc(positions_new, box_length)
    velocities_new = velocities + 0.5 * (forces + forces_new) * dt / mass
    return positions_new, velocities_new, forces_new


def integrate_nve(positions, velocities, box_length, dt, n_steps, mass=1.0, stride=100):
    forces = lj_force_pbc(positions, box_length)
    n_frames = n_steps // stride
    traj_positions = np.empty((n_frames + 1, *positions.shape))
    traj_velocities = np.empty((n_frames + 1, *velocities.shape))
    times = np.empty(n_frames + 1)

    traj_positions[0] = positions
    traj_velocities[0] = velocities
    times[0] = 0.0

    frame = 1
    for step in range(1, n_steps + 1):
        positions, velocities, forces = velocity_verlet(
            positions, velocities, forces, box_length, dt, mass
        )
        if step % stride == 0:
            traj_positions[frame] = positions
            traj_velocities[frame] = velocities
            times[frame] = step * dt
            frame += 1
    return traj_positions, traj_velocities, times
