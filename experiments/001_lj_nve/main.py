import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.initialise import fcc_lattice, assign_velocities
from src.core.integrators import integrate_nve
from src.core.potentials import lj_potential_pbc
from src.core.observables import compute_energy, compute_temperature, compute_pressure, compute_rdf
from src.core.io import write_xyz

N_PARTICLES = 864
DENSITY = 0.8
TEMPERATURE = 1.0
DT = 0.004
N_STEPS = 10000
STRIDE = 100

box_length = (N_PARTICLES / DENSITY) ** (1.0 / 3.0)
n_cells = int(round((N_PARTICLES / 4) ** (1.0 / 3.0)))
positions = fcc_lattice(n_cells, box_length)
positions = positions[:N_PARTICLES]
velocities = assign_velocities(N_PARTICLES, TEMPERATURE)

print(f"Box length: {box_length:.4f}")
print(f"FCC cells: {n_cells}")
print(f"Actual particles: {positions.shape[0]}")
print(f"Running {N_STEPS} steps with dt={DT}...")

traj_pos, traj_vel, times = integrate_nve(positions, velocities, box_length, DT, N_STEPS, stride=STRIDE)

ke, pe, total = compute_energy(traj_pos[-1], traj_vel[-1], box_length)
T = compute_temperature(ke, N_PARTICLES)
print(f"\nFinal state:")
print(f"  KE={ke:.4f} PE={pe:.4f} Total={total:.4f} T*={T:.4f}")

energies = np.array([compute_energy(traj_pos[i], traj_vel[i], box_length) for i in range(len(times))])
ke_arr, pe_arr, total_arr = energies[:, 0], energies[:, 1], energies[:, 2]
temp_arr = np.array([compute_temperature(ke_arr[i], N_PARTICLES) for i in range(len(times))])

ke_avg = np.mean(ke_arr[1:])
ke_std = np.std(ke_arr[1:])
total_avg = np.mean(total_arr[1:])
total_std = np.std(total_arr[1:])
T_avg = np.mean(temp_arr[1:])
T_std = np.std(temp_arr[1:])
print(f"\nEquilibration (steps {STRIDE}-{N_STEPS}):")
print(f"  KE = {ke_avg:.4f} +/- {ke_std:.4f}")
print(f"  Total Energy = {total_avg:.4f} +/- {total_std:.4f}")
print(f"  T* = {T_avg:.4f} +/- {T_std:.4f}")

r_centers, rdf = compute_rdf(traj_pos[-1], box_length, n_bins=200)

np.savez("output.npz", times=times, ke=ke_arr, pe=pe_arr, total=total_arr,
         temp=temp_arr, r_centers=r_centers, rdf=rdf, traj_pos=traj_pos)
write_xyz("trajectory.xyz", traj_pos[-1], box_length, "Final frame")
print("\nSaved: output.npz, trajectory.xyz")
