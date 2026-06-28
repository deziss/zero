import numpy as np
from src.core.potentials import lj_force_pbc, lj_potential_pbc
from src.core.integrators import velocity_verlet, integrate_nve


def _pe(pos, box):
    e, _ = lj_potential_pbc(pos, box)
    return e


def test_velocity_verlet_symplectic():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    velocities = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    box_length = 10.0
    dt = 0.001
    forces = lj_force_pbc(positions, box_length)
    p0, v0, f0 = velocity_verlet(positions, velocities, forces, box_length, dt)
    e0 = 0.5 * np.sum(v0 * v0) + _pe(p0, box_length)
    p1, v1, f1 = velocity_verlet(p0, v0, f0, box_length, dt)
    e1 = 0.5 * np.sum(v1 * v1) + _pe(p1, box_length)
    assert np.isclose(e0, e1, rtol=1e-6), f"Energy drift too large: {e0} -> {e1}"


def test_integrate_nve_energy_conservation():
    # Best stability test: two particles at the LJ potential minimum
    # r_min = 2^(1/6) ≈ 1.122, F=0 at minimum -> linear restoring force
    # With tiny perturbation → small oscillations ≈ harmonic -> VE is integrable
    positions = np.array([[0.0, 0.0, 0.0],
                          [2.0**(1.0/6.0), 0.0, 0.0]], dtype=np.float64)
    velocities = np.array([[0.01, 0.0, 0.0],
                           [-0.01, 0.0, 0.0]], dtype=np.float64)
    box_length = 10.0
    dt = 0.002
    forces = lj_force_pbc(positions, box_length)
    e0 = 0.5 * np.sum(velocities * velocities) + lj_potential_pbc(positions, box_length)[0]
    
    # Run 5000 steps (10 time units)
    n_steps = 5000
    for step in range(n_steps):
        positions, velocities, forces = velocity_verlet(
            positions, velocities, forces, box_length, dt
        )
    e1 = 0.5 * np.sum(velocities * velocities) + lj_potential_pbc(positions, box_length)[0]
    drift = abs(e1 - e0)
    assert drift < abs(e0) * 1e-6, f"Symplectic NVE failed: |dE|={drift:.2e} rel={drift/abs(e0):.2e}"


def test_integrate_nve_returns_correct_shape():
    n = 32
    box_length = 6.0
    positions = np.random.rand(n, 3).astype(np.float64) * box_length
    velocities = np.random.randn(n, 3).astype(np.float64)
    traj_pos, traj_vel, times = integrate_nve(positions, velocities, box_length, dt=0.01, n_steps=50, stride=10)
    n_frames = 50 // 10 + 1
    assert traj_pos.shape == (n_frames, n, 3), f"traj_pos shape {traj_pos.shape}"
    assert traj_vel.shape == (n_frames, n, 3), f"traj_vel shape {traj_vel.shape}"
    assert len(times) == n_frames, f"times length {len(times)}"
