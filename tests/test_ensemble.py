import numpy as np
from src.core.potentials import lj_force_pbc
from src.core.integrators import velocity_verlet
from src.core.ensemble import andersen_thermostat, langevin_thermostat
from src.core.observables import compute_kinetic_energy, compute_temperature


def _fcc_lattice(n_cells, box):
    a = box / n_cells
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    pos = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    pos.append(b + np.array([i, j, k]) * a)
    return np.array(pos)


def _sc_lattice(n_cells, box):
    a = box / n_cells
    pos = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                pos.append(np.array([i, j, k]) * a)
    return np.array(pos)


def test_andersen_thermalises():
    np.random.seed(42)
    n = 108
    box = 10.0
    dt = 0.01
    pos = _fcc_lattice(3, box)
    vel = np.random.randn(n, 3) * 0.5
    forces = lj_force_pbc(pos, box)
    target_T = 2.0
    nu = 1.0
    n_steps = 5000
    for _ in range(n_steps):
        pos, vel, forces = velocity_verlet(pos, vel, forces, box, dt)
        vel = andersen_thermostat(vel, target_T, nu, dt)
    ke = compute_kinetic_energy(vel)
    T = compute_temperature(ke, n)
    assert np.isclose(T, target_T, rtol=1e-1), f"Andersen should thermalise to {target_T}, got {T}"


def test_andersen_maxwellian():
    np.random.seed(42)
    n = 216
    box = 15.0
    dt = 0.01
    pos = _sc_lattice(6, box)
    vel = np.zeros((n, 3))
    forces = lj_force_pbc(pos, box)
    target_T = 3.0
    nu = 10.0
    n_steps = 1000
    for _ in range(n_steps):
        pos, vel, forces = velocity_verlet(pos, vel, forces, box, dt)
        vel = andersen_thermostat(vel, target_T, nu, dt)
    ke = compute_kinetic_energy(vel)
    T = compute_temperature(ke, n)
    assert np.isclose(T, target_T, rtol=1e-1), f"Andersen T={T}, expected {target_T}"


def test_langevin_thermalises():
    np.random.seed(42)
    n = 108
    box = 10.0
    dt = 0.01
    pos = _fcc_lattice(3, box)
    vel = np.random.randn(n, 3) * 0.5
    target_T = 2.0
    gamma = 1.0
    n_steps = 5000
    for i in range(n_steps):
        forces = lj_force_pbc(pos, box)
        pos, vel = langevin_thermostat(pos, vel, forces, gamma, target_T, dt)
    ke = compute_kinetic_energy(vel)
    T = compute_temperature(ke, n)
    assert np.isclose(T, target_T, rtol=1e-1), f"Langevin should thermalise to {target_T}, got {T}"


def test_langevin_different_temperature():
    np.random.seed(42)
    n = 108
    box = 10.0
    dt = 0.01
    target_temp = 5.0
    gamma = 1.0
    pos = _fcc_lattice(3, box)
    vel = np.random.randn(n, 3) * 0.5
    target_T = 5.0
    gamma = 1.0
    n_steps = 5000
    for i in range(n_steps):
        forces = lj_force_pbc(pos, box)
        pos, vel = langevin_thermostat(pos, vel, forces, gamma, target_T, dt)
    ke = compute_kinetic_energy(vel)
    T = compute_temperature(ke, n)
    assert np.isclose(T, target_T, rtol=1e-1), f"Langevin T={T}, expected {target_T}"
