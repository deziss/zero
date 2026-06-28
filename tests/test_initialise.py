import numpy as np
from src.core.initialise import fcc_lattice, assign_velocities


def test_fcc_lattice_counts():
    pos = fcc_lattice(2, 4.0)
    assert pos.shape == (32, 3), f"2x2x2 FCC should give 32 atoms, got {pos.shape}"


def test_fcc_lattice_density():
    n_cells = 2
    box = 4.0
    pos = fcc_lattice(n_cells, box)
    volume = box ** 3
    density = pos.shape[0] / volume
    assert np.isclose(density, 0.5, rtol=1e-10), f"FCC density should be 0.5, got {density}"


def test_fcc_lattice_within_box():
    n_cells = 3
    box = 6.0
    pos = fcc_lattice(n_cells, box)
    assert np.all(pos >= 0.0), "Positions should be non-negative"
    assert np.all(pos <= box + 1e-14), f"Positions should be within [0, {box}]"


def test_fcc_lattice_different_sizes():
    for n in [1, 2, 3]:
        box = 2.0 * n
        pos = fcc_lattice(n, box)
        expected = 4 * n ** 3
        assert pos.shape == (expected, 3), f"{n}x{n}x{n} FCC: expected {expected}, got {pos.shape[0]}"


def test_assign_velocities_zero_momentum():
    np.random.seed(42)
    n = 100
    vel = assign_velocities(n, temperature=1.0)
    com_vel = np.mean(vel, axis=0)
    assert np.allclose(com_vel, 0.0, atol=1e-14), f"COM velocity should be zero, got {com_vel}"


def test_assign_velocities_temperature():
    np.random.seed(42)
    n = 500
    vel = assign_velocities(n, temperature=2.0)
    ke = 0.5 * np.sum(vel * vel)
    dof = 3 * n - 3
    T = 2.0 * ke / dof
    assert np.isclose(T, 2.0, rtol=1e-2), f"Temperature should be 2.0, got {T}"
