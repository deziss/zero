import numpy as np
from src.core.potentials import lj_potential, lj_force, lj_potential_pbc, lj_force_pbc, SIGMA, EPSILON, RCUT


def test_lj_potential_at_sigma():
    r2 = SIGMA * SIGMA
    e = lj_potential(r2)
    assert np.isclose(e, 0.0), f"LJ potential at r=sigma should be 0, got {e}"


def test_lj_potential_at_minimum():
    r2 = (2.0 ** (1.0 / 6.0) * SIGMA) ** 2
    e = lj_potential(r2)
    assert np.isclose(e, -EPSILON), f"LJ potential minimum should be -epsilon, got {e}"


def test_lj_force_zero_at_minimum():
    r2 = (2.0 ** (1.0 / 6.0) * SIGMA) ** 2
    f = lj_force(r2)
    assert np.abs(f) < 1e-10, f"LJ force at minimum should be 0, got {f}"


def test_lj_force_attractive_beyond_minimum():
    r2 = (2.0 * SIGMA) ** 2
    f = lj_force(r2)
    assert f < 0, f"LJ force beyond minimum should be attractive (f<0), got {f}"


def test_lj_force_repulsive_below_sigma():
    r2 = (0.9 * SIGMA) ** 2
    f = lj_force(r2)
    assert f > 0, f"LJ force below sigma should be repulsive (f>0), got {f}"


def test_lj_cutoff():
    positions = np.array([[0.0, 0.0, 0.0], [RCUT + 0.1, 0.0, 0.0]])
    box_length = 10.0 * RCUT
    e, _ = lj_potential_pbc(positions, box_length)
    assert np.isclose(e, 0.0), f"Energy beyond cutoff should be 0, got {e}"


def test_newton_third_law():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    box_length = 10.0
    forces = lj_force_pbc(positions, box_length)
    assert np.allclose(forces[0], -forces[1]), "Forces should obey Newton's third law"


def test_energy_conservation_potential():
    positions = np.array([[0.0, 0.0, 0.0], [1.2 * SIGMA, 0.0, 0.0]])
    e, f = lj_potential_pbc(positions, 10.0 * SIGMA)
    assert e < 0, "Energy at r=1.2 sigma should be negative (attractive regime)"
