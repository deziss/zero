import numpy as np
from src.core.observables import (
    compute_temperature, compute_kinetic_energy,
    compute_potential_energy, compute_pressure, compute_rdf,
    compute_energy
)


def test_temperature_ideal_gas():
    n = 1000
    velocities = np.random.randn(n, 3)
    v_com = np.mean(velocities, axis=0)
    velocities -= v_com
    v2 = np.sum(velocities * velocities)
    target = 1.0
    scale = np.sqrt(3.0 * (n - 1) * target / np.sum(velocities * velocities))
    velocities *= scale
    ke = compute_kinetic_energy(velocities)
    T = compute_temperature(ke, n)
    assert np.isclose(T, target, rtol=1e-3), f"Temperature should be {target}, got {T}"


def test_rdf_normalization():
    n = 500
    box_length = 10.0
    np.random.seed(7)
    positions = np.random.rand(n, 3).astype(np.float64) * box_length
    r_centers, rdf = compute_rdf(positions, box_length, n_bins=100)
    dr = r_centers[1] - r_centers[0]
    # For an ideal gas, RDF should approach 1.0 at intermediate radii
    # (between 0.5*dr and rmax*0.8 to avoid boundary effects at r=0 and r=rmax)
    intermediate = slice(int(0.5 * len(rdf)), int(0.8 * len(rdf)))
    avg_rdf = np.mean(rdf[intermediate])
    assert np.isclose(avg_rdf, 1.0, rtol=0.1), f"RDF should approach 1.0 for ideal gas, got {avg_rdf:.3f}"


def test_rdf_zero_at_small_r():
    box_length = 10.0
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    r_centers, rdf = compute_rdf(positions, box_length, n_bins=50)
    assert rdf[0] == 0.0, "RDF should be zero at r=0"


def test_pressure_ideal_gas():
    # Test the ideal gas law: P = rho * T
    # Strategy: place particles so far apart that LJ forces are EXACTLY zero
    # (beyond RCUT). Then P = P_ideal by definition.
    n = 10
    box_length = 100.0  # huge box
    rho = n / (box_length ** 3)
    target_T = 2.0
    velocities = np.random.randn(n, 3)
    velocities -= velocities.mean(axis=0)
    scale = np.sqrt(3.0 * (n - 1) * target_T / np.sum(velocities * velocities))
    velocities *= scale

    # Place each particle at least RCUT+1 apart on a line
    positions = np.zeros((n, 3), dtype=np.float64)
    separation = (2.5 * 1.0) + 1.0
    for i in range(n):
        positions[i, 0] = i * separation
        
    from src.core.potentials import lj_force_pbc
    forces = lj_force_pbc(positions, box_length)
    T = compute_temperature(compute_kinetic_energy(velocities), n)
    virial = np.sum(positions * forces)
    assert np.isclose(virial, 0.0, atol=1e-10), f"LJ forces should be zero, virial={virial}"
    P = compute_pressure(positions, forces, box_length, T, n)
    P_ideal = rho * T
    assert np.isclose(P, P_ideal, rtol=0.05), f"Pressure {P:.6f} != ideal {P_ideal:.6f}"


def test_energy_consistency():
    n = 100
    box_length = 8.0
    positions = np.random.rand(n, 3).astype(np.float64) * box_length
    velocities = np.random.randn(n, 3).astype(np.float64)
    ke = compute_kinetic_energy(velocities)
    pe = compute_potential_energy(positions, box_length)
    ke2, pe2, total = compute_energy(positions, velocities, box_length)
    assert np.isclose(ke, ke2), f"KE mismatch: {ke} vs {ke2}"
    assert np.isclose(pe, pe2), f"PE mismatch: {pe} vs {pe2}"
    assert np.isclose(total, ke + pe), f"Total mismatch: {total} vs {ke + pe}"
