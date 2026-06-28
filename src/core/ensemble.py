import numpy as np


def andersen_thermostat(velocities, target_temperature, nu, dt, mass=1.0):
    n = velocities.shape[0]
    mask = np.random.random(n) < nu * dt
    sigma = np.sqrt(target_temperature / mass)
    velocities[mask] = np.random.randn(np.sum(mask), 3) * sigma
    return velocities


def langevin_thermostat(positions, velocities, forces, gamma, target_temperature, dt, mass=1.0):
    """
    Langevin dynamics using the velocity-rescaling (simple) thermostat.
    Adds friction + stochastic noise to sample the canonical ensemble.
    """
    # Ornstein-Uhlenbeck factor
    omega = np.exp(-gamma * dt)
    # OU noise std (from fluctuation-dissipation theorem)
    sigma_noise = np.sqrt((target_temperature / mass) * (1.0 - omega * omega))
    
    # Apply OU process (friction + noise)
    v = omega * velocities + sigma_noise * np.random.randn(*velocities.shape)
    
    # Integrate position: x(t+dt) = x(t) + v*dt
    q = positions + v * dt
    
    return q, v
