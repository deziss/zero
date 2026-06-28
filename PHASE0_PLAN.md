# Phase 0 — Foundations (1–2 Months)

**Master Plan**: Adaptive Multiscale Biological Simulation AI
**Strategy**: Build vertically first (one correct simulation), then scale horizontally
**Stack**: Python (NumPy / SciPy / JAX)

---

## Guiding Principle

Solve one small simulation **correctly** end-to-end, then layer on complexity.
The vertical slice for Phase 0 is:

> **1D → 2D Lennard-Jones fluid simulation with thermodynamic validation**

This touches:
- Molecular dynamics (integration, force fields)
- Thermodynamics (temperature, pressure, radial distribution)
- Quantum approximations (via LJ parameters)
- Infrastructure (GPU, Docker, orchestration)
- Future AI layer (replace LJ with GNN-predicted potentials)

---

## 0.1 — Learn Core Scientific Math

### Quantum Mechanics (Weeks 1–2)
| Topic | Goal | Verification |
|-------|------|-------------|
| Schrödinger equation & Hamiltonians | Solve 1D infinite well numerically | Plot eigenfunctions |
| Wavefunctions & probability density | Compute expectation values | Match analytical solutions |
| Electron density & DFT basics | Understand Hohenberg–Kohn theorems | Write a simple LDA total energy for H₂ |
| Quantum tunneling | Solve finite barrier numerically | Compute transmission coefficient |
| Density Functional Theory (DFT) | Run PySCF on H₂ molecule | Compare bond length vs experiment |

**Deliverable**: Jupyter notebook `notebooks/01_quantum_basics.ipynb`

### Statistical Thermodynamics (Weeks 3–4)
| Topic | Goal | Verification |
|-------|------|-------------|
| Entropy & Boltzmann distribution | Sample from canonical ensemble | Histogram matches Boltzmann factor |
| Gibbs free energy | Compute ΔG for simple reaction | Analytical check |
| Partition functions | Compute Z for ideal gas | Known formula |
| Langevin dynamics | Simulate Brownian particle | Mean-square displacement linear in t |
| Thermodynamic integration | Compute free energy difference | Compare to brute-force sampling |

**Deliverable**: Jupyter notebook `notebooks/02_statmech_basics.ipynb`

### Molecular Dynamics (Weeks 5–6)
| Topic | Goal | Verification |
|-------|------|-------------|
| Force fields & potentials | Implement LJ + Coulomb in Python | Energy conservation test |
| Integration methods | Verlet, Velocity-Verlet, BAOAB | Compare stability regions |
| Periodic boundary conditions | Implement minimum-image convention | No drift over 1M steps |
| Thermostats & barostats | Andersen, Nosé-Hoover, Parrinello-Rahman | NVT / NPT ensembles match expected T/P |
| Trajectory analysis | RDF, MSD, velocity autocorrelation | LJ fluid RDF matches literature |

**Deliverable**: Jupyter notebook `notebooks/03_md_basics.ipynb`

### AI Architecture (Weeks 7–8)
| Topic | Goal | Verification |
|-------|------|-------------|
| GNN fundamentals | Implement message-passing on toy graph | Train to predict LJ energy |
| Equivariant networks (e3nn) | Build SO(3) equivariant layer | Rotational invariance test |
| Physics-Informed Neural Networks (PINNs) | Solve 1D Poisson equation with DeepXDE | Compare to analytical |
| Neural Operators | Learn mapping between function spaces | Fourier neural operator on Darcy flow |
| Diffusion models | Understand score matching + reverse process | Generate valid LJ configurations |

**Deliverable**: Jupyter notebook `notebooks/04_ai_architecture.ipynb`

---

## 0.2 — Build Infrastructure

### Directory Structure
```
zero/
├── notebooks/               # 0.1 learning notebooks
│   ├── 01_quantum_basics.ipynb
│   ├── 02_statmech_basics.ipynb
│   ├── 03_md_basics.ipynb
│   └── 04_ai_architecture.ipynb
├── src/
│   ├── core/               # Simulation engine
│   │   ├── __init__.py
│   │   ├── potentials.py   # LJ, Coulomb, custom
│   │   ├── integrators.py  # Verlet, Langevin, BAOAB
│   │   ├── ensemble.py     # NVT, NPT thermostats/barostats
│   │   ├── observables.py  # RDF, MSD, structure factor
│   │   └── io.py           # Trajectory readers/writers
│   ├── quantum/            # QM wrappers (PySCF)
│   ├── ml/                 # AI models (GNN, PINN, neural op)
│   └── orchestration/      # Ray, Slurm wrappers
├── tests/
│   ├── test_potentials.py
│   ├── test_integrators.py
│   ├── test_ensemble.py
│   └── test_observables.py
├── experiments/
│   ├── 001_lj_nve/         # NVE LJ simulation
│   ├── 002_lj_nvt/         # NVT LJ simulation
│   └── 003_water_spce/     # SPC/E water benchmark
├── docker/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── configs/
│   ├── simulation.yaml     # Default simulation parameters
│   └── cluster.yaml        # Ray cluster config
├── docs/
│   ├── math_notes/         # Derivation notes (LaTeX)
│   └── architecture.md     # System design doc
├── requirements.txt
├── setup.py
└── README.md
```

### Installation Plan
| Tool | Purpose | Install Command |
|------|---------|----------------|
| Python 3.11+ | Runtime | `apt install python3.11` |
| CUDA 12.x | GPU compute | NVIDIA driver + cuda-toolkit |
| PyTorch 2.x | Deep learning | `pip install torch` |
| JAX | GPU-accelerated autodiff | `pip install jax[cuda12]` |
| PySCF | Quantum chemistry | `pip install pyscf` |
| OpenMM | Molecular dynamics engine | `conda install -c conda-forge openmm` |
| TorchMD-Net | Equivariant neural networks | `pip install torchmd-net` |
| DeepXDE | PINNs | `pip install deepxde` |
| e3nn | Equivariant math | `pip install e3nn` |
| Ray | Distributed compute | `pip install "ray[default]"` |
| Docker | Containerization | `apt install docker.io` |
| Triton Inference Server | Model serving | Via NGC container |

### Validation Milestones
1. **Week 1–2**: Quantum notebook passes all self-checks
2. **Week 3–4**: Statmech notebook reproduces Boltzmann distribution from MD
3. **Week 5–6**: LJ fluid simulation conserves energy to 1e-5 relative; RDF matches literature
4. **Week 7–8**: PINN solves Poisson equation to 1% relative error
5. **Infrastructure**: Full `docker-compose up` launches Jupyter + Ray cluster; GPU-enabled

---

## Vertical Slice: LJ Fluid NVE Simulation

The first **end-to-end correct simulation** is:

> Simulate 864 LJ particles in a cubic box at reduced density ρ* = 0.8, temperature T* = 1.0, in the NVE ensemble. Verify energy conservation to machine precision over 10,000 steps. Compute radial distribution function and compare to standard reference (e.g., Verlet 1968).

### Implementation Phasing
1. **`src/core/potentials.py`** — Lennard-Jones with cutoff + tail correction
2. **`src/core/integrators.py`** — Velocity-Verlet integrator
3. **`src/core/observables.py`** — RDF, temperature, pressure, energy
4. **`experiments/001_lj_nve/`** — Full simulation script + analysis notebook
5. **`tests/`** — Unit tests for each component

### Verification Criteria
- Total energy drift < 1e-5 per 1000 steps
- Temperature fluctuation consistent with equipartition
- RDF peak positions match Verlet (1968) within 1%
- All tests pass with `pytest`

---

## Next Decision Points

After Phase 0 deliverables are validated:

1. **Scale**: Move LJ from CPU to JAX/GPU (10–100x speedup)
2. **Layer QM**: Replace LJ with potential from PySCF DFT calculation
3. **Layer AI**: Train GNN surrogate for DFT potential
4. **Layer biology**: Add water model → solvated protein fragment
5. **Orchestrate**: Distribute across Ray cluster
