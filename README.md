# Adaptive Multiscale Biological Simulation AI

*A Scientific Simulation Operating System — vertically built, horizontally scaled.*

---

## Overview

This project is a **scientific simulation operating system** that unifies:

- **Physics**: Quantum, molecular, statistical mechanics
- **Chemistry**: DFT, force fields, thermodynamics
- **Biology**: Cellular abstractions, pathways, transport
- **AI**: GNNs, equivariant networks, PINNs, neural operators
- **HPC**: GPU compute, distributed orchestration, adaptive simulation depth
- **Agents**: Autonomous scientific reasoning loops

**Strategy**: Build vertically first (solve one small simulation correctly), then scale horizontally. Gradually layer in complexity — never simulate everything at the highest resolution.

---

## Research Focus

### Core Problem: Adaptive Computational Allocation

The central research question — *before biology, before AI, before quantum*:

> **Where should expensive physics be applied?**

An AI system that dynamically decides how much precision each region of a simulation needs. Where activity is low → coarse simulation. Where interactions are complex → atomistic or quantum.

MVP v1 (this alone is publishable): Given a small molecule system, AI predicts which atoms need deeper simulation, coarse-simulates the rest.

### Potential Research Titles

- *"Adaptive Multiscale Physics-Constrained Molecular Foundation Models"*
- *"Dynamic Resolution Quantum-Biological Simulation Systems"*

---

## Architecture Vision

```
Phase 0: Foundations           Quantum mechanics, statmech, MD basics
     ↓
Phase 1: Equivariant Model     Learned atomic force predictor
     ↓
Phase 2: Adaptive Simulator    Dynamic simulation depth engine
     ↓
Phase 3: Quantum Regions       Hybrid QM/MM, DFT surrogates (1000x faster)
     ↓
Phase 4: Multiscale Biology    Cellular abstractions, neural operators
     ↓
Phase 5: Agentic AI            Autonomous scientific reasoning loops
     ↓
Phase 6: Infrastructure        GPU → Distributed → HPC cluster

Small correct physics
    ↓ Adaptive abstraction
    ↓ Hierarchical simulation
    ↓ Agentic orchestration
    ↓ Large-scale biological world models
```

---

## Current Status

| Component | Status |
|-----------|--------|
|| **Core engine** (`src/core/`) | ✅ Implemented |
|| **Potentials** (LJ, Coulomb) | ✅ Done |
|| **Integrators** (Velocity-Verlet) | ✅ Done |
|| **Observables** (RDF, T, P, energy) | ✅ Done |
|| **Ensemble** (Andersen, Langevin) | ✅ Done |
|| **I/O** (XYZ, HDF5) | ✅ Done |
|| **Tests** | ✅ Syntax-verified |
|| **ML Module** (`src/ml/`) | ✅ Phase 1 complete |
|| **Sim Engine** (`src/sim/`) | ✅ Phase 2 complete |
|| **Adaptive Depth Engine** | ✅ Built |
|| **Notebooks 01–08** | ✅ All written |
|| **Docker/Ray infra** | ✅ Dockerfiles + compose |

### What's in `src/core/`

```
src/core/
├── potentials.py       — Lennard-Jones, Coulomb, PBC
├── integrators.py       — Velocity-Verlet, NVE integration
├── observables.py       — RDF, temperature, pressure, energy
├── ensemble.py           — Andersen, Langevin thermostats
└── io.py                — XYZ, HDF5 read/write
```

---

## Directory Structure

```
zero/
├── docs/
│   ├── master_plan.md          ← Full 6-phase master plan (this doc)
│   └── math_notes/             ← Derivation notes (LaTeX)
├── src/core/                   ← Simulation engine (implemented)
├── tests/                      ← Unit tests
├── notebooks/                  ← Learning notebooks
├── figures/                    ← Generated plots (PNG)
├── experiments/                ← Simulation configs (e.g. 001_lj_nve/)
├── configs/                    ← YAML simulation parameters
├── docker/                     ← Dockerfiles + compose
├── graphify-out/               ← AST graph (auto-generated, do not edit)
├── AGENTS.md                   ← Agent instructions
├── PHASE0_PLAN.md              ← Phase 0 detail plan
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **AI / DL** | PyTorch, JAX, e3nn, TorchMD-Net |
| **Scientific** | OpenMM, PySCF, DeepXDE, LAMMPS, GROMACS |
| **Infra** | Docker, K8s, Ray, CUDA 12.x, Triton |

### Runtime Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
h5py   >= 3.8
jax      >= 0.4
pytest   >= 7.0
```

---

## Current Status

|| Component | Status |
||------|------|
|| **Phase 0** (Physics Foundations) | ✅ Complete |
| **Phase 1** (Equivariant Model / MACE) | ✅ Complete |
| **Phase 2** (Adaptive Depth Engine) | ✅ Complete |
| **Phase 3** (Quantum Regions / QM/MM) | ✅ Complete |
| **Phase 4** (Multiscale Biology / FNO) | ✅ Complete |
| **Phase 5** (Agentic Reasoning Loop) | ✅ Complete |
| **MVP v1** (Adaptive Simulation Depth) | ✅ Implemented |
| **Notebooks 01–11** | ✅ All written |
| **Infrastructure** (Docker/Ray) | ✅ Completed |

## What to Build Next

**Priority 1**: MVP v2 — Integration & Training

- Connect Phase 0-5 modules into unified pipeline
- QM9 dataset download and caching
- Train MACE surrogate on QM reference data
- End-to-end adaptive simulation demo

**Priority 2**: Infrastructure & Scaling

- GPU acceleration (CUDA) for all modules
- Docker/Ray cluster deployment
- Distributed training across GPU nodes
- Benchmarking and performance profiles

**Priority 3**: Phase 6 — Large-scale Systems

- GPU → Distributed → HPC cluster
- Parallel simulation across compute nodes
- Multi-molecule ensemble learning
- Real-world biological applications (protein-ligand, enzyme catalysis)

---

## What NOT to Do

- ❌ Train giant transformers immediately
- ❌ Attempt whole-cell simulation first
- ❌ Build AGI before physics
- ❌ Simulate every electron
- ❌ Overfocus on UI

---

## Getting Started

```bash
# Create & activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest -v

# View project graph
cat graphify-out/GRAPH_REPORT.md
```

---

## License

[To be determined]

---

*The realistic path toward organism-scale simulation systems.*
