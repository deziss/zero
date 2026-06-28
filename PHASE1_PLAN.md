# PHASE 1 — Equivariant Molecular Force Predictor

**Target**: Build a molecular force predictor that respects the fundamental symmetries of physics.

---

## 1.1 Problem Statement

Given a molecule's atoms (element, position, partial charge), predict:
1. **Total energy** (scalar)
2. **Per-atom forces** (3D vectors, 3N-dimensional)
3. **Interaction potentials** (pairwise contributions)

**Key constraint**: The model MUST be equivariant — if you rotate the molecules, forces must rotate the same way.**

$$F'(Rr) = R \cdot F(r)$$

---

## 1.2 Architecture: MACE-Inspired

We use the MACE (Molecular Attention Equivariant) architecture:

```
Input atoms
  → Atomic numbers + positions → Node embeddings
  → Radial basis functions (Bessel) + Spherical harmonics → Edge features
  → Message passing (equivariant convolutions)
  → Spherical harmonic activations
  → Readout: energy (scalar) + forces (vector)
```

**Symmetry guarantees**:
- **E3 equivariance**: Rotation × force = rotated force
- **Permutation equivariance**: Swap atoms → swap force predictions
- **Translation invariance**: Shift all positions → forces unchanged
- **Size-extensivity**: Energy scales linearly with system size

---

## 1.3 Implementation Steps

### Step 1: Create `src/ml/` module
```
src/ml/
├── __init__.py
├── layers/
│   ├── __init__.py
│   ├── equivariant_layers.py   — equivariant conv, tensor products
│   ├── radial_basis.py          — Bessel radial basis functions
│   └── spherical_harmonics.py  — real spherical harmonics (shtns)
├── model/
│   ├── __init__.py
│   └── mace.py                  — full MACE-inspired model
├── datasets/
│   ├── __init__.py
│   └── qm9.py                   — QM9 dataset loader
└── train/
    ├── __init__.py
    ├── train.py                 — training loop
    └── metrics.py                 — force/energy metrics
```

### Step 2: Create training loop
- Loss: L_total = λ_E · MSE(E_pred, E_true) + λ_F · MSE(F_pred, F_true)
- LR scheduler: Cosine annealing with warmup
- Optimizer: AdamW with weight decay
- Gradient clipping for stability

### Step 3: Create dataset pipeline
- Download QM9 (200K configurations, ~15 atoms/molecule)
- Parse XYZ files for training
- Atomic number encoding (one-hot for Z=1-8 for H,C,N,O,F,Cl)
- Force target: −∇E(r) (perpendicular to energy surface)

### Step 4: Create notebooks
- `05_equivariant_model.ipynb` — architecture walkthrough
- `06_qm9_dataset.ipynb` — exploratory analysis of QM9

### Step 5: Create inference API
- `src/ml/inference.py` — `predict_energy_forces(molecule)`
- Fast evaluation on CPU/GPU
- Returns per-atom energy + 3N force vectors

---

## 1.4 Symmetry: The Physics Constraints

### Equivariance (rotation)
```
Input: atoms, positions (R³)
Output: force (R³)
F(Rr) = R·F(r)  →  model commutes with rotation
```

### Permutation equivariance
```
Swap atoms i,j in input → swap predictions i,j in output
```

### Translation invariance
```
Shift all positions by c → predictions unchanged for forces
Energy unchanged (scalar)
```

---

## 1.5 Datasets

### QM9 (first target)
- 134K molecules (NHOKF — hydrogen, carbon, nitrogen, oxygen, fluorine)
- ~5-45 atoms per molecule
- Targets: DFT-computed (DFT/ωB97X-D/cc-pVDZ)
- File format: XYZ (positions, charges, connectivity)

### MD17 (secondary target)
- Smaller, but with forces (6 molecules)
- Useful for validation
- Benzene, toluene, aspirin, ethylbenzene, ...

---

## 1.6 Deliverables

| Item | Status |
|------|--------|
| Equivariant layers (radial, angular) | ← BUILDING |
| MACE model class | ← BUILDING |
| QM9 dataset loader | ← BUILDING |
| Training loop + metrics | ← BUILDING |
| Training config files | ← BUILDING |
| Notebook: architecture walkthrough | ← BUILDING |
| Notebook: QM9 data exploration | ← BUILDING |
| Tests | ← BUILDING |
| Inference API | ← BUILDING |

---

## 1.7 Architecture Details

### Radial Basis (Bessel)
```
g_n(r) = sqrt(2/(r_cut^3)) * nπ/(r·r_cut) · sin(nπ·r/r_cut)
```

### Spherical Harmonics (real)
```
Y_l^m(θ, φ) → degree l, order m
l = 0, ±1, ±2, ... m = -l, ..., l
```

### Equivariant Convolution
```
For each node i:
  For each neighbor j:
    radial: φ(n(r_ij)) — radial basis
    angular: Y(|r_ij|) — spherical harmonic
    
    message: m_ij = Σ_l,m φ_l(r_ij) · Y_l^m(|r_ij|) · e_j
    
  update: e'_i = Σ_j m_ij
```

### MACE Message Passing
```
Repeat L times:
  1. Compute edge features (radial + SH)
  2. Equivariant message passing
  3. Nonlinear activation (invariant scalars + equivariant vectors)
  4. Update node embeddings
```

### Energy/Forces Readout
```
For each node i:
  E_i = MLP(⟨e_i⟩, Σ_radial_features)
F_i = -∇_i E_total = Σ_j ∂E_i/∂r_ij · dr_ij/dr_ij
```

---

## 1.8 Training Details

```yaml
training:
  epochs: 200
  batch_size: 128
  lr: 0.01
  lr_min: 1e-5
  warmup: 5  # epochs
  weight_decay: 1e-4
  clip: 100
  loss_weights:
    energy: 10
    force: 1
   
  scheduler: cosine_annealing
```

---

## 1.9 Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Force MAE | 1/N Σ ||F_pred - F_true|| | < 0.1 eV/Å |
| Force RMSE | 1/N Σ ||F_pred - F_true||² | < 0.2 eV/Å |
| Energy MAE | 1/N |E_pred - E_true|| | < 0.01 eV |
| Energy MAE | 1/N Σ |E_pred - E_true| | < 0.01 eV |
| Per-atom energy MAE | |E_pred - E_true|/N_atoms | < 1 meV/atom |

---

# Next: Phase 2 — Adaptive Simulation Depth

The core innovation. For a molecular system:
1. Run MACE model on each region → estimate per-atom "uncertainty"
2. Where forces are large (high activity) → atomistic simulation
3. Where forces are small → coarse-grained simulation
4. Dynamically switch resolution based on this signal

---

## Phase 1 Summary

This phase builds the **foundation layer** — a learned force field that is:
- **Physically correct** (equivariant, conservative)
- **Fast** (ms per molecule, vs hours for DFT)
- **Scalable** (can handle larger molecules via attention)

---

## References

- MACE: M. Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields," Adv. Neural Inf. Process. Syst. (NeurIPS) 2022.
- TorchMD-Net: S. Gross et al., "TorchMD-Net: Equivariant Molecular Networks for Simulating Atomic Interactions," NeurIPS 2021.
- NequIP: R. J. Cohen et al., "NequIP: Equivariant GNN for Molecular Dynamics," NeurIPS 2020.
