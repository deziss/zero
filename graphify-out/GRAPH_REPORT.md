# Graph Report - .  (2026-06-03)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 792 nodes · 924 edges · 73 communities (65 shown, 8 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 69 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 68|Community 68]]

## God Nodes (most connected - your core abstractions)
1. `AdaptiveSimulationPipeline` - 27 edges
2. `Adaptive Multiscale Biological Simulation AI` - 15 edges
3. `Cell` - 14 edges
4. `AgentLoop` - 14 edges
5. `lj_force_pbc()` - 12 edges
6. `Adaptive Multiscale Biological Simulation AI` - 12 edges
7. `PHASE 1 — Equivariant Molecular Force Predictor` - 10 edges
8. `QM9Dataset` - 9 edges
9. `MACE` - 9 edges
10. `lj_potential_pbc()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_fcc_lattice_counts()` --calls--> `fcc_lattice()`  [INFERRED]
  tests/test_initialise.py → src/core/initialise.py
- `test_fcc_lattice_density()` --calls--> `fcc_lattice()`  [INFERRED]
  tests/test_initialise.py → src/core/initialise.py
- `test_fcc_lattice_within_box()` --calls--> `fcc_lattice()`  [INFERRED]
  tests/test_initialise.py → src/core/initialise.py
- `test_fcc_lattice_different_sizes()` --calls--> `fcc_lattice()`  [INFERRED]
  tests/test_initialise.py → src/core/initialise.py
- `test_lj_force_zero_at_minimum()` --calls--> `lj_force()`  [INFERRED]
  tests/test_potentials.py → src/core/potentials.py

## Communities (73 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (37): EquivariantConvLayer, EquivariantNonLinearity, RadialTensorProduct, src/ml/layers/equivariant_layers.py  Equivariant message-passing layers for mole, Compute equivariant edge features from node features and radial basis., Equivariant non-linearity.          Applies non-linear activation only to invari, Args:             x: (n_nodes, hidden_dim) = first half scalars + second half ve, Equivariant convolutional layer for molecular models.      Commutes with rotatio (+29 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (39): andersen_thermostat(), langevin_thermostat(), Langevin dynamics using the velocity-rescaling (simple) thermostat.     Adds fri, integrate_nve(), velocity_verlet(), compute_energy(), compute_kinetic_energy(), compute_potential_energy() (+31 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (22): AdaptiveSimulationPipeline, Convert configuration to dictionary., Initialize Phase 0 physics engine.                  Uses actual function-based i, Initialize Phase 1 ML model.                  MACE model from src.ml.model for f, Initialize Phase 2 adaptive depth engine.                  Uses fallback depth s, Initialize Phase 3 quantum components.                  Uses fallback QM/MM mapp, Initialize Phase 4 neural operators.                  Uses fallback population m, Initialize Phase 5 agent loop.                  Uses fallback agent logic when f (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (26): AdaptiveCoarsening, CoarseGrainer, CoarseningLevel, src/bio/coarse_grain/coarse_grainer.py — Hierarchical coarse-graining  Maps atom, Find clusters using radius-based grouping., Decompose system into all coarse-graining levels.                  Returns:, Dynamically adjusts coarse-graining level per region.          Uses the importan, Perform adaptive coarse-graining based on importance.                  Args: (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (39): 1.1 Problem Statement, 1.2 Architecture: MACE-Inspired, 1.3 Implementation Steps, 1.4 Symmetry: The Physics Constraints, 1.5 Datasets, 1.6 Deliverables, 1.7 Architecture Details, 1.8 Training Details (+31 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (30): main(), src/integrate/api.py — Unified API entry point, Main entry point for the unified integration pipeline., AdaptiveConfig, AgentConfig, BioConfig, dump(), dump_yaml_file() (+22 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (18): Cell, CellState, src/bio/cellular/cell.py — Cellular abstractions  Models cells as objects with:, Divide the cell into two., Check if the cell should undergo apoptosis., Release signaling molecule into environment., Receive signaling molecule from environment., Differentiate into a new cell type. (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (16): AgentLoop, Experiment, Run the agent loop for the specified number of iterations.                  Args, Generate experiment plan from current knowledge., Execute experiments and collect results., Run a single experiment., Analyze experiment results., Update/evolve theories based on results. (+8 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (14): DepthSelector, DepthSelectorLayer, src/sim/layers/depth_selector.py  Depth selector layer: maps per-atom features t, Neural layer that predicts simulation depth for each atom.          Takes equiva, Args:             x: (n_atoms, hidden_dim) equivariant node features, Simpler depth selector using deterministic feature thresholds.          Uses fix, Args:             force_norm: (n_atoms,) magnitude of local forces             g, ImportanceScorer (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (13): from_checkpoint(), InferenceAPI, MolecularForcePredictor, no_grad(), src/ml/inference.py  Inference API for molecular force prediction.  Provides a u, Switch between cuda/cpu devices., Set prediction mode.                  - 'energy': Only energy (no gradient compu, Simpler functional API for molecular force prediction.          Usage:         p (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (21): Adaptive Multiscale Biological Simulation AI, Architecture Vision, code:block1 (Phase 0: Foundations           Quantum mechanics, statmech, ), code:block2 (src/core/), code:block3 (zero/), code:block4 (numpy >= 1.24), code:bash (# Create & activate environment), Core Problem: Adaptive Computational Allocation (+13 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (9): QM9Dataset, QM9DatasetSubset, src/ml/datasets/qm9.py  QM9 dataset loader for molecular force prediction.  QM9, QM9 dataset for molecular force prediction., Download QM9 dataset from GitHub., Load processed QM9 data., Get molecule configuration., Split dataset into train/validation/test. (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (9): DFTSurrogate, src/qm/surrogates/dft_surrogate.py — DFT surrogate model  Trains a fast neural n, Dataset for training DFT surrogate models.          Stores tuples of:       - at, Add one molecule to the dataset.                  Args:             atomic_numbe, Return basic statistics of the dataset., Neural network surrogate for DFT calculations.          Trained on DFT data to p, Encode distances using Gaussian radial basis functions., Predict energy and forces for a molecule.                  Args:             ato (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (16): 0.1 — Learn Core Scientific Math, 0.2 — Build Infrastructure, AI Architecture (Weeks 7–8), code:block1 (zero/), Directory Structure, Guiding Principle, Implementation Phasing, Installation Plan (+8 more)

### Community 14 - "Community 14"
Cohesion: 0.16
Nodes (8): FourierNeuralOperator, src/bio/neural_ops/fno.py — Fourier Neural Operator for field-level simulation, 1D spectral convolution for 1D field operators., 2D spectral convolution for FNO., Fourier Neural Operator for continuous field operators.          Maps molecular, Args:             x: (batch, h, w, in_channels) input field, SpectralConv1d, SpectralConv2d

### Community 15 - "Community 15"
Cohesion: 0.15
Nodes (8): DepthSelector, ImportanceNet, src/sim/deepness/simulation_depth.py  Simulation depth levels for the adaptive e, Enumeration of simulation depth levels., Selects simulation depth per atom/region based on importance scoring., Args:             node_features: (n_atoms, hidden_dim) node features, Network that predicts importance from node features., SimulationDepth

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (8): PySCFInterface, src/qm/dft/pyscf_interface.py — PySCF quantum chemistry interface  Provides a wr, Full DFT calculation with all outputs.                  Args:             atoms:, Wrapper for PySCF quantum chemistry calculations.          Provides high-fidelit, Args:             method: DFT functional (b3lyp, pbe0, wb97x-d, etc.), Calculate total energy using DFT.                  Args:             atoms: list, Calculate atomic forces via analytical gradients.                  Returns:, Calculate wave function and molecular orbitals.                  Returns:

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (12): compute_force_angle_error(), compute_metrics(), src/ml/train/metrics.py  Force and energy metrics for molecular force predictor, Compute force and energy metrics.      Args:         pred_energy: predicted ener, Compute force angle error (between prediction and ground truth)., compute_metrics(), src/ml/train/train.py  Training loop for molecular force prediction.  Loss: L =, Run validation and return metrics. (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (8): Evaluate experimental result against hypothesis., Update belief confidence using Bayesian update., Propose a new scientific question., Summarize agent's current state., Agent that performs scientific reasoning.          Capabilities:     1. Generate, Deduce a hypothesis from observations., Design experiment to test a hypothesis., ScientistAgent

### Community 19 - "Community 19"
Cohesion: 0.17
Nodes (6): src/qm/train/surrogate_trainer.py — Training pipeline for DFT surrogates  Trains, Training loop for DFT surrogate models.          Handles:     - Data loading fro, Evaluate model on test dataset., Save training configuration., Train surrogate model.                  Args:             dataset: SurrogateData, SurrogateTrainer

### Community 20 - "Community 20"
Cohesion: 0.2
Nodes (6): QM_MMMapper, src/qm/mapping/qm_mm_mapper.py — QM/MM region mapping  Maps atoms between quantu, Maps atoms between QM and MM regions for hybrid simulation.          Strategy:, Identify which atoms belong in the QM region., Identify link atoms (boundary between QM and MM)., Split full system into QM/MM regions.

### Community 21 - "Community 21"
Cohesion: 0.2
Nodes (5): PhenomenologicalTrainer, src/bio/train/pheno_trainer.py — Training pipeline for neural operators  Trains, Evaluate on validation dataloader., Training loop for FNO-based biological simulation models.          Tasks:     -, Train the FNO model.                  Args:             dataloader: training Dat

### Community 22 - "Community 22"
Cohesion: 0.31
Nodes (8): assign_velocities(), fcc_lattice(), test_assign_velocities_temperature(), test_assign_velocities_zero_momentum(), test_fcc_lattice_counts(), test_fcc_lattice_density(), test_fcc_lattice_different_sizes(), test_fcc_lattice_within_box()

### Community 23 - "Community 23"
Cohesion: 0.36
Nodes (7): read_xyz(), write_hdf5(), write_xyz(), test_hdf5_positions_only(), test_hdf5_write_and_read(), test_xyz_round_trip(), test_xyz_with_box_length()

### Community 24 - "Community 24"
Cohesion: 0.25
Nodes (7): Adaptive Multiscale Biological Simulation AI, code:block4 (Small correct physics), Core Research Problem, Final Strategy, Guiding Philosophy, Potential Research Titles, What NOT to Do

### Community 25 - "Community 25"
Cohesion: 0.32
Nodes (4): Constructs theories from supported hypotheses.          Builds theories by:, Add a hypothesis test to the theory., Update theory statements from supporting tests., TheoryBuilder

### Community 26 - "Community 26"
Cohesion: 0.29
Nodes (7): 0.1 Learn Core Scientific Math, 0.2 Infrastructure, AI Architecture, Molecular Dynamics, PHASE 0 — Foundations (1–2 Months), Quantum Mechanics, Statistical Thermodynamics

### Community 27 - "Community 27"
Cohesion: 0.29
Nodes (5): Tests for src.ml.radial_basis — syntax checks only., RadialBasisNetwork class exists., bessel_basis exists and returns correct shape., test_bessel_basis(), test_RadialBasisNetwork()

### Community 28 - "Community 28"
Cohesion: 0.33
Nodes (6): 1.1 Target, 1.2 Architecture Lever, 1.3 Dataset, 1.4 Symmetry Physics, 1.5 Deliverables, PHASE 1 — First Real Model (2–4 Months)

### Community 29 - "Community 29"
Cohesion: 0.33
Nodes (6): 2.1 Molecular Dynamics, 2.2 Adaptive Simulation Depth (THE INNOVATION), 2.3 Region Importance Scoring, 2.4 Dynamic Simulation Layers, 2.5 Thermodynamic Constraints, PHASE 2 — First Simulator (4–8 Months)

### Community 30 - "Community 30"
Cohesion: 0.33
Nodes (6): 3.1 Hybrid QM/MM Engine, 3.2 PySCF Integration, 3.3 Quantum Surrogate Models, 3.4 Constraints, code:block1 (Whole molecule → classical), PHASE 3 — Quantum Region Engine (8–14 Months)

### Community 31 - "Community 31"
Cohesion: 0.33
Nodes (5): Tests for src.bio.coarse_grain — syntax checks only., AdaptiveCoarsening class exists., CoarseGrainer class exists., test_AdaptiveCoarsening(), test_CoarseGrainer()

### Community 32 - "Community 32"
Cohesion: 0.33
Nodes (5): Tests for src.sim.layers.depth_selector — syntax checks only., DepthSelector exists., DepthSelectorLayer exists., test_DepthSelector(), test_DepthSelectorLayer()

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (5): Tests for src.qm.surrogates — syntax checks only., SurrogateDataset class exists., DFTSurrogate class exists., test_DFTSurrogate(), test_SurrogateDataset()

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (5): Tests for src.ml.layers.equivariant_layers — syntax checks only., RadialTensorProduct class exists., EquivariantConvLayer class exists., test_EquivariantConvLayer(), test_RadialTensorProduct()

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (5): Tests for src.ml.inference — syntax checks only., MolecularForcePredictor class exists., InferenceAPI class exists., test_InferenceAPI_class(), test_MolecularForcePredictor_class()

### Community 36 - "Community 36"
Cohesion: 0.33
Nodes (5): Tests for src.ml.train.metrics — syntax checks only., compute_force_angle_error function exists., compute_metrics function exists., test_compute_force_angle_error(), test_compute_metrics()

### Community 37 - "Community 37"
Cohesion: 0.33
Nodes (5): Tests for src.sim.deepness.simulation_depth — syntax checks only., DepthSelector class exists., SimulationDepth class exists with correct depth codes., test_DepthSelector(), test_SimulationDepth()

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (5): Tests for src.sim.engine.switchoff_engine — syntax checks only., SimRegion class exists., SwitchoffEngine class exists., test_SimRegion(), test_SwitchoffEngine()

### Community 39 - "Community 39"
Cohesion: 0.33
Nodes (5): Tests for src.ml.train.train — syntax checks only., compute_metrics function exists., train_loop function exists., test_compute_metrics(), test_train_loop()

### Community 40 - "Community 40"
Cohesion: 0.33
Nodes (5): Tests for src.sim.layers.uncertainty_estimator — syntax checks only., ImportanceScorer exists., UncertaintyEstimator exists., test_ImportanceScorer(), test_UncertaintyEstimator()

### Community 41 - "Community 41"
Cohesion: 0.4
Nodes (5): 5.1 Scientific Agents, 5.2 Multi-Agent Design, 5.3 Closed-Loop Scientific Engine, code:block3 (Observe → Simulate → Detect anomaly → Increase precision), PHASE 5 — Agentic Scientific AI (2–3 Years)

### Community 42 - "Community 42"
Cohesion: 0.4
Nodes (5): MVP Roadmap, MVP v1, MVP v2, MVP v3, MVP v4

### Community 43 - "Community 43"
Cohesion: 0.4
Nodes (5): 4.1 Cellular Abstractions, 4.2 Neural Operators, 4.3 Dynamic Zooming, code:block2 (Zoom out  → cellular abstraction), PHASE 4 — Multiscale Biology (1–2 Years)

### Community 44 - "Community 44"
Cohesion: 0.4
Nodes (3): Tests for src.bio.cellular — syntax checks only., CellPopulation class exists., test_CellPopulation()

### Community 45 - "Community 45"
Cohesion: 0.4
Nodes (3): Tests for src.bio.neural_ops.fno — syntax checks only., SpectralConv2d class exists., test_SpectralConv2d()

### Community 46 - "Community 46"
Cohesion: 0.4
Nodes (3): Tests for src.ml.layers.spherical_harmonics — syntax checks only., real_spherical_harmonics exists., test_real_spherical_harmonics()

### Community 47 - "Community 47"
Cohesion: 0.5
Nodes (4): Long-Term: HPC Cluster, Mid-Term: Distributed, PHASE 6 — Infrastructure Scaling, Short-Term: Single GPU

### Community 48 - "Community 48"
Cohesion: 0.5
Nodes (4): AI / Deep Learning, Infrastructure, Scientific, Stack Overview

### Community 49 - "Community 49"
Cohesion: 0.5
Nodes (3): Tests for src.ml.model.mace — syntax checks only., MACE class exists and is a torch.nn.Module., test_MACE_class()

### Community 50 - "Community 50"
Cohesion: 0.5
Nodes (3): Tests for src.bio.train.pheno_trainer — syntax checks only., PhenomenologicalTrainer class exists., test_PhenomenologicalTrainer()

### Community 51 - "Community 51"
Cohesion: 0.5
Nodes (3): Tests for src.qm.dft.pyscf_interface — syntax checks only., PySCFInterface class exists., test_PySCFInterface()

### Community 52 - "Community 52"
Cohesion: 0.5
Nodes (3): Tests for src.ml.datasets.qm9 — syntax checks only., QM9Dataset class exists., test_QM9Dataset_class()

### Community 53 - "Community 53"
Cohesion: 0.5
Nodes (3): Tests for src.qm.mapping.qm_mm_mapper — syntax checks only., QM_MMMapper class exists., test_QM_MMMapper()

### Community 54 - "Community 54"
Cohesion: 0.5
Nodes (3): Tests for src.sim — Phase 2 syntax checks only., sim module exposes expected components., test_sim_module()

### Community 55 - "Community 55"
Cohesion: 0.5
Nodes (3): Tests for src.qm.train.surrogate_trainer — syntax checks only., SurrogateTrainer class exists., test_SurrogateTrainer()

## Knowledge Gaps
- **334 isolated node(s):** `Tests for src.ml.layers.equivariant_layers — syntax checks only.`, `EquivariantConvLayer class exists.`, `RadialTensorProduct class exists.`, `Tests for src.qm.mapping.qm_mm_mapper — syntax checks only.`, `QM_MMMapper class exists.` (+329 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AdaptiveSimulationPipeline` connect `Community 2` to `Community 1`, `Community 3`, `Community 5`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `PipelineConfig` connect `Community 5` to `Community 2`, `Community 3`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `AdaptiveSimulationPipeline` (e.g. with `main()` and `PipelineConfig`) actually correct?**
  _`AdaptiveSimulationPipeline` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `lj_force_pbc()` (e.g. with `test_newton_third_law()` and `test_andersen_thermalises()`) actually correct?**
  _`lj_force_pbc()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tests for src.ml.layers.equivariant_layers — syntax checks only.`, `EquivariantConvLayer class exists.`, `RadialTensorProduct class exists.` to the rest of the system?**
  _334 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._