"""src/sim/engine/switchoff_engine.py

Switchoff Engine — the core adaptive simulation depth engine.

Decides per-atom which level of simulation to use:
  depth 0: coarse-grained statistical
  depth 1: cellular/mesoscopic
  depth 2: atomistic MD
  depth 3: high-res MD
  depth 4: quantum (DFT)

The engine switches depth dynamically based on:
- AI predictions (MACE force predictions)
- Physical indicators (force, gradient, activity)
- Uncertainty estimates
"""