"""test_integration.py — Verify unified integration pipeline"""
import sys, os
import numpy as np

sys.path.insert(0, "/home/anshukushwaha/95095/Backup/Desktop/learn/zero")

# Test imports
from src.integrate.config import PipelineConfig
from src.integrate.pipeline import AdaptiveSimulationPipeline, PipelineState

# Test config
config = PipelineConfig.default()
print(f"✓ Config loaded: {len(config.to_dict())} keys")
print(f"  Physics: {config.physics.n_particles} particles, T={config.physics.temperature}K, dt={config.physics.dt}")
print(f"  ML: degree={config.ml.max_degree}, layers={config.ml.num_interactions}, hidden={config.ml.hidden_dim}")
print(f"  Adaptive: {config.adaptive.n_levels} levels, thresholds f={config.adaptive.force_threshold}")
print(f"  QM: {config.qm.n_qm_atoms} atoms/method={config.qm.method}, basis={config.qm.basis}")
print(f"  Bio: channels={config.bio.in_channels}→{config.bio.out_channels}, modes={config.bio.modes}")
print(f"  Agent: iterations={config.agent.max_iterations}, goal={config.agent.goal}")

# Test pipeline instantiation
pipeline = AdaptiveSimulationPipeline(config)
print(f"\n✓ Pipeline created: {len(pipeline.to_dict())} keys")
print(f"  Physics engine: {type(pipeline.physics).__name__}")
print(f"  ML model: {type(pipeline.ml_model).__name__}")
print(f"  Adaptive: {type(pipeline.adaptive).__name__}")
print(f"  QM: {type(pipeline.qm).__name__}")
print(f"  Bio: {type(pipeline.bio).__name__}")
print(f"  Agent: {type(pipeline.agent).__name__}")

# Test system init
print("\n--- System Initialization ---")
atomic_numbers = np.array([6, 8, 1, 1])
positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]])
pipeline.initialize_system(atomic_numbers, positions)
assert pipeline.state == PipelineState.IDLE
assert pipeline.step_count == 0
assert pipeline.current_atoms is not None
assert pipeline.current_positions is not None
print(f"✓ System initialized: {len(atomic_numbers)} atoms, state={pipeline.state.value}")
print(f"  Atoms: {atomic_numbers.tolist()}, positions shape: {positions.shape}")

# Run simulation steps
print("\n--- Running 10 Simulation Steps ---")
for i in range(10):
    result = pipeline.step()
    metrics_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                             for k, v in result.metrics.items() if isinstance(v, float)])
    print(f"  Step {i+1:3d}: E={result.energy:10.4f}, qm={result.n_active_qm:3d}, "
          f"fine={result.n_fine:3d}, coarse={result.n_coarse:3d}, "
          f"metrics=[{metrics_str}]")

# Verify pipeline state
assert pipeline.state == PipelineState.IDLE
assert pipeline.step_count == 10
assert len(pipeline.results) == 10
print(f"\n✓ Pipeline state: step_count={pipeline.step_count}, results={len(pipeline.results)}")

# Test checkpoint save/load with pickle
print("\n--- Checkpoint Save/Load ---")
import tempfile, pickle
with tempfile.TemporaryDirectory() as tmpdir:
    pipeline.checkpoint_path = os.path.join(tmpdir, 'last_checkpoint.p')
    pipeline.save_checkpoint()
    
    # Verify checkpoint file exists
    assert os.path.exists(pipeline.checkpoint_path), "Checkpoint file not created"
    print(f"✓ Checkpoint saved to: {os.path.basename(pipeline.checkpoint_path)}")
    
    # Load checkpoint
    with open(pipeline.checkpoint_path, 'rb') as f:
        loaded_data = pickle.load(f)
    print(f"✓ Checkpoint loaded: {len(loaded_data)} keys with step_count={loaded_data.get('step_count', 'N/A')}")

# Test run_full (with smaller step count for speed)
print("\n--- Testing run_full() ---")
pipeline.stop()  # Reset for clean state
pipeline.state = PipelineState.IDLE
pipeline.step_count = 0

# Initialize again
pipeline.initialize_system(atomic_numbers, positions)
summary = pipeline.run(n_steps=5)

print(f"✓ run_full() returned: steps={summary['total_steps']}, "
      f"time={summary['time_elapsed']:.3f}s, steps_per_sec={summary['total_steps']/summary['time_elapsed']:.1f}")
assert summary['total_steps'] == 5
assert len(summary['results']) == 5
assert pipeline.state == PipelineState.COMPLETE

print("\n✅ ALL INTEGRATION TESTS PASSED!")
