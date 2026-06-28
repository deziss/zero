"""src/integrate/api.py — Unified API entry point"""

from src.integrate.config import PipelineConfig
from src.integrate.pipeline import AdaptiveSimulationPipeline


def main(config_path: str = 'configs/integrate/default.yaml'):
    """Main entry point for the unified integration pipeline."""
    import os
    if os.path.exists(config_path):
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig.default()
    
    pipeline = AdaptiveSimulationPipeline(config)
    return pipeline.run_full()


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'configs/integrate/default.yaml'
    result = main(path)
    print(f"Pipeline complete: {result.get('total_steps', 0)} steps in {result.get('time_elapsed', 0):.1f}s")
