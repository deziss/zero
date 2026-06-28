"""src/integrate/config.py — Pipeline configuration

Manages unified configuration for all phases:
- Phase 0 (physics): simulation parameters, potentials
- Phase 1 (ML): model architecture, training
- Phase 2 (adaptive): depth thresholds, scorching
- Phase 3 (QM): DFT settings, surrogate
- Phase 4 (bio): cell population, FNO settings
- Phase 5 (agent): goal, iteration limits

Configuration is loaded from YAML files or defaults
and merged hierarchically by phase.
"""

import os
import json
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    def parse_yaml_file(path):
        """Fallback YAML parser using configobj or dict literal."""
        import ast
        # Try loading as simple dict first (fallback for minimal YAML-like)
        with open(path) as f:
            content = f.read()
        return ast.literal_eval(content.replace('true', 'True').replace('false', 'False').replace(': null', ': None'))
    
    def dump_yaml_file(path, data):
        """Fallback YAML dumper using json."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    class yaml:
        @staticmethod
        def safe_load(f_obj):
            f_obj.seek(0)
            return parse_yaml_file(f_obj)
        @staticmethod
        def dump(data, f_obj, **kwargs):
            dump_yaml_file(f_obj, data)
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class PhysicsConfig:
    """Phase 0: Physics parameters."""
    dt: float = 0.001
    n_particles: int = 100
    temperature: float = 300.0
    pressure: float = 1.0
    potential: str = 'lj'  # lj, coulomb, combined
    cutoff: float = 2.5
    boundary: str = 'pbc'  # pbc, fixed, free
    thermostat: Optional[str] = 'langevin'
    thermostat_params: Dict[str, float] = field(default_factory=lambda: {
        'gamma': 0.1,
        'target_temp': 300.0,
    })


@dataclass
class MLConfig:
    """Phase 1: ML model parameters."""
    max_degree: int = 4
    max_l: int = 4
    num_interactions: int = 2
    hidden_dim: int = 128
    n_rbf: int = 20
    r_cut: float = 5.0
    cutoff_mult: float = 2.5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 300


@dataclass
class AdaptiveConfig:
    """Phase 2: Adaptive depth parameters."""
    n_levels: int = 5
    force_threshold: float = 0.5
    gradient_threshold: float = 1.0
    uncertainty_threshold: float = 0.3
    density_threshold: float = 0.2
    auto_depth: bool = True
    scoring: str = 'hybrid'
    depth_map_freq: int = 10


@dataclass
class QMConfig:
    """Phase 3: Quantum region parameters."""
    n_qm_atoms: int = 50
    n_layer_atoms: int = 10
    cutoff_angstrom: float = 5.0
    method: str = 'b3lyp'
    basis: str = '6-31g'
    surrogate_type: str = 'dft_surrogate'
    max_n_atoms: int = 1000
    hidden_dim: int = 256
    n_layers: int = 6


@dataclass
class BioConfig:
    """Phase 4: Biology parameters."""
    in_channels: int = 5
    out_channels: int = 5
    latent_dim: int = 64
    modes: int = 16
    width: int = 128
    n_layers: int = 6
    grid_size: int = 128
    padding: int = 25
    n_cells_initial: int = 200
    spatial_grid: int = 100
    dt: float = 1.0
    signaling_range: float = 10.0


@dataclass
class AgentConfig:
    """Phase 5: Agent parameters."""
    goal: str = 'Discover underlying physics principles.'
    max_iterations: int = 100
    verbosity: int = 1
    n_memories: int = 100
    exploration_rate: float = 0.3


@dataclass
class PipelineConfig:
    """Unified configuration for the entire pipeline."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    qm: QMConfig = field(default_factory=QMConfig)
    bio: BioConfig = field(default_factory=BioConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Global settings
    device: str = 'cpu'
    seed: int = 42
    log_level: str = 'INFO'
    checkpoint_dir: str = 'checkpoints/integrate/'
    log_dir: str = 'logs/integrate/'
    
    # Workflow
    run_physics: bool = True
    train_ml: bool = True
    enable_adaptive: bool = True
    enable_qm: bool = True
    enable_bio: bool = True
    enable_agent: bool = True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'physics': self.physics.__dict__,
            'ml': self.ml.__dict__,
            'adaptive': self.adaptive.__dict__,
            'qm': self.qm.__dict__,
            'bio': self.bio.__dict__,
            'agent': self.agent.__dict__,
            'device': self.device,
            'seed': self.seed,
            'run_physics': self.run_physics,
            'train_ml': self.train_ml,
            'enable_adaptive': self.enable_adaptive,
            'enable_qm': self.enable_qm,
            'enable_bio': self.enable_bio,
            'enable_agent': self.enable_agent,
        }
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'physics' in data:
            config.physics = PhysicsConfig(**data['physics'])
        if 'ml' in data:
            config.ml = MLConfig(**data['ml'])
        if 'adaptive' in data:
            config.adaptive = AdaptiveConfig(**data['adaptive'])
        if 'qm' in data:
            config.qm = QMConfig(**data['qm'])
        if 'bio' in data:
            config.bio = BioConfig(**data['bio'])
        if 'agent' in data:
            config.agent = AgentConfig(**data['agent'])
        
        if 'device' in data:
            config.device = data['device']
        if 'seed' in data:
            config.seed = data['seed']
        
        return config
    
    @classmethod
    def default(cls) -> 'PipelineConfig':
        """Create default configuration."""
        return cls()
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        data = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
