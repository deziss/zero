# src/ml — ML/AI module for molecular force prediction
# NOTE: Torch-dependent modules are imported lazily to allow non-torch workflows.

# Pure NumPy predictor (MVP v1, no torch needed)
from .predictor.numeric_depth_predictor import NumericDepthPredictor

# --- Lazy imports for torch-dependent modules ---
_lazy_modules = [
    'layers', 'model', 'datasets', 'train', 'inference',
    'equivariant_layers', 'radial_basis', 'spherical_harmonics',
    'mace', 'qm9', 'train_loop', 'compute_metrics', 'InferenceAPI',
    'bessel_basis', 'airy_basis',
    'real_spherical_harmonics', 'sph_harm',
    'EquivariantConvLayer', 'RadialTensorProduct',
    'MACE', 'QM9Dataset',
]

def __getattr__(name):
    if name == '_lazy_modules':
        return _lazy_modules
    # Lazy import of torch-dependent modules
    import importlib
    # Map name -> module path
    _name_to_module = {
        'bessel_basis': '.layers.radial_basis',
        'airy_basis': '.layers.radial_basis',
        'real_spherical_harmonics': '.layers.spherical_harmonics',
        'sph_harm': '.layers.spherical_harmonics',
        'EquivariantConvLayer': '.layers.equivariant_layers',
        'RadialTensorProduct': '.layers.equivariant_layers',
        'EquivariantConvLayer': '.layers.equivariant_layers',
        'MACE': '.model.mace',
        'QM9Dataset': '.datasets.qm9',
        'train_loop': '.train.train',
        'compute_metrics': '.train.train',
        'InferenceAPI': '.inference',
    }
    if name in _name_to_module:
        mod = importlib.import_module(_name_to_module[name], __name__)
        return getattr(mod, name, None)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['NumericDepthPredictor',
    "bessel_basis", "airy_basis",
    "real_spherical_harmonics", "sph_harm",
    "EquivariantConvLayer", "RadialTensorProduct",
    "MACE",
    "QM9Dataset",
    "train_loop", "compute_metrics",
    "InferenceAPI",
    "NumericDepthPredictor",
]
