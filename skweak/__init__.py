"""
skweak/
├── core/                   # Abstract base classes and interfaces
├── utils/                  # Utility functions and helpers
├── labelmodel/             # Label model implementations
├── endmodel/               # End model implementations
├── pipelines/              # Experiment pipelines
│   ├── endmodel_experiment/
│   ├── semisupervised_experiment/
│   └── adversarial_experiment/
"""
from functools import lru_cache
from . import dataset
from . import evaluation
from . import experiments
from .constant import set_pdb

@lru_cache(maxsize=1)
def _lazy_load_all_models():
    from .core.patch import _ensure_models_loaded
    _ensure_models_loaded()

_lazy_load_all_models()


__version__ = "0.0.1"


__all__ = [
    "dataset",
    "evaluation",
    "experiments",
    "transform",
    "core",
    "utils",
    "labelmodel",
    "endmodel",
    "set_pdb",
]