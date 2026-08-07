"""
neuro_framework
===============
Unified framework for connectome-constrained neural network modelling
of the Drosophila visual system.

Target conference: NeurIPS 2026

Quick start
-----------
    from neuro_framework.utils import setup_logging
    setup_logging()   # routes logs to neuro_framework/logs/

    from neuro_framework.connectome import ConnectomeLoader
    from neuro_framework.models import ConnectomeNetwork
    from neuro_framework.stimulus import build_stimulus_tensor
    from neuro_framework.training import TorchTrainer
"""

__version__ = "0.1.0"

# Expose top-level logging helper so users can call
# ``import neuro_framework; neuro_framework.setup_logging()``
from neuro_framework.utils.logging import setup_logging, get_logger

__all__ = ["setup_logging", "get_logger", "__version__"]
