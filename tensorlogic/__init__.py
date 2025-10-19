"""
Tensor Logic: A unified programming language for AI

Combines neural and symbolic reasoning through tensor equations.
Based on "Tensor Logic: The Language of AI" by Pedro Domingos (arXiv:2510.12269)
"""

from .core.program import TensorProgram
from .core.tensor import TensorWrapper
from .reasoning.embed import EmbeddingSpace
from .reasoning.composer import GatedMultiHopComposer, stack_relation_bank
from .learn.trainer import Trainer
from .utils.io import save_model, load_model, save_checkpoint, load_checkpoint, export_embeddings

__version__ = "0.1.0"
__all__ = [
    "TensorProgram",
    "TensorWrapper",
    "EmbeddingSpace",
    "Trainer",
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    "export_embeddings",
    "GatedMultiHopComposer",
    "stack_relation_bank",
]
