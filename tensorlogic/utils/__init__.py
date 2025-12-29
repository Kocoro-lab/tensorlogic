from .diagnostics import GradientHealthChecker, check_embedding_quality, diagnose_training_stuck
from .init_strategies import (
    init_embeddings,
    init_relation_matrix,
    init_embedding_space,
    init_composer,
    InitConfig
)
from .sparse import (
    adjacency_from_pairs_sparse,
    sparse_mm_aggregate,
)


def plot_attention_weights(*args, **kwargs):
    from .visualization import plot_attention_weights as _plot_attention_weights

    return _plot_attention_weights(*args, **kwargs)


def plot_embedding_similarity(*args, **kwargs):
    from .visualization import plot_embedding_similarity as _plot_embedding_similarity

    return _plot_embedding_similarity(*args, **kwargs)


def plot_training_curves(*args, **kwargs):
    from .visualization import plot_training_curves as _plot_training_curves

    return _plot_training_curves(*args, **kwargs)


def plot_relation_composition(*args, **kwargs):
    from .visualization import plot_relation_composition as _plot_relation_composition

    return _plot_relation_composition(*args, **kwargs)


def print_attention_summary(*args, **kwargs):
    from .visualization import print_attention_summary as _print_attention_summary

    return _print_attention_summary(*args, **kwargs)


__all__ = [
    # Diagnostics
    "GradientHealthChecker",
    "check_embedding_quality",
    "diagnose_training_stuck",

    # Initialization
    "init_embeddings",
    "init_relation_matrix",
    "init_embedding_space",
    "init_composer",
    "InitConfig",

    # Visualization
    "plot_attention_weights",
    "plot_embedding_similarity",
    "plot_training_curves",
    "plot_relation_composition",
    "print_attention_summary",

    # Sparse utils
    "adjacency_from_pairs_sparse",
    "sparse_mm_aggregate",
]
