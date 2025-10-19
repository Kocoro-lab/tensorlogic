from .diagnostics import GradientHealthChecker, check_embedding_quality, diagnose_training_stuck
from .init_strategies import (
    init_embeddings,
    init_relation_matrix,
    init_embedding_space,
    init_composer,
    InitConfig
)
from .visualization import (
    plot_attention_weights,
    plot_embedding_similarity,
    plot_training_curves,
    plot_relation_composition,
    print_attention_summary
)
from .sparse import (
    adjacency_from_pairs_sparse,
    sparse_mm_aggregate,
)

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
