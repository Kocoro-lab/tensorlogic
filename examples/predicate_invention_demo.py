"""
Demo: predicate invention via RESCAL and registration.

Build triples for a simple setting (only 'parent' relation), run RESCAL to
discover latent predicates, select top candidates by validation F1, and
register them into EmbeddingSpace.
"""

import torch

from tensorlogic.reasoning import (
    EmbeddingSpace,
    invent_and_register_rescal,
)
from tensorlogic import save_model, export_embeddings
from tensorlogic.utils.visualization import plot_embedding_similarity
from tensorlogic.utils.viz_helper import ensure_viz_directory, print_viz_summary


def main():
    torch.manual_seed(123)

    # Simple family chain
    N = 6
    R = 1  # only 'parent'
    parent = 0
    parent_facts = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]

    # Triples list (i, r, j)
    triples = [(i, parent, j) for (i, j) in parent_facts]

    # Embedding space
    space = EmbeddingSpace(num_objects=N, embedding_dim=32, temperature=0.2)

    # Run invention and register top-2 predicates
    result = invent_and_register_rescal(
        triples=triples,
        num_objects=N,
        num_relations=R,
        rank=16,
        top_k=2,
        val_ratio=0.3,
        steps=3,
        embedding_space=space,
        tensor_program=None,
        verbose=True,
    )

    print("\nSelected invented predicates:")
    for name, metrics in zip(result["names"], result["metrics"]):
        print(f"  {name}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")

    print("\nRegistered relations in EmbeddingSpace:")
    print(list(space.relations.keys()))

    # Save embedding space with invented predicates
    print("\n" + "=" * 70)
    print("Saving model with invented predicates...")
    print("=" * 70)

    metadata = {
        'description': 'Family tree model with invented predicates via RESCAL',
        'num_entities': N,
        'embedding_dim': 32,
        'num_training_triples': len(triples),
        'rescal_rank': 16,
        'invented_relations': result["names"],
        'num_invented': len(result["names"]),
        'discovery_method': 'RESCAL tensor factorization'
    }

    save_model(space, '../models/predicate_invention.pt', metadata=metadata)
    export_embeddings(space, '../models/predicate_invention.json')
    print("✓ Model with invented predicates saved to models/predicate_invention.pt")
    print("✓ Embeddings exported to models/predicate_invention.json")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    try:
        viz_dir = ensure_viz_directory('predicate_invention')

        # Plot embedding similarities
        plot_embedding_similarity(
            space.object_embeddings.weight,
            save_path=f"{viz_dir}/embeddings_similarity_with_invented.png",
            show=False
        )

        descriptions = {
            'embeddings_similarity_with_invented.png': 'Entity embeddings learned via RESCAL with invented predicates',
        }
        print_viz_summary('predicate_invention', descriptions)

    except ImportError:
        print("⚠️  Matplotlib not available. Skipping visualizations.")
        print("   Install with: pip install matplotlib")


if __name__ == "__main__":
    main()

