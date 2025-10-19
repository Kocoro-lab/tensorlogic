"""
Demo: learn automatic two-hop composition (grandparent = parent ∘ parent)
without hand-written rules, using embedding-space + gated multi-hop composer.

Steps:
1) Train an EmbeddingSpace to model the 'parent' relation from facts.
2) Freeze EmbeddingSpace; train a GatedMultiHopComposer to score (s,o) pairs
   that hold under the (implicit) grandparent relation, using only the
   'parent' relation bank and two hops.
"""

import torch

from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.utils.sparse import adjacency_from_pairs_sparse, sparse_mm_aggregate
from tensorlogic.reasoning.composer import GatedMultiHopComposer
from tensorlogic.learn.losses import ContrastiveLoss
from tensorlogic.learn.trainer import EmbeddingTrainer, PairScoringTrainer
from tensorlogic import save_model, export_embeddings
from tensorlogic.utils.visualization import plot_attention_weights, plot_embedding_similarity, print_attention_summary
from tensorlogic.utils.viz_helper import ensure_viz_directory, print_viz_summary


def main():
    torch.manual_seed(42)

    # Entities: 0:Alice, 1:Bob, 2:Charlie, 3:Diana, 4:Eve
    num_objects = 5
    space = EmbeddingSpace(num_objects=num_objects, embedding_dim=32, temperature=0.2)

    # Define and train 'parent' relation from facts
    parent_facts = [
        (0, 1),  # Alice -> Bob
        (1, 2),  # Bob -> Charlie
        (2, 3),  # Charlie -> Diana
        (0, 4),  # Alice -> Eve (to add some variety)
    ]

    space.add_relation('parent', init='random')
    emb_trainer = EmbeddingTrainer(space, learning_rate=0.01)
    emb_trainer.train_relation('parent', parent_facts, epochs=150, verbose=False)

    # (Optional) Show sparse adjacency and a single sparse message passing step
    adj_sparse = adjacency_from_pairs_sparse(num_objects, parent_facts)
    msgs = sparse_mm_aggregate(adj_sparse, space.object_embeddings)
    print("Sparse aggregate shape:", msgs.shape)

    # Prepare grandparent training data: (s, o) pairs that should be true
    grandparent_pos = [
        (0, 2),  # Alice -> Charlie
        (1, 3),  # Bob -> Diana
    ]

    # Negative samples (automatic inside loss if not provided)

    # Create composer over the available relation bank (only 'parent')
    composer = GatedMultiHopComposer(embedding_dim=space.embedding_dim, num_relations=1, num_hops=2)

    # Loss and trainer for composer
    loss_fn = ContrastiveLoss(margin=0.5, negative_samples=6, use_logits=False)
    pair_trainer = PairScoringTrainer(composer, learning_rate=0.01)

    # Scorer closure uses the EmbeddingSpace for embeddings and relation bank
    def scorer(model, subjects, objects):
        return space.score_with_composer(
            composer=model,
            subjects=subjects,
            objects=objects,
            relation_order=['parent'],
            use_sigmoid=True,
        )

    # Build tiny batch list
    pos_tensor = torch.tensor(grandparent_pos, dtype=torch.long)
    batches = [{'pos': pos_tensor} for _ in range(200)]
    pair_trainer.train_epoch(loss_fn=loss_fn, scorer=scorer, batches=batches, num_objects=num_objects, verbose=True)

    # Save trained embedding space and composer
    print("\n" + "=" * 70)
    print("Saving trained models...")
    print("=" * 70)

    metadata = {
        'description': 'Family tree embedding space with learned parent relation',
        'num_entities': num_objects,
        'embedding_dim': space.embedding_dim,
        'temperature': space.temperature,
        'num_training_examples': len(parent_facts),
        'training_epochs': 150,
        'relations': ['parent'],
        'composer_info': 'GatedMultiHopComposer trained to learn grandparent composition'
    }

    save_model(space, 'models/learnable_composer.pt', metadata=metadata)
    export_embeddings(space, 'models/learnable_composer.json')

    # Also save the composer weights
    torch.save(composer.state_dict(), 'models/learnable_composer_weights.pt')

    print("✓ Embedding space saved to models/learnable_composer.pt")
    print("✓ Embeddings exported to models/learnable_composer.json")
    print("✓ Composer weights saved to models/learnable_composer_weights.pt")

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating learned composer...")
    print("=" * 70)

    test_pairs = [
        (0, 2),  # true
        (1, 3),  # true
        (0, 3),  # false
        (2, 4),  # likely false
    ]
    subjects = torch.tensor([p[0] for p in test_pairs], dtype=torch.long)
    objects = torch.tensor([p[1] for p in test_pairs], dtype=torch.long)
    scores = scorer(composer, subjects, objects)
    for (s, o), sc in zip(test_pairs, scores.tolist()):
        print(f"Score grandparent({s}->{o}) = {sc:.3f}")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    try:
        viz_dir = ensure_viz_directory('learnable_composer')

        # Plot attention weights for a sample query
        s_idx = torch.tensor([0])
        o_idx = torch.tensor([2])
        subj_emb = space.object_embeddings[s_idx]
        obj_emb = space.object_embeddings[o_idx]
        bank = space.relation_bank_tensor(order=['parent'])
        logits, attns = composer.forward_with_attn(subj_emb, obj_emb, bank)

        plot_attention_weights(
            attns,
            ['parent'],
            query_pair=(f'Entity {s_idx.item()}', f'Entity {o_idx.item()}'),
            save_path=f"{viz_dir}/attention_weights_query.png",
            show=False
        )

        # Print attention summary
        print_attention_summary(attns, ['parent'])

        # Plot embedding similarities
        plot_embedding_similarity(
            space.object_embeddings,
            save_path=f"{viz_dir}/embeddings_similarity_heatmap.png",
            show=False
        )

        descriptions = {
            'attention_weights_query.png': 'Learned attention weights across hops for (0→2) query',
            'embeddings_similarity_heatmap.png': 'Cosine similarity of learned entity embeddings',
        }
        print_viz_summary('learnable_composer', descriptions)

    except ImportError:
        print("⚠️  Matplotlib not available. Skipping visualizations.")
        print("   Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
