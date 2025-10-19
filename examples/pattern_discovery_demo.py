"""
Unsupervised pattern discovery via closure-based self-supervision.

We start with only single-hop 'parent' facts. We derive 2-hop closure by
Boolean composition (parent ∘ parent) to create positive pairs for training a
GatedMultiHopComposer. We do NOT hand-write the grandparent pattern.

The learned per-hop attention should concentrate on 'parent' at both hops.
"""

import torch

from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.reasoning.composer import GatedMultiHopComposer
from tensorlogic.reasoning.closure import compose_sequence, sample_pairs_from_adjacency
from tensorlogic.learn.losses import ContrastiveLoss
from tensorlogic.utils.sparse import adjacency_from_pairs_sparse, sparse_mm_aggregate
from tensorlogic.learn.trainer import PairScoringTrainer
from tensorlogic import save_model, export_embeddings


def main():
    torch.manual_seed(7)

    N = 6
    D = 32
    space = EmbeddingSpace(num_objects=N, embedding_dim=D, temperature=0.2)

    # Base parent facts
    parent_facts = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]

    # Add 'parent' relation and embed from facts (superposition) for an initial structure
    space.add_relation('parent', init='zeros')
    space.embed_relation_from_facts('parent', parent_facts)

    # Compute 2-hop closure as pseudo-labels (no manual rule)
    # Build Boolean adjacency directly from raw parent facts (no thresholding)
    adj_parent = torch.zeros((N, N), dtype=torch.float32)
    for i, j in parent_facts:
        adj_parent[i, j] = 1.0
    # (Optional) Demonstrate sparse adjacency and aggregation
    adj_sparse = adjacency_from_pairs_sparse(N, parent_facts)
    msgs = sparse_mm_aggregate(adj_sparse, space.object_embeddings)
    print("Sparse aggregate shape:", msgs.shape)

    adj_gparent = compose_sequence({'parent': adj_parent}, ['parent', 'parent'])

    # Positive pairs from closure
    pos_pairs = sample_pairs_from_adjacency(adj_gparent, num_samples=64)
    pos = torch.tensor(pos_pairs, dtype=torch.long)

    # Composer with 2 hops over a bank with only 'parent'
    composer = GatedMultiHopComposer(embedding_dim=D, num_relations=1, num_hops=2)
    loss_fn = ContrastiveLoss(margin=0.5, negative_samples=32, use_logits=False)
    trainer = PairScoringTrainer(composer, learning_rate=0.01)

    def scorer(model, subjects, objects):
        return space.score_with_composer(model, subjects, objects, relation_order=['parent'], use_sigmoid=True)

    # Train for a few epochs
    batches = [{'pos': pos} for _ in range(150)]
    trainer.train_epoch(loss_fn=loss_fn, scorer=scorer, batches=batches, num_objects=N, verbose=True)

    # Save trained embedding space and composer
    print("\n" + "=" * 70)
    print("Saving trained pattern discovery model...")
    print("=" * 70)

    metadata = {
        'description': 'Pattern discovery model: learns grandparent pattern from parent facts via closure',
        'num_entities': N,
        'embedding_dim': D,
        'temperature': space.temperature,
        'num_training_examples': len(parent_facts),
        'relations': ['parent'],
        'discovery_method': 'Closure-based self-supervision'
    }

    save_model(space, '../models/pattern_discovery.pt', metadata=metadata)
    export_embeddings(space, '../models/pattern_discovery.json')

    # Also save the composer weights
    torch.save(composer.state_dict(), '../models/pattern_discovery_composer_weights.pt')

    print("✓ Embedding space saved to models/pattern_discovery.pt")
    print("✓ Embeddings exported to models/pattern_discovery.json")
    print("✓ Composer weights saved to models/pattern_discovery_composer_weights.pt")

    # Inspect attention weights on a sample (0 -> 2 should be positive)
    s = torch.tensor([0])
    o = torch.tensor([2])
    subj_emb = space.object_embeddings[s]
    obj_emb = space.object_embeddings[o]
    bank = space.relation_bank_tensor(order=['parent'])
    logits, attns = composer.forward_with_attn(subj_emb, obj_emb, bank)
    print("Logit:", logits.item())
    for i, w in enumerate(attns, 1):
        print(f"Hop {i} attention over ['parent']:", w.squeeze(0).tolist())


if __name__ == "__main__":
    main()
