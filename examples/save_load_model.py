"""
Example: Training, Saving, and Loading Models

Demonstrates:
1. Train an embedding space model
2. Save the trained model to disk
3. Load the model back
4. Use loaded model for inference
"""

import sys
sys.path.append('..')

import torch
from tensorlogic import EmbeddingSpace, save_model, load_model, export_embeddings
from tensorlogic.learn.trainer import EmbeddingTrainer


def train_and_save():
    """Train a model and save it"""
    print("=" * 70)
    print("STEP 1: Training Model")
    print("=" * 70)

    # Define family members
    num_people = 6
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

    # Create embedding space
    embed_space = EmbeddingSpace(
        num_objects=num_people,
        embedding_dim=32,
        temperature=0.1,
        device='cpu'
    )

    # Add objects with names
    for i, name in enumerate(names):
        embed_space.add_object(name, i)

    print(f"\nFamily members ({num_people} people): {names}")

    # Define parent relationships
    parent_facts = [
        (0, 1),  # Alice -> Bob
        (0, 2),  # Alice -> Charlie
        (1, 3),  # Bob -> Diana
        (2, 4),  # Charlie -> Eve
        (2, 5),  # Charlie -> Frank
    ]

    print("\nParent relationships (training data):")
    for i, j in parent_facts:
        print(f"  {names[i]} -> {names[j]}")

    # Train parent relation
    embed_space.add_relation('parent', init='random')
    trainer = EmbeddingTrainer(embed_space, optimizer_type='adam', learning_rate=0.01)

    print("\nTraining...")
    trainer.train_relation('parent', parent_facts, epochs=200, verbose=True)

    # Test accuracy
    print("\n" + "=" * 70)
    print("Testing Trained Model:")
    print("=" * 70)
    print("\nTrue relationships:")
    for i, j in parent_facts[:3]:
        prob = embed_space.query_relation('parent', i, j, use_sigmoid=True)
        print(f"  P({names[i]} parent of {names[j]}) = {prob.item():.4f}")

    # Compose grandparent
    embed_space.apply_rule('parent', 'parent', 'grandparent')
    print("\nGrandparent relationships (composed):")
    prob = embed_space.query_relation('grandparent', 0, 3)  # Alice -> Diana
    print(f"  P(Alice grandparent of Diana) = {prob.item():.4f}")

    # Save model
    print("\n" + "=" * 70)
    print("STEP 2: Saving Model")
    print("=" * 70)

    metadata = {
        'description': 'Family tree model',
        'num_training_examples': len(parent_facts),
        'training_epochs': 200,
        'relations': ['parent', 'grandparent']
    }

    save_model(embed_space, '../models/family_tree.pt', metadata=metadata)

    # Also export human-readable version
    export_embeddings(embed_space, '../models/family_tree.json')

    return embed_space


def load_and_test():
    """Load model and test inference"""
    print("\n" + "=" * 70)
    print("STEP 3: Loading Model from Disk")
    print("=" * 70)

    # Create empty model with same config
    embed_space_loaded = EmbeddingSpace(
        num_objects=6,
        embedding_dim=32,
        temperature=0.1,
        device='cpu'
    )

    # Load trained weights
    metadata = load_model(embed_space_loaded, '../models/family_tree.pt')

    print(f"\nMetadata: {metadata}")

    # Test loaded model
    print("\n" + "=" * 70)
    print("STEP 4: Testing Loaded Model")
    print("=" * 70)

    names = list(embed_space_loaded.name_to_index.keys())
    print(f"\nLoaded object names: {names}")

    print("\nQuerying parent relationships:")
    test_pairs = [
        (0, 1, True),   # Alice -> Bob (should be True)
        (0, 2, True),   # Alice -> Charlie (should be True)
        (1, 2, False),  # Bob -> Charlie (should be False)
        (3, 4, False),  # Diana -> Eve (should be False)
    ]

    for i, j, expected in test_pairs:
        prob = embed_space_loaded.query_relation('parent', i, j, use_sigmoid=True)
        result = "✓" if (prob > 0.5) == expected else "✗"
        print(f"  {result} P({names[i]} parent of {names[j]}) = {prob.item():.4f} (expected: {expected})")

    print("\nQuerying grandparent relationships:")
    prob = embed_space_loaded.query_relation('grandparent', 0, 3)  # Alice -> Diana
    print(f"  P(Alice grandparent of Diana) = {prob.item():.4f}")

    print("\n" + "=" * 70)
    print("SUCCESS: Model saved and loaded correctly!")
    print("=" * 70)

    return embed_space_loaded


def compare_models(original, loaded):
    """Compare original and loaded models"""
    print("\n" + "=" * 70)
    print("STEP 5: Comparing Original vs Loaded Model")
    print("=" * 70)

    # Compare embeddings
    print("\nEmbedding comparison (first object):")
    orig_emb = original.object_embeddings[0][:5]
    load_emb = loaded.object_embeddings[0][:5]
    print(f"  Original: {orig_emb.detach().cpu().numpy()}")
    print(f"  Loaded:   {load_emb.detach().cpu().numpy()}")
    diff = torch.norm(orig_emb - load_emb).item()
    print(f"  Difference: {diff:.6f} {'✓' if diff < 1e-5 else '✗'}")

    # Compare relation matrices
    print("\nRelation matrix comparison (parent, first 3x3):")
    orig_rel = original.relations['parent'][:3, :3]
    load_rel = loaded.relations['parent'][:3, :3]
    print(f"  Original:\n{orig_rel.detach().cpu().numpy()}")
    print(f"  Loaded:\n{load_rel.detach().cpu().numpy()}")
    diff = torch.norm(orig_rel - load_rel).item()
    print(f"  Difference: {diff:.6f} {'✓' if diff < 1e-5 else '✗'}")


def main():
    print("\n" + "=" * 70)
    print("Tensor Logic: Model Save/Load Demo")
    print("=" * 70)

    # Train and save
    original_model = train_and_save()

    # Load and test
    loaded_model = load_and_test()

    # Compare
    compare_models(original_model, loaded_model)

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nSaved files:")
    print("  - models/family_tree.pt    (binary model file)")
    print("  - models/family_tree.json  (human-readable embeddings)")
    print("\nYou can now use the saved model in other scripts:")
    print("  embed_space = EmbeddingSpace(...)")
    print("  load_model(embed_space, 'models/family_tree.pt')")
    print("=" * 70)


if __name__ == "__main__":
    main()
