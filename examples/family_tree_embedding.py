"""
Family Tree Reasoning Example - Embedding Space with Learning

Demonstrates:
1. Learning object embeddings from relation facts
2. Reasoning in embedding space
3. Analogical inference
4. Composing relations (parent -> grandparent)
"""

import sys
sys.path.append('..')

import torch
from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.learn.trainer import EmbeddingTrainer
from tensorlogic import save_model, export_embeddings


def main():
    print("=" * 70)
    print("Family Tree Reasoning - Embedding Space with Learning")
    print("=" * 70)

    # Define family members
    num_people = 8
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

    print(f"\nFamily members ({num_people} people):")
    for i, name in enumerate(names):
        print(f"  {i}: {name}")

    # Create embedding space
    embedding_dim = 64
    temperature = 0.1  # Low temperature = more strict reasoning

    embed_space = EmbeddingSpace(
        num_objects=num_people,
        embedding_dim=embedding_dim,
        temperature=temperature,
        device='cpu'
    )

    # Add objects with names
    for i, name in enumerate(names):
        embed_space.add_object(name, i)

    # Define parent relationships (ground truth)
    parent_facts = [
        (0, 1),  # Alice -> Bob
        (0, 2),  # Alice -> Charlie
        (1, 3),  # Bob -> Diana
        (1, 4),  # Bob -> Eve
        (2, 5),  # Charlie -> Frank
        (5, 6),  # Frank -> Grace
        (5, 7),  # Frank -> Henry
    ]

    print("\nParent relationships (training data):")
    for i, j in parent_facts:
        print(f"  {names[i]} is parent of {names[j]}")

    # Initialize parent relation
    embed_space.add_relation('parent', init='random')

    # Train the parent relation
    print("\n" + "=" * 70)
    print("Training parent relation embedding...")
    print("=" * 70)

    trainer = EmbeddingTrainer(
        embed_space,
        optimizer_type='adam',
        learning_rate=0.01
    )

    trainer.train_relation(
        relation_name='parent',
        positive_pairs=parent_facts,
        epochs=200,
        verbose=True
    )

    # Test learned parent relation
    print("\n" + "=" * 70)
    print("Testing learned parent relation:")
    print("=" * 70)

    print("\nTrue relationships (should be ~1.0):")
    for i, j in parent_facts[:5]:  # Test subset
        prob = embed_space.query_relation('parent', i, j, use_sigmoid=True)
        print(f"  P({names[i]} parent of {names[j]}) = {prob.item():.4f}")

    print("\nFalse relationships (should be ~0.0):")
    false_pairs = [(1, 2), (3, 4), (0, 5)]
    for i, j in false_pairs:
        prob = embed_space.query_relation('parent', i, j, use_sigmoid=True)
        print(f"  P({names[i]} parent of {names[j]}) = {prob.item():.4f}")

    # Compose relations: grandparent = parent ∘ parent
    print("\n" + "=" * 70)
    print("Composing relations: grandparent = parent ∘ parent")
    print("=" * 70)

    embed_space.apply_rule('parent', 'parent', 'grandparent')

    # Query grandparent relationships
    print("\nGrandparent relationships (inferred):")
    grandparent_expected = [
        (0, 3),  # Alice -> Diana (via Bob)
        (0, 4),  # Alice -> Eve (via Bob)
        (0, 5),  # Alice -> Frank (via Charlie)
        (2, 6),  # Charlie -> Grace (via Frank)
        (2, 7),  # Charlie -> Henry (via Frank)
    ]

    for i, j in grandparent_expected:
        prob = embed_space.query_relation('grandparent', i, j, use_sigmoid=True)
        print(f"  P({names[i]} grandparent of {names[j]}) = {prob.item():.4f}")

    # Analogical reasoning
    print("\n" + "=" * 70)
    print("Analogical Reasoning:")
    print("=" * 70)

    print("\nKnown: Alice is parent of Bob")
    print("Query: If Alice is similar to Charlie, who might Charlie be parent of?")

    # Find similar person to Bob based on Alice->Bob pattern
    source = (0, 1)  # Alice -> Bob
    target_subject = 2  # Charlie
    inferred_obj = embed_space.analogical_inference(source, 'parent', target_subject)
    print(f"Answer: {names[inferred_obj]} (actual: Frank, idx={names.index('Frank')})")

    # Similarity analysis
    print("\n" + "=" * 70)
    print("Similarity Analysis:")
    print("=" * 70)

    print(f"\nMost similar people to Bob:")
    similar = embed_space.find_similar(1, top_k=3)
    for idx, score in similar:
        print(f"  {names[idx]}: {score:.4f}")

    print(f"\nMost similar people to Alice:")
    similar = embed_space.find_similar(0, top_k=3)
    for idx, score in similar:
        print(f"  {names[idx]}: {score:.4f}")

    # Query all pairs
    print("\n" + "=" * 70)
    print("All grandparent relationships (threshold=0.5):")
    print("=" * 70)

    all_grandparents = embed_space.query_all_pairs('grandparent', threshold=0.5)
    print("\nGrandparent matrix:")
    for i in range(num_people):
        for j in range(num_people):
            if all_grandparents[i, j] > 0.5:
                print(f"  {names[i]} is grandparent of {names[j]}")

    # Save trained model
    print("\n" + "=" * 70)
    print("Saving trained model...")
    print("=" * 70)

    metadata = {
        'description': 'Family tree embedding model with learned parent relation',
        'num_entities': num_people,
        'embedding_dim': embedding_dim,
        'temperature': temperature,
        'num_training_examples': len(parent_facts),
        'training_epochs': 200,
        'relations': ['parent', 'grandparent']
    }

    save_model(embed_space, '../models/family_tree_embedding.pt', metadata=metadata)
    export_embeddings(embed_space, '../models/family_tree_embedding.json')
    print("✓ Model saved to models/family_tree_embedding.pt")
    print("✓ Embeddings exported to models/family_tree_embedding.json")

    print("\n" + "=" * 70)
    print("Embedding space reasoning complete!")
    print("=" * 70)
    print(f"\nKey insights:")
    print(f"  - Embeddings learned structure of family tree")
    print(f"  - Composition (grandparent) works via matrix multiplication")
    print(f"  - Analogical reasoning finds similar patterns")
    print(f"  - Low temperature ({temperature}) gives strict logical reasoning")
    print("=" * 70)


if __name__ == "__main__":
    main()
