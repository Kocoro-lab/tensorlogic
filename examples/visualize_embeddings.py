"""
Visualize Learned Embeddings and Relations

Shows:
1. Relation matrices as heatmaps
2. Object embeddings
3. Similarity matrix between objects
4. 2D projection of embeddings (PCA)
"""

import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional demo dependency
    plt = None
from tensorlogic import EmbeddingSpace, load_model


def _require_matplotlib():
    if plt is None:
        raise ImportError("This example requires matplotlib. Install with: pip install matplotlib")


def visualize_relation_matrix(model, relation_name, figsize=(10, 8)):
    """Visualize a relation matrix as a heatmap"""
    _require_matplotlib()
    if relation_name not in model.relations:
        print(f"Relation '{relation_name}' not found in model")
        return

    # Get relation matrix
    matrix = model.relations[relation_name].detach().cpu().numpy()

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight Value', rotation=270, labelpad=20)

    # Labels and title
    ax.set_xlabel('Child Embedding Dimensions', fontsize=12)
    ax.set_ylabel('Parent Embedding Dimensions', fontsize=12)
    ax.set_title(f'Learned "{relation_name}" Relation Matrix\n[{matrix.shape[0]} × {matrix.shape[1]}]',
                 fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(False)

    # Add some statistics
    stats_text = f'Mean: {matrix.mean():.4f}\nStd: {matrix.std():.4f}\nMin: {matrix.min():.4f}\nMax: {matrix.max():.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def visualize_embeddings_heatmap(model, figsize=(12, 6)):
    """Visualize all object embeddings as a heatmap"""
    _require_matplotlib()
    embeddings = model.object_embeddings.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(embeddings, cmap='viridis', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Embedding Value', rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel('Embedding Dimensions', fontsize=12)
    ax.set_ylabel('Objects', fontsize=12)

    # Y-axis labels (object names)
    names = [model.object_names.get(i, f'Object {i}') for i in range(len(embeddings))]
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    ax.set_title(f'Object Embeddings\n[{embeddings.shape[0]} objects × {embeddings.shape[1]} dimensions]',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def visualize_similarity_matrix(model, figsize=(8, 7)):
    """Visualize cosine similarity between all objects"""
    _require_matplotlib()
    embeddings = model.object_embeddings.detach().cpu().numpy()

    # Compute cosine similarity matrix
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Similarity = normalized @ normalized.T
    similarity = normalized @ normalized.T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(similarity, cmap='YlOrRd', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)

    # Labels
    names = [model.object_names.get(i, f'Object {i}') for i in range(len(embeddings))]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)

    # Add values in cells
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, f'{similarity[i, j]:.2f}',
                          ha="center", va="center", color="black" if similarity[i, j] < 0.5 else "white",
                          fontsize=9)

    ax.set_title('Object Similarity Matrix\n(Cosine Similarity)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def visualize_embeddings_2d(model, figsize=(10, 8)):
    """Project embeddings to 2D using PCA and visualize"""
    _require_matplotlib()
    embeddings = model.object_embeddings.detach().cpu().numpy()

    # PCA to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        s=200, c=range(len(embeddings)), cmap='tab10',
                        edgecolors='black', linewidths=2, alpha=0.7)

    # Add labels
    names = [model.object_names.get(i, f'Object {i}') for i in range(len(embeddings))]
    for i, name in enumerate(names):
        ax.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Object Embeddings in 2D (PCA Projection)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_query_scores(model, relation_name, figsize=(10, 8)):
    """Visualize query scores for all object pairs"""
    _require_matplotlib()
    num_objects = model.num_objects
    names = [model.object_names.get(i, f'Object {i}') for i in range(num_objects)]

    # Compute scores for all pairs
    scores = np.zeros((num_objects, num_objects))
    for i in range(num_objects):
        for j in range(num_objects):
            score = model.query_relation(relation_name, i, j, use_sigmoid=True)
            scores[i, j] = score.item()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(scores, cmap='RdYlGn', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', rotation=270, labelpad=20)

    # Labels
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)

    # Add values in cells
    for i in range(num_objects):
        for j in range(num_objects):
            color = "white" if scores[i, j] > 0.5 else "black"
            text = ax.text(j, i, f'{scores[i, j]:.2f}',
                          ha="center", va="center", color=color, fontsize=10)

    ax.set_xlabel('Object (child)', fontsize=12)
    ax.set_ylabel('Subject (parent)', fontsize=12)
    ax.set_title(f'Query Results: "{relation_name}" Relation\nP(subject {relation_name} object)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("Embedding Visualization")
    print("=" * 70)

    # Load the trained model
    print("\nLoading trained model...")
    model = EmbeddingSpace(num_objects=6, embedding_dim=32, temperature=0.1)
    metadata = load_model(model, '../models/family_tree.pt')

    print(f"Loaded: {metadata.get('description', 'Family tree model')}")
    print(f"Objects: {list(model.name_to_index.keys())}")
    print(f"Relations: {list(model.relations.keys())}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)

    figures = []

    # 1. Relation matrices
    print("\n1. Visualizing relation matrices...")
    for relation_name in model.relations.keys():
        fig = visualize_relation_matrix(model, relation_name)
        figures.append((f'{relation_name}_matrix', fig))
        print(f"   ✓ {relation_name} relation matrix")

    # 2. Object embeddings heatmap
    print("\n2. Visualizing object embeddings...")
    fig = visualize_embeddings_heatmap(model)
    figures.append(('embeddings_heatmap', fig))
    print(f"   ✓ Embeddings heatmap")

    # 3. Similarity matrix
    print("\n3. Computing similarity matrix...")
    fig = visualize_similarity_matrix(model)
    figures.append(('similarity_matrix', fig))
    print(f"   ✓ Similarity matrix")

    # 4. 2D projection
    print("\n4. Creating 2D projection (PCA)...")
    try:
        fig = visualize_embeddings_2d(model)
        figures.append(('embeddings_2d', fig))
        print(f"   ✓ 2D projection")
    except ImportError:
        print(f"   ⚠ Skipping 2D projection (scikit-learn not installed)")
        print(f"     Install with: pip install scikit-learn")

    # 5. Query score matrices
    print("\n5. Computing query scores...")
    for relation_name in ['parent', 'grandparent']:
        if relation_name in model.relations:
            fig = visualize_query_scores(model, relation_name)
            figures.append((f'{relation_name}_queries', fig))
            print(f"   ✓ {relation_name} query scores")

    # Save all figures
    print("\n" + "=" * 70)
    print("Saving figures...")
    print("=" * 70)

    import os
    os.makedirs('../visualizations', exist_ok=True)

    for name, fig in figures:
        path = f'../visualizations/{name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {path}")

    # Show all figures
    print("\n" + "=" * 70)
    print("Displaying figures...")
    print("=" * 70)
    print("\nClose the figure windows to continue...")

    plt.show()

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"\nGenerated {len(figures)} visualizations")
    print(f"Saved to: visualizations/")
    print("\nKey insights:")
    print("  - Relation matrices show learned transformations")
    print("  - Similarity matrix shows which objects are close in embedding space")
    print("  - Query scores show predicted relationships")
    print("=" * 70)


if __name__ == "__main__":
    main()
