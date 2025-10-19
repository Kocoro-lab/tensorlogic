"""
Visualization tools for attention weights, embeddings, and training progress

Requires matplotlib (optional dependency)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    """Check if matplotlib is available"""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_attention_weights(
    attention_weights: List[torch.Tensor],
    relation_names: List[str],
    query_pair: Optional[Tuple[str, str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot attention weights for each hop

    Args:
        attention_weights: List of [batch, num_relations] tensors (one per hop)
        relation_names: Names of relations in order
        query_pair: Optional (subject, object) names for title
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> logits, attn = composer.forward_with_attn(subj_emb, obj_emb, bank)
        >>> plot_attention_weights(attn, ['parent', 'works_at'])
    """
    _check_matplotlib()

    num_hops = len(attention_weights)
    fig, axes = plt.subplots(1, num_hops, figsize=(5 * num_hops, 4))

    if num_hops == 1:
        axes = [axes]

    for hop_idx, (ax, weights) in enumerate(zip(axes, attention_weights)):
        # Convert to numpy and average over batch
        weights_np = weights.detach().cpu().numpy()
        if weights_np.ndim == 2:
            weights_np = weights_np.mean(axis=0)  # Average over batch

        # Create bar plot
        x = np.arange(len(relation_names))
        bars = ax.bar(x, weights_np, color='steelblue', alpha=0.7)

        # Highlight max attention
        max_idx = weights_np.argmax()
        bars[max_idx].set_color('orange')
        bars[max_idx].set_alpha(1.0)

        ax.set_xlabel('Relation')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Hop {hop_idx + 1}')
        ax.set_xticks(x)
        ax.set_xticklabels(relation_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, weights_np)):
            if val > 0.05:  # Only show significant values
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Overall title
    if query_pair:
        fig.suptitle(f'Attention Weights: {query_pair[0]} → {query_pair[1]}',
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Multi-Hop Attention Weights', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_embedding_similarity(
    embeddings: torch.Tensor,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    cmap: str = 'RdYlBu_r'
):
    """
    Plot pairwise similarity heatmap of embeddings

    Args:
        embeddings: [N, D] tensor of embeddings
        labels: Optional list of N labels
        save_path: Optional path to save figure
        show: Whether to display the plot
        cmap: Colormap name

    Example:
        >>> plot_embedding_similarity(space.object_embeddings.weight)
    """
    _check_matplotlib()

    # Compute pairwise similarities
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    similarity = torch.matmul(normalized, normalized.t())
    similarity_np = similarity.detach().cpu().numpy()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(similarity_np, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)

    # Set ticks and labels
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    ax.set_title('Embedding Similarity Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Object')
    ax.set_ylabel('Object')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    history: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training loss curves from curriculum history

    Args:
        history: List of stage histories from CurriculumTrainer
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> curriculum = CurriculumTrainer(model)
        >>> history = curriculum.train_all(...)
        >>> plot_training_curves(history)
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange']
    offset = 0

    for i, stage_hist in enumerate(history):
        losses = stage_hist['losses']
        stage_name = stage_hist['stage_name']
        num_hops = stage_hist['num_hops']

        x = np.arange(offset, offset + len(losses))

        color = colors[i % len(colors)]
        ax.plot(x, losses, label=f"{stage_name} ({num_hops}-hop)",
               color=color, linewidth=2)

        # Mark stage boundaries
        if i < len(history) - 1:
            ax.axvline(x=offset + len(losses), color='gray',
                      linestyle='--', alpha=0.5, linewidth=1)

        offset += len(losses)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Curriculum Training Progress', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Add stage labels at top
    offset = 0
    for stage_hist in history:
        mid = offset + len(stage_hist['losses']) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95, stage_hist['stage_name'],
               ha='center', va='top', fontsize=9, alpha=0.7)
        offset += len(stage_hist['losses'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_relation_composition(
    space: 'EmbeddingSpace',
    relation1_name: str,
    relation2_name: str,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize relation composition: R1 @ R2

    Shows three heatmaps: R1, R2, and R1@R2

    Args:
        space: EmbeddingSpace instance
        relation1_name: Name of first relation
        relation2_name: Name of second relation
        labels: Optional object labels
        threshold: Threshold for binary visualization
        save_path: Optional path to save
        show: Whether to display

    Example:
        >>> plot_relation_composition(space, 'parent', 'parent')  # Shows grandparent
    """
    _check_matplotlib()

    # Get relation matrices
    R1 = space.relations[relation1_name]
    R2 = space.relations[relation2_name]

    # Compute composition
    R_composed = torch.matmul(R1, R2)

    # Get embeddings and compute scores
    emb = space.object_embeddings
    N = emb.shape[0]

    with torch.no_grad():
        # Scores for each relation
        scores1 = torch.sigmoid(torch.matmul(torch.matmul(emb, R1), emb.t()) / space.temperature)
        scores2 = torch.sigmoid(torch.matmul(torch.matmul(emb, R2), emb.t()) / space.temperature)
        scores_comp = torch.sigmoid(torch.matmul(torch.matmul(emb, R_composed), emb.t()) / space.temperature)

        # Threshold to binary
        adj1 = (scores1 > threshold).float().cpu().numpy()
        adj2 = (scores2 > threshold).float().cpu().numpy()
        adj_comp = (scores_comp > threshold).float().cpu().numpy()

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, adj, title in zip(axes, [adj1, adj2, adj_comp],
                              [relation1_name, relation2_name, f'{relation1_name} @ {relation2_name}']):
        im = ax.imshow(adj, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Object')
        ax.set_ylabel('Object')

        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_attention_summary(
    attention_weights: List[torch.Tensor],
    relation_names: List[str],
    top_k: int = 3
):
    """
    Print text summary of attention weights (no matplotlib required)

    Args:
        attention_weights: List of attention tensors
        relation_names: Relation names
        top_k: Number of top relations to show

    Example:
        >>> print_attention_summary(attn, ['parent', 'works_at', 'located_in'])
    """
    print("\n" + "="*60)
    print("🔍 Attention Weights Summary")
    print("="*60)

    for hop_idx, weights in enumerate(attention_weights, 1):
        print(f"\nHop {hop_idx}:")

        # Average over batch if needed
        weights_np = weights.detach().cpu().numpy()
        if weights_np.ndim == 2:
            weights_np = weights_np.mean(axis=0)

        # Get top-k
        sorted_indices = np.argsort(weights_np)[::-1][:top_k]

        for rank, idx in enumerate(sorted_indices, 1):
            rel_name = relation_names[idx]
            weight = weights_np[idx]
            bar = "█" * int(weight * 20)  # Simple text bar
            print(f"  {rank}. {rel_name:15s} {weight:.3f} {bar}")

    print("="*60 + "\n")
