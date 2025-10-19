"""
Helper utilities for organizing and saving visualizations by example.

Provides centralized directory management for example-specific visualizations.
"""

import os
from pathlib import Path


def ensure_viz_directory(example_name: str) -> str:
    """
    Ensure visualization subdirectory exists for an example.

    Creates: visualizations/{example_name}/

    Args:
        example_name: Name of the example (e.g., 'family_tree_embedding')

    Returns:
        Path to the visualization directory

    Example:
        >>> viz_dir = ensure_viz_directory('learnable_composer')
        >>> save_path = f"{viz_dir}/attention_weights.png"
    """
    # Create path relative to examples directory
    base_path = Path(__file__).parent.parent.parent / "visualizations" / example_name
    base_path.mkdir(parents=True, exist_ok=True)
    return str(base_path)


def get_viz_path(example_name: str, filename: str) -> str:
    """
    Get full path for a visualization file.

    Args:
        example_name: Name of the example
        filename: Filename (e.g., 'embeddings_heatmap.png')

    Returns:
        Full path to visualization file

    Example:
        >>> path = get_viz_path('learnable_composer', 'embeddings_heatmap.png')
    """
    viz_dir = ensure_viz_directory(example_name)
    return os.path.join(viz_dir, filename)


def print_viz_summary(example_name: str, descriptions: dict):
    """
    Print summary of generated visualizations.

    Args:
        example_name: Name of the example
        descriptions: Dict of {filename: description}

    Example:
        >>> descriptions = {
        ...     'embeddings_heatmap.png': 'Pairwise similarity of learned embeddings',
        ...     'relation_composition.png': 'Parent @ Parent = Grandparent composition'
        ... }
        >>> print_viz_summary('learnable_composer', descriptions)
    """
    viz_dir = ensure_viz_directory(example_name)

    print("\n" + "=" * 70)
    print(f"📊 Visualizations saved to: visualizations/{example_name}/")
    print("=" * 70)

    for filename, description in descriptions.items():
        print(f"  ✓ {filename}")
        print(f"    └─ {description}")

    print("=" * 70 + "\n")
