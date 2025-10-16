"""
Input/Output utilities for saving and loading models
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any


def save_model(model, path: str, metadata: Dict[str, Any] = None):
    """
    Save a trained model (TensorProgram or EmbeddingSpace)

    Args:
        model: Model to save
        path: File path (e.g., 'models/my_model.pt')
        metadata: Optional metadata dict
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
    }

    # Add model-specific config
    if hasattr(model, 'num_objects'):
        # EmbeddingSpace
        checkpoint['config'] = {
            'num_objects': model.num_objects,
            'embedding_dim': model.embedding_dim,
            'temperature': model.temperature,
        }
        # Save object name mappings
        if hasattr(model, 'object_names'):
            checkpoint['object_names'] = model.object_names
            checkpoint['name_to_index'] = model.name_to_index

    # Add metadata
    if metadata:
        checkpoint['metadata'] = metadata

    # Save
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  - Size: {path.stat().st_size / 1024:.2f} KB")


def load_model(model, path: str) -> Dict[str, Any]:
    """
    Load a trained model

    Args:
        model: Model instance to load into (must be initialized with same config)
        path: File path to load from

    Returns:
        Metadata dict (if any)
    """
    checkpoint = torch.load(path, weights_only=False)

    # For EmbeddingSpace, need to add relations first
    if checkpoint['model_type'] == 'EmbeddingSpace':
        state_dict = checkpoint['model_state_dict']
        # Find all relation names from state dict keys
        relation_names = set()
        for key in state_dict.keys():
            if key.startswith('relations.'):
                rel_name = key.split('.', 1)[1]
                relation_names.add(rel_name)

        # Add relations to model (with dummy data - will be overwritten)
        for rel_name in relation_names:
            if rel_name not in model.relations:
                model.add_relation(rel_name, init='zeros')

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore object name mappings if present
    if 'object_names' in checkpoint:
        model.object_names = checkpoint['object_names']
        model.name_to_index = checkpoint['name_to_index']

    print(f"Model loaded from {path}")
    print(f"  - Type: {checkpoint['model_type']}")
    if 'config' in checkpoint:
        print(f"  - Config: {checkpoint['config']}")

    return checkpoint.get('metadata', {})


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    path: str,
    metadata: Dict[str, Any] = None
):
    """
    Save training checkpoint (model + optimizer state)

    Useful for resuming training later.

    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: File path
        metadata: Optional metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_type': type(model).__name__,
    }

    # Add config if present
    if hasattr(model, 'num_objects'):
        checkpoint['config'] = {
            'num_objects': model.num_objects,
            'embedding_dim': model.embedding_dim,
            'temperature': model.temperature,
        }
        if hasattr(model, 'object_names'):
            checkpoint['object_names'] = model.object_names
            checkpoint['name_to_index'] = model.name_to_index

    if metadata:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} (epoch {epoch}, loss {loss:.4f})")


def load_checkpoint(model, optimizer, path: str) -> tuple:
    """
    Load training checkpoint

    Args:
        model: Model instance
        optimizer: Optimizer instance
        path: File path

    Returns:
        (epoch, loss, metadata)
    """
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore mappings
    if 'object_names' in checkpoint:
        model.object_names = checkpoint['object_names']
        model.name_to_index = checkpoint['name_to_index']

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metadata = checkpoint.get('metadata', {})

    print(f"Checkpoint loaded from {path}")
    print(f"  - Epoch: {epoch}, Loss: {loss:.4f}")

    return epoch, loss, metadata


def export_embeddings(model, path: str):
    """
    Export embeddings and relations to human-readable JSON

    Args:
        model: EmbeddingSpace model
        path: Output JSON file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'config': {
            'num_objects': model.num_objects,
            'embedding_dim': model.embedding_dim,
            'temperature': model.temperature,
        },
        'object_embeddings': {},
        'relations': {}
    }

    # Export object embeddings
    for idx, name in model.object_names.items():
        emb = model.object_embeddings[idx].detach().cpu().numpy().tolist()
        data['object_embeddings'][name] = emb

    # Export relations (first 5 rows/cols for readability)
    for rel_name, rel_matrix in model.relations.items():
        matrix = rel_matrix.detach().cpu().numpy()
        data['relations'][rel_name] = {
            'shape': list(matrix.shape),
            'sample_5x5': matrix[:5, :5].tolist()  # Just show corner
        }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Embeddings exported to {path}")
