#!/usr/bin/env python3
"""
Transformer-based reasoning with TensorLogic.

This example demonstrates:
1. Using transformers with EmbeddingSpace for relational reasoning
2. Combining attention with symbolic constraints (knowledge graph masks)
3. Learning relation transformations via attention patterns
4. Visualizing attention as relation discovery

The key insight: Attention weights reveal implicit relations between entities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# TensorLogic imports
from tensorlogic.core import TensorProgram
from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.transformers import (
    MultiHeadAttention,
    TransformerEncoder,
    SinusoidalPositionalEncoding
)
from tensorlogic.transformers.utils import causal_mask
from tensorlogic.utils.visualization import plot_attention_weights
from tensorlogic.learn.trainer import EmbeddingTrainer
from tensorlogic.learn.losses import ContrastiveLoss

class RelationalTransformer(nn.Module):
    """
    Transformer that learns relations between entities in an EmbeddingSpace.

    Key idea: Each attention head discovers a different type of relation.
    """

    def __init__(
        self,
        embedding_space: EmbeddingSpace,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        mode: str = 'continuous'
    ):
        super().__init__()
        self.space = embedding_space
        self.d_model = embedding_space.embedding_dim
        self.num_heads = num_heads

        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim or 4 * self.d_model,
            dropout=dropout,
            mode=mode
        )

        # Optional: Learn to map attention patterns to relation types
        self.relation_classifier = nn.Linear(num_heads, len(embedding_space.relations))

    def forward(
        self,
        entity_indices: torch.Tensor,
        knowledge_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            entity_indices: [batch_size, seq_len] indices into embedding space
            knowledge_mask: [batch_size, seq_len, seq_len] from knowledge graph

        Returns:
            output: [batch_size, seq_len, d_model] transformed embeddings
            attention_weights: List of [batch_size, num_heads, seq_len, seq_len]
        """
        # Get entity embeddings
        batch_size, seq_len = entity_indices.shape
        embeddings = self.space.object_embeddings[entity_indices]  # [B, L, D]

        # Apply transformer with optional knowledge graph constraints
        output, attention_weights = self.encoder(
            embeddings,
            src_mask=knowledge_mask,
            return_attention=True
        )

        return output, attention_weights

    def discover_relations(
        self,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Map attention patterns to relation types.

        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]

        Returns:
            relation_scores: [batch_size, seq_len, seq_len, num_relations]
        """
        B, H, L, L2 = attention_weights.shape

        # Reshape to [B*L*L, H]
        weights_flat = attention_weights.permute(0, 2, 3, 1).reshape(-1, H)

        # Classify each attention pattern
        relation_scores = self.relation_classifier(weights_flat)  # [B*L*L, R]

        # Reshape back
        return relation_scores.reshape(B, L, L2, -1)


class KnowledgeGraphTransformer(nn.Module):
    """
    Transformer that respects knowledge graph structure.
    Uses symbolic facts to constrain attention patterns.
    """

    def __init__(
        self,
        program: TensorProgram,
        embedding_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        self.program = program
        self.d_model = embedding_dim

        # Entity embeddings
        num_entities = self._count_entities()
        self.embeddings = nn.Embedding(num_entities, embedding_dim)

        # Attention that respects graph structure
        self.structured_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mode='continuous',
            return_attention_weights=True
        )

    def _count_entities(self) -> int:
        """Count unique entities in the knowledge graph."""
        # Scan all relation tensors to find the maximum dimension
        max_size = 0
        # Check both learnable tensors and constants
        all_tensors = {**self.program.tensors, **self.program.constants}
        for name, tensor in all_tensors.items():
            if tensor.numel() > 0 and len(tensor.shape) >= 2:
                max_size = max(max_size, tensor.shape[0], tensor.shape[1])
        # max_size is already the number of entities (e.g., 10x10 tensor = 10 entities)
        return max(max_size, 1)

    def get_knowledge_mask(self, relation: str) -> torch.Tensor:
        """
        Create attention mask from knowledge graph structure.

        Args:
            relation: Name of relation to use as mask

        Returns:
            mask: [1, seq_len, seq_len] where True = masked out (disallowed)
        """
        # Get tensor from either tensors or constants
        if relation in self.program.tensors:
            adjacency = self.program.tensors[relation]
        elif relation in self.program.constants:
            adjacency = self.program.constants[relation]
        else:
            raise ValueError(f"Relation '{relation}' not found in program")

        # Convert to boolean mask (mask out where relation doesn't exist)
        # Invert logic: True means no connection, so mask out
        if self.program.mode == 'continuous':
            mask = adjacency <= 0.5
        else:
            mask = ~adjacency.bool()

        # Add batch dimension
        return mask.unsqueeze(0)

    def forward(
        self,
        entity_ids: torch.Tensor,
        relation_for_mask: str = 'connected'
    ) -> torch.Tensor:
        """
        Apply attention constrained by knowledge graph.

        Args:
            entity_ids: [batch_size, seq_len] entity indices
            relation_for_mask: Which relation to use for masking

        Returns:
            output: [batch_size, seq_len, embedding_dim]
        """
        # Get embeddings
        x = self.embeddings(entity_ids)  # [B, L, D]

        # Get mask from knowledge graph
        mask = self.get_knowledge_mask(relation_for_mask)

        # Apply structured attention
        output, weights = self.structured_attention(x, mask=mask)

        return output, weights


def demo_relational_reasoning():
    """Demo: Discover relations via attention patterns."""
    print("\n" + "="*60)
    print("Demo: Relational Reasoning with Transformers")
    print("="*60)

    # Create embedding space with some relations
    num_entities = 20
    embedding_dim = 64
    space = EmbeddingSpace(
        num_objects=num_entities,
        embedding_dim=embedding_dim,
        temperature=0.1
    )

    # Add known relations
    space.add_relation('parent', init='random')
    space.add_relation('child', init='random')
    space.add_relation('sibling', init='random')

    # Create relational transformer
    model = RelationalTransformer(
        embedding_space=space,
        num_heads=4,  # Multiple heads for relation discovery
        num_layers=2,
        mode='continuous'
    )

    # Example: sequence of family members
    batch_size = 2
    seq_len = 5
    entity_seq = torch.randint(0, num_entities, (batch_size, seq_len))

    print(f"\nInput: {batch_size} sequences of {seq_len} entities")
    print(f"Entity indices shape: {entity_seq.shape}")

    # Forward pass
    output, attention_weights = model(entity_seq)

    print(f"\nOutput embeddings shape: {output.shape}")
    print(f"Number of layers with attention: {len(attention_weights)}")

    # Discover what relations the attention learned
    for layer_idx, attn in enumerate(attention_weights):
        print(f"\nLayer {layer_idx} attention shape: {attn.shape}")

        # Average over batch for visualization
        avg_attention = attn.mean(dim=0)  # [num_heads, seq_len, seq_len]

        # Each head might learn a different relation
        for head_idx in range(model.num_heads):
            head_attn = avg_attention[head_idx].detach().numpy()

            # Find strongest attention pattern
            max_val = head_attn.max()
            import numpy as np
            max_idx = np.argmax(head_attn)
            max_pos = np.unravel_index(max_idx, head_attn.shape)
            print(f"  Head {head_idx}: Strongest attention from position {max_pos[0]} to {max_pos[1]} (score: {max_val:.3f})")

    # Map attention to relation types
    relation_scores = model.discover_relations(attention_weights[0])
    print(f"\nRelation discovery scores shape: {relation_scores.shape}")

    # Find most likely relation for each pair
    best_relations = relation_scores.argmax(dim=-1)  # [B, L, L]
    relation_names = list(space.relations.keys())

    print("\nDiscovered relations (first sequence):")
    for i in range(min(3, seq_len)):  # Show first 3 positions
        for j in range(min(3, seq_len)):
            if i != j:
                rel_idx = best_relations[0, i, j].item()
                rel_name = relation_names[rel_idx] if rel_idx < len(relation_names) else 'unknown'
                score = relation_scores[0, i, j, rel_idx].item()
                print(f"  Entity {i} -> Entity {j}: {rel_name} (score: {score:.3f})")


def demo_knowledge_constrained_attention():
    """Demo: Use knowledge graph to constrain attention."""
    print("\n" + "="*60)
    print("Demo: Knowledge Graph Constrained Attention")
    print("="*60)

    # Create a simple knowledge graph
    program = TensorProgram(mode='boolean')

    # Add a connectivity relation (sparse graph)
    num_nodes = 10
    connected = torch.zeros(num_nodes, num_nodes, dtype=torch.int8)
    # Create some connections
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (3, 6), (6, 7), (7, 8), (8, 9)]
    for i, j in edges:
        connected[i, j] = 1
        connected[j, i] = 1  # Symmetric

    program.add_tensor('connected', data=connected)

    print(f"\nKnowledge graph: {num_nodes} nodes, {len(edges)} edges")
    print("Edges:", edges)

    # Create transformer that respects graph structure
    model = KnowledgeGraphTransformer(
        program=program,
        embedding_dim=32,
        num_heads=4
    )

    # Process a sequence of nodes
    batch_size = 1
    seq_len = num_nodes
    node_sequence = torch.arange(num_nodes).unsqueeze(0)  # [1, 10]

    print(f"\nProcessing sequence of all {num_nodes} nodes")

    # Forward pass with graph-constrained attention
    output, weights = model(node_sequence, relation_for_mask='connected')

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Visualize how graph structure constrains attention
    mask = model.get_knowledge_mask('connected').squeeze(0)
    print(f"\nGraph mask shape: {mask.shape}")
    print(f"Number of allowed attention connections: {mask.sum().item()}")
    print(f"Total possible connections: {mask.numel()}")

    # Show sparsity
    sparsity = 1 - (mask.sum().float() / mask.numel()).item()
    print(f"Attention sparsity due to graph structure: {sparsity:.1%}")

    # Example: attention from node 0
    attn_from_0 = weights[0, 0, 0, :].detach()  # First head, from node 0
    print(f"\nAttention from node 0 (first head):")
    for i, score in enumerate(attn_from_0):
        if score > 0.01:  # Only show significant attention
            is_connected = connected[0, i].item() > 0
            print(f"  To node {i}: {score:.3f} {'(connected)' if is_connected else '(not directly connected)'}")


def demo_sequential_reasoning():
    """Demo: Sequential reasoning with positional encoding."""
    print("\n" + "="*60)
    print("Demo: Sequential Reasoning with Transformers")
    print("="*60)

    # Create a sequence of logical facts
    embedding_dim = 128
    seq_len = 10

    # Use sinusoidal positional encoding for order awareness
    pos_encoder = SinusoidalPositionalEncoding(
        d_model=embedding_dim,
        max_len=100
    )

    # Create transformer encoder
    encoder = TransformerEncoder(
        num_layers=2,
        d_model=embedding_dim,
        nhead=4,
        dim_feedforward=256,
        mode='continuous'
    )

    # Example: sequence of facts to reason over
    batch_size = 1
    fact_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    print(f"\nProcessing {seq_len} sequential facts")
    print(f"Input shape: {fact_embeddings.shape}")

    # Add positional encoding
    fact_embeddings_pos = pos_encoder(fact_embeddings)

    # Apply causal mask for autoregressive reasoning
    mask = causal_mask(seq_len)
    print(f"Causal mask shape: {mask.shape}")

    # Forward pass
    output, attention_weights = encoder(
        fact_embeddings_pos,
        src_mask=mask,
        return_attention=True
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")

    # Analyze attention patterns
    last_layer_attn = attention_weights[-1][0]  # [num_heads, seq_len, seq_len]

    print("\nAttention pattern analysis (last layer):")
    for head_idx in range(last_layer_attn.shape[0]):
        head_attn = last_layer_attn[head_idx]

        # Compute attention distance (how far back each position looks)
        distances = []
        for i in range(1, seq_len):  # Skip first position
            # Find where position i attends most
            attn_scores = head_attn[i, :i+1]  # Can only look at previous positions
            if len(attn_scores) > 0:
                max_pos = attn_scores.argmax().item()
                distance = i - max_pos
                distances.append(distance)

        if distances:
            avg_distance = sum(distances) / len(distances)
            print(f"  Head {head_idx}: Average attention distance = {avg_distance:.1f} positions")


def main():
    """Run all demos."""

    # 1. Relational reasoning
    demo_relational_reasoning()

    # 2. Knowledge graph constraints
    demo_knowledge_constrained_attention()

    # 3. Sequential reasoning
    demo_sequential_reasoning()

    print("\n" + "="*60)
    print("Transformer Integration Complete!")
    print("="*60)
    print("\nKey insights demonstrated:")
    print("1. Attention heads can discover different relation types")
    print("2. Knowledge graphs can constrain attention patterns")
    print("3. Positional encoding enables sequential reasoning")
    print("4. Transformers integrate seamlessly with TensorLogic components")


if __name__ == "__main__":
    main()
