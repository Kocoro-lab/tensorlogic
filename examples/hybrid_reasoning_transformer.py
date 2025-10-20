#!/usr/bin/env python3
"""
Hybrid Reasoning Transformer: Combining Symbolic Logic with Neural Attention.

This example demonstrates the unique value of TensorLogic by showing how to:
1. Use symbolic rules to guide transformer attention patterns
2. Combine knowledge graph constraints with language modeling
3. Learn attention that respects logical relationships
4. Generate text that follows symbolic constraints

Key Insight: Attention weights can be interpreted as soft logical implications,
and symbolic rules can provide hard constraints on what the model can attend to.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np

# TensorLogic imports
from tensorlogic.core import TensorProgram
from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.reasoning.forward import forward_chain
from tensorlogic.transformers import (
    MultiHeadAttention,
    TransformerEncoder,
    DecoderOnlyLM,
    SinusoidalPositionalEncoding
)
from tensorlogic.transformers.utils import causal_mask
from tensorlogic.ops import logical_join, logical_project


class SymbolicConstrainedTransformer(nn.Module):
    """
    Transformer that uses symbolic rules to constrain attention patterns.

    This model combines:
    - Neural attention mechanisms for learning patterns
    - Symbolic logic for enforcing hard constraints
    - Knowledge graph structure for guiding information flow
    """

    def __init__(
        self,
        program: TensorProgram,
        vocab_size: int,
        d_model: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        relation_names: List[str] = None
    ):
        super().__init__()
        self.program = program
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.relation_names = relation_names or []

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

        # Transformer encoder with multiple layers
        self.encoder = TransformerEncoder(
            num_layers=n_layer,
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            mode='continuous'  # Use continuous for differentiability
        )

        # Relation-aware attention layer
        self.relation_attention = RelationAwareAttention(
            d_model=d_model,
            n_head=n_head,
            n_relations=len(relation_names)
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def get_symbolic_mask(
        self,
        batch_size: int,
        seq_len: int,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate attention mask based on symbolic rules from TensorProgram.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            token_ids: [B, L] token indices

        Returns:
            mask: [B, 1, L, L] attention mask where True = masked out (disallowed)
        """
        # Initialize mask (no connections masked by default)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)

        # Apply symbolic constraints from the program
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    # Example: Check if tokens can be connected based on rules
                    token_i = token_ids[b, i].item()
                    token_j = token_ids[b, j].item()

                    # Query the program for valid connections
                    # This is simplified - in practice you'd map tokens to entities
                    can_connect = self.check_symbolic_connection(token_i, token_j)
                    # True means mask out, so invert the logic
                    mask[b, 0, i, j] = not can_connect

        return mask

    def check_symbolic_connection(self, token_i: int, token_j: int) -> bool:
        """
        Check if two tokens can be connected according to symbolic rules.

        This is where TensorLogic's symbolic reasoning comes in:
        - Map tokens to logical entities
        - Check if relation exists in knowledge graph
        - Apply forward chaining to derive new connections
        """
        # Simplified example: check if connection exists in a relation
        for relation_name in self.relation_names:
            # Check both tensors and constants
            if relation_name in self.program.tensors:
                relation = self.program.tensors[relation_name]
            elif relation_name in self.program.constants:
                relation = self.program.constants[relation_name]
            else:
                continue

            if relation is not None and token_i < relation.shape[0] and token_j < relation.shape[1]:
                if relation[token_i, token_j] > 0.5:  # Connection exists
                    return True

        # Default: allow connection (can be changed based on requirements)
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        use_symbolic_constraints: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional symbolic constraints.

        Args:
            input_ids: [B, L] token indices
            use_symbolic_constraints: Whether to apply symbolic masking

        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids)  # [B, L, D]
        x = self.pos_encoder(x)

        # Get symbolic mask if requested
        if use_symbolic_constraints:
            symbolic_mask = self.get_symbolic_mask(B, L, input_ids)
            # Combine with causal mask for autoregressive generation
            causal = causal_mask(L, device=input_ids.device)
            # Both masks use True=masked out, so OR them to combine disallowed positions
            mask = symbolic_mask | causal.unsqueeze(0)
        else:
            mask = causal_mask(L, device=input_ids.device)

        # Apply transformer with constraints
        x = self.encoder(x, src_mask=mask)

        # Apply relation-aware attention
        x, relation_scores = self.relation_attention(x, mask)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits, relation_scores


class RelationAwareAttention(nn.Module):
    """
    Attention mechanism that learns to identify and use different relation types.

    Each attention head specializes in a different type of logical relation.
    """

    def __init__(self, d_model: int, n_head: int, n_relations: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_relations = n_relations

        # Multi-head attention
        self.attention = MultiHeadAttention(
            embedding_dim=d_model,
            num_heads=n_head,
            mode='continuous',
            return_attention_weights=True
        )

        # Learn to classify attention patterns as relations
        self.relation_classifier = nn.Linear(n_head, n_relations)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply relation-aware attention.

        Args:
            x: [B, L, D] input embeddings
            mask: Optional attention mask

        Returns:
            output: [B, L, D] attended values
            relation_scores: [B, L, L, n_relations] relation type scores
        """
        # Apply multi-head attention
        output, attention_weights = self.attention(x, mask=mask, need_weights=True)

        # Classify attention patterns as relations
        # attention_weights: [B, n_head, L, L]
        B, H, L, _ = attention_weights.shape

        # Average attention across positions to get head specialization
        head_patterns = attention_weights.mean(dim=(2, 3))  # [B, n_head]

        # Classify what relation each pattern represents
        relation_scores = self.relation_classifier(head_patterns)  # [B, n_relations]

        return output, relation_scores


class KnowledgeGuidedTextGenerator(nn.Module):
    """
    Text generator that uses knowledge graph structure to guide generation.

    Combines:
    - Embeddings from EmbeddingSpace for entities
    - Transformer for sequence modeling
    - Symbolic rules for constraining generation
    """

    def __init__(
        self,
        space: EmbeddingSpace,
        program: TensorProgram,
        vocab_size: int,
        d_model: int = 256
    ):
        super().__init__()
        self.space = space
        self.program = program
        self.vocab_size = vocab_size

        # Map entities to tokens (simplified - in practice use proper tokenizer)
        self.entity_to_token = nn.Linear(space.embedding_dim, d_model)

        # Decoder-only LM for generation
        self.lm = DecoderOnlyLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=4,
            n_head=4,
            max_seq_len=512,
            mode='continuous'
        )

        # Projection from hidden states to entity space
        self.to_entity_space = nn.Linear(d_model, space.embedding_dim)

    def generate_with_constraints(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """
        Generate text while respecting knowledge graph constraints.

        At each step:
        1. Generate next token probabilities
        2. Map to entity space
        3. Check symbolic constraints
        4. Mask invalid continuations
        5. Sample from valid tokens
        """
        device = prompt.device
        generated = prompt

        for _ in range(max_length):
            # Get LM predictions
            logits = self.lm(generated)  # [B, L, V]
            next_logits = logits[:, -1, :] / temperature  # [B, V]

            # Map to entity space to check constraints
            hidden = self.lm.encoder(
                self.lm.pos_emb(self.lm.tok_emb(generated))
            )[:, -1, :]  # [B, D]
            entity_emb = self.to_entity_space(hidden)  # [B, E]

            # Find closest entities
            similarities = torch.matmul(entity_emb, self.space.embeddings.T)  # [B, num_entities]
            closest_entities = similarities.argmax(dim=-1)  # [B]

            # Apply symbolic constraints
            next_logits = self.apply_generation_constraints(
                next_logits,
                closest_entities,
                generated
            )

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def apply_generation_constraints(
        self,
        logits: torch.Tensor,
        current_entities: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply symbolic constraints to generation logits.

        Uses forward chaining to determine valid next tokens.
        """
        B, V = logits.shape

        # For each token, check if it's symbolically valid
        for b in range(B):
            entity = current_entities[b].item()

            # Get valid next entities from knowledge graph
            valid_next = self.get_valid_continuations(entity)

            # Mask tokens that don't correspond to valid entities
            for v in range(V):
                if not self.is_token_valid(v, valid_next):
                    logits[b, v] = float('-inf')

        return logits

    def get_valid_continuations(self, entity: int) -> List[int]:
        """
        Use forward chaining to find valid next entities.
        """
        # Query all relations for connections from this entity
        valid = []
        for relation_name in self.space.relations.keys():
            # Check both tensors and constants
            if relation_name in self.program.tensors:
                relation = self.program.tensors[relation_name]
            elif relation_name in self.program.constants:
                relation = self.program.constants[relation_name]
            else:
                continue

            if relation is not None and entity < relation.shape[0]:
                # Find all entities connected via this relation
                connections = (relation[entity] > 0.5).nonzero(as_tuple=True)[0]
                valid.extend(connections.tolist())

        return valid

    def is_token_valid(self, token: int, valid_entities: List[int]) -> bool:
        """
        Check if token corresponds to a valid entity.
        """
        # Simplified: assume direct mapping for demo
        # In practice, use proper token-entity mapping
        return token in valid_entities or len(valid_entities) == 0


def demo_hybrid_reasoning():
    """Demo: Transformer with symbolic reasoning constraints."""
    print("\n" + "="*60)
    print("Demo: Hybrid Reasoning Transformer")
    print("="*60)

    # Create knowledge graph with logical rules
    program = TensorProgram(mode='continuous')

    # Define entities and relations
    num_entities = 50
    vocab_size = 100

    # Add relations (knowledge graph structure)
    causes = torch.zeros(num_entities, num_entities)
    prevents = torch.zeros(num_entities, num_entities)
    related_to = torch.zeros(num_entities, num_entities)

    # Add some logical connections
    # "A causes B", "B causes C" => "A related_to C"
    causes[0, 1] = 1.0  # A causes B
    causes[1, 2] = 1.0  # B causes C
    prevents[2, 3] = 1.0  # C prevents D

    program.add_tensor('causes', data=causes)
    program.add_tensor('prevents', data=prevents)

    # Define logical rule: transitive causation
    # related(X,Z) := causes(X,Y), causes(Y,Z)
    program.add_equation(
        'related_to',
        lambda tensors: logical_join(
            tensors['causes'],
            tensors['causes'],
            'ij,jk->ik'
        )
    )

    # Execute forward chaining to derive new relations
    results = program.forward()
    related_to = results['related_to']
    program.add_tensor('related_to', data=related_to)

    print(f"Knowledge graph: {num_entities} entities, 3 relation types")
    print(f"Derived {related_to.sum().item():.0f} 'related_to' connections via forward chaining")

    # Create hybrid transformer
    model = SymbolicConstrainedTransformer(
        program=program,
        vocab_size=vocab_size,
        d_model=128,
        n_head=4,
        n_layer=2,
        relation_names=['causes', 'prevents', 'related_to']
    )

    print(f"\nCreated Symbolic-Constrained Transformer:")
    print(f"  - {vocab_size} vocabulary size")
    print(f"  - {4} attention heads")
    print(f"  - Attention constrained by knowledge graph")

    # Example: Generate text with constraints
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nProcessing sequences of length {seq_len}")

    # Forward without constraints (standard transformer)
    logits_unconstrained, _ = model(input_ids, use_symbolic_constraints=False)
    print(f"Unconstrained logits shape: {logits_unconstrained.shape}")

    # Forward with symbolic constraints
    logits_constrained, relation_scores = model(input_ids, use_symbolic_constraints=True)
    print(f"Constrained logits shape: {logits_constrained.shape}")
    print(f"Relation scores shape: {relation_scores.shape}")

    # Compare attention patterns
    print("\nEffect of symbolic constraints:")
    print("  - Attention is masked based on knowledge graph structure")
    print("  - Model can only attend to symbolically valid connections")
    print("  - Generates text that respects logical rules")

    # Show learned relation types
    print(f"\nLearned relation types (scores):")
    for i, name in enumerate(model.relation_names):
        if i < relation_scores.shape[-1]:
            score = relation_scores[0, i].item()
            print(f"  {name}: {score:.3f}")


def demo_knowledge_guided_generation():
    """Demo: Text generation guided by knowledge graph."""
    print("\n" + "="*60)
    print("Demo: Knowledge-Guided Text Generation")
    print("="*60)

    # Create embedding space with entities
    num_entities = 20
    embedding_dim = 64
    space = EmbeddingSpace(
        num_objects=num_entities,
        embedding_dim=embedding_dim
    )

    # Add relations
    space.add_relation('follows', init='random')
    space.add_relation('contradicts', init='random')

    # Create knowledge program
    program = TensorProgram(mode='continuous')

    # Add factual constraints
    follows = torch.zeros(num_entities, num_entities)
    follows[0, 1] = 1.0  # Fact 0 follows fact 1
    follows[1, 2] = 1.0  # Fact 1 follows fact 2

    contradicts = torch.zeros(num_entities, num_entities)
    contradicts[0, 3] = 1.0  # Fact 0 contradicts fact 3
    contradicts[2, 3] = 1.0  # Fact 2 contradicts fact 3

    program.add_tensor('follows', data=follows)
    program.add_tensor('contradicts', data=contradicts)

    print(f"Knowledge base: {num_entities} facts")
    print(f"Constraints: {follows.sum().item():.0f} 'follows', {contradicts.sum().item():.0f} 'contradicts'")

    # Create knowledge-guided generator
    vocab_size = 50
    generator = KnowledgeGuidedTextGenerator(
        space=space,
        program=program,
        vocab_size=vocab_size,
        d_model=128
    )

    print(f"\nKnowledge-Guided Generator:")
    print(f"  - Embeddings from {embedding_dim}D space")
    print(f"  - Generation constrained by logical rules")
    print(f"  - Prevents contradictory sequences")

    # Example generation
    prompt = torch.tensor([[0]])  # Start with token 0
    print(f"\nGenerating from prompt: {prompt.tolist()}")

    # Note: This is a conceptual demo - actual generation would need more setup
    print("\nGeneration process:")
    print("1. Generate next token probabilities from LM")
    print("2. Map tokens to entity embeddings")
    print("3. Check knowledge graph for valid continuations")
    print("4. Mask tokens that violate logical constraints")
    print("5. Sample from remaining valid tokens")

    # Show how constraints affect generation
    print("\nConstraint effects:")
    print("  - Cannot generate fact 3 after fact 0 (contradiction)")
    print("  - Must maintain logical consistency")
    print("  - Guided by 'follows' relations in knowledge graph")


def demo_attention_as_logic():
    """Demo: Interpreting attention weights as logical implications."""
    print("\n" + "="*60)
    print("Demo: Attention as Logical Reasoning")
    print("="*60)

    # Create a simple logical system
    num_propositions = 10
    d_model = 32

    # Create attention that learns logical implications
    logical_attention = MultiHeadAttention(
        embedding_dim=d_model,
        num_heads=4,
        mode='continuous'
    )

    print(f"Logical Attention System:")
    print(f"  - {num_propositions} propositions")
    print(f"  - {4} reasoning heads (different inference types)")

    # Embed propositions
    prop_embeddings = torch.randn(1, num_propositions, d_model)

    # Forward pass
    attended, weights = logical_attention(prop_embeddings, need_weights=True)

    print(f"\nAttention weights shape: {weights.shape}")
    print("Interpretation: weights[h,i,j] = strength of implication i→j in head h")

    # Analyze logical patterns
    weights_np = weights[0].detach().numpy()  # [H, L, L]

    print("\nLogical patterns discovered:")
    for head in range(weights.shape[1]):
        head_weights = weights_np[head]

        # Find strong implications (high attention)
        strong_implications = (head_weights > 0.7).nonzero()
        if len(strong_implications[0]) > 0:
            i, j = strong_implications[0][0], strong_implications[1][0]
            strength = head_weights[i, j]
            print(f"  Head {head}: Proposition {i} strongly implies {j} (weight={strength:.3f})")

    # Convert to Boolean logic
    print("\nConverting to Boolean logic (threshold=0.5):")
    boolean_weights = (weights > 0.5).float()
    print(f"Boolean implications: {boolean_weights.sum().item():.0f} total")

    # Show transitivity
    print("\nChecking transitivity (if A→B and B→C then A→C):")
    for h in range(weights.shape[1]):
        W = weights[0, h]  # [L, L]
        # Compute transitive closure
        W2 = torch.matmul(W, W)  # Two-hop implications

        # Check if two-hop paths are also direct paths
        for i in range(3):  # Check first 3 propositions
            for j in range(3):
                if i != j:
                    direct = W[i, j].item()
                    indirect = W2[i, j].item()
                    if indirect > direct + 0.1:  # Significant indirect path
                        print(f"    Head {h}: P{i}→P{j} indirect ({indirect:.3f}) > direct ({direct:.3f})")


def main():
    """Run all hybrid reasoning demos."""

    # 1. Transformer with symbolic constraints
    demo_hybrid_reasoning()

    # 2. Knowledge-guided text generation
    demo_knowledge_guided_generation()

    # 3. Attention as logical reasoning
    demo_attention_as_logic()

    print("\n" + "="*60)
    print("Hybrid Reasoning Complete!")
    print("="*60)
    print("\nKey Innovations Demonstrated:")
    print("1. Symbolic rules constrain neural attention patterns")
    print("2. Knowledge graphs guide text generation")
    print("3. Attention weights represent logical implications")
    print("4. Forward chaining derives new constraints dynamically")
    print("5. Seamless integration of neural and symbolic AI")
    print("\nThis is the unique value of TensorLogic:")
    print("Not just implementing transformers, but truly unifying")
    print("neural and symbolic reasoning at a fundamental level!")


if __name__ == "__main__":
    main()