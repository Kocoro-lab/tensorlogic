#!/usr/bin/env python3
"""
RNN/LSTM sequence reasoning with TensorLogic.

This example demonstrates:
1. Using RNNs (SimpleRNN, LSTM, GRU) for temporal reasoning
2. Combining RNNs with symbolic logic for rule-based sequence processing
3. Learning temporal patterns and dependencies
4. Visualizing RNN hidden states as evolving embeddings

Key insight: RNN hidden states can be viewed as temporal embeddings that
evolve according to tensor equations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

# TensorLogic imports
from tensorlogic.core import TensorProgram
from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.transformers import (
    SimpleRNN,
    LSTM,
    GRU,
    BidirectionalWrapper,
    export_rnn_as_equations
)
from tensorlogic.learn.trainer import EmbeddingTrainer


class TemporalReasoningRNN(nn.Module):
    """
    RNN that learns temporal patterns in logical sequences.

    Combines RNN with TensorLogic's reasoning capabilities for
    processing sequences of logical facts.
    """

    def __init__(
        self,
        embedding_space: EmbeddingSpace,
        rnn_type: str = 'lstm',
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = False,
        mode: str = 'continuous'
    ):
        super().__init__()
        self.space = embedding_space
        self.input_size = embedding_space.embedding_dim
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        # Choose RNN type
        if rnn_type == 'lstm':
            rnn_class = LSTM
        elif rnn_type == 'gru':
            rnn_class = GRU
        else:
            rnn_class = SimpleRNN

        # Create RNN
        self.rnn = rnn_class(
            input_size=self.input_size,
            hidden_size=hidden_size,
            mode=mode
        )

        # Optionally make bidirectional
        if bidirectional:
            self.rnn = BidirectionalWrapper(self.rnn, merge_mode='concat')
            output_size = hidden_size * 2
        else:
            output_size = hidden_size

        # Output projection to predict next entity/relation
        self.output_projection = nn.Linear(output_size, embedding_space.num_objects)

        # Optional: Learn temporal relations
        self.temporal_relation = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )

    def forward(
        self,
        entity_sequence: torch.Tensor,
        return_all_hiddens: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence of entities through RNN.

        Args:
            entity_sequence: [batch_size, seq_len] entity indices
            return_all_hiddens: If True, return all hidden states

        Returns:
            predictions: [batch_size, seq_len, num_objects] next entity predictions
            hidden_states: RNN hidden states
        """
        # Get entity embeddings
        batch_size, seq_len = entity_sequence.shape
        embeddings = self.space.object_embeddings[entity_sequence]  # [B, L, D]

        # Process through RNN
        if isinstance(self.rnn, BidirectionalWrapper):
            rnn_output = self.rnn(embeddings)  # [B, L, H*2] or [B, L, H]
            hidden_states = None  # Bidirectional doesn't return single hidden
        elif self.rnn_type == 'lstm':
            rnn_output, (h_n, c_n) = self.rnn(embeddings, return_sequences=True)
            hidden_states = (h_n, c_n)
        else:
            rnn_output, h_n = self.rnn(embeddings, return_sequences=True)
            hidden_states = h_n

        # Predict next entities
        predictions = self.output_projection(rnn_output)

        if return_all_hiddens:
            return predictions, rnn_output
        else:
            return predictions, hidden_states

    def reason_temporally(
        self,
        fact_sequence: List[Tuple[int, str, int]]
    ) -> torch.Tensor:
        """
        Apply temporal reasoning over a sequence of facts.

        Args:
            fact_sequence: List of (subject, relation, object) tuples

        Returns:
            temporal_scores: Scores for temporal consistency
        """
        # Extract entities from facts
        entities = []
        for subj, rel, obj in fact_sequence:
            entities.extend([subj, obj])

        entity_tensor = torch.tensor(entities).unsqueeze(0)  # [1, seq_len]

        # Get RNN representations
        _, all_hiddens = self.forward(entity_tensor, return_all_hiddens=True)

        # Compute temporal consistency scores
        # Score = h[t]^T × R_temporal × h[t+1]
        scores = []
        for t in range(all_hiddens.shape[1] - 1):
            h_t = all_hiddens[0, t, :]      # [H]
            h_next = all_hiddens[0, t+1, :]  # [H]

            # Bilinear scoring with temporal relation
            score = h_t @ self.temporal_relation @ h_next
            scores.append(score.item())

        return torch.tensor(scores)


class SymbolicSequenceRNN(nn.Module):
    """
    RNN that combines with symbolic rules for sequence processing.

    Uses TensorProgram rules to constrain RNN predictions.
    """

    def __init__(
        self,
        program: TensorProgram,
        vocab_size: int,
        hidden_size: int = 64,
        rnn_type: str = 'gru'
    ):
        super().__init__()
        self.program = program
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding layer for symbolic tokens
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Choose RNN
        if rnn_type == 'lstm':
            self.rnn = LSTM(hidden_size, hidden_size, mode='continuous')
        elif rnn_type == 'gru':
            self.rnn = GRU(hidden_size, hidden_size, mode='continuous')
        else:
            self.rnn = SimpleRNN(hidden_size, hidden_size, mode='continuous')

        self.rnn_type = rnn_type

        # Output layer
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, mask_invalid: bool = True):
        """
        Forward pass with optional symbolic constraint masking.

        Args:
            x: [batch_size, seq_len] input sequence
            mask_invalid: If True, use program rules to mask invalid predictions

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed input
        embedded = self.embedding(x)  # [B, L, H]

        # RNN forward
        if self.rnn_type == 'lstm':
            rnn_out, _ = self.rnn(embedded, return_sequences=True)
        else:
            rnn_out, _ = self.rnn(embedded, return_sequences=True)

        # Project to vocabulary
        logits = self.output(rnn_out)  # [B, L, V]

        # Apply symbolic constraints
        if mask_invalid:
            logits = self.apply_symbolic_constraints(logits, x)

        return logits

    def apply_symbolic_constraints(
        self,
        logits: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Use TensorProgram rules to mask invalid next tokens.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            context: [batch_size, seq_len] context sequence

        Returns:
            Masked logits where invalid transitions have -inf
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Get valid transitions from program
        # This is a simplified example - in practice you'd query the program
        # for valid next states given the current state

        # Example: Create transition mask based on symbolic rules
        mask = torch.ones_like(logits, dtype=torch.bool)

        # Apply some example rules (simplified)
        for b in range(batch_size):
            for t in range(seq_len):
                if t > 0:
                    prev_token = context[b, t-1].item()
                    # Example rule: token 0 can only be followed by tokens 1 or 2
                    if prev_token == 0:
                        mask[b, t, 3:] = False  # Mask out tokens 3 and above

        # Apply mask
        logits = logits.masked_fill(~mask, float('-inf'))

        return logits


def demo_temporal_reasoning():
    """Demo: Temporal reasoning with RNN over entity sequences."""
    print("\n" + "="*60)
    print("Demo: Temporal Reasoning with RNN")
    print("="*60)

    # Create embedding space
    num_entities = 10
    embedding_dim = 32
    space = EmbeddingSpace(
        num_objects=num_entities,
        embedding_dim=embedding_dim
    )

    # Add some relations
    space.add_relation('follows', init='random')
    space.add_relation('causes', init='random')

    # Create temporal reasoning RNN
    model = TemporalReasoningRNN(
        embedding_space=space,
        rnn_type='lstm',
        hidden_size=64,
        bidirectional=False
    )

    print(f"\nCreated LSTM for temporal reasoning")
    print(f"Input: Entity embeddings ({embedding_dim}D)")
    print(f"Hidden size: 64")
    print(f"Output: Next entity predictions ({num_entities} classes)")

    # Example sequence of entities
    batch_size = 2
    seq_len = 8
    entity_seq = torch.randint(0, num_entities, (batch_size, seq_len))

    print(f"\nProcessing {batch_size} sequences of length {seq_len}")

    # Forward pass
    predictions, hidden_states = model(entity_seq)

    print(f"Predictions shape: {predictions.shape}")
    if hidden_states is not None:
        if isinstance(hidden_states, tuple):  # LSTM
            h, c = hidden_states
            print(f"Final hidden state shape: {h.shape}")
            print(f"Final cell state shape: {c.shape}")
        else:
            print(f"Final hidden state shape: {hidden_states.shape}")

    # Temporal reasoning over facts
    fact_sequence = [
        (0, 'follows', 1),
        (1, 'causes', 2),
        (2, 'follows', 3)
    ]

    print("\n\nTemporal reasoning over fact sequence:")
    for fact in fact_sequence:
        print(f"  {fact}")

    temporal_scores = model.reason_temporally(fact_sequence)
    print(f"\nTemporal consistency scores: {temporal_scores}")

    # Export as equations
    equations = export_rnn_as_equations(model.rnn)
    print("\n\nRNN as Tensor Equations:")
    for eq in equations:
        print(f"  {eq}")


def demo_symbolic_constrained_rnn():
    """Demo: RNN with symbolic constraints from TensorProgram."""
    print("\n" + "="*60)
    print("Demo: Symbolically Constrained RNN")
    print("="*60)

    # Create a simple program with rules
    program = TensorProgram(mode='boolean')

    # Add some transition rules (simplified)
    # In practice, these would define valid state transitions
    vocab_size = 10
    transitions = torch.zeros(vocab_size, vocab_size, dtype=torch.int8)
    # Example rules: 0->1, 1->2, 2->0 (cyclic), 3->4, etc.
    transitions[0, 1] = 1
    transitions[1, 2] = 1
    transitions[2, 0] = 1
    transitions[3, 4] = 1
    transitions[4, 5] = 1

    program.add_tensor('valid_transitions', data=transitions)

    # Create constrained RNN
    model = SymbolicSequenceRNN(
        program=program,
        vocab_size=vocab_size,
        hidden_size=32,
        rnn_type='gru'
    )

    print(f"\nCreated GRU with symbolic constraints")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Hidden size: 32")

    # Example input sequence
    batch_size = 1
    seq_len = 6
    input_seq = torch.tensor([[0, 1, 2, 0, 1, 2]])  # Following valid pattern

    print(f"\nInput sequence: {input_seq[0].tolist()}")

    # Forward without constraints
    logits_unconstrained = model(input_seq, mask_invalid=False)
    probs_unconstrained = F.softmax(logits_unconstrained[0], dim=-1)

    print("\nUnconstrained predictions (top-3 per position):")
    for t in range(seq_len):
        top3_probs, top3_idx = probs_unconstrained[t].topk(3)
        print(f"  Position {t}: {[(idx.item(), prob.item()) for idx, prob in zip(top3_idx, top3_probs)]}")

    # Forward with constraints
    logits_constrained = model(input_seq, mask_invalid=True)
    probs_constrained = F.softmax(logits_constrained[0], dim=-1)

    print("\nConstrained predictions (top-3 per position):")
    for t in range(seq_len):
        top3_probs, top3_idx = probs_constrained[t].topk(3)
        valid_preds = [(idx.item(), prob.item()) for idx, prob in zip(top3_idx, top3_probs) if not torch.isinf(logits_constrained[0, t, idx])]
        print(f"  Position {t}: {valid_preds}")


def demo_bidirectional_lstm():
    """Demo: Bidirectional LSTM for context-aware sequence processing."""
    print("\n" + "="*60)
    print("Demo: Bidirectional LSTM")
    print("="*60)

    # Create a simple LSTM
    input_size = 16
    hidden_size = 32
    lstm = LSTM(input_size, hidden_size, mode='continuous')

    # Make it bidirectional
    bi_lstm = BidirectionalWrapper(lstm, merge_mode='concat')

    print(f"Created Bidirectional LSTM")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size} (per direction)")
    print(f"Output size: {hidden_size * 2} (concatenated)")

    # Example input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_size)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = bi_lstm(x)

    print(f"Output shape: {output.shape}")
    print(f"Processing both forward and backward directions")

    # Compare with unidirectional
    uni_output, _ = lstm(x, return_sequences=True)
    print(f"\nUnidirectional output shape: {uni_output.shape}")

    # Show that bidirectional captures more context
    print("\nBidirectional advantages:")
    print("  - Sees both past and future context")
    print("  - Better for tasks like sequence labeling")
    print("  - Doubles the representational capacity")


def demo_boolean_mode_rnn():
    """Demo: RNN in boolean mode for discrete reasoning."""
    print("\n" + "="*60)
    print("Demo: Boolean Mode RNN")
    print("="*60)

    # Create RNN in boolean mode
    input_size = 8
    hidden_size = 16
    rnn = SimpleRNN(
        input_size,
        hidden_size,
        mode='boolean',  # Boolean mode!
        activation='sigmoid'  # Use sigmoid for thresholding
    )

    print(f"Created Boolean RNN")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Mode: boolean (outputs are 0 or 1)")

    # Binary input
    batch_size = 1
    seq_len = 5
    x = (torch.randn(batch_size, seq_len, input_size) > 0).float()

    print(f"\nBinary input shape: {x.shape}")
    print(f"Input values: {torch.unique(x).tolist()}")

    # Forward pass
    output, final_hidden = rnn(x, return_sequences=True)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output values: {torch.unique(output).tolist()}")
    print(f"Final hidden shape: {final_hidden.shape}")
    print(f"Final hidden values: {torch.unique(final_hidden).tolist()}")

    # Show discrete state transitions
    print("\nDiscrete state evolution:")
    for t in range(seq_len):
        state_vector = output[0, t, :8]  # First 8 dims
        binary_state = ''.join([str(int(s.item())) for s in state_vector])
        print(f"  Time {t}: {binary_state}")


def main():
    """Run all RNN demos."""

    # 1. Temporal reasoning with LSTM
    demo_temporal_reasoning()

    # 2. Symbolic constraints with GRU
    demo_symbolic_constrained_rnn()

    # 3. Bidirectional LSTM
    demo_bidirectional_lstm()

    # 4. Boolean mode RNN
    demo_boolean_mode_rnn()

    print("\n" + "="*60)
    print("RNN/LSTM Integration Complete!")
    print("="*60)
    print("\nKey insights demonstrated:")
    print("1. RNNs as tensor equations with temporal indices")
    print("2. Combining RNNs with symbolic constraints")
    print("3. Bidirectional processing for full context")
    print("4. Boolean mode for discrete state machines")
    print("5. Temporal reasoning over fact sequences")


if __name__ == "__main__":
    main()
