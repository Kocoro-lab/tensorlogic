# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Setup
```bash
# Development installation (recommended for contributors)
pip install -e .

# Or manual dependency installation
pip install -r requirements.txt
```

### Running Examples
After development install with `pip install -e .`, run examples directly:
```bash
python3 examples/family_tree_symbolic.py          # Boolean reasoning
python3 examples/family_tree_embedding.py         # Embedding learning
python3 examples/learnable_demo.py                # Learnable composition
python3 examples/pattern_discovery_demo.py        # Self-supervised pattern discovery
python3 examples/predicate_invention_demo.py      # RESCAL predicate invention
python3 examples/benchmark_suite.py               # Performance benchmarks
python3 examples/three_body_kb_demo.py            # Real-world knowledge base from text
python3 examples/transformer_reasoning_demo.py    # Transformers + KG constraints + relation discovery
python3 examples/rnn_sequence_reasoning_demo.py   # RNN/LSTM/GRU temporal reasoning + symbolic masks
# Shakespeare character-level language model (nanoGPT-like):
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py             # Train on input.txt
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py --checkpoint checkpoints/shakespeare/best.pt
```

Or with manual install, set `PYTHONPATH`:
```bash
PYTHONPATH=. python3 examples/<name>.py
```

### Testing & Validation
```bash
# Run all examples to validate the installation
for example in examples/*.py; do python3 "$example"; done

# Run a specific example with output capture
python3 examples/benchmark_suite.py > results.txt 2>&1
```

Note: `examples/three_body_kb_demo.py` reads an external local text file. Skip it or adjust the input path if the file is not available on your machine.

## Architecture Overview

### Core Design Principle
Tensorlogic implements a dual‑mode tensor‑based reasoning framework that unifies neural and symbolic AI. It supports both **Boolean mode** (binary logical operations) and **continuous mode** (differentiable, fuzzy reasoning). This lets systems learn from noisy data while maintaining interpretability.

### Module Structure and Data Flow

**Core Layer** (`tensorlogic/core/`)
- `TensorProgram`: Base abstraction for both Boolean and differentiable programs; stores tensors (facts/relations) and equations (rules)
- `TensorWrapper`: Helper that wraps tensors with metadata to track Boolean vs. continuous mode; optional in current pipelines (modules primarily use raw `torch.Tensor`).

**Operations Layer** (`tensorlogic/ops/`)
- Implements logical operations: `logical_join`, `logical_project`, `logical_union`, `logical_negation` using Einstein summation (`torch.einsum`)
- Abstracts away dimensional complexity—same operations handle 2D matrices, 3D tensors, or arbitrary‑rank tensors uniformly
- Supports both Boolean and continuous modes. In Boolean mode results are thresholded to 0/1. In continuous mode operations return real‑valued tensors; use nonlinearities (e.g., sigmoid) or thresholds when probabilities are desired. Mode is controlled by the caller (e.g., via `TensorProgram.mode` or an explicit `mode` argument).

**Reasoning Layer** (`tensorlogic/reasoning/`)
- **EmbeddingSpace** (`embed.py`): Maps objects to learned vectors, relations to learnable matrices; enables analogical reasoning via bilinear scoring `s^T W_r o`
- **Forward/Backward Chaining** (`forward.py`, `backward.py`): Traditional logical inference mechanisms that execute rules in sequence, supporting deductive reasoning and simple goal‑directed queries (abduction is not implemented)
- **Composition** (`composer.py`): GatedMultiHopComposer learns multi-hop relation paths with learned attention gates; enables "grandparent = parent ∘ parent" discovery
- **Closure** (`closure.py`): Computes transitive closures and implicit relations (e.g., inferring "sister" from "parent" facts)
- **Predicate Invention** (`predicate_invention.py`): RESCAL-based tensor factorization discovers hidden relations and latent structure in knowledge bases
- **Decomposition** (`decomposition.py`): Breaks complex predicates into simpler components for hierarchical reasoning

**Learning Layer** (`tensorlogic/learn/`)
- **Trainers** (`trainer.py`): Generic nn.Module trainers supporting embedding‑space and pair‑scoring learning (see `EmbeddingTrainer` and `PairScoringTrainer`); orchestrate data loading, forward passes, gradient computation, and parameter updates
- **Losses** (`losses.py`): Contrastive loss, MSE, BCE—all differentiable through the entire computational graph
- **Curriculum** (`curriculum.py`): Progressive training strategies that start with simple facts and gradually introduce complex reasoning tasks

**Utilities Layer** (`tensorlogic/utils/`)
- **Visualization** (`visualization.py`, `viz_helper.py`): 2D/3D visualization of learned embeddings and reasoning paths
- **Diagnostics** (`diagnostics.py`): Monitor gradient health, loss trends, and convergence during training
- **Init Strategies** (`init_strategies.py`): Specialized initialization schemes for embeddings and relation matrices
- **Sparse Utils** (`sparse.py`): Efficient handling of sparse fact tensors (memory-optimized)
- **I/O** (`io.py`): Save/load models preserving both symbolic structure and learned parameters

**Transformers & Recurrent Layer** (`tensorlogic/transformers/`)
- Attention as tensor equations (`tensor_equations.py`):
  - Scores: `einsum('bhid,bhjd->bhij', Q, K) / sqrt(d)`
  - Weights: softmax (continuous) or argmax one-hot (boolean); masking supported
  - Apply: `einsum('bhij,bhjd->bhid', A, V)`
- Components:
  - `MultiHeadAttention` (scaled dot-product), boolean/continuous modes
  - Positional encodings: `SinusoidalPositionalEncoding`, `LearnedPositionalEncoding`
  - Encoder/Decoder/Full Transformer in `layers.py`
  - `DecoderOnlyLM` (GPT-like) in `lm.py` with tied embeddings and generation
  - Recurrent modules in `recurrent.py`: `SimpleRNN`, `LSTM`, `GRU`, `BidirectionalWrapper`
  - Utilities: `utils.causal_mask(L)` for autoregressive masking

Export tensor equations (for transparency and audit):
```python
from tensorlogic.transformers import Transformer, export_rnn_as_equations

eqs = Transformer().to_tensor_equations()
print('\n'.join(eqs))

# For RNNs
from tensorlogic.transformers import LSTM
lstm = LSTM(input_size=128, hidden_size=256)
print('\n'.join(export_rnn_as_equations(lstm)))
```

### Data Flow Example: From Input to Reasoning

1. User creates a `TensorProgram` and adds tensors (facts as Boolean or continuous tensors) and equations (symbolic rules)
2. For **symbolic reasoning**: TensorProgram executes equations using ops layer (logical_join, logical_project) with forward/backward chaining
3. For **embedding-space reasoning**: Objects are represented as vectors in EmbeddingSpace, relations as learned matrices; queries scored via bilinear form
4. For **multi-hop**: GatedMultiHopComposer applies learned attention gates across relation sequences
5. During **training**: Trainer feeds data through chosen reasoning mechanism, computes loss (contrastive/MSE/BCE), backpropagates through entire computation graph via PyTorch autograd
6. **Integration**: All mechanisms can be combined—hybrid systems use both hand-crafted symbolic rules and learned embeddings in parallel, then combine predictions

### Key Design Patterns

- **Mode Abstraction**: Operations and tensors handle Boolean vs. continuous seamlessly; no need for separate code paths
- **Universal Einstein Summation Interface**: All logical operations (join, project) reduce to `torch.einsum` calls, enabling consistent handling of arbitrary tensor ranks
- **Trainable Modules as Parameters**: Relations are nn.Parameter objects; program weights are differentiable and optimizable via standard PyTorch optimizers
- **Extensibility via Inheritance**: Custom reasoning methods inherit from a Trainer base class; custom losses extend from base loss classes
- **Composable Primitives**: Forward chaining, embeddings, composers, and predicate invention are independent but composable—users can chain them together as needed

## Important Code Patterns & Conventions

### Working with TensorProgram
```python
from tensorlogic.core.program import TensorProgram

# Boolean mode (strict logic)
prog = TensorProgram(mode='boolean', device='cpu')
prog.add_tensor("parent", data=torch.tensor([[1, 0], [0, 1]], dtype=torch.int8))

# Continuous mode (learnable)
prog = TensorProgram(mode='continuous', device='cuda')
prog.add_tensor("similarity", data=torch.randn(10, 10, requires_grad=True))
```

### Working with EmbeddingSpace
```python
from tensorlogic.reasoning.embed import EmbeddingSpace

space = EmbeddingSpace(num_objects=100, embedding_dim=16, device='cpu')
space.add_relation("knows", init='random')   # nn.Parameter, learnable
space.add_relation("likes", init='random')
# Query and compose relations
p = space.query_relation("knows", i, j, use_sigmoid=True)
space.apply_rule("knows", "works_with", "knows_via_work")
```

### Training Patterns
```python
from tensorlogic.learn.trainer import EmbeddingTrainer, PairScoringTrainer
from tensorlogic.learn.losses import ContrastiveLoss
from tensorlogic.reasoning.composer import GatedMultiHopComposer

# Embedding trainer for a single relation
trainer = EmbeddingTrainer(space, optimizer_type='adam', learning_rate=0.01)
trainer.train_relation("parent", positive_pairs, epochs=100, verbose=True)

# Pair-scoring trainer for a multi-relation composer
composer = GatedMultiHopComposer(
    embedding_dim=space.embedding_dim,
    num_relations=len(space.relations),
    num_hops=2,
)
pair_trainer = PairScoringTrainer(composer, optimizer_type='adam', learning_rate=1e-3)
loss_fn = ContrastiveLoss()
def scorer(model, subjects, objects):
    return space.score_with_composer(model, subjects, objects)  # probabilities
# See examples/curriculum for batching; call pair_trainer.train_epoch(loss_fn, scorer, batches, num_objects=space.num_objects)
```

### Saving and Loading
Use the I/O utilities to persist learned state. EmbeddingSpace models are fully restored; TensorProgram saves learnable parameters (you must recreate constants and equations in code before loading).
```python
from tensorlogic.utils.io import save_model, load_model

# EmbeddingSpace
save_model(space, "embedding_space.pt")
space2 = EmbeddingSpace(num_objects=..., embedding_dim=..., temperature=...)
metadata = load_model(space2, "embedding_space.pt")

# TensorProgram (parameters only)
save_model(program, "program_params.pt")
# ...recreate constants and equations in code...
load_model(program, "program_params.pt")
```

## When to Use Different Approaches

**Boolean Mode** → Rules, audits, compliance, binary logical ops (determinism depends on facts/rules)
- Hard-coded rules: "grandfather(X,Z) := father(X,Y), father(Y,Z)"
- Symbolic deduction for tasks requiring deterministic, rule-based behavior
- Use `TensorProgram` in Boolean mode with forward/backward chaining

**Embedding Learning** → Fast learning from labeled examples, analogical reasoning
- Knowledge base completion with positive/negative pairs
- Similarity-based reasoning where approximate answers are acceptable
- Use `EmbeddingSpace` with `EmbeddingTrainer` or `PairScoringTrainer`

**Learnable Composition** → Discover relation transformations from supervised examples
- Learn that "grandparent = parent ∘ parent" from examples
- Multi-hop path prediction over a learned relation matrix bank
- Use `GatedMultiHopComposer` trained on supervised examples

**Sequence Modeling (Neural Text/Time Series)** → Decoder-only Transformer or RNNs
- Next-token prediction with `DecoderOnlyLM` (GPT-like). Suited for demos (Tiny Shakespeare) and research; for large-scale use, add KV cache/AMP/DDP.
- Temporal embeddings with `SimpleRNN`/`LSTM`/`GRU`; combine with symbolic masks from `TensorProgram` for constrained prediction.
- Knowledge-graph-aware attention: mask attention with program facts for structure-aware sequence modeling.

**Predicate Invention** → Automatically discover hidden structure, no labels needed
- Knowledge graph refinement and latent relation discovery
- Self-supervised learning of implicit predicates
- Use `invent_and_register_rescal` on existing relation tensors

## Common Pitfalls & Solutions

1. **Mixing Boolean and continuous tensors**: Ensure all tensors in a program are either Boolean (int8/int32) or continuous (float32/float64), not mixed.
   - Fix: Cast tensors explicitly before adding to program

2. **Training on unlabeled data**: EmbeddingTrainer and PairScoringTrainer require labeled positive/negative pairs.
   - Fix: Use predicate invention (RESCAL) or pattern discovery for self-supervised learning

3. **OOM with large embeddings**: Large embedding spaces can exhaust GPU memory.
   - Fix: Use sparse tensors via `sparse.py` utilities, reduce batch size, or switch `device='cpu'`

4. **Forgetting to register entities**: EmbeddingSpace requires entities to be pre‑registered.
   - Fix: Call `space.add_object(name, index)` (optionally with an initial embedding)

5. **Mode mismatch in operations**: Operations assume consistent tensor types.
   - Fix: Ensure all input tensors to an operation are in the same mode (Boolean or continuous)

6. **Probability interpretation in continuous mode**: Some ops (e.g., joins, projections) return unnormalized real values and can exceed 1.
   - Fix: Apply nonlinearities (e.g., sigmoid/softmax) or thresholds where a probabilistic interpretation is required

7. **Attention mask shapes/broadcasting**: Masks must be broadcastable to `[B, H, Lq, Lk]`.
   - Fix: Provide `[B, Lq, Lk]` or `[Lq, Lk]`; `MultiHeadAttention` will expand to heads. Use `utils.causal_mask(L)` for `[1, L, L]`.

8. **Attention weights not returned**: By default components may return only outputs.
   - Fix: Pass `need_weights=True` to `MultiHeadAttention` or `return_attention=True` to encoder/decoder stacks.

9. **Decoder-only LM generation speed**: `generate()` is O(T^2) without KV cache.
   - Fix: For longer generations, implement KV cache or reduce context; top-k sampling is supported.

## Paper Reference

Implementation based on: ["Tensor Logic: The Language of AI"](https://arxiv.org/abs/2510.12269) (Domingos)


### Strengths, Gaps vs. Paper

Strengths
- Core tensor-logic primitives implemented (einsum rules, tensor equations)
- Dual-mode support present (Boolean thresholding + continuous reasoning)
- Forward/backward chaining available (backward is simple, goal-directed)
- Embedding-space reasoning and learning integrated
- RESCAL-based predicate invention working end-to-end
- Training utilities (generic, embedding, pair-scoring) available
- Diagnostics and visualization helpers included
- Examples cover symbolic, embeddings, composition, invention, and benchmarking
- Benchmarking suite provided for quick comparisons
 - Transformers: encoder, decoder, and full architecture with explicit tensor equations
 - Decoder-only LM (GPT-like) with tied embeddings and generation
 - Recurrent modules: SimpleRNN/LSTM/GRU with tensor-equation cells and boolean mode

Gaps
- Advanced tensor decompositions (e.g., Tucker, CP) not implemented
- Loopy belief propagation for cyclic inference not implemented
- Sampling via selective projection not exposed
- Typed embedding spaces not implemented
- Full Datalog-style backward chaining not implemented
- Abductive reasoning not implemented
- GPU-optimized sparse ops/large-graph kernels missing
- TensorProgram save/load does not serialize constants/equations (manual rebuild required)
- TensorWrapper is not used pervasively as a language-level carrier
- Embeddings/composers not integrated as first-class language constructs
 - LM performance optimizations (KV cache, FlashAttention, AMP, DDP) are not included by default
