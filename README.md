# Tensor Logic

Unified neural + symbolic reasoning via tensor equations.
Based on ["Tensor Logic: The Language of AI"](https://arxiv.org/abs/2510.12269) (Domingos).

Key capabilities
- Tensor equations as programs (einsum joins + projections) with forward/backward chaining
- Boolean mode (deductive, no hallucinations) and continuous mode (learnable with temperature)
- Embedding‑space reasoning and gated multi‑hop composer (learns compositions)
- Automatic predicate invention via RESCAL (no labels, no manual patterns)
- **FB15k-237 benchmark**: RESCAL outperforms the LibKGE reference and is competitive with RotatE/TuckER (MRR 0.347)
- Sparse facts, AMP, batching, and a simple benchmark suite

## Requirements
- Python 3.8+, PyTorch 2.0+, NumPy

## Installation

### Option 1: Development Installation (Recommended for Contributors)
```bash
# Clone the repository
git clone https://github.com/Kocoro-lab/tensorlogic.git
cd tensorlogic

# Install in editable mode (changes to code are immediately reflected)
pip install -e .
```

Then use in any Python project:
```python
import tensorlogic
from tensorlogic.reasoning.embed import EmbeddingSpace
```

### Option 2: Direct GitHub Installation
```bash
pip install git+https://github.com/Kocoro-lab/tensorlogic.git
```

### Option 3: Manual Dependency Installation
```bash
git clone https://github.com/Kocoro-lab/tensorlogic.git
cd tensorlogic
pip install -r requirements.txt
```

Then set `PYTHONPATH` when running examples:
```bash
PYTHONPATH=. python3 examples/family_tree_symbolic.py
```

## Quick Start Examples

After installation with `pip install -e .`, run examples directly:
```bash
python3 examples/family_tree_symbolic.py
```

Or if using manual dependency installation, set `PYTHONPATH`:
```bash
PYTHONPATH=. python3 examples/<name>.py
```

### Foundation: Boolean Reasoning
```bash
python3 examples/family_tree_symbolic.py
```
**Learn:** Forward chaining, Boolean logic, guaranteed correctness. No training required.
**Use case:** Rules, audits, symbolic deduction (grandparent = parent ∘ parent).

### Foundation: Embedding Learning
```bash
python3 examples/family_tree_embedding.py
```
**Learn:** Train embeddings from data, learn relation matrices, compose relations.
**Use case:** Fast learning from positive/negative pairs, analogical reasoning via similarity.

### Advanced: Learnable Composition
```bash
python3 examples/learnable_demo.py
```
**Learn:** GatedMultiHopComposer learns multi-hop paths from examples.
**Use case:** Predict compositions from a few supervised examples (e.g., grandparent from parent pairs).

### Advanced: Pattern Discovery (Self-Supervised)
```bash
python3 examples/pattern_discovery_demo.py
```
**Learn:** Closure-based discovery—infer patterns like "sister" from "parent" facts.
**Use case:** Find implicit relations without labels or manual enumeration.

### Advanced: Predicate Invention (RESCAL)
```bash
python3 examples/predicate_invention_demo.py
```
**Learn:** Automatically invent hidden predicates via RESCAL tensor factorization.
**Use case:** Discover latent structure and missing relations in knowledge bases.

### FB15k-237 Knowledge Graph Benchmark
```bash
python3 examples/fb15k237_benchmark.py                                  # Full training
python3 examples/fb15k237_benchmark.py --epochs 2 --eval_interval 1     # Quick smoke test
python3 examples/fb15k237_benchmark.py --eval_only                      # Evaluate checkpoint
python3 examples/fb15k237_benchmark.py --resume                         # Resume training
```
**Standard KG link prediction benchmark** (14,541 entities, 237 relations, 310K triples).
Auto-downloads dataset, trains RESCAL with 1vsAll scoring, evaluates with filtered MRR/Hits@K.

| Metric | TensorLogic RESCAL | LibKGE RESCAL | DistMult | ComplEx | RotatE |
|--------|-------------------|---------------|----------|---------|--------|
| MRR    | **0.347**         | 0.304         | 0.241    | 0.247   | 0.338  |
| H@1    | **0.258**         | 0.242         | 0.155    | 0.158   | 0.241  |
| H@3    | **0.382**         | 0.331         | 0.263    | 0.275   | 0.375  |
| H@10   | **0.524**         | 0.419         | 0.419    | 0.428   | 0.533  |

### Internal Benchmarks & Comparison
```bash
python3 examples/benchmark_suite.py
```
**Learn:** Compare Boolean, Embedding, Composer, and Composer+Invented on multiple scenarios.
**Metrics:** AUC, Hits@K, F1, training time, query speed, memory usage.

### Real-World Demo: Knowledge Base from Text (A Chinese Sci-Fi)
```bash
python3 examples/three_body_kb_demo.py
```
**Learn:** Extract entities/relations from text, train a KB, perform multi-hop queries.
**Use case:** End-to-end pipeline: text → structured KB → reasoning.

### Transformer & RNN

TensorLogic now includes full **Transformer** and **RNN/LSTM/GRU** implementations, all expressed as tensor equations!

```bash
# Transformer with knowledge graph constraints
python3 examples/transformer_reasoning_demo.py

# RNN/LSTM temporal reasoning with symbolic masks
python3 examples/rnn_sequence_reasoning_demo.py

# Hybrid reasoning: symbolic logic + neural attention
python3 examples/hybrid_reasoning_transformer.py

# Shakespeare language model (nanoGPT-like, ~1.5 val loss)
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py  # Train
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py --checkpoint checkpoints/shakespeare/best.pt

# TensorLogic unique features with Shakespeare - see what makes it special
PYTHONPATH=. python3 examples/shakespeare/generate_tensorlogic_shakespeare.py
```

**Features:**
- **Multi-head attention** as tensor equations: `Q×K^T/√d → softmax → ×V`
- **Transformers**: Full encoder-decoder architecture with cross-attention
- **Decoder-only LM**: GPT-style autoregressive models with generation
- **RNN/LSTM/GRU**: Temporal reasoning with tensor equation cells
- **Boolean mode**: Hard attention for interpretable reasoning
- **Symbolic constraints**: Use knowledge graphs to mask attention
- **Export to equations**: See any model as pure tensor equations

```python
from tensorlogic.transformers import Transformer, LSTM, DecoderOnlyLM

# Build a transformer
transformer = Transformer(d_model=512, nhead=8, num_encoder_layers=6)

# Export as tensor equations
equations = transformer.to_tensor_equations()
print('\n'.join(equations))  # See the math!

# RNN with boolean mode (discrete states)
lstm = LSTM(input_size=128, hidden_size=256, mode='boolean')

# Decoder-only language model
lm = DecoderOnlyLM(vocab_size=50304, d_model=768, n_layer=12)
generated = lm.generate(prompt, max_new_tokens=100)
```

## When to Use Which Approach

| Approach | When to Use | Example |
|----------|------------|---------|
| **Boolean** | Hard rules, audits, compliance, zero hallucinations | Tax rules, medical contraindications, legal logic |
| **Embedding** | Known relations to learn, fast training, few labels | Knowledge base completion, similarity search |
| **Composer** | Learning multi-hop paths from examples | Predict "grandparent" from "parent" facts, relation composition |
| **Composer + Invented** | Fully automatic—discover structure, no labels | Knowledge graph refinement, find hidden relations |
| **Transformer** | Sequence modeling, attention patterns, language tasks | Text generation, seq2seq, attention as relation discovery |
| **RNN/LSTM** | Temporal sequences, state machines, time series | Sequential reasoning, temporal embeddings, constrained generation |
| **Hybrid** | Neural + symbolic, constrained generation | Knowledge-aware language models, attention with logical masks |

**Quick decision tree:**
1. Only logic rules? → Boolean mode
2. Have labeled examples? → Embedding + optionally Composer
3. Need automatic structure? → Add predicate invention (RESCAL)

Why Tensor Logic?
- Tiny models: 10-500 KB (vs GBs for LLMs), training in seconds/minutes
- Zero hallucinations in Boolean mode (guaranteed correct deductions)
- Learns relation compositions automatically (no hand-written rules)
- Discovers hidden predicates from data (no manual feature engineering)
- Differentiable throughout (integrate into neural pipelines)

## Core Concepts

**Tensor Programs** combine tensors (facts, relations, weights) with equations (rules).
- **Boolean mode**: Strict logic (0 or 1), forward/backward chaining, guaranteed correctness
- **Continuous mode**: Probabilistic reasoning, learnable embeddings and relation matrices, differentiable

**Embedding Space** reasoning represents objects and relations as vectors/matrices.
- Objects → embeddings (learned vectors encoding identity)
- Relations → matrices (learned transformations)
- Scoring: `score(subject, relation, object) = subject^T × relation_matrix × object`
- Composition: `grandparent = parent @ parent` (matrix multiplication)

**Training Process**: Learn embeddings and relation matrices from positive/negative pairs.
- Initialize: random embeddings + relation matrices
- For each epoch: compute scores for pairs, backpropagate loss, update parameters
- Result: embeddings encode structure, relations encode transformations, enables multi-hop reasoning

**Predicate Invention** discovers hidden relations via RESCAL factorization.
- Input: triples (head, relation, tail)
- Process: factorize knowledge graph tensor, extract candidate predicates
- Output: novel relations that improve structure learning

Core APIs (minimal)
- Program (Boolean / differentiable): `tensorlogic.core.program.TensorProgram`
- Embedding space: `tensorlogic.reasoning.embed.EmbeddingSpace`
- Composer: `tensorlogic.reasoning.composer.GatedMultiHopComposer`
- Predicate invention: `tensorlogic.reasoning.predicate_invention.invent_and_register_rescal`

Tools (see `tensorlogic/utils` and `tensorlogic/learn`)
- Diagnostics (gradient health), curriculum training, init strategies, visualization, sparse utils

Project structure (essentials)
```
tensorlogic/
  core/         # TensorProgram, tensor ops
  reasoning/    # EmbeddingSpace, Composer, Closure, Decomposition, Invention
  learn/        # Trainers (Embedding/Pair), Losses, Curriculum
  utils/        # Diagnostics, Init, Sparse, Visualization, I/O
examples/       # Boolean / Embedding / Composer / Invention / Benchmarks
```

## Benchmarks

**Standard KG benchmark** (FB15k-237):
```bash
python3 examples/fb15k237_benchmark.py
```
RESCAL achieves MRR 0.347, beating LibKGE reference (0.304) and exceeding RotatE (0.338).

**Internal benchmark suite**:
```bash
python3 examples/benchmark_suite.py
```
Reports: Speed (train/query time, pairs/s), Memory (MB), Quality (AUC/Hits@K/F1) on family, small‑KG, synthetic scenarios.

How to ask queries
- Parse questions into a structured form: (head, relation, ?) or (?, relation, tail). Example: "Who founded TechCorp?" → relation="founded", tail="TechCorp".
- Define a candidate set: use all entities or filter by type (e.g., only People or only Products). Larger sets are fine; use batched scoring for scale.
- Score with relations, not plain cosine: use the bilinear scorer s^T W_r o (EmbeddingSpace), then sigmoid/threshold as needed.
- Rank or classify: compute scores for all candidates, take Top‑K or apply a threshold for yes/no.
- Multi‑hop reasoning: if the path is known, specify a relation sequence (e.g., [founded, develops]); if unknown, use a trained multi‑hop composer over a relation bank to learn the path. You still provide anchors (head/tail and candidate set).
- Prerequisites: entities must be registered; relations must exist (trained or invented); map synonyms to canonical relation names; truly unseen entities are out‑of‑scope.
- Examples:
  - "Who founded TechCorp?" → relation=founded, tail=TechCorp, candidates=People → Top‑K subjects.
  - "What products do companies founded by Alice develop?" → head=Alice, path=[founded, develops], candidates=Products → Top‑K objects.
- Non‑goals: open‑ended web search or free‑form QA; this is a learnable relation/path scorer over a known KB.

**Not Yet Implemented**
- CNNs, kernel machines, and probabilistic graphical models (PGMs) as tensor equations
- Typed embedding spaces with rectangular relation matrices across entity types
- First‑class integration of embeddings/composers as `TensorProgram` operators
- General Datalog solver with full backward‑chaining and abductive reasoning
- Advanced tensor decompositions (Tucker/CP) and optimized sparse GPU kernels

License & citation
- MIT License (see LICENSE)
- Paper: https://arxiv.org/abs/2510.12269
- Please cite the associated paper if you use this repo
