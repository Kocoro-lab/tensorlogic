# Tensor Logic

Unified neural + symbolic reasoning via tensor equations.
Based on ["Tensor Logic: The Language of AI"](https://arxiv.org/abs/2510.12269) (Domingos).

Key capabilities
- Tensor equations as programs (einsum joins + projections) with forward/backward chaining
- Boolean mode (deductive, no hallucinations) and continuous mode (learnable with temperature)
- Embedding‑space reasoning and gated multi‑hop composer (learns compositions)
- Automatic predicate invention via RESCAL (no labels, no manual patterns)
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

### Benchmarks & Comparison
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

## When to Use Which Approach

| Approach | When to Use | Example |
|----------|------------|---------|
| **Boolean** | Hard rules, audits, compliance, zero hallucinations | Tax rules, medical contraindications, legal logic |
| **Embedding** | Known relations to learn, fast training, few labels | Knowledge base completion, similarity search |
| **Composer** | Learning multi-hop paths from examples | Predict "grandparent" from "parent" facts, relation composition |
| **Composer + Invented** | Fully automatic—discover structure, no labels | Knowledge graph refinement, find hidden relations |

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
- Ready‑made modules for other paradigms (Transformers/CNN/RNN/kernel machines/PGMs). Current focus is symbolic reasoning, embedding‑space reasoning, multi‑hop composition, and RESCAL‑style decomposition.
- Typed embedding spaces with rectangular relation matrices across types. Use a single entity table and filter candidates by type externally for now.
- First‑class integration of embedding/composer as language‑layer operators. Today these are parallel modules rather than ops inside `TensorProgram`.
- General backward‑chaining Datalog solver and large‑scale structure learning. Provided are forward chaining and RESCAL‑based predicate invention.
- Advanced large‑scale tensor decompositions (e.g., Tucker/CP) and optimized sparse GPU backends.

License & citation
- MIT License (see LICENSE)
- Paper: https://arxiv.org/abs/2510.12269
- Please cite the associated paper if you use this repo
