# Shakespeare Language Model Training

Character-level language model training using TensorLogic Transformers, comparable to nanoGPT, plus demonstrations of TensorLogic's unique neural-symbolic capabilities.

## Quick Start

```bash
# Download Shakespeare dataset (if not already present)
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Quick test (50 iterations) - run from project root
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py --max_iters 50

# Full training (5000 iterations, ~6 hours on M-series Mac)
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py --max_iters 5000

# Generate text from trained model
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py --checkpoint checkpoints/shakespeare/best.pt
```

## Features

- **Causal Masking**: Implementation prevents attention to future tokens
- **MPS/CUDA/CPU Support**: Automatic device detection, optimized for Apple Silicon
- **Learning Rate Schedule**: Warmup + cosine decay for stable training
- **Gradient Clipping**: Prevents training instability
- **Checkpoint Management**: Saves best and final models
- **Temperature Control**: Adjustable generation randomness

## Configuration

Key hyperparameters (defaults based on nanoGPT):

- `--d_model 384`: Model dimension
- `--n_layer 6`: Number of transformer layers
- `--n_head 6`: Number of attention heads
- `--block_size 128`: Context length
- `--batch_size 64`: Batch size
- `--learning_rate 1e-3`: Initial learning rate
- `--dropout 0.2`: Dropout rate

## Custom Training Examples

```bash
# Smaller model for faster training
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py --d_model 256 --n_layer 4 --batch_size 32

# Longer context window
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py --block_size 256 --batch_size 32

# More aggressive training
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py --learning_rate 3e-3 --dropout 0.1
```

## Generation Options

```bash
# Generate with different temperatures
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py --checkpoint checkpoints/shakespeare/best.pt --temperature 0.5  # More focused
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py --checkpoint checkpoints/shakespeare/best.pt --temperature 1.0  # More creative

# Generate longer text
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py --checkpoint checkpoints/shakespeare/best.pt --max_tokens 1000

# Generate from a prompt
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py --checkpoint checkpoints/shakespeare/best.pt --prompt "To be or not to be"
```

## TensorLogic Features Demo (Shakespeare)

Explore what makes TensorLogic special compared to vanilla transformers:

```bash
# Run the interactive demo showing TensorLogic's capabilities
PYTHONPATH=. python3 examples/shakespeare/generate_tensorlogic_shakespeare.py
```

This demo shows:
1. **Standard vs Boolean Mode** — Soft vs hard attention mechanisms
2. **Grammar Constraints** — Force valid character sequences
3. **Learned Relations** — Characters that naturally follow each other
4. **Hybrid Generation** — Combine neural + symbolic constraints
5. **Tensor Equations** — See your model as pure math
6. **Romeo & Juliet Constraints** — Prevent impossible scenes

### Example Output

```
PYTHONPATH=. python3 examples/shakespeare/generate_tensorlogic_shakespeare.py
======================================================================
TensorLogic Features Demo (Shakespeare)
======================================================================

✓ Loading trained Shakespeare model from checkpoints/shakespeare/best.pt
  Shows coherent Shakespeare-like text with TensorLogic features

Vocabulary size: 65 characters

1. STANDARD TEXT GENERATION (vanilla-like)
--------------------------------------------------
Using trained Shakespeare model — expect coherent text

Softmax attention (temperature=0.8):
  Prompt: 'To be or not to be'
  Generated: 'to be or not to be success;
Or her is him.

Clown:

HERMIONE:
And we'

2. BOOLEAN MODE (hard, interpretable attention)
--------------------------------------------------
Same weights, boolean attention per head (one-hot)

Argmax decoding (deterministic given input):
  Prompt: 'To be or not to be'
  Generated: 'to be or not to be
The service shall be the court.

KING RICHARD III'

Key differences:
  • One-hot attention per head
  • Deterministic argmax decoding
  • Traceable attention paths

3. KNOWLEDGE-CONSTRAINED GENERATION
--------------------------------------------------
Grammar mask + 2-hop reachability via tensor rules

Unconstrained:
  'thou art not the grief of the king?

GLOUCESTER:
'

Grammar-constrained (enforces 'th', 'qu', spacing):
  'thou art forth, or know
Thy pill into your head t'

4. TENSOR EQUATIONS
--------------------------------------------------
Model as equations:
  E_tok[b,i,d] := TokenEmbed[tokens[b,i], d]
  E_pos[i,d]   := PosEmbed[i, d]
  X_0[b,i,d]   := E_tok[b,i,d] + E_pos[i,d]

Enables:
  - Property checks
  - Op dedup/optimization
  - Transparent computation

5. RELATION LEARNING (embeddings)
--------------------------------------------------
Learn 3 relations in embedding space
  - Each relation is D×D (embedding_dim×embedding_dim)

Generation without relations:
  'lord in my majesty,
Tell the Vargar of land o'

With learned relation 'follows_well':
  'lord for the libers,
To that worship Mortague'

6. HYBRID NEURAL–SYMBOLIC GENERATION
--------------------------------------------------
Combine:
  1) Neural LM
  2) Grammar rules (must follow)
  3) Learned relations (soft preferences)
  4) Hard constraints

Examples:
  - No constraints: 'romeo: he arms, thought that I knock
His more e'
  - Grammar only:   'romeo: sir love!

AUTOLYCUS:
We live him in her'
  - Relations only: 'romeo: God he yet nurse!

PERDITA:
The Volsce o'
  - Both (hybrid):  'romeo: the court of utter man now they do not f'

7. PRACTICAL: ROMEO & JULIET CONSTRAINTS
--------------------------------------------------
Character interaction rules (boolean program)
  Direct interactions: 8
  Derived conflicts: 18

Prevents impossible scenes like:
  ✗ "Romeo and Tybalt laughed together merrily"
  ✓ "Romeo and Mercutio jested as friends do"

======================================================================
SUMMARY
======================================================================

Vanilla Transformer:
  - Soft attention only
  - No explicit constraints
  - Opaque operations
  - Learns everything from data

TensorLogic Transformer:
  - Boolean OR continuous attention (switchable)
  - Hard symbolic constraints (knowledge graphs)
  - Transparent tensor equations
  - Combines learned + programmed knowledge
  - Can enforce logical consistency
  - Prevents impossible/contradictory outputs
  - Full interpretability on demand

Use TensorLogic when you need:
  ✓ Logical consistency guarantees
  ✓ Integrate domain knowledge without retraining
  ✓ Interpretable reasoning paths
  ✓ Hybrid neural–symbolic control
  ✓ Safer, constrained generation

======================================================================
Run with your trained Shakespeare model:
  1. Load checkpoints/shakespeare/best.pt
  2. Add constraints as above
  3. Generate logically consistent Shakespeare
======================================================================
```

## Architecture

The model uses TensorLogic's `DecoderOnlyLM` which implements:

- Multi-head self-attention with correct causal masking
- Learned positional encodings
- Weight tying between token embeddings and output projection
- Layer normalization and residual connections
- **Unique**: Switchable boolean/continuous modes for interpretability
- **Unique**: Integration with symbolic constraints via TensorProgram

## Files

- `train_shakespeare.py`: Main training script
- `generate_shakespeare.py`: Standard text generation utility
- `generate_tensorlogic_shakespeare.py`: TensorLogic's unique features demonstration with Shakespeare
- `checkpoints/shakespeare/`: Saved model checkpoints (created during training)
- `input.txt`: Shakespeare dataset (download separately to project root)

## Directory Structure

```
tensorlogic/
├── input.txt                          # Shakespeare dataset (download here)
├── examples/
│   └── shakespeare/
│       ├── README.md                  # This file
│       ├── train_shakespeare.py       # Training script
│       └── generate_shakespeare.py    # Generation script
└── checkpoints/
    └── shakespeare/
        ├── best.pt                    # Best model checkpoint
        └── final.pt                   # Final model checkpoint
```

## Running from Different Directories

All commands should be run from the project root directory (`tensorlogic/`):

```bash
cd /Users/wayland/Code_Ptmind/tensorlogic
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py
```
