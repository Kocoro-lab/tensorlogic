# Shakespeare Language Model Training

Character-level language model training using TensorLogic Transformers, comparable to nanoGPT.

## Quick Start

```bash
# Download Shakespeare dataset (if not already present)
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Quick test (50 iterations) - run from project root
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py --max_iters 50

# Full training (5000 iterations, ~6 hours on M-series Mac)
PYTHONPATH=. python3 examples/shakespeare/train_shakespeare.py --max_iters 5000

# Generate text from trained model
PYTHONPATH=. python3 examples/shakespeare/generate_shakespeare.py checkpoints/shakespeare/best.pt
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

## Architecture

The model uses TensorLogic's `DecoderOnlyLM` which implements:

- Multi-head self-attention with correct causal masking
- Learned positional encodings
- Weight tying between token embeddings and output projection
- Layer normalization and residual connections

## Files

- `train_shakespeare.py`: Main training script
- `generate_shakespeare.py`: Text generation utility
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

