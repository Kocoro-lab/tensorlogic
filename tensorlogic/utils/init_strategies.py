"""
Initialization strategies for embeddings and relation matrices

Best practices encapsulated for optimal training convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union


def init_embeddings(
    embeddings: Union[nn.Embedding, nn.Parameter, torch.Tensor],
    strategy: Literal['normalized_random', 'xavier', 'orthogonal', 'uniform'] = 'normalized_random',
    scale: float = 1.0
):
    """
    Initialize embeddings with best practices

    Args:
        embeddings: Embedding weights to initialize (nn.Embedding, nn.Parameter, or tensor)
        strategy: Initialization strategy
            - 'normalized_random': Random unit vectors (RECOMMENDED)
            - 'xavier': Xavier/Glorot initialization
            - 'orthogonal': Orthogonal initialization
            - 'uniform': Uniform distribution
        scale: Scaling factor

    Strategies explained:
        - normalized_random: Good for similarity-based models (default)
        - xavier: Good for general neural networks
        - orthogonal: Good for preventing gradient explosion
        - uniform: Simple uniform distribution
    """
    with torch.no_grad():
        weight = embeddings.weight if isinstance(embeddings, nn.Embedding) else embeddings
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

        if strategy == 'normalized_random':
            # Random then normalize to unit vectors
            nn.init.normal_(weight, mean=0, std=scale)
            weight.copy_(F.normalize(weight, p=2, dim=1))

        elif strategy == 'xavier':
            nn.init.xavier_uniform_(weight, gain=scale)

        elif strategy == 'orthogonal':
            nn.init.orthogonal_(weight, gain=scale)

        elif strategy == 'uniform':
            nn.init.uniform_(weight, -scale, scale)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def init_relation_matrix(
    relation: nn.Parameter,
    strategy: Literal['small_random', 'identity', 'xavier', 'scaled_identity'] = 'small_random',
    scale: float = 0.1
):
    """
    Initialize a single relation matrix

    Args:
        relation: nn.Parameter of shape [D, D]
        strategy: Initialization strategy
            - 'small_random': Small random values (RECOMMENDED for learning)
            - 'identity': Identity matrix (good for fine-tuning)
            - 'xavier': Xavier initialization
            - 'scaled_identity': Identity + small noise
        scale: Scaling factor

    Strategies explained:
        - small_random: Start with weak relations, let training strengthen them
        - identity: Preserve embedding similarity initially
        - scaled_identity: Mix of both above
    """
    with torch.no_grad():
        if strategy == 'small_random':
            nn.init.normal_(relation, mean=0, std=scale)

        elif strategy == 'identity':
            dim = relation.shape[0]
            relation.data = torch.eye(dim, device=relation.device)

        elif strategy == 'xavier':
            nn.init.xavier_uniform_(relation, gain=scale)

        elif strategy == 'scaled_identity':
            dim = relation.shape[0]
            relation.data = torch.eye(dim, device=relation.device) * scale
            # Add small noise
            noise = torch.randn_like(relation) * (scale * 0.1)
            relation.data += noise

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def init_embedding_space(
    space: 'EmbeddingSpace',
    embedding_strategy: str = 'normalized_random',
    relation_strategy: str = 'small_random',
    embedding_scale: float = 1.0,
    relation_scale: float = 0.1
):
    """
    Initialize an entire EmbeddingSpace with best practices

    Args:
        space: EmbeddingSpace instance
        embedding_strategy: Strategy for object embeddings
        relation_strategy: Strategy for relation matrices
        embedding_scale: Scale for embedding init
        relation_scale: Scale for relation init

    Recommended combinations:
        - For similarity learning: normalized_random + small_random
        - For fine-tuning: normalized_random + scaled_identity
        - For transfer learning: keep pretrained embeddings, reinit relations
    """
    # Initialize object embeddings
    init_embeddings(
        space.object_embeddings,
        strategy=embedding_strategy,
        scale=embedding_scale
    )

    # Initialize all relation matrices
    for relation in space.relations.values():
        init_relation_matrix(
            relation,
            strategy=relation_strategy,
            scale=relation_scale
        )


def init_composer(
    composer: 'GatedMultiHopComposer',
    strategy: Literal['default', 'zero_bias', 'uniform_attention'] = 'default'
):
    """
    Initialize a GatedMultiHopComposer

    Args:
        composer: GatedMultiHopComposer instance
        strategy: Initialization strategy
            - 'default': PyTorch default (Kaiming)
            - 'zero_bias': Zero biases, helps with initial stability
            - 'uniform_attention': Start with uniform attention over relations

    Strategies explained:
        - zero_bias: Prevents initial bias toward specific relations
        - uniform_attention: Each relation equally weighted initially
    """
    with torch.no_grad():
        if strategy == 'zero_bias':
            # Zero out all biases
            for module in composer.modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

        elif strategy == 'uniform_attention':
            # Initialize hop attention to produce uniform weights
            for hop_gate in composer.hop_attention:
                # Set weights small so softmax is nearly uniform
                hop_gate.weight.data.normal_(0, 0.01)
                if hop_gate.bias is not None:
                    hop_gate.bias.data.zero_()

        elif strategy != 'default':
            raise ValueError(f"Unknown strategy: {strategy}")


class InitConfig:
    """
    Configuration bundle for initialization

    Usage:
        config = InitConfig.for_similarity_learning()
        config.apply(space)
    """

    def __init__(
        self,
        embedding_strategy: str = 'normalized_random',
        relation_strategy: str = 'small_random',
        composer_strategy: str = 'default',
        embedding_scale: float = 1.0,
        relation_scale: float = 0.1
    ):
        self.embedding_strategy = embedding_strategy
        self.relation_strategy = relation_strategy
        self.composer_strategy = composer_strategy
        self.embedding_scale = embedding_scale
        self.relation_scale = relation_scale

    def apply_to_space(self, space: 'EmbeddingSpace'):
        """Apply config to EmbeddingSpace"""
        init_embedding_space(
            space,
            embedding_strategy=self.embedding_strategy,
            relation_strategy=self.relation_strategy,
            embedding_scale=self.embedding_scale,
            relation_scale=self.relation_scale
        )

    def apply_to_composer(self, composer: 'GatedMultiHopComposer'):
        """Apply config to Composer"""
        init_composer(composer, strategy=self.composer_strategy)

    @classmethod
    def for_similarity_learning(cls) -> 'InitConfig':
        """Preset for similarity-based learning (RECOMMENDED)"""
        return cls(
            embedding_strategy='normalized_random',
            relation_strategy='small_random',
            composer_strategy='uniform_attention',
            embedding_scale=1.0,
            relation_scale=0.1
        )

    @classmethod
    def for_fine_tuning(cls) -> 'InitConfig':
        """Preset for fine-tuning pre-trained models"""
        return cls(
            embedding_strategy='normalized_random',  # Keep if training from scratch
            relation_strategy='scaled_identity',     # Preserve similarity initially
            composer_strategy='zero_bias',
            embedding_scale=1.0,
            relation_scale=0.5  # Stronger initial relations
        )

    @classmethod
    def for_large_scale(cls) -> 'InitConfig':
        """Preset for large-scale models (>10K objects)"""
        return cls(
            embedding_strategy='xavier',         # Better for large dims
            relation_strategy='small_random',
            composer_strategy='zero_bias',
            embedding_scale=1.0,
            relation_scale=0.05  # Smaller init for stability
        )


def reinitialize_failed_parameters(
    model: nn.Module,
    grad_threshold: float = 1e-7,
    strategy: str = 'small_random'
):
    """
    Reinitialize parameters with vanishing gradients

    Useful for recovering from training issues

    Args:
        model: Model with potentially stuck parameters
        grad_threshold: Threshold for detecting vanished gradients
        strategy: Reinitialization strategy
    """
    num_reinitialized = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.norm().item()

            if grad_norm < grad_threshold:
                print(f"Reinitializing {name} (grad_norm={grad_norm:.2e})")

                if 'weight' in name:
                    if len(param.shape) == 2:  # Matrix
                        nn.init.normal_(param, mean=0, std=0.1)
                    else:
                        nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.zero_()

                num_reinitialized += 1

    print(f"Reinitialized {num_reinitialized} parameters")
    return num_reinitialized
