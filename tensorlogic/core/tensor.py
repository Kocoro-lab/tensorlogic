"""
Core tensor wrapper for Tensor Logic

Provides Boolean and continuous tensor modes with logical operations.
"""

import torch
import numpy as np
from typing import Union, Optional


class TensorWrapper:
    """
    Wrapper around PyTorch tensors that supports both Boolean (logical) and
    continuous (neural) modes.

    In Boolean mode, operations are strict (0/1 values).
    In continuous mode, operations use real values for fuzzy logic and learning.
    """

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, list],
        mode: str = 'continuous',
        learnable: bool = False,
        device: str = 'cpu'
    ):
        """
        Args:
            data: Initial tensor data
            mode: 'boolean' for strict logic, 'continuous' for neural/fuzzy
            learnable: Whether this tensor has learnable parameters
            device: 'cpu' or 'cuda'
        """
        assert mode in ['boolean', 'continuous'], f"Mode must be 'boolean' or 'continuous', got {mode}"

        self.mode = mode
        self.device = device

        # Convert to tensor
        if isinstance(data, torch.Tensor):
            self.tensor = data.to(device)
        elif isinstance(data, np.ndarray):
            self.tensor = torch.from_numpy(data).float().to(device)
        else:
            self.tensor = torch.tensor(data, dtype=torch.float32).to(device)

        # In Boolean mode, threshold to 0 or 1
        if mode == 'boolean':
            self.tensor = (self.tensor > 0.5).float()

        # Make learnable if requested
        if learnable:
            self.tensor = torch.nn.Parameter(self.tensor)

        self.learnable = learnable

    def to_boolean(self, threshold: float = 0.5) -> 'TensorWrapper':
        """Convert to Boolean mode by thresholding"""
        boolean_data = (self.tensor > threshold).float()
        return TensorWrapper(boolean_data, mode='boolean', learnable=False, device=self.device)

    def to_continuous(self) -> 'TensorWrapper':
        """Convert to continuous mode"""
        return TensorWrapper(self.tensor.detach(), mode='continuous', learnable=self.learnable, device=self.device)

    def detach(self) -> 'TensorWrapper':
        """Detach from computation graph"""
        return TensorWrapper(self.tensor.detach(), mode=self.mode, learnable=False, device=self.device)

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    def __repr__(self):
        mode_str = "Boolean" if self.mode == 'boolean' else "Continuous"
        learnable_str = " (learnable)" if self.learnable else ""
        return f"TensorWrapper({mode_str}{learnable_str}, shape={self.shape})"


def create_tensor(
    shape: tuple,
    mode: str = 'continuous',
    learnable: bool = False,
    init: str = 'zeros',
    device: str = 'cpu'
) -> TensorWrapper:
    """
    Factory function to create tensors

    Args:
        shape: Tensor shape
        mode: 'boolean' or 'continuous'
        learnable: Whether tensor parameters are learnable
        init: Initialization method: 'zeros', 'ones', 'random', 'randn'
        device: 'cpu' or 'cuda'
    """
    if init == 'zeros':
        data = torch.zeros(shape)
    elif init == 'ones':
        data = torch.ones(shape)
    elif init == 'random':
        data = torch.rand(shape)
    elif init == 'randn':
        data = torch.randn(shape)
    else:
        raise ValueError(f"Unknown init method: {init}")

    return TensorWrapper(data, mode=mode, learnable=learnable, device=device)
