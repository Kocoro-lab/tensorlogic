"""
Logical tensor operations for symbolic reasoning

Implements join, project, and other logical operations using Einstein summation.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def logical_join(
    t1: torch.Tensor,
    t2: torch.Tensor,
    equation: Optional[str] = None,
    mode: str = 'continuous'
) -> torch.Tensor:
    """
    Join two tensors (analogous to logical AND or database join)

    For example, Parent[i,j] * Parent[j,k] -> Grandparent[i,k]

    Args:
        t1: First tensor
        t2: Second tensor
        equation: Einstein summation equation (e.g., 'ij,jk->ik')
        mode: 'boolean' for strict logic, 'continuous' for fuzzy
    """
    if equation is None:
        # Simple case: assume matrix multiplication pattern
        # Works for 2D tensors: ij,jk->ik
        if t1.dim() == 2 and t2.dim() == 2:
            result = torch.matmul(t1, t2)
        else:
            raise ValueError("equation parameter required for non-2D tensors")
    else:
        result = torch.einsum(equation, t1, t2)

    # In Boolean mode, threshold the result
    if mode == 'boolean':
        result = (result > 0.5).float()

    return result


def logical_project(
    tensor: torch.Tensor,
    dim: int,
    mode: str = 'continuous'
) -> torch.Tensor:
    """
    Project tensor along a dimension (analogous to existential quantification)

    For example, exists Y: Parent(X,Y) -> HasChild(X)

    Args:
        tensor: Input tensor
        dim: Dimension to project (sum over)
        mode: 'boolean' for strict logic (max), 'continuous' for sum
    """
    if mode == 'boolean':
        # Existential: true if any value is true
        result, _ = torch.max(tensor, dim=dim)
    else:
        # Continuous: sum probabilities
        result = torch.sum(tensor, dim=dim)

    return result


def logical_select(
    tensor: torch.Tensor,
    condition: torch.Tensor,
    mode: str = 'continuous'
) -> torch.Tensor:
    """
    Select elements satisfying a condition (analogous to WHERE clause)

    Args:
        tensor: Input tensor
        condition: Boolean/continuous condition tensor (same shape as tensor)
        mode: 'boolean' or 'continuous'
    """
    if mode == 'boolean':
        result = tensor * (condition > 0.5).float()
    else:
        result = tensor * condition

    return result


def logical_union(
    t1: torch.Tensor,
    t2: torch.Tensor,
    mode: str = 'continuous'
) -> torch.Tensor:
    """
    Union of two tensors (analogous to logical OR)

    Args:
        t1: First tensor
        t2: Second tensor
        mode: 'boolean' for max, 'continuous' for probabilistic OR
    """
    if mode == 'boolean':
        result = torch.maximum(t1, t2)
    else:
        # Probabilistic OR: P(A or B) = P(A) + P(B) - P(A)*P(B)
        result = t1 + t2 - t1 * t2

    return result


def logical_negation(
    tensor: torch.Tensor,
    mode: str = 'continuous'
) -> torch.Tensor:
    """
    Logical negation (NOT)

    Args:
        tensor: Input tensor
        mode: 'boolean' or 'continuous'
    """
    if mode == 'boolean':
        result = 1.0 - tensor
    else:
        # Probabilistic NOT: P(not A) = 1 - P(A)
        result = 1.0 - tensor

    return result


def forward_chain_step(
    antecedent_tensors: list,
    equation: str
) -> torch.Tensor:
    """
    Single step of forward chaining: apply a rule

    Args:
        antecedent_tensors: List of tensors for rule antecedents
        equation: Einstein equation for joining antecedents

    Returns:
        Consequent tensor
    """
    if len(antecedent_tensors) == 1:
        return antecedent_tensors[0]
    elif len(antecedent_tensors) == 2:
        return torch.einsum(equation, antecedent_tensors[0], antecedent_tensors[1])
    else:
        # Multiple antecedents - chain them
        result = antecedent_tensors[0]
        for t in antecedent_tensors[1:]:
            result = torch.einsum(equation, result, t)
        return result


def apply_nonlinearity(
    tensor: torch.Tensor,
    func: str,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply nonlinearity function

    Args:
        tensor: Input tensor
        func: Function name: 'step', 'sigmoid', 'relu', 'softmax', 'tanh'
        temperature: Temperature for sigmoid/softmax (lower = sharper)
    """
    if func == 'step':
        return (tensor > 0.0).float()
    elif func == 'sigmoid':
        return torch.sigmoid(tensor / temperature)
    elif func == 'relu':
        return F.relu(tensor)
    elif func == 'softmax':
        # Assume last dimension for softmax
        return F.softmax(tensor / temperature, dim=-1)
    elif func == 'tanh':
        return torch.tanh(tensor)
    elif func == 'identity' or func is None:
        return tensor
    else:
        raise ValueError(f"Unknown nonlinearity: {func}")
