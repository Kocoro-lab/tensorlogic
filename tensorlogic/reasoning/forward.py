"""
Forward chaining for logical inference

Applies rules iteratively until no new facts are derived.
"""

import torch
from typing import Dict, Set


def forward_chain(
    program,
    facts: Dict[str, torch.Tensor],
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Forward chaining inference

    Repeatedly applies rules until convergence or max iterations.

    Args:
        program: TensorProgram with rules
        facts: Initial facts (ground truth tensors)
        max_iterations: Maximum number of iterations
        convergence_threshold: Stop when changes are below this

    Returns:
        Dictionary of all derived facts
    """
    current = facts.copy()
    prev_keys = set(current.keys())

    for iteration in range(max_iterations):
        # Execute one forward pass
        new_results = program.forward(current)

        # Check for convergence
        converged = True
        for key, tensor in new_results.items():
            if key in current:
                # Check if tensor changed significantly
                diff = torch.abs(tensor - current[key]).max().item()
                if diff > convergence_threshold:
                    converged = False
            else:
                # New fact derived
                converged = False

        # Update current facts
        current = new_results

        if converged:
            break

    return current


def iterative_deepening(
    program,
    facts: Dict[str, torch.Tensor],
    goal: str,
    max_depth: int = 10
) -> torch.Tensor:
    """
    Iterative deepening search for goal

    Args:
        program: TensorProgram with rules
        facts: Initial facts
        goal: Goal tensor name
        max_depth: Maximum search depth

    Returns:
        Goal tensor if found
    """
    for depth in range(1, max_depth + 1):
        result = forward_chain(program, facts, max_iterations=depth)
        if goal in result:
            return result[goal]

    raise ValueError(f"Goal {goal} not derivable within depth {max_depth}")
