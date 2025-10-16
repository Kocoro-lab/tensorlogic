"""
Backward chaining for goal-directed reasoning

Starts from a query and works backwards through rules.
"""

import torch
from typing import Dict, Optional, List, Tuple


def backward_chain(
    program,
    goal: str,
    query_indices: Optional[torch.Tensor] = None,
    facts: Optional[Dict[str, torch.Tensor]] = None,
    max_depth: int = 10
) -> torch.Tensor:
    """
    Backward chaining from a goal

    Args:
        program: TensorProgram with rules
        goal: Goal tensor name to prove
        query_indices: Specific indices to query (e.g., [i, j] for relation(i,j))
        facts: Known facts
        max_depth: Maximum recursion depth

    Returns:
        Tensor with query results
    """
    if facts is None:
        facts = {}

    # Simple implementation: check if goal is in facts
    if goal in facts:
        result = facts[goal]
        if query_indices is not None:
            # Extract specific indices
            return result[tuple(query_indices)]
        return result

    # Check if goal can be computed from equations
    if goal in program.equations:
        # Get dependencies
        func, inputs = program.equations[goal]

        # Recursively evaluate inputs
        input_tensors = {}
        for inp in inputs:
            input_tensors[inp] = backward_chain(
                program, inp, None, facts, max_depth - 1
            )

        # Apply rule
        result = func(input_tensors)

        if query_indices is not None:
            return result[tuple(query_indices)]
        return result

    # Goal not derivable
    raise ValueError(f"Goal {goal} not derivable from available facts and rules")


def query(
    program,
    goal_expr: str,
    facts: Dict[str, torch.Tensor],
    variables: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    Query the knowledge base

    Examples:
        query(program, "grandparent", facts) # All grandparent relationships
        query(program, "grandparent", facts, {"i": 0, "k": 2}) # Is 0 grandparent of 2?

    Args:
        program: TensorProgram
        goal_expr: Goal expression (tensor name)
        facts: Known facts
        variables: Variable bindings for specific queries

    Returns:
        Query result tensor
    """
    if variables:
        # Extract specific indices
        indices = list(variables.values())
        return backward_chain(program, goal_expr, torch.tensor(indices), facts)
    else:
        # Return full tensor
        return backward_chain(program, goal_expr, None, facts)
