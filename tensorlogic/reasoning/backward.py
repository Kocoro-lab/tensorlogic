"""
Recursive goal-directed evaluation for TensorProgram

Evaluates a goal tensor by recursively resolving its dependencies through
the program's equation graph. This is a simple recursive forward evaluation
(not Prolog-style backward chaining with unification or variable grounding).

Limitations:
- No unification or variable binding engine
- No memoization of intermediate results
- Cyclic equations exhaust max_depth with a ValueError
"""

import warnings
import torch
from typing import Dict, Optional, List, Tuple


def resolve_goal(
    program,
    goal: str,
    query_indices: Optional[torch.Tensor] = None,
    facts: Optional[Dict[str, torch.Tensor]] = None,
    max_depth: int = 10
) -> torch.Tensor:
    """
    Recursively evaluate a goal tensor by resolving its equation dependencies.

    Walks the program's equation graph depth-first, evaluating each dependency
    before applying the goal's equation function. Equivalent to lazy forward
    evaluation rooted at the goal.

    Args:
        program: TensorProgram with rules
        goal: Goal tensor name to evaluate
        query_indices: Specific indices to extract from result (e.g., [i, j])
        facts: Known facts (base tensors that don't need derivation)
        max_depth: Maximum recursion depth (guards against cycles)

    Returns:
        Tensor with evaluation results

    Raises:
        ValueError: If max_depth is exceeded (likely cyclic equations)
            or if the goal cannot be derived from facts and rules
    """
    if facts is None:
        facts = {}

    # Check if goal is directly available as a fact or constant
    all_available = {**program.constants, **facts}
    if goal in all_available:
        result = all_available[goal]
        if query_indices is not None:
            return result[tuple(query_indices)]
        return result

    if max_depth < 0:
        raise ValueError(
            f"Max depth exceeded while evaluating '{goal}'. "
            "This usually indicates cyclic equation dependencies."
        )

    # Check if goal can be computed from equations
    if goal in program.equations:
        func, inputs = program.equations[goal]

        # Recursively evaluate inputs
        input_tensors = {}
        for inp in inputs:
            input_tensors[inp] = resolve_goal(
                program, inp, None, facts, max_depth - 1
            )

        result = func(input_tensors)

        if query_indices is not None:
            return result[tuple(query_indices)]
        return result

    raise ValueError(f"Goal '{goal}' not derivable from available facts and rules")


def backward_chain(
    program,
    goal: str,
    query_indices: Optional[torch.Tensor] = None,
    facts: Optional[Dict[str, torch.Tensor]] = None,
    max_depth: int = 10,
) -> torch.Tensor:
    """Deprecated alias for :func:`resolve_goal`.

    This function performs recursive forward evaluation, not Prolog-style
    backward chaining. Use ``resolve_goal`` instead.
    """
    warnings.warn(
        "backward_chain is deprecated and will be removed in a future release. "
        "Use resolve_goal instead (same signature, same behavior).",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolve_goal(program, goal, query_indices, facts, max_depth)


def query(
    program,
    goal_expr: str,
    facts: Dict[str, torch.Tensor],
    variables: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    Evaluate a goal expression against known facts.

    Convenience wrapper around backward_chain that translates variable
    bindings into tensor indices.

    Examples:
        query(program, "grandparent", facts)  # Full relation tensor
        query(program, "grandparent", facts, {"i": 0, "k": 2})  # Specific entry

    Args:
        program: TensorProgram
        goal_expr: Goal expression (tensor name)
        facts: Known facts
        variables: Variable bindings mapping dimension names to indices

    Returns:
        Query result tensor
    """
    if variables:
        indices = list(variables.values())
        return resolve_goal(program, goal_expr, torch.tensor(indices), facts)
    else:
        return resolve_goal(program, goal_expr, None, facts)
