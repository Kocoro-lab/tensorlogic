"""
Core TensorProgram class for defining and executing tensor logic programs
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Union
from ..ops.logical import (
    logical_join, logical_project, logical_union, logical_negation,
    apply_nonlinearity, forward_chain_step
)


class TensorProgram(nn.Module):
    """
    A Tensor Logic program consisting of tensor equations

    Programs define:
    - Tensors (facts, relations, weights)
    - Equations (rules, transformations)
    - Inference modes (forward/backward chaining)
    """

    def __init__(self, mode: str = 'continuous', device: str = 'cpu'):
        """
        Args:
            mode: 'boolean' for strict logic, 'continuous' for neural/fuzzy
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        self.mode = mode
        self.device = device

        # Storage for tensors (facts, relations, weights)
        self.tensors = nn.ParameterDict()  # Learnable tensors
        self.constants = {}  # Fixed tensors

        # Equations (rules) stored as functions
        self.equations = {}
        self.equation_order = []  # Execution order

    def add_tensor(
        self,
        name: str,
        shape: Optional[tuple] = None,
        data: Optional[torch.Tensor] = None,
        learnable: bool = False,
        init: str = 'zeros'
    ):
        """
        Add a tensor to the program

        Args:
            name: Tensor name
            shape: Shape of tensor (if creating new)
            data: Initial data (if provided)
            learnable: Whether this tensor has learnable parameters
            init: Initialization: 'zeros', 'ones', 'random', 'randn'
        """
        if data is not None:
            tensor = data.to(self.device)
        elif shape is not None:
            if init == 'zeros':
                tensor = torch.zeros(shape, device=self.device)
            elif init == 'ones':
                tensor = torch.ones(shape, device=self.device)
            elif init == 'random':
                tensor = torch.rand(shape, device=self.device)
            elif init == 'randn':
                tensor = torch.randn(shape, device=self.device) * 0.1
            else:
                raise ValueError(f"Unknown init: {init}")
        else:
            raise ValueError("Must provide either shape or data")

        if learnable:
            self.tensors[name] = nn.Parameter(tensor)
        else:
            self.constants[name] = tensor

    def add_equation(
        self,
        name: str,
        func: Union[str, Callable],
        inputs: Optional[List[str]] = None,
        equation_str: Optional[str] = None
    ):
        """
        Add an equation (rule) to the program

        Args:
            name: Name of output tensor
            func: Function to compute output, or string expression
            inputs: List of input tensor names (if func is callable)
            equation_str: Einstein equation string (e.g., 'ij,jk->ik')
        """
        if isinstance(func, str):
            # Parse simple expressions
            self.equations[name] = self._parse_expression(name, func, inputs)
        else:
            # Custom function
            self.equations[name] = (func, inputs)

        self.equation_order.append(name)

    def _parse_expression(self, name: str, expr: str, inputs: Optional[List[str]]):
        """Parse simple string expressions into functions"""

        # Handle simple cases like 'parent @ parent' or 'parent * parent'
        if '@' in expr:
            # Matrix multiplication
            parts = expr.split('@')
            if len(parts) == 2:
                t1_name, t2_name = [p.strip() for p in parts]

                def rule_func(tensors):
                    t1 = tensors.get(t1_name, self.constants.get(t1_name))
                    t2 = tensors.get(t2_name, self.constants.get(t2_name))
                    if t1 is None or t2 is None:
                        raise ValueError(f"Unknown tensor name(s) in expression: {expr}")
                    return torch.matmul(t1, t2)

                return (rule_func, [t1_name, t2_name])

            raise ValueError(f"Invalid matrix multiplication expression: {expr}")

        if '*' in expr:
            # Alias for matrix multiplication
            parts = expr.split('*')
            if len(parts) == 2:
                t1_name, t2_name = [p.strip() for p in parts]

                def rule_func(tensors):
                    t1 = tensors.get(t1_name, self.constants.get(t1_name))
                    t2 = tensors.get(t2_name, self.constants.get(t2_name))
                    if t1 is None or t2 is None:
                        raise ValueError(f"Unknown tensor name(s) in expression: {expr}")
                    return torch.matmul(t1, t2)

                return (rule_func, [t1_name, t2_name])

            raise ValueError(f"Invalid matrix multiplication expression: {expr}")

        # If inputs provided, create a direct evaluator
        if inputs:
            if len(inputs) > 1 and not expr:
                raise ValueError(
                    f"Cannot evaluate multiple inputs for {name} without an explicit expression"
                )

            def eval_func(tensors):
                # Resolve input tensors from current results, with constants as fallback.
                values = [tensors.get(k, self.constants.get(k)) for k in inputs]
                if any(v is None for v in values):
                    missing = [k for k, v in zip(inputs, values) if v is None]
                    raise ValueError(f"Unknown tensor name(s): {missing}")

                if len(values) == 1:
                    return values[0]

                return forward_chain_step(values, equation=expr)

            return (eval_func, inputs)

        raise ValueError(f"Cannot parse expression: {expr}")

    def forward(self, input_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Execute the program (forward chaining)

        Args:
            input_data: Dictionary of input tensor data

        Returns:
            Dictionary of all computed tensors
        """
        # Initialize with constants and inputs
        results = {**self.constants}

        if input_data:
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    results[k] = v.to(self.device)
                else:
                    results[k] = torch.tensor(v, device=self.device, dtype=torch.float32)

        # Add learnable parameters
        for k, v in self.tensors.items():
            results[k] = v

        # Execute equations in order
        for eq_name in self.equation_order:
            func, inputs = self.equations[eq_name]
            result = func(results)

            # Apply mode-specific processing
            if self.mode == 'boolean':
                result = (result > 0.5).float()

            results[eq_name] = result

        return results

    def query(self, goal: str, bindings: Optional[Dict] = None) -> torch.Tensor:
        """
        Query the program (backward chaining)

        Args:
            goal: Goal tensor name to query
            bindings: Optional variable bindings

        Returns:
            Result tensor
        """
        # Simple implementation: run forward and extract goal
        results = self.forward(bindings)
        return results.get(goal)

    def get_learnable_parameters(self):
        """Get all learnable parameters"""
        return list(self.tensors.parameters())

    def to(self, *args, **kwargs):
        """Move program to device and/or dtype."""
        super().to(*args, **kwargs)

        if len(args) >= 1 and isinstance(args[0], (str, torch.device)):
            self.device = str(args[0])
        elif "device" in kwargs and kwargs["device"] is not None:
            self.device = str(kwargs["device"])

        for k, v in self.constants.items():
            self.constants[k] = v.to(*args, **kwargs)
        return self


class RuleBasedProgram(TensorProgram):
    """
    Specialized program for rule-based symbolic reasoning

    Provides convenience methods for defining logical rules.
    """

    def __init__(self, mode: str = 'boolean', device: str = 'cpu'):
        super().__init__(mode=mode, device=device)

    def add_rule(
        self,
        name: str,
        antecedents: List[str],
        equation: str
    ):
        """
        Add a logical rule

        Args:
            name: Name of consequent (output)
            antecedents: List of antecedent tensor names
            equation: Einstein equation for joining antecedents
        """
        def rule_func(tensors):
            ant_tensors = [tensors[a] for a in antecedents]
            if len(ant_tensors) == 2:
                return torch.einsum(equation, ant_tensors[0], ant_tensors[1])
            else:
                return forward_chain_step(ant_tensors, equation)

        self.equations[name] = (rule_func, antecedents)
        self.equation_order.append(name)

    def add_fact(
        self,
        name: str,
        data: torch.Tensor
    ):
        """Add a fact (ground truth data)"""
        self.add_tensor(name, data=data, learnable=False)
