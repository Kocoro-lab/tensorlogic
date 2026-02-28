"""Tests for TensorProgram forward execution, boolean mode, and logical ops."""

import pytest
import torch

from tensorlogic.core.program import TensorProgram, RuleBasedProgram
from tensorlogic.ops.logical import (
    logical_join, logical_project, logical_union, logical_negation,
)


# --- TensorProgram ---

class TestTensorProgram:
    def test_forward_single_equation(self):
        prog = TensorProgram(mode='continuous')
        prog.add_tensor("a", data=torch.eye(3))
        prog.add_tensor("b", data=torch.ones(3, 3))
        prog.add_equation("c", "a @ b")

        results = prog.forward()
        assert "c" in results
        assert torch.allclose(results["c"], torch.ones(3, 3))

    def test_forward_chained_equations(self):
        """Two chained equations: c = a @ b, d = c @ a."""
        prog = TensorProgram(mode='continuous')
        prog.add_tensor("a", data=torch.eye(3))
        prog.add_tensor("b", data=torch.ones(3, 3) * 2)
        prog.add_equation("c", "a @ b")
        prog.add_equation("d", "c @ a")

        results = prog.forward()
        assert torch.allclose(results["d"], torch.ones(3, 3) * 2)

    def test_forward_with_input_data(self):
        prog = TensorProgram(mode='continuous')
        prog.add_equation("out", "x @ w")

        results = prog.forward({
            "x": torch.eye(2),
            "w": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        })
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(results["out"], expected)

    def test_boolean_mode_thresholds_final_only(self):
        """Boolean thresholding should only apply to final outputs,
        not to intermediates consumed by downstream equations."""
        prog = TensorProgram(mode='boolean')
        # parent has a path A->B->C
        parent = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        prog.add_tensor("parent", data=parent)
        prog.add_equation("grandparent", "parent @ parent")

        results = prog.forward()
        # grandparent[0,2] should be 1 (A is grandparent of C)
        assert results["grandparent"][0, 2].item() == 1.0
        # grandparent[0,0] should be 0 (A is not grandparent of A)
        assert results["grandparent"][0, 0].item() == 0.0

    def test_boolean_intermediate_not_thresholded(self):
        """Intermediate used by another equation should preserve raw values."""
        prog = TensorProgram(mode='boolean')
        # Values that would be affected by premature thresholding
        prog.add_tensor("a", data=torch.tensor([[0.0, 0.3], [0.7, 0.0]]))
        prog.add_tensor("b", data=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        prog.add_equation("mid", "a @ b")   # intermediate
        prog.add_equation("out", "mid @ b")  # final

        results = prog.forward()
        # "mid" is consumed by "out", so it should NOT be thresholded
        # "out" IS final, so it should be thresholded
        assert results["out"].dtype == torch.float32

    def test_add_tensor_requires_shape_or_data(self):
        prog = TensorProgram()
        with pytest.raises(ValueError, match="Must provide"):
            prog.add_tensor("bad")

    def test_learnable_parameters(self):
        prog = TensorProgram()
        prog.add_tensor("w", shape=(3, 3), learnable=True, init='randn')
        params = prog.get_learnable_parameters()
        assert len(params) == 1
        assert params[0].requires_grad


class TestRuleBasedProgram:
    def test_add_rule_and_forward(self):
        prog = RuleBasedProgram(mode='boolean')
        parent = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        prog.add_fact("parent", parent)
        prog.add_rule("grandparent", ["parent", "parent"], "ij,jk->ik")

        results = prog.forward()
        assert results["grandparent"][0, 2].item() == 1.0


# --- Logical ops ---

class TestLogicalOps:
    def test_join_2d_continuous(self):
        a = torch.eye(3)
        b = torch.ones(3, 3)
        result = logical_join(a, b, mode='continuous')
        assert torch.allclose(result, torch.ones(3, 3))

    def test_join_2d_boolean(self):
        a = torch.tensor([[0.6, 0.2], [0.8, 0.1]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = logical_join(a, b, mode='boolean')
        # matmul then threshold
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_join_einsum(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        result = logical_join(a, b, equation='ij,jk->ik', mode='continuous')
        expected = torch.einsum('ij,jk->ik', a, b)
        assert torch.allclose(result, expected)

    def test_join_non2d_requires_equation(self):
        with pytest.raises(ValueError, match="equation parameter required"):
            logical_join(torch.randn(2, 3, 4), torch.randn(2, 3, 4))

    def test_project_boolean(self):
        t = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        result = logical_project(t, dim=1, mode='boolean')
        assert torch.equal(result, torch.tensor([1.0, 0.0]))

    def test_project_continuous(self):
        t = torch.tensor([[0.3, 0.7], [0.1, 0.2]])
        result = logical_project(t, dim=1, mode='continuous')
        assert torch.allclose(result, torch.tensor([1.0, 0.3]))

    def test_union_boolean(self):
        a = torch.tensor([1.0, 0.0, 1.0])
        b = torch.tensor([0.0, 1.0, 1.0])
        result = logical_union(a, b, mode='boolean')
        assert torch.equal(result, torch.tensor([1.0, 1.0, 1.0]))

    def test_union_continuous(self):
        a = torch.tensor([0.5, 0.0])
        b = torch.tensor([0.0, 0.5])
        result = logical_union(a, b, mode='continuous')
        # P(A or B) = P(A) + P(B) - P(A)*P(B)
        expected = torch.tensor([0.5, 0.5])
        assert torch.allclose(result, expected)

    def test_negation_boolean(self):
        t = torch.tensor([1.0, 0.0, 0.8])
        result = logical_negation(t, mode='boolean')
        assert torch.equal(result, torch.tensor([0.0, 1.0, 0.0]))

    def test_negation_continuous(self):
        t = torch.tensor([0.3, 0.7])
        result = logical_negation(t, mode='continuous')
        assert torch.allclose(result, torch.tensor([0.7, 0.3]))
