import sys

import pytest
import torch


def test_tensorlogic_import_does_not_import_matplotlib():
    # Ensure matplotlib isn't already in sys.modules (and clear any remnants).
    for key in list(sys.modules.keys()):
        if key.startswith("matplotlib"):
            sys.modules.pop(key, None)

    import tensorlogic  # noqa: F401

    assert not any(key.startswith("matplotlib") for key in sys.modules.keys())


def test_init_embedding_space_initializes_parameter_embeddings():
    from tensorlogic.reasoning.embed import EmbeddingSpace
    from tensorlogic.utils.init_strategies import init_embedding_space

    space = EmbeddingSpace(num_objects=10, embedding_dim=8, device="cpu")
    space.add_relation("r", init="zeros")

    init_embedding_space(space)

    norms = space.object_embeddings.detach().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def test_embedding_space_ignores_stale_device_attribute():
    from tensorlogic.reasoning.embed import EmbeddingSpace

    space = EmbeddingSpace(num_objects=5, embedding_dim=4, device="cpu")
    space.device = "cuda"  # simulate stale attribute after a .to() or user override

    space.add_relation("parent", init="zeros")
    assert space.relations["parent"].device.type == "cpu"

    space.embed_relation_from_facts("parent", [(0, 1), (1, 2)])
    assert space.relations["parent"].device.type == "cpu"


def test_resolve_goal_max_depth_cycle_raises_value_error():
    from tensorlogic.core.program import TensorProgram
    from tensorlogic.reasoning.backward import resolve_goal

    prog = TensorProgram(mode="continuous")
    prog.add_equation("a", lambda t: t["b"], inputs=["b"])
    prog.add_equation("b", lambda t: t["a"], inputs=["a"])

    with pytest.raises(ValueError):
        resolve_goal(prog, "a", facts={}, max_depth=2)


def test_backward_chain_emits_deprecation_warning():
    import warnings
    from tensorlogic.core.program import TensorProgram
    from tensorlogic.reasoning.backward import backward_chain

    prog = TensorProgram(mode="continuous")
    prog.add_tensor("x", data=torch.ones(2, 2))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        backward_chain(prog, "x", facts={})
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "resolve_goal" in str(w[0].message)

