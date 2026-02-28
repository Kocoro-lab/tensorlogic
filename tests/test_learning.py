"""Tests for EmbeddingTrainer, ContrastiveLoss, GatedMultiHopComposer, and io."""

import pytest
import tempfile
import torch
import torch.nn as nn

from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.reasoning.composer import GatedMultiHopComposer, stack_relation_bank
from tensorlogic.learn.trainer import Trainer, EmbeddingTrainer, PairScoringTrainer
from tensorlogic.learn.losses import ContrastiveLoss
from tensorlogic.utils.io import save_model, load_model, save_checkpoint, load_checkpoint
from tensorlogic.core.program import TensorProgram


# --- EmbeddingTrainer ---

class TestEmbeddingTrainer:
    def _make_space(self, n=6, dim=8):
        space = EmbeddingSpace(num_objects=n, embedding_dim=dim, device='cpu')
        space.add_relation("parent", init='random')
        return space

    def test_rejects_non_embedding_space(self):
        prog = TensorProgram(mode='continuous')
        prog.add_tensor("w", shape=(4, 4), learnable=True)
        with pytest.raises(TypeError, match="EmbeddingTrainer requires an EmbeddingSpace"):
            EmbeddingTrainer(prog)

    def test_train_relation_reduces_loss(self):
        space = self._make_space()
        trainer = EmbeddingTrainer(space, learning_rate=0.05)
        pos = [(0, 1), (1, 2), (2, 3)]
        neg = [(0, 3), (3, 0), (2, 0)]

        # Score before training
        with torch.no_grad():
            scores_before = [space.query_relation("parent", s, o).item() for s, o in pos]

        trainer.train_relation("parent", pos, neg, epochs=50, verbose=False)

        with torch.no_grad():
            scores_after = [space.query_relation("parent", s, o).item() for s, o in pos]

        # Positive scores should increase on average
        assert sum(scores_after) > sum(scores_before)

    def test_train_relation_no_negatives(self):
        """Auto-sampled negatives should work."""
        space = self._make_space()
        trainer = EmbeddingTrainer(space, learning_rate=0.01)
        pos = [(0, 1), (2, 3)]
        trainer.train_relation("parent", pos, epochs=10, verbose=False)

    def test_train_relation_empty_positives(self):
        """Empty positives should return without error."""
        space = self._make_space()
        trainer = EmbeddingTrainer(space, learning_rate=0.01)
        trainer.train_relation("parent", [], epochs=10, verbose=False)


# --- GatedMultiHopComposer ---

class TestGatedMultiHopComposer:
    def test_forward_shape(self):
        dim, R, hops, B = 8, 3, 2, 4
        composer = GatedMultiHopComposer(dim, R, hops)
        subj = torch.randn(B, dim)
        obj = torch.randn(B, dim)
        bank = torch.randn(R, dim, dim)
        scores = composer(subj, obj, bank)
        assert scores.shape == (B,)

    def test_forward_with_attn(self):
        dim, R, hops, B = 8, 3, 2, 4
        composer = GatedMultiHopComposer(dim, R, hops)
        subj = torch.randn(B, dim)
        obj = torch.randn(B, dim)
        bank = torch.randn(R, dim, dim)
        scores, weights = composer.forward_with_attn(subj, obj, bank)
        assert scores.shape == (B,)
        assert len(weights) == hops
        for w in weights:
            assert w.shape == (B, R)
            assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_gradients_flow(self):
        dim, R, hops, B = 8, 3, 2, 4
        composer = GatedMultiHopComposer(dim, R, hops)
        subj = torch.randn(B, dim)
        obj = torch.randn(B, dim)
        bank = torch.randn(R, dim, dim, requires_grad=True)
        scores = composer(subj, obj, bank)
        loss = scores.sum()
        loss.backward()
        assert bank.grad is not None
        for p in composer.parameters():
            assert p.grad is not None


# --- ContrastiveLoss ---

class TestContrastiveLoss:
    def _dummy_scorer(self, model, subjects, objects):
        return torch.sigmoid(subjects.float() - objects.float())

    def test_forward_with_negatives(self):
        loss_fn = ContrastiveLoss(negative_samples=3)
        pos = torch.tensor([[0, 1], [2, 3]])
        neg = torch.tensor([[[0, 2], [0, 3], [0, 4]],
                            [[2, 0], [2, 1], [2, 4]]])
        loss = loss_fn(None, pos, neg, scorer=self._dummy_scorer, num_objects=5)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_forward_auto_sample(self):
        loss_fn = ContrastiveLoss(negative_samples=2)
        pos = torch.tensor([[0, 1], [2, 3]])
        loss = loss_fn(None, pos, scorer=self._dummy_scorer, num_objects=5)
        assert loss.item() > 0

    def test_requires_scorer(self):
        loss_fn = ContrastiveLoss()
        pos = torch.tensor([[0, 1]])
        with pytest.raises(ValueError, match="scorer callable is required"):
            loss_fn(None, pos)


# --- PairScoringTrainer ---

class TestPairScoringTrainer:
    def test_train_epoch(self):
        dim, R = 8, 2
        composer = GatedMultiHopComposer(dim, R, num_hops=1)
        trainer = PairScoringTrainer(composer, learning_rate=1e-3)
        loss_fn = ContrastiveLoss(negative_samples=2)

        bank = torch.randn(R, dim, dim)

        def scorer(model, subjects, objects):
            subj_emb = torch.randn(len(subjects), dim)
            obj_emb = torch.randn(len(objects), dim)
            return torch.sigmoid(model(subj_emb, obj_emb, bank))

        batches = [{'pos': torch.tensor([[0, 1], [2, 3]])} for _ in range(3)]
        avg_loss = trainer.train_epoch(loss_fn, scorer, batches, num_objects=5, verbose=False)
        assert avg_loss > 0


# --- I/O ---

class TestIO:
    def test_save_load_embedding_space(self):
        space = EmbeddingSpace(num_objects=5, embedding_dim=4, device='cpu')
        space.add_relation("parent", init='random')
        space.add_object("alice", 0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_model(space, f.name, metadata={"epoch": 10})
            space2 = EmbeddingSpace(num_objects=5, embedding_dim=4, device='cpu')
            meta = load_model(space2, f.name)

        assert meta["epoch"] == 10
        assert torch.allclose(
            space.object_embeddings.data,
            space2.object_embeddings.data,
        )
        assert "parent" in space2.relations

    def test_save_load_checkpoint(self):
        space = EmbeddingSpace(num_objects=5, embedding_dim=4, device='cpu')
        space.add_relation("r", init='random')
        opt = torch.optim.Adam(space.parameters(), lr=0.01)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_checkpoint(space, opt, epoch=5, loss=0.123, path=f.name)

            space2 = EmbeddingSpace(num_objects=5, embedding_dim=4, device='cpu')
            space2.add_relation("r", init='zeros')
            opt2 = torch.optim.Adam(space2.parameters(), lr=0.01)
            epoch, loss, meta = load_checkpoint(space2, opt2, f.name)

        assert epoch == 5
        assert abs(loss - 0.123) < 1e-6

    def test_load_model_rejects_corrupt_file(self):
        """load_model should raise on a file that isn't a valid torch checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"not a valid torch file")
            f.flush()
            space = EmbeddingSpace(num_objects=5, embedding_dim=4)
            with pytest.raises(Exception):
                load_model(space, f.name)


# --- stack_relation_bank ---

class TestStackRelationBank:
    def test_stack(self):
        pd = nn.ParameterDict({
            "a": nn.Parameter(torch.eye(3)),
            "b": nn.Parameter(torch.ones(3, 3)),
        })
        bank = stack_relation_bank(pd, order=["a", "b"])
        assert bank.shape == (2, 3, 3)
        assert torch.equal(bank[0], torch.eye(3))
