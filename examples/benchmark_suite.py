"""
Benchmark Suite: Boolean vs Embedding vs Composer vs Composer+Invented

Metrics:
  - Speed: training time (s), query throughput (pairs/s)
  - Memory: model param size (MB)
  - Quality: AUC, Hits@K, F1 (on target pairs)

Datasets:
  - Family tree (N≈10, relation: parent, target: grandparent)
  - Small KG (N≈300, R≈3, target: r0∘r1)
  - Synthetic multi-hop (N≈200, R≈2, target: r0∘r0)

Keep it straightforward; avoid over-engineering.
"""

import math
import time
from typing import Dict, List, Tuple

import torch

from tensorlogic.core.program import TensorProgram
from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.reasoning.composer import GatedMultiHopComposer
from tensorlogic.reasoning.closure import compose_sequence
from tensorlogic.learn.losses import ContrastiveLoss
from tensorlogic.learn.trainer import PairScoringTrainer
from tensorlogic.reasoning.predicate_invention import invent_and_register_rescal


Pair = Tuple[int, int]


def model_param_mb(model) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 * 1024)


def pairs_from_adj(adj: torch.Tensor) -> List[Pair]:
    idx = (adj > 0).nonzero(as_tuple=False)
    return [(int(i.item()), int(j.item())) for i, j in idx]


def sparse_auc_hits_f1(scores: torch.Tensor, target_pairs: List[Pair], k: int = 50) -> Dict[str, float]:
    N = scores.shape[0]
    target = set(target_pairs)
    # AUC: compare target vs random negs
    if len(target) == 0:
        auc = 0.5
        f1 = 0.0
    else:
        pos = torch.tensor(list(target), dtype=torch.long)
        pos_scores = scores[pos[:, 0], pos[:, 1]]
        # Sample negatives uniformly
        sample_neg = min(10000, N * N)
        neg_i = torch.randint(0, N, (sample_neg,))
        neg_j = torch.randint(0, N, (sample_neg,))
        neg_scores = scores[neg_i, neg_j]
        cmp = (pos_scores.view(-1, 1) > neg_scores.view(1, -1)).float()
        ties = (pos_scores.view(-1, 1) == neg_scores.view(1, -1)).float() * 0.5
        auc = (cmp + ties).mean().item()
        # F1 via threshold at mean score
        thr = scores.mean().item()
        pred = (scores >= thr)
        pred_pairs = set((int(i.item()), int(j.item())) for i, j in pred.nonzero(as_tuple=False))
        tp = len(pred_pairs & target)
        pp = max(len(pred_pairs), 1)
        vp = max(len(target), 1)
        prec = tp / pp
        rec = tp / vp
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # Hits@K
    flat_scores = scores.flatten()
    K = min(k, flat_scores.numel())
    top_vals, top_idx = torch.topk(flat_scores, K)
    ii = (top_idx // N).tolist()
    jj = (top_idx % N).tolist()
    hits = len(set(zip(ii, jj)) & set(target_pairs)) / max(len(target_pairs), 1)
    return {"auc": float(auc), "hits@k": float(hits), "f1": float(f1)}


def timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0


def build_family(N: int = 10):
    parent = [(i, i + 1) for i in range(N - 1)]
    # target grandparent closure
    adj = torch.zeros((N, N))
    for i, j in parent:
        adj[i, j] = 1.0
    gp = compose_sequence({"parent": adj}, ["parent", "parent"])
    return N, {"parent": adj}, pairs_from_adj(gp)


def build_small_kg(N: int = 300, R: int = 3, density: float = 0.002):
    rels = {}
    for r in range(R):
        adj = (torch.rand(N, N) < density).float()
        adj.fill_diagonal_(0.0)
        rels[f"r{r}"] = adj
    # Target: r0 ∘ r1
    tgt = compose_sequence({"r0": rels["r0"], "r1": rels["r1"]}, ["r0", "r1"])
    return N, rels, pairs_from_adj(tgt)


def boolean_benchmark(N: int, rels: Dict[str, torch.Tensor], target_pairs: List[Pair]) -> Dict[str, float]:
    prog = TensorProgram(mode="boolean")
    for name, adj in rels.items():
        prog.add_tensor(name, data=(adj > 0).float(), learnable=False)
    # Assume target is r0@r1 if two relations exist, else parent@parent
    if set(rels.keys()) == {"parent"}:
        prog.add_equation("target", "parent @ parent")
    else:
        prog.add_equation("target", "r0 @ r1")
    results, dt = timeit(prog.forward)
    target = results["target"]
    scores = target  # boolean scores in {0,1}
    metrics = sparse_auc_hits_f1(scores, target_pairs)
    metrics.update({"train_time": 0.0, "query_time": dt, "pairs_per_s": N * N / max(dt, 1e-9), "model_mb": 0.0})
    return metrics


def embedding_benchmark(N: int, rels: Dict[str, torch.Tensor], target_pairs: List[Pair]) -> Dict[str, float]:
    space = EmbeddingSpace(num_objects=N, embedding_dim=32, temperature=0.2)
    for name, adj in rels.items():
        pairs = pairs_from_adj(adj)
        space.add_relation(name, init="zeros")
        space.embed_relation_from_facts(name, pairs)
    # Compose to get target
    if set(rels.keys()) == {"parent"}:
        space.apply_rule("parent", "parent", "target")
    else:
        space.apply_rule("r0", "r1", "target")
    with torch.no_grad():
        emb = space.object_embeddings
        R = space.relations["target"]
        scores = torch.sigmoid((emb @ R) @ emb.t() / space.temperature)
    metrics = sparse_auc_hits_f1(scores, target_pairs)
    metrics.update({"train_time": 0.0, "query_time": 0.0, "pairs_per_s": float("nan"), "model_mb": model_param_mb(space)})
    return metrics


def composer_benchmark(N: int, rels: Dict[str, torch.Tensor], target_pairs: List[Pair], with_invented: bool = False) -> Dict[str, float]:
    # Embedding space to host relation bank
    space = EmbeddingSpace(num_objects=N, embedding_dim=32, temperature=0.2)
    rel_names = sorted(list(rels.keys()))
    for name in rel_names:
        space.add_relation(name, init="random")
    # Optionally invent predicates
    train_time = 0.0
    if with_invented:
        # Build triples from rels; assign ids by rel_names index
        triples: List[Tuple[int, int, int]] = []
        for ridx, name in enumerate(rel_names):
            pairs = pairs_from_adj(rels[name])
            triples += [(i, ridx, j) for i, j in pairs]
        _, dt = timeit(
            invent_and_register_rescal,
            triples=triples,
            num_objects=N,
            num_relations=len(rel_names),
            rank=16,
            top_k=2,
            val_ratio=0.2,
            steps=3,
            embedding_space=space,
            tensor_program=None,
            verbose=False,
        )
        train_time += dt
        # Extend relation bank with invented ones
        rel_names = sorted(list(space.relations.keys()))

    # Train composer on target pairs
    composer = GatedMultiHopComposer(embedding_dim=space.embedding_dim, num_relations=len(rel_names), num_hops=2)
    loss_fn = ContrastiveLoss(margin=0.5, negative_samples=64, use_logits=False)
    trainer = PairScoringTrainer(composer, learning_rate=0.01)
    pos = torch.tensor(target_pairs, dtype=torch.long)
    batches = [{"pos": pos} for _ in range(100)]

    def scorer(model, subjects, objects):
        return space.score_with_composer_batched(model, subjects, objects, relation_order=rel_names, use_sigmoid=True, batch_size=8192)

    _, dt = timeit(trainer.train_epoch, loss_fn=loss_fn, scorer=scorer, batches=batches, num_objects=N, verbose=False)
    train_time += dt

    # Evaluate on all pairs
    all_i, all_j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    subjects = all_i.reshape(-1)
    objects = all_j.reshape(-1)
    probs, qt = timeit(space.score_with_composer_batched, composer, subjects, objects, rel_names, True, 65536)
    scores = probs.reshape(N, N)
    metrics = sparse_auc_hits_f1(scores, target_pairs)
    metrics.update({
        "train_time": train_time,
        "query_time": qt,
        "pairs_per_s": (N * N) / max(qt, 1e-9),
        "model_mb": model_param_mb(composer) + model_param_mb(space),
    })
    return metrics


def run_benchmarks():
    torch.manual_seed(0)

    scenarios = [
        ("family",) + build_family(N=10),
        ("smallkg",) + build_small_kg(N=300, R=3, density=0.002),
        ("synthetic",) + build_small_kg(N=200, R=2, density=0.003),
    ]

    for name, N, rels, target_pairs in scenarios:
        print(f"\n=== Scenario: {name} (N={N}, rels={list(rels.keys())}) ===")
        # Boolean
        bool_metrics = boolean_benchmark(N, rels, target_pairs)
        print("Boolean:", bool_metrics)
        # Embedding
        emb_metrics = embedding_benchmark(N, rels, target_pairs)
        print("Embedding:", emb_metrics)
        # Composer
        comp_metrics = composer_benchmark(N, rels, target_pairs, with_invented=False)
        print("Composer:", comp_metrics)
        # Composer + Invented
        inv_metrics = composer_benchmark(N, rels, target_pairs, with_invented=True)
        print("Composer+Invented:", inv_metrics)


if __name__ == "__main__":
    run_benchmarks()

