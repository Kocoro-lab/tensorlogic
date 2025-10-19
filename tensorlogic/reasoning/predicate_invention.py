"""
Predicate invention pipeline using RESCAL factorization.

Steps:
 1) Split triples into train/validation
 2) Train RESCAL on train triples
 3) Extract candidate invented predicates from embeddings
 4) Rank candidates by validation F1 (pair-wise), select top-K
 5) Register selected predicates into EmbeddingSpace and/or TensorProgram
"""

from typing import List, Tuple, Optional, Dict

import torch

from .decomposition import RESCALModel, RESCALTrainer, extract_predicates_from_embeddings, latent_outer_scores


Triple = Tuple[int, int, int]  # (head, rel, tail)


def split_triples(triples: List[Triple], val_ratio: float = 0.2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split triples list into train and validation tensors [M,3]."""
    if not triples:
        return torch.empty((0, 3), dtype=torch.long), torch.empty((0, 3), dtype=torch.long)
    t = torch.tensor(triples, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(t.shape[0], generator=g)
    t = t[perm]
    n_val = int(t.shape[0] * val_ratio)
    val = t[:n_val]
    train = t[n_val:]
    return train, val


def triples_to_pair_set(triples: torch.Tensor) -> set:
    """Convert [M,3] triples to a set of (i,j) pairs, ignoring relation."""
    return set((int(i.item()), int(j.item())) for i, _, j in triples)


def evaluate_candidate(adj: torch.Tensor, val_pairs: set) -> Dict[str, float]:
    """Compute precision, recall, F1 of candidate adjacency against validation pairs."""
    # Predicted positives
    idx_pred = (adj > 0).nonzero(as_tuple=False)
    pred_pairs = set((int(i.item()), int(j.item())) for i, j in idx_pred)

    tp = len(pred_pairs & val_pairs)
    pp = max(len(pred_pairs), 1)  # avoid div0
    vp = max(len(val_pairs), 1)
    precision = tp / pp
    recall = tp / vp
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "pp": float(pp), "vp": float(vp)}


def select_top_predicates(
    candidates: List[torch.Tensor],
    val_pairs: set,
    top_k: int = 5
) -> List[Tuple[int, Dict[str, float]]]:
    """Rank candidates by F1 and return top_k [(index, metrics)]."""
    scored: List[Tuple[int, Dict[str, float]]] = []
    for idx, adj in enumerate(candidates):
        metrics = evaluate_candidate(adj, val_pairs)
        scored.append((idx, metrics))
    scored.sort(key=lambda x: x[1]["f1"], reverse=True)
    return scored[:top_k]


def iou(adj_a: torch.Tensor, adj_b: torch.Tensor) -> float:
    """Intersection over Union for Boolean adjacencies."""
    a = (adj_a > 0).float()
    b = (adj_b > 0).float()
    inter = (a * b).sum().item()
    union = ((a + b).clamp_max_(1.0)).sum().item()
    if union == 0:
        return 0.0
    return inter / union


def dedup_candidates(candidates: List[torch.Tensor], indices: List[int], iou_threshold: float = 0.9) -> List[int]:
    """Prune near-duplicate candidates by IoU threshold; keep earlier indices."""
    kept: List[int] = []
    for idx in indices:
        dup = False
        for j in kept:
            if iou(candidates[idx], candidates[j]) >= iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(idx)
    return kept


def auc_and_hits(scores: torch.Tensor, val_pairs: set, sample_neg: int = 10000, hits_k: int = 50) -> Dict[str, float]:
    """
    Compute AUC (pos vs random neg) and Hits@K (global) from a continuous score
    matrix. For large N, sampling negatives is used.
    """
    N = scores.shape[0]
    # Positives
    pos = torch.tensor(list(val_pairs), dtype=torch.long)
    pos_scores = scores[pos[:, 0], pos[:, 1]] if pos.numel() > 0 else torch.tensor([])

    # Negatives
    neg_i = torch.randint(0, N, (sample_neg,))
    neg_j = torch.randint(0, N, (sample_neg,))
    neg_scores = scores[neg_i, neg_j]

    # AUC via pairwise comparisons
    if pos_scores.numel() == 0:
        auc = 0.5
    else:
        # Expand and compare: pos_scores[:,None] vs neg_scores[None,:]
        cmp = (pos_scores.view(-1, 1) > neg_scores.view(1, -1)).float()
        ties = (pos_scores.view(-1, 1) == neg_scores.view(1, -1)).float() * 0.5
        auc = (cmp + ties).mean().item()

    # Hits@K (global top-K)
    flat_scores = scores.flatten()
    K = min(hits_k, flat_scores.numel())
    topk_vals, topk_idx = torch.topk(flat_scores, K)
    # Map topk linear indices to (i,j)
    ii = (topk_idx // N).tolist()
    jj = (topk_idx % N).tolist()
    top_pairs = set(zip(ii, jj))
    hits = len(top_pairs & val_pairs) / max(len(val_pairs), 1)

    return {"auc": float(auc), "hits@k": float(hits)}


def register_predicates(
    names: List[str],
    adjs: List[torch.Tensor],
    embedding_space=None,
    tensor_program=None,
):
    """
    Register invented predicates into EmbeddingSpace and/or TensorProgram.

    For EmbeddingSpace: add relation via superposition from fact pairs.
    For TensorProgram: add fixed Boolean tensors as facts.
    """
    if embedding_space is None and tensor_program is None:
        return

    for name, adj in zip(names, adjs):
        # Extract fact pairs
        idx = (adj > 0).nonzero(as_tuple=False)
        pairs = [(int(i.item()), int(j.item())) for i, j in idx]

        if embedding_space is not None:
            if name in embedding_space.relations:
                continue
            embedding_space.add_relation(name, init='zeros')
            embedding_space.embed_relation_from_facts(name, pairs)

        if tensor_program is not None:
            # Store as float Boolean matrix
            tensor_program.add_tensor(name, data=adj.float(), learnable=False)


def invent_and_register_rescal(
    triples: List[Triple],
    num_objects: int,
    num_relations: int,
    rank: int = 16,
    top_k: int = 3,
    val_ratio: float = 0.2,
    steps: int = 3,
    negative_k: int = 5,
    embedding_space=None,
    tensor_program=None,
    verbose: bool = True,
) -> Dict:
    """
    End-to-end pipeline for RESCAL-based predicate invention.

    Returns a dict with selected predicate indices, metrics, and names.
    """
    train, val = split_triples(triples, val_ratio=val_ratio)
    val_pairs = triples_to_pair_set(val)

    model = RESCALModel(num_objects=num_objects, num_relations=num_relations, rank=rank)
    trainer = RESCALTrainer(model, lr=1e-2)

    for s in range(steps):
        loss = trainer.train_epoch(train, num_objects=num_objects, negative_k=negative_k, batch_size=2048, verbose=verbose)
        if verbose:
            print(f"[RESCAL] Step {s+1}/{steps} loss={loss:.4f}")

    # Extract candidates (binary) and continuous scores per latent dim
    with torch.no_grad():
        score_mats = latent_outer_scores(model.E)
        cands = extract_predicates_from_embeddings(model.E, percentile=95.0)

    # Rank by F1, then enrich metrics with AUC/Hits, and deduplicate by IoU
    selected = select_top_predicates(cands, val_pairs, top_k=len(cands))
    indices_all = [i for i, _ in selected]
    # Dedup
    indices_dedup = dedup_candidates(cands, indices_all, iou_threshold=0.9)
    # Compute AUC/Hits for deduped
    enriched = []
    for i in indices_dedup:
        m = evaluate_candidate(cands[i], val_pairs)
        ah = auc_and_hits(score_mats[i], val_pairs, sample_neg=min(10000, num_objects * num_objects), hits_k=50)
        m.update(ah)
        enriched.append((i, m))
    # Final top_k by F1 then AUC
    enriched.sort(key=lambda x: (x[1]["f1"], x[1]["auc"]), reverse=True)
    selected = enriched[:top_k]
    indices = [i for i, _ in selected]
    metrics = [m for _, m in selected]

    names = [f"invented_{i}" for i in indices]
    adjs = [cands[i] for i in indices]

    register_predicates(names, adjs, embedding_space=embedding_space, tensor_program=tensor_program)

    return {
        "indices": indices,
        "metrics": metrics,
        "names": names,
        "count": len(indices),
    }
