"""
Embedding space reasoning for analogical inference

Represents concepts and relations as learned vectors/tensors in embedding space.
Enables fuzzy reasoning via similarity (dot products).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class EmbeddingSpace(nn.Module):
    """
    Embedding space for tensor logic reasoning

    Objects and relations are represented as vectors/tensors.
    Reasoning happens via dot products and tensor operations.
    """

    def __init__(
        self,
        num_objects: int,
        embedding_dim: int = 128,
        temperature: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Args:
            num_objects: Number of objects/concepts to embed
            embedding_dim: Dimensionality of embedding space
            temperature: Temperature for fuzzy reasoning (lower = stricter)
            device: 'cpu' or 'cuda'
        """
        super().__init__()

        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.device = device

        # Object embeddings: [num_objects, embedding_dim]
        # Initialize as random unit vectors (normalized)
        embeddings = torch.randn(num_objects, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        self.object_embeddings = nn.Parameter(embeddings)

        # Relation embeddings: stored as matrices [embedding_dim, embedding_dim]
        self.relations = nn.ParameterDict()

        # Index to name mapping
        self.object_names = {}
        self.name_to_index = {}

    def add_object(self, name: str, index: int, embedding: Optional[torch.Tensor] = None):
        """
        Add or update an object embedding

        Args:
            name: Object name
            index: Object index
            embedding: Optional initial embedding vector
        """
        self.object_names[index] = name
        self.name_to_index[name] = index

        if embedding is not None:
            with torch.no_grad():
                self.object_embeddings[index] = F.normalize(embedding, p=2, dim=0)

    def add_relation(self, name: str, init: str = 'identity'):
        """
        Add a relation

        Args:
            name: Relation name
            init: Initialization: 'identity', 'random', 'zeros'
        """
        if init == 'identity':
            matrix = torch.eye(self.embedding_dim)
        elif init == 'random':
            matrix = torch.randn(self.embedding_dim, self.embedding_dim) * 0.1
        elif init == 'zeros':
            matrix = torch.zeros(self.embedding_dim, self.embedding_dim)
        else:
            raise ValueError(f"Unknown init: {init}")

        self.relations[name] = nn.Parameter(matrix)

    def embed_relation_from_facts(
        self,
        name: str,
        fact_pairs: List[Tuple[int, int]]
    ):
        """
        Create relation embedding from ground truth facts

        Uses superposition: sum of outer products of object embeddings

        Args:
            name: Relation name
            fact_pairs: List of (subject_idx, object_idx) pairs that are true
        """
        # Initialize relation matrix
        relation_matrix = torch.zeros(
            self.embedding_dim, self.embedding_dim,
            device=self.device
        )

        # Superpose outer products
        for i, j in fact_pairs:
            emb_i = self.object_embeddings[i]  # [dim]
            emb_j = self.object_embeddings[j]  # [dim]
            outer = torch.outer(emb_i, emb_j)  # [dim, dim]
            relation_matrix += outer

        self.relations[name] = nn.Parameter(relation_matrix)

    def query_relation(
        self,
        relation_name: str,
        subject: int,
        obj: int,
        use_sigmoid: bool = True
    ) -> torch.Tensor:
        """
        Query if a relation holds between two objects

        Args:
            relation_name: Name of relation
            subject: Subject object index
            obj: Object object index
            use_sigmoid: Apply sigmoid for probability

        Returns:
            Scalar tensor (probability or logit)
        """
        if relation_name not in self.relations:
            raise ValueError(f"Relation {relation_name} not found")

        # Get embeddings
        emb_subj = self.object_embeddings[subject]  # [dim]
        emb_obj = self.object_embeddings[obj]      # [dim]
        relation = self.relations[relation_name]    # [dim, dim]

        # Compute: emb_subj^T @ relation @ emb_obj
        temp = torch.matmul(relation, emb_obj)     # [dim]
        score = torch.dot(emb_subj, temp)          # scalar

        if use_sigmoid:
            return torch.sigmoid(score / self.temperature)
        return score

    def query_all_pairs(
        self,
        relation_name: str,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Query all pairs for a relation

        Args:
            relation_name: Name of relation
            threshold: Threshold for binary output

        Returns:
            Binary matrix [num_objects, num_objects]
        """
        relation = self.relations[relation_name]  # [dim, dim]
        embeddings = self.object_embeddings       # [num_objects, dim]

        # Compute all pairs: embeddings @ relation @ embeddings^T
        temp = torch.matmul(embeddings, relation)  # [num_objects, dim]
        scores = torch.matmul(temp, embeddings.T)  # [num_objects, num_objects]

        # Apply sigmoid and threshold
        probs = torch.sigmoid(scores / self.temperature)
        return (probs > threshold).float()

    def apply_rule(
        self,
        relation1_name: str,
        relation2_name: str,
        output_name: str
    ):
        """
        Apply a composition rule: R3(x,z) = R1(x,y) ∧ R2(y,z)

        In embedding space: compose by matrix multiplication

        Args:
            relation1_name: First relation
            relation2_name: Second relation
            output_name: Output relation name
        """
        r1 = self.relations[relation1_name]  # [dim, dim]
        r2 = self.relations[relation2_name]  # [dim, dim]

        # Compose relations by multiplication
        composed = torch.matmul(r1, r2)  # [dim, dim]

        self.relations[output_name] = nn.Parameter(composed)

    def similarity(self, obj1: int, obj2: int) -> torch.Tensor:
        """
        Compute similarity between two objects

        Args:
            obj1: First object index
            obj2: Second object index

        Returns:
            Similarity score (cosine similarity)
        """
        emb1 = self.object_embeddings[obj1]
        emb2 = self.object_embeddings[obj2]
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))

    def find_similar(self, obj: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar objects

        Args:
            obj: Object index
            top_k: Number of similar objects to return

        Returns:
            List of (index, similarity_score) tuples
        """
        emb = self.object_embeddings[obj]  # [dim]
        all_embs = self.object_embeddings  # [num_objects, dim]

        # Compute similarities
        similarities = torch.matmul(all_embs, emb)  # [num_objects]

        # Get top-k (excluding self)
        similarities[obj] = -float('inf')
        top_scores, top_indices = torch.topk(similarities, k=top_k)

        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append((idx.item(), score.item()))

        return results

    def analogical_inference(
        self,
        source: Tuple[int, int],
        relation_name: str,
        target_subject: int
    ) -> int:
        """
        Analogical reasoning: If (a, b) in R, and c is similar to a,
        infer that (c, ?) might be in R where ? is similar to b

        Args:
            source: Known (subject, object) pair
            relation_name: Relation name
            target_subject: New subject to infer object for

        Returns:
            Most likely object index
        """
        src_subj, src_obj = source

        # Get embeddings
        src_subj_emb = self.object_embeddings[src_subj]
        src_obj_emb = self.object_embeddings[src_obj]
        target_emb = self.object_embeddings[target_subject]
        relation = self.relations[relation_name]

        # Compute transformation: src_obj_emb = relation @ src_subj_emb
        # Apply to target: target_obj_emb = relation @ target_emb
        inferred_obj_emb = torch.matmul(relation, target_emb)

        # Find most similar object to inferred embedding
        similarities = torch.matmul(self.object_embeddings, inferred_obj_emb)
        most_similar = torch.argmax(similarities)

        return most_similar.item()
