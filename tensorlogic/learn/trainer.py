"""
Training utilities for Tensor Logic programs

Provides learning capabilities using PyTorch autograd.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Callable, List
from tqdm import tqdm


class Trainer:
    """
    Trainer for Tensor Logic programs

    Supports learning facts, relations, and rules from data.
    """

    def __init__(
        self,
        program,
        optimizer_type: str = 'adam',
        learning_rate: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Args:
            program: TensorProgram or EmbeddingSpace to train
            optimizer_type: 'sgd', 'adam', 'adamw'
            learning_rate: Learning rate
            device: 'cpu' or 'cuda'
        """
        self.program = program
        self.device = device
        self.program.to(device)

        # Get learnable parameters
        params = list(program.parameters())

        # Initialize optimizer
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(params, lr=learning_rate)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(params, lr=learning_rate)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(params, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.history = {'loss': [], 'accuracy': []}

    def train_step(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_fn: Optional[Callable] = None
    ) -> float:
        """
        Single training step

        Args:
            inputs: Input tensors
            targets: Target tensors
            loss_fn: Custom loss function (default: MSE)

        Returns:
            Loss value
        """
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.program.forward(inputs)

        # Compute loss
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = 0.0
        for target_name, target_value in targets.items():
            if target_name in outputs:
                pred = outputs[target_name]
                target = target_value.to(self.device)
                loss = loss_fn(pred, target)
                total_loss += loss
            else:
                raise ValueError(f"Target {target_name} not in outputs")

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train(
        self,
        train_data: List[Dict],
        epochs: int = 100,
        batch_size: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        validation_data: Optional[List[Dict]] = None,
        verbose: bool = True
    ):
        """
        Train the program

        Args:
            train_data: List of {'inputs': {...}, 'targets': {...}} dicts
            epochs: Number of epochs
            batch_size: Batch size (None = full batch)
            loss_fn: Custom loss function
            validation_data: Optional validation data
            verbose: Show progress bar
        """
        iterator = tqdm(range(epochs)) if verbose else range(epochs)

        for epoch in iterator:
            total_loss = 0.0
            num_batches = 0

            # Simple full-batch training
            if batch_size is None or batch_size >= len(train_data):
                for data in train_data:
                    loss = self.train_step(
                        data['inputs'],
                        data['targets'],
                        loss_fn
                    )
                    total_loss += loss
                    num_batches += 1
            else:
                # Mini-batch training
                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i:i+batch_size]
                    for data in batch:
                        loss = self.train_step(
                            data['inputs'],
                            data['targets'],
                            loss_fn
                        )
                        total_loss += loss
                        num_batches += 1

            # Guard against empty training data
            if num_batches == 0:
                continue

            avg_loss = total_loss / num_batches
            self.history['loss'].append(avg_loss)

            # Validation
            if validation_data:
                val_loss = self.evaluate(validation_data, loss_fn)
                if verbose:
                    iterator.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}'
                    })
            else:
                if verbose:
                    iterator.set_postfix({'loss': f'{avg_loss:.4f}'})

    def evaluate(
        self,
        data: List[Dict],
        loss_fn: Optional[Callable] = None
    ) -> float:
        """
        Evaluate on data without updating weights

        Args:
            data: List of data examples
            loss_fn: Loss function

        Returns:
            Average loss
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            for example in data:
                outputs = self.program.forward(example['inputs'])
                for target_name, target_value in example['targets'].items():
                    if target_name in outputs:
                        pred = outputs[target_name]
                        target = target_value.to(self.device)
                        loss = loss_fn(pred, target)
                        total_loss += loss.item()
                        num_examples += 1

        return total_loss / num_examples if num_examples > 0 else 0.0

    def compute_accuracy(
        self,
        data: List[Dict],
        threshold: float = 0.5
    ) -> float:
        """
        Compute classification accuracy

        Args:
            data: List of data examples
            threshold: Threshold for binary classification

        Returns:
            Accuracy (0-1)
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for example in data:
                outputs = self.program.forward(example['inputs'])
                for target_name, target_value in example['targets'].items():
                    if target_name in outputs:
                        pred = outputs[target_name]
                        target = target_value.to(self.device)

                        # Binary classification
                        pred_binary = (pred > threshold).float()
                        correct += (pred_binary == target).sum().item()
                        total += target.numel()

        return correct / total if total > 0 else 0.0


class EmbeddingTrainer(Trainer):
    """
    Specialized trainer for EmbeddingSpace

    Trains embeddings to match relation facts.
    """

    def train_relation(
        self,
        relation_name: str,
        positive_pairs: List[tuple],
        negative_pairs: Optional[List[tuple]] = None,
        epochs: int = 100,
        verbose: bool = True
    ):
        """
        Train a relation from positive/negative examples

        Args:
            relation_name: Relation to train
            positive_pairs: List of (subject, object) pairs that are true
            negative_pairs: List of (subject, object) pairs that are false
            epochs: Number of epochs
            verbose: Show progress
        """
        # Materialize relation tensor once
        relation = self.program.relations[relation_name]
        temperature = self.program.temperature

        # Create negative pairs if not provided
        if negative_pairs is None:
            negative_pairs = self._sample_negative_pairs(
                positive_pairs,
                n_samples=len(positive_pairs)
            )

        pos_pairs = torch.tensor(positive_pairs, dtype=torch.long, device=self.device).reshape(-1, 2)
        neg_pairs = torch.tensor(negative_pairs, dtype=torch.long, device=self.device).reshape(-1, 2)
        if pos_pairs.numel() == 0:
            return
        has_negatives = neg_pairs.numel() > 0

        # Training loop
        iterator = tqdm(range(epochs)) if verbose else range(epochs)

        for epoch in iterator:
            self.optimizer.zero_grad()

            pos_subj = pos_pairs[:, 0]
            pos_obj = pos_pairs[:, 1]
            pos_emb_subj = self.program.object_embeddings[pos_subj]
            pos_emb_obj = self.program.object_embeddings[pos_obj]

            pos_scores = torch.einsum("bi,ij,bj->b", pos_emb_subj, relation, pos_emb_obj)
            pos_scores = torch.sigmoid(pos_scores / temperature)

            if has_negatives:
                neg_subj = neg_pairs[:, 0]
                neg_obj = neg_pairs[:, 1]
                neg_emb_subj = self.program.object_embeddings[neg_subj]
                neg_emb_obj = self.program.object_embeddings[neg_obj]
                neg_scores = torch.einsum("bi,ij,bj->b", neg_emb_subj, relation, neg_emb_obj)
                neg_scores = torch.sigmoid(neg_scores / temperature)
                neg_loss = -torch.log(1.0 - neg_scores + 1e-10).sum()
                total_loss = -torch.log(pos_scores + 1e-10).sum() + neg_loss
            else:
                total_loss = -torch.log(pos_scores + 1e-10).sum()

            # Optimize
            total_loss.backward()
            self.optimizer.step()

            if verbose:
                iterator.set_postfix({'loss': f'{total_loss.item():.4f}'})

    def _sample_negative_pairs(
        self,
        positive_pairs: List[tuple],
        n_samples: int
    ) -> List[tuple]:
        """Sample negative pairs (not in positive set)"""
        if n_samples <= 0:
            return []

        num_objects = self.program.num_objects
        positive_set = set(positive_pairs)
        mask = torch.ones((num_objects, num_objects), dtype=torch.bool, device=self.device)
        for i, j in positive_set:
            if 0 <= i < num_objects and 0 <= j < num_objects:
                mask[i, j] = False
        mask.fill_diagonal_(False)

        available = mask.nonzero(as_tuple=False)
        if available.shape[0] == 0:
            return []

        if available.shape[0] <= n_samples:
            selected = available
        else:
            perm = torch.randperm(available.shape[0], device=self.device)
            selected = available[perm[:n_samples]]

        return [(int(i.item()), int(j.item())) for i, j in selected]


class PairScoringTrainer:
    """
    Generic trainer for pair-scoring models using a provided loss and scorer.

    Suitable for training a multi-hop composer over an EmbeddingSpace by
    optimizing scores on (subject, object) pairs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_type: str = 'adam',
        learning_rate: float = 1e-3,
        device: str = 'cpu',
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self._amp_device_type = torch.device(device).type

        params = list(model.parameters())
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(params, lr=learning_rate)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(params, lr=learning_rate)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(params, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        self._scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    def train_epoch(
        self,
        loss_fn: nn.Module,
        scorer: Callable,
        batches: List[dict],
        num_objects: int,
        verbose: bool = True,
    ) -> float:
        """
        Train for one epoch.

        batches: list of dicts with keys:
          - 'pos': Tensor[B,2]
          - optional 'neg': Tensor[B,K,2]
        """
        iterator = tqdm(batches) if verbose else batches
        running = 0.0
        n = 0

        for batch in iterator:
            self.optimizer.zero_grad()
            pos = batch['pos'].to(self.device)
            neg = batch.get('neg')
            if neg is not None:
                neg = neg.to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type=self._amp_device_type):
                    loss = loss_fn(
                        self.model,
                        positive_pairs=pos,
                        negative_pairs=neg,
                        scorer=scorer,
                        num_objects=num_objects,
                    )
                self._scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss = loss_fn(
                    self.model,
                    positive_pairs=pos,
                    negative_pairs=neg,
                    scorer=scorer,
                    num_objects=num_objects,
                )
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            running += loss.item()
            n += 1
            if verbose:
                iterator.set_postfix({'loss': f'{(running / n):.4f}'})

        return running / max(n, 1)
