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
        # Create negative pairs if not provided
        if negative_pairs is None:
            negative_pairs = self._sample_negative_pairs(
                positive_pairs,
                n_samples=len(positive_pairs)
            )

        # Training loop
        iterator = tqdm(range(epochs)) if verbose else range(epochs)

        for epoch in iterator:
            self.optimizer.zero_grad()

            total_loss = 0.0

            # Positive examples
            for subj, obj in positive_pairs:
                score = self.program.query_relation(
                    relation_name, subj, obj, use_sigmoid=True
                )
                loss = -torch.log(score + 1e-10)  # Negative log likelihood
                total_loss += loss

            # Negative examples
            for subj, obj in negative_pairs:
                score = self.program.query_relation(
                    relation_name, subj, obj, use_sigmoid=True
                )
                loss = -torch.log(1 - score + 1e-10)
                total_loss += loss

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
        positive_set = set(positive_pairs)
        negatives = []

        num_objects = self.program.num_objects

        while len(negatives) < n_samples:
            i = torch.randint(0, num_objects, (1,)).item()
            j = torch.randint(0, num_objects, (1,)).item()
            if (i, j) not in positive_set and i != j:
                negatives.append((i, j))

        return negatives
