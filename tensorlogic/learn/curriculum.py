"""
Curriculum learning for progressive multi-hop reasoning

Trains models progressively: 1-hop → 2-hop → 3-hop
Prevents overwhelming the model with complex patterns early on
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """Configuration for one curriculum stage"""
    name: str
    num_hops: int
    epochs: int
    learning_rate: float
    description: str = ""


class CurriculumTrainer:
    """
    Progressive training from simple to complex patterns

    Phase 1: Learn basic 1-hop relations
    Phase 2: Learn 2-hop compositions
    Phase 3: Learn complex multi-hop patterns

    Usage:
        curriculum = CurriculumTrainer(model)
        curriculum.add_stage("basic", num_hops=1, epochs=100, lr=0.01)
        curriculum.add_stage("composition", num_hops=2, epochs=100, lr=0.005)
        curriculum.train_all(data, loss_fn, scorer)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        verbose: bool = True
    ):
        self.model = model
        self.optimizer_class = optimizer_class
        self.verbose = verbose
        self.stages: List[CurriculumStage] = []
        self.history = []

    def add_stage(
        self,
        name: str,
        num_hops: int,
        epochs: int,
        learning_rate: float,
        description: str = ""
    ):
        """Add a curriculum stage"""
        stage = CurriculumStage(
            name=name,
            num_hops=num_hops,
            epochs=epochs,
            learning_rate=learning_rate,
            description=description
        )
        self.stages.append(stage)

    def train_stage(
        self,
        stage: CurriculumStage,
        batches: List[Dict],
        loss_fn: nn.Module,
        scorer: Callable,
        num_objects: int,
        validation_fn: Optional[Callable] = None
    ) -> Dict:
        """
        Train one curriculum stage

        Args:
            stage: Curriculum stage config
            batches: Training batches
            loss_fn: Loss function
            scorer: Scoring function
            num_objects: Number of objects for negative sampling
            validation_fn: Optional validation function

        Returns:
            stage_history: Training metrics for this stage
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎓 Stage: {stage.name} ({stage.num_hops}-hop)")
            print(f"{'='*60}")
            if stage.description:
                print(f"Description: {stage.description}")
            print(f"Epochs: {stage.epochs}, LR: {stage.learning_rate}")

        # Set model's num_hops if supported
        if hasattr(self.model, 'num_hops'):
            original_hops = self.model.num_hops
            self.model.num_hops = stage.num_hops
        else:
            original_hops = None

        # Create optimizer for this stage
        optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=stage.learning_rate
        )

        stage_losses = []

        # Training loop
        for epoch in range(stage.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in batches:
                optimizer.zero_grad()

                # Get positive pairs
                pos_pairs = batch['pos']

                # Compute loss
                loss = loss_fn(
                    self.model,
                    pos_pairs,
                    negative_pairs=batch.get('neg'),
                    scorer=scorer,
                    num_objects=num_objects
                )

                # Backward and optimize
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            stage_losses.append(avg_loss)

            # Validation
            val_metric = None
            if validation_fn is not None and epoch % 10 == 0:
                val_metric = validation_fn(self.model)

            # Logging
            if self.verbose and (epoch % 10 == 0 or epoch == stage.epochs - 1):
                msg = f"  Epoch {epoch:3d}/{stage.epochs}: Loss={avg_loss:.4f}"
                if val_metric is not None:
                    msg += f", Val={val_metric:.4f}"
                print(msg)

        # Restore original num_hops
        if original_hops is not None:
            self.model.num_hops = original_hops

        if self.verbose:
            improvement = (stage_losses[0] - stage_losses[-1]) / (stage_losses[0] + 1e-10)
            print(f"✅ Stage complete! Improvement: {improvement*100:.1f}%\n")

        return {
            'stage_name': stage.name,
            'num_hops': stage.num_hops,
            'losses': stage_losses,
            'final_loss': stage_losses[-1],
            'initial_loss': stage_losses[0]
        }

    def train_all(
        self,
        batches: List[Dict],
        loss_fn: nn.Module,
        scorer: Callable,
        num_objects: int,
        validation_fn: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Train all curriculum stages sequentially

        Returns:
            Full training history for all stages
        """
        if not self.stages:
            raise ValueError("No curriculum stages defined. Use add_stage() first.")

        if self.verbose:
            print(f"\n🎯 Starting Curriculum Training")
            print(f"Total stages: {len(self.stages)}")
            print(f"Progression: {' → '.join(s.name for s in self.stages)}\n")

        self.history = []

        for i, stage in enumerate(self.stages, 1):
            if self.verbose:
                print(f"\n[Stage {i}/{len(self.stages)}]")

            stage_history = self.train_stage(
                stage=stage,
                batches=batches,
                loss_fn=loss_fn,
                scorer=scorer,
                num_objects=num_objects,
                validation_fn=validation_fn
            )

            self.history.append(stage_history)

        if self.verbose:
            print(f"\n{'='*60}")
            print("🎉 Curriculum Training Complete!")
            print(f"{'='*60}")
            self._print_summary()

        return self.history

    def _print_summary(self):
        """Print training summary"""
        print("\n📊 Training Summary:")
        for hist in self.history:
            improvement = (hist['initial_loss'] - hist['final_loss']) / (hist['initial_loss'] + 1e-10)
            print(f"  {hist['stage_name']:15s} ({hist['num_hops']}-hop): "
                  f"{hist['initial_loss']:.4f} → {hist['final_loss']:.4f} "
                  f"({improvement*100:+.1f}%)")


def create_standard_curriculum(
    model: nn.Module,
    max_hops: int = 3,
    epochs_per_stage: int = 100
) -> CurriculumTrainer:
    """
    Create standard curriculum: 1-hop → 2-hop → 3-hop

    Args:
        model: Model to train
        max_hops: Maximum number of hops
        epochs_per_stage: Epochs per stage

    Returns:
        Configured CurriculumTrainer
    """
    curriculum = CurriculumTrainer(model)

    # Stage 1: Single-hop relations
    curriculum.add_stage(
        name="basic_relations",
        num_hops=1,
        epochs=epochs_per_stage,
        learning_rate=0.01,
        description="Learn basic 1-hop relation matrices"
    )

    # Stage 2+: Multi-hop compositions
    for hop in range(2, max_hops + 1):
        # Decrease LR as we go deeper
        lr = 0.01 * (0.5 ** (hop - 1))

        curriculum.add_stage(
            name=f"{hop}_hop_composition",
            num_hops=hop,
            epochs=epochs_per_stage,
            learning_rate=lr,
            description=f"Learn {hop}-hop relation compositions"
        )

    return curriculum


def create_adaptive_curriculum(
    model: nn.Module,
    convergence_threshold: float = 0.01,
    max_hops: int = 3
) -> CurriculumTrainer:
    """
    Create adaptive curriculum that progresses based on convergence

    Args:
        model: Model to train
        convergence_threshold: Loss change threshold for progression
        max_hops: Maximum number of hops

    Returns:
        Configured CurriculumTrainer
    """
    # Note: Full adaptive implementation would require custom training loop
    # This is a simplified version with reasonable defaults

    curriculum = CurriculumTrainer(model)

    # Adaptive: more epochs for harder stages
    epochs_schedule = {
        1: 50,   # Quick warmup
        2: 100,  # Main learning
        3: 150,  # Fine-tuning complex patterns
    }

    for hop in range(1, max_hops + 1):
        curriculum.add_stage(
            name=f"adaptive_{hop}_hop",
            num_hops=hop,
            epochs=epochs_schedule.get(hop, 100),
            learning_rate=0.01 * (0.7 ** (hop - 1)),
            description=f"Adaptive {hop}-hop training"
        )

    return curriculum
