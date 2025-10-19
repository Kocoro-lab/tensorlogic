"""
Diagnostic tools for training health monitoring and debugging
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from collections import defaultdict


class GradientHealthChecker:
    """
    Monitor gradient flow and detect training issues

    Detects:
    - Vanishing gradients (norm < threshold)
    - Exploding gradients (norm > threshold)
    - Dead parameters (no gradient updates)
    - NaN/Inf gradients

    Usage:
        checker = GradientHealthChecker(model)
        loss.backward()
        report = checker.check()
        if report['has_issues']:
            print(report['warnings'])
    """

    def __init__(
        self,
        model: nn.Module,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0
    ):
        self.model = model
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.history = defaultdict(list)

    def check(self, step: Optional[int] = None) -> Dict:
        """
        Check gradient health after backward pass

        Returns:
            report: Dict with warnings, statistics, and recommendations
        """
        warnings = []
        stats = {}
        has_issues = False

        grad_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                warnings.append(f"⚠️  {name}: No gradient (not in computation graph)")
                has_issues = True
                continue

            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm

            # Track history
            self.history[name].append(grad_norm)

            # Check for NaN/Inf
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                warnings.append(f"🔴 {name}: NaN/Inf gradient detected!")
                has_issues = True
                continue

            # Check vanishing
            if grad_norm < self.vanishing_threshold:
                warnings.append(f"❄️  {name}: Vanishing gradient (norm={grad_norm:.2e})")
                has_issues = True

            # Check exploding
            if grad_norm > self.exploding_threshold:
                warnings.append(f"💥 {name}: Exploding gradient (norm={grad_norm:.2e})")
                has_issues = True

        # Compute statistics
        if grad_norms:
            norms = list(grad_norms.values())
            stats = {
                'mean_norm': sum(norms) / len(norms),
                'max_norm': max(norms),
                'min_norm': min(norms),
                'num_params': len(norms)
            }

        # Generate recommendations
        recommendations = []
        if has_issues:
            if any('Vanishing' in w for w in warnings):
                recommendations.append("→ Try increasing learning rate or changing initialization")
            if any('Exploding' in w for w in warnings):
                recommendations.append("→ Try gradient clipping: torch.nn.utils.clip_grad_norm_()")
            if any('NaN/Inf' in w for w in warnings):
                recommendations.append("→ Check for numerical instability (log(0), divide by zero)")

        return {
            'has_issues': has_issues,
            'warnings': warnings,
            'stats': stats,
            'grad_norms': grad_norms,
            'recommendations': recommendations,
            'step': step
        }

    def print_report(self, report: Dict):
        """Print formatted health report"""
        print("\n" + "="*60)
        print("🔍 Gradient Health Report")
        print("="*60)

        if report.get('step') is not None:
            print(f"Step: {report['step']}")

        # Statistics
        stats = report.get('stats', {})
        if stats:
            print(f"\n📊 Statistics:")
            print(f"  Mean gradient norm: {stats['mean_norm']:.6f}")
            print(f"  Max gradient norm:  {stats['max_norm']:.6f}")
            print(f"  Min gradient norm:  {stats['min_norm']:.6f}")

        # Warnings
        warnings = report.get('warnings', [])
        if warnings:
            print(f"\n⚠️  Issues detected ({len(warnings)}):")
            for w in warnings[:10]:  # Show first 10
                print(f"  {w}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")
        else:
            print("\n✅ No gradient issues detected!")

        # Recommendations
        recs = report.get('recommendations', [])
        if recs:
            print(f"\n💡 Recommendations:")
            for r in recs:
                print(f"  {r}")

        print("="*60 + "\n")

    def get_history(self, param_name: str) -> List[float]:
        """Get gradient norm history for a parameter"""
        return self.history.get(param_name, [])

    def reset_history(self):
        """Clear gradient history"""
        self.history.clear()


def check_embedding_quality(embeddings: torch.Tensor) -> Dict:
    """
    Check quality of learned embeddings

    Args:
        embeddings: [N, D] tensor

    Returns:
        report: Dict with quality metrics
    """
    N, D = embeddings.shape

    # Compute pairwise similarities
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    similarities = torch.matmul(normalized, normalized.t())

    # Remove diagonal (self-similarity)
    mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    off_diag = similarities[mask]

    # Compute statistics
    report = {
        'num_objects': N,
        'embedding_dim': D,
        'mean_similarity': off_diag.mean().item(),
        'max_similarity': off_diag.max().item(),
        'min_similarity': off_diag.min().item(),
        'std_similarity': off_diag.std().item(),
    }

    # Check for issues
    warnings = []

    # Check if embeddings collapsed (all similar)
    if report['mean_similarity'] > 0.9:
        warnings.append("⚠️  Embeddings may have collapsed (high mean similarity)")

    # Check if embeddings are too sparse (all orthogonal)
    if report['mean_similarity'] < 0.1:
        warnings.append("⚠️  Embeddings may be too sparse (low mean similarity)")

    # Check for duplicates
    duplicate_count = (similarities > 0.99)[mask].sum().item() // 2
    if duplicate_count > 0:
        warnings.append(f"⚠️  {duplicate_count} pairs of near-duplicate embeddings found")

    report['warnings'] = warnings
    report['has_issues'] = len(warnings) > 0

    return report


def diagnose_training_stuck(loss_history: List[float], window: int = 10) -> Dict:
    """
    Diagnose if training is stuck or diverging

    Args:
        loss_history: List of loss values
        window: Window size for moving average

    Returns:
        diagnosis: Dict with status and recommendations
    """
    if len(loss_history) < window:
        return {
            'status': 'insufficient_data',
            'message': f'Need at least {window} steps to diagnose'
        }

    recent = loss_history[-window:]

    # Check for NaN/Inf
    if any(not torch.isfinite(torch.tensor(x)) for x in recent):
        return {
            'status': 'diverged',
            'message': 'Loss is NaN or Inf',
            'recommendations': [
                'Reduce learning rate',
                'Check for numerical instability',
                'Add gradient clipping'
            ]
        }

    # Check if stuck (no improvement)
    improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-10)

    if improvement < 0.001:  # Less than 0.1% improvement
        return {
            'status': 'stuck',
            'message': f'No improvement in last {window} steps',
            'recent_losses': recent,
            'recommendations': [
                'Try increasing learning rate',
                'Check if model capacity is sufficient',
                'Add regularization to prevent overfitting'
            ]
        }

    # Check if increasing (diverging)
    if recent[-1] > recent[0] * 1.5:
        return {
            'status': 'increasing',
            'message': 'Loss is increasing',
            'recent_losses': recent,
            'recommendations': [
                'Reduce learning rate',
                'Check for data quality issues',
                'Add gradient clipping'
            ]
        }

    # Check convergence rate
    avg_change = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent))) / (len(recent) - 1)

    if avg_change < 0.0001:
        return {
            'status': 'converged',
            'message': 'Training appears to have converged',
            'final_loss': recent[-1]
        }

    return {
        'status': 'healthy',
        'message': 'Training is progressing normally',
        'improvement': improvement,
        'avg_change': avg_change
    }
