import numpy as np
import torch as th


from typing import Sequence, List, Optional
from imitation.data.types import TrajectoryWithRew
from imitation.util import logger as imit_logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

class PriorityReplayBuffer:
    """
    Replay buffer that implements DUO-style priority sampling.
    Trajectories are sampled with probability proportional to their on-policiness
    under the current policy, as described in Feng et al. (2025).
    """
    def __init__(
        self,
        base_algorithm: BaseAlgorithm,
        rng: np.random.Generator,
        max_size: int,
        custom_logger=None,
    ):
        self.base_algorithm = base_algorithm
        self.rng = rng
        self.max_size = max_size
        self.device = next(base_algorithm.policy.parameters()).device
        self.trajectories = []
        self.logger = custom_logger or imit_logger.configure()

    def _on_policiness(self, traj: TrajectoryWithRew) -> float:
        """
        Compute O(τ) = sum_t log π(a_t | s_t) for a trajectory.
        Handles numerical instability by clipping log probabilities to a reasonable range.
        """
        obs = th.as_tensor(traj.obs[:-1], device=self.device)
        acts = th.as_tensor(traj.acts, device=self.device)
        # Check policy type
        if not isinstance(self.base_algorithm.policy, ActorCriticPolicy):
            raise TypeError("The policy must be an instance of ActorCriticPolicy or a compatible subclass.")
        with th.no_grad():
            dist = self.base_algorithm.policy.get_distribution(obs)
            log_probs = dist.log_prob(acts)
            # Clip log probabilities to avoid numerical instability
            # -100 is a reasonable lower bound as exp(-100) ≈ 3.7e-44
            # This prevents -inf values while still allowing very low probabilities
            log_probs = th.clamp(log_probs, min=-100.0)
        return log_probs.sum().item()

    def add(self, trajectories: Sequence[TrajectoryWithRew]) -> None:
        """Add trajectories to the buffer."""
        self.trajectories.extend(trajectories)
        if len(self.trajectories) > self.max_size:
            # Remove oldest trajectories to maintain max_size
            self.trajectories = self.trajectories[-self.max_size:]

    def sample(self, size: int) -> Sequence[TrajectoryWithRew]:
        """Sample trajectories with probability proportional to their on-policiness."""
        if len(self.trajectories) == 0:
            return []

        # Compute O(τ) for all trajectories
        on_policiness = np.array([
            self._on_policiness(traj) for traj in self.trajectories
        ])
        mu = on_policiness.mean()
        sigma = on_policiness.std() + 1e-8
        # Compute rectified Z-score
        z_scores = np.maximum(0, (on_policiness - mu) / sigma)
        # Normalize to get probabilities
        if z_scores.sum() == 0:
            probs = np.ones_like(z_scores) / len(z_scores)
        else:
            probs = z_scores / z_scores.sum()

        # Sample trajectories with these probabilities
        sampled_indices = self.rng.choice(
            len(self.trajectories), size=size, p=probs
        )
        return [self.trajectories[i] for i in sampled_indices]

    def get_all(self) -> Sequence[TrajectoryWithRew]:
        """Get all trajectories in the buffer."""
        return self.trajectories

    def __len__(self) -> int:
        return len(self.trajectories)