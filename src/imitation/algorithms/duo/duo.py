import numpy as np
import torch as th
import os
from typing import Sequence, List, Optional

from imitation.data.rollout import discounted_sum
from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair
from imitation.algorithms.preference_comparisons import Fragmenter
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair
from imitation.algorithms.preference_comparisons import Fragmenter, PreferenceModel
from imitation.data import rollout
from imitation.util import logger as imit_logger
import matplotlib.pyplot as plt

class PriorityFragmenter(Fragmenter):
    """
    Fragmenter that implements DUO-style priority sampling over the replay buffer.
    Trajectories are sampled with probability proportional to their on-policiness
    under the current policy, as described in Feng et al. (2025).
    """
    def __init__(
        self,
        base_algorithm: BaseAlgorithm,
        rng: np.random.Generator,
        fragment_length: int,
        custom_logger=None,
    ):
        super().__init__(custom_logger)
        self.base_algorithm = base_algorithm
        self.rng = rng
        self.fragment_length = fragment_length
        self.device = next(base_algorithm.policy.parameters()).device

    def _on_policiness(self, traj: TrajectoryWithRew) -> float:
        """
        Compute O(τ) = sum_t log π(a_t | s_t) for a trajectory.
        """
        obs = th.as_tensor(traj.obs[:-1], device=self.device)
        acts = th.as_tensor(traj.acts, device=self.device)
        # Check policy type
        if not isinstance(self.base_algorithm.policy, ActorCriticPolicy):
            raise TypeError("The policy must be an instance of ActorCriticPolicy or a compatible subclass.")
        with th.no_grad():
            dist = self.base_algorithm.policy.get_distribution(obs)
            log_probs = dist.log_prob(acts)
        return log_probs.sum().item()

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        self.logger.log("trajectories", len(trajectories))

        # Compute O(τ) for all trajectories
        on_policiness = np.array([
            self._on_policiness(traj) for traj in trajectories
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
            len(trajectories), size=2 * num_pairs, p=probs
        )
        fragments = []
        for idx in sampled_indices:
            traj = trajectories[idx]
            if len(traj) < fragment_length:
                continue  # skip too-short
            start = self.rng.integers(0, len(traj) - fragment_length + 1)
            end = start + fragment_length
            fragment = TrajectoryWithRew(
                obs=traj.obs[start:end+1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=(end == len(traj) and traj.terminal),
            )
            fragments.append(fragment)
        # Pair up fragments
        iterator = iter(fragments)
        return list(zip(iterator, iterator))
    
class ElbowVisualizer:
    """Visualizes the elbow method for KMeans clustering."""
    def __init__(self, logger=None):
        self.logger = logger

    def plot_elbow(self, K_range, inertias, save_path=None, show=False, title="Elbow Method for Optimal k"):
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertias, 'bo-', markersize=6)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(K_range)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if show:
            plt.show()
        plt.close()
        if self.logger:
            self.logger.log(f"Elbow plot saved to {save_path}")

class RewardDifferenceDiversityFragmenter(Fragmenter):
    """Selects diverse queries by clustering in the space of predicted reward differences (DUO/ξD)."""
    def __init__(
        self,
        preference_model: PreferenceModel,
        base_fragmenter: Fragmenter,
        max_k: int = 10,
        min_k: int = 2,
        elbow_tol: float = 0.05,
        random_state: int = 42,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        visualize_elbow: bool = True,
    ):
        """
        Args:
            preference_model: The reward model used to compute predicted rewards.
            base_fragmenter: Fragmenter to generate candidate fragment pairs (e.g., UncertaintyFragmenter).
            max_k: Maximum number of clusters to consider for elbow method.
            min_k: Minimum number of clusters to consider for elbow method.
            elbow_tol: Tolerance for elbow detection (fractional drop in inertia).
            random_state: Random seed for KMeans.
            custom_logger: Logger.
            visualize_elbow: Whether to visualize the elbow method.
        """
        super().__init__(custom_logger=custom_logger)
        self.preference_model = preference_model
        self.base_fragmenter = base_fragmenter
        self.max_k = max_k
        self.min_k = min_k
        self.elbow_tol = elbow_tol
        self.random_state = random_state
        self.visualize_elbow = visualize_elbow
        self.elbow_visualizer = ElbowVisualizer(logger=custom_logger)
        self.current_iteration = 0

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        # Step 1: Get candidate queries from base fragmenter
        candidate_pairs = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=max(self.max_k * num_pairs, num_pairs * 2),  # oversample
        )
        if len(candidate_pairs) == 0:
            return []

        # Step 2: Compute reward difference vectors for each pair
        diff_vecs = []
        for frag1, frag2 in candidate_pairs:
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            with th.no_grad():
                r1 = self.preference_model.rewards(trans1).cpu().numpy().flatten()
                r2 = self.preference_model.rewards(trans2).cpu().numpy().flatten()
            # Pad to same length if needed
            minlen = min(len(r1), len(r2))
            r1, r2 = r1[:minlen], r2[:minlen]
            diff_vecs.append(r2 - r1)
        diff_vecs = np.stack(diff_vecs)

        # Normalize diff_vecs before clustering using z-score
        diff_vecs_normalized = (diff_vecs - diff_vecs.mean(axis=0)) / (diff_vecs.std(axis=0) + 1e-8)

        # Step 3: Find K using elbow method (standard)
        inertias = []
        # distortions = []
        K_range = range(self.min_k, min(self.max_k, len(diff_vecs_normalized)) + 1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300)
            kmeans.fit(diff_vecs_normalized)
            inertias.append(kmeans.inertia_)
            # distortions.append(sum(np.min(cdist(diff_vecs_normalized, kmeans.cluster_centers_, 'euclidean'), axis=1) ** 2) / diff_vecs_normalized.shape[0])

        # Optionally visualize the elbow
        if self.visualize_elbow:
            # Always create visualization directory
            output_dir = self.logger.get_dir()
            os.makedirs(output_dir, exist_ok=True)

            plot_path = os.path.join(
                output_dir, 
                f"elbow_iteration_{self.current_iteration:04d}.png"
            )
            self.elbow_visualizer.plot_elbow(
                list(K_range), inertias, save_path=plot_path, show=False,
                title="Elbow Method for KMeans (Reward Difference Space)"
            )

        try:
            from kneed import KneeLocator
            kl = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
            optimal_k = kl.knee
        except Exception as e:
            self.logger.warn(f"KneeLocator failed: {e}")
            optimal_k = None

        if optimal_k is None:
            # if no knee is found, use some reasonable default
            max_k = min(self.max_k, len(diff_vecs_normalized))
            optimal_k = max(self.min_k + (max_k - self.min_k) // 2 - 1, 1)

        self.logger.info(f"optimal k suggested by KneeLocator: {optimal_k}")

        # Step 4: Cluster with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10, max_iter=300)
        kmeans.fit(diff_vecs_normalized)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Step 5: For each cluster, select the closest query
        # selected_indices = []
        # for i in range(optimal_k):
        #     cluster_idxs = np.where(labels == i)[0]
        #     if len(cluster_idxs) == 0:
        #         continue
        #     center = centers[i]
        #     dists = np.linalg.norm(diff_vecs_normalized[cluster_idxs] - center, axis=1)
        #     closest_idx = cluster_idxs[np.argmin(dists)]
        #     selected_indices.append(closest_idx)
        #
        # # Step 6: Return up to num_pairs most diverse queries
        # if len(selected_indices) > num_pairs:
        #     # If more than needed, pick the most uncertain (largest norm of diff_vec)
        #     norms = [np.linalg.norm(diff_vecs[idx]) for idx in selected_indices]
        #     top_idxs = np.argsort(norms)[-num_pairs:][::-1]
        #     selected_indices = [selected_indices[i] for i in top_idxs]
        # selected_pairs = [candidate_pairs[idx] for idx in selected_indices]

        selected_indices = []
        cluster_buckets = {i: np.where(labels == i)[0].tolist() for i in range(optimal_k)}
        while cluster_buckets and len(selected_indices) < num_pairs:
            for c, bucket in list(cluster_buckets.items()):
                if bucket:
                    selected_indices.append(bucket.pop(0))
                if not bucket:
                    cluster_buckets.pop(c)

        selected_pairs = [candidate_pairs[idx] for idx in selected_indices]

        self.current_iteration += 1

        self.logger.log("selected_pairs by reward difference", len(selected_pairs))

        return selected_pairs 
