import numpy as np
import torch as th
from typing import Sequence, List, Optional
from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair
from imitation.algorithms.preference_comparisons import Fragmenter
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from sklearn.cluster import KMeans
from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair
from imitation.algorithms.preference_comparisons import Fragmenter, PreferenceModel
from imitation.data import rollout
from imitation.util import logger as imit_logger

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

    def _on_policiness(self, traj: TrajectoryWithRew) -> float:
        """
        Compute O(τ) = sum_t log π(a_t | s_t) for a trajectory.
        """
        obs = th.as_tensor(traj.obs[:-1])
        acts = th.as_tensor(traj.acts)
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
    
class RewardDifferenceDiversityFragmenter(Fragmenter):
    """Selects diverse queries by clustering in the space of predicted reward differences (DUO/ξD)."""
    def __init__(
        self,
        preference_model: PreferenceModel,
        base_fragmenter: Fragmenter,
        max_k: int = 10,
        min_k: int = 2,
        elbow_tol: float = 0.05,
        random_state: int = 0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
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
        """
        super().__init__(custom_logger=custom_logger)
        self.preference_model = preference_model
        self.base_fragmenter = base_fragmenter
        self.max_k = max_k
        self.min_k = min_k
        self.elbow_tol = elbow_tol
        self.random_state = random_state

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

        # Step 3: Find K using elbow method
        inertias = []
        K_range = range(self.min_k, min(self.max_k, len(diff_vecs)) + 1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
            kmeans.fit(diff_vecs)
            inertias.append(kmeans.inertia_)
        # Elbow: look for largest fractional drop
        drops = np.diff(inertias) / inertias[:-1]
        if len(drops) == 0:
            best_k = self.min_k
        else:
            elbow_idx = np.argmin(drops > -self.elbow_tol)
            best_k = K_range[elbow_idx] if elbow_idx < len(K_range) else K_range[-1]

        # Step 4: Cluster with best_k
        kmeans = KMeans(n_clusters=best_k, random_state=self.random_state, n_init='auto')
        kmeans.fit(diff_vecs)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Step 5: For each cluster, select the closest query
        selected_indices = []
        for i in range(best_k):
            cluster_idxs = np.where(labels == i)[0]
            if len(cluster_idxs) == 0:
                continue
            center = centers[i]
            dists = np.linalg.norm(diff_vecs[cluster_idxs] - center, axis=1)
            closest_idx = cluster_idxs[np.argmin(dists)]
            selected_indices.append(closest_idx)
            
        # Step 6: Return up to num_pairs most diverse queries
        if len(selected_indices) > num_pairs:
            # If more than needed, pick the most uncertain (largest norm of diff_vec)
            norms = [np.linalg.norm(diff_vecs[idx]) for idx in selected_indices]
            top_idxs = np.argsort(norms)[-num_pairs:][::-1]
            selected_indices = [selected_indices[i] for i in top_idxs]
        selected_pairs = [candidate_pairs[idx] for idx in selected_indices]
        return selected_pairs 
