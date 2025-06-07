import numpy as np
import torch as th
import os
from typing import Sequence, List, Optional

from imitation.data.rollout import discounted_sum
from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair
from imitation.algorithms.preference_comparisons import Fragmenter
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair
from imitation.algorithms.preference_comparisons import Fragmenter, PreferenceModel
from imitation.data import rollout
from imitation.util import logger as imit_logger
import matplotlib.pyplot as plt

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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if show:
            plt.show()
        plt.close()
        if self.logger:
            self.logger.log(f"Elbow plot saved to {save_path}")

class RewardDifferenceSelector:
    def __init__(
        self,
        preference_model: PreferenceModel,
        logger: Optional[imit_logger.HierarchicalLogger] = None,
        min_k: int = 2,
        max_k: int = 10,
        random_state: int = 42,
        use_elbow: bool = False,
    ):
        """
        Args:
            preference_model: The reward model used to compute predicted rewards.
        """
        self.preference_model = preference_model
        self.logger = logger or imit_logger.configure()
        self.elbow_visualizer = ElbowVisualizer(logger=logger)
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state
        self.use_elbow = use_elbow

    def select_pairs(
        self,
        candidate_pairs: Sequence[TrajectoryWithRewPair],
        num_pairs: int,
        current_iteration: int,
    ):
        """
        Selects pairs based on the reward difference.

        Args:
            candidate_pairs: List of candidate trajectory pairs.
            num_pairs: Number of pairs to select.

        Returns:
            List of selected trajectory pairs.
        """
        if len(candidate_pairs) == 0:
            return []

        # Compute reward difference vectors for each pair
        reward_diffs = self._calculate_distances(candidate_pairs)
        reward_diffs = reward_diffs.cpu().numpy()

        # Find optimal k using elbow method
        optimal_k = self._find_optimal_k(reward_diffs, num_pairs=num_pairs, current_iteration=current_iteration) if self.use_elbow else num_pairs

        # Perform KMeans clustering on the reward differences
        clustering = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10, max_iter=300)
        clustering.fit(reward_diffs)
        labels = clustering.labels_  # array [N]
        centers = clustering.cluster_centers_  # array [k, D]

        # For each cluster, pick the member whose feature vector is closest to its center
        selected_indices = []
        for cluster_id in range(optimal_k):
            member_mask = (labels == cluster_id)
            if not np.any(member_mask):
                continue

            members = reward_diffs[member_mask]  # [M, D]
            center = centers[cluster_id]  # [D]
            # compute Euclidean distances from center
            dists = np.linalg.norm(members - center, axis=1)  # [M]
            # find the original index of the closest member
            member_indices = np.nonzero(member_mask)[0]  # [M]
            closest_member = member_indices[np.argmin(dists)]
            selected_indices.append(int(closest_member))

        # Map back to trajectory pairs
        selected_pairs = [candidate_pairs[i] for i in selected_indices]
        return selected_pairs


    def _calculate_distances(self, pairs: Sequence[TrajectoryWithRewPair]):
        diff_vecs = []
        for frag1, frag2 in pairs:
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            with th.no_grad():
                r1 = self.preference_model.rewards(trans1)
                r2 = self.preference_model.rewards(trans2)
                # Pad to same length if needed
                minlen = min(len(r1), len(r2))
                r1, r2 = r1[:minlen], r2[:minlen]
                diff = r2 - r1
                # Flatten the difference vector for each pair
                diff_vecs.append(diff.flatten())

        # Stack all differences into a single tensor
        diff_vecs = th.stack(diff_vecs)  # Shape: (num_pairs, flattened_diff_length)
        diff_vecs = (diff_vecs - diff_vecs.mean(dim=0)) / (diff_vecs.std(dim=0) + 1e-8)
        return diff_vecs

    def _find_optimal_k(self, reward_diffs: np.array, num_pairs: int, current_iteration: int):
        B = reward_diffs.shape[0]

        inertias = []
        min_k = max(self.min_k, 2)
        max_k = min(self.max_k, B)
        K_range = range(min_k, max_k + 1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300)
            kmeans.fit(reward_diffs)
            inertias.append(kmeans.inertia_)

        output_dir = self.logger.get_dir()
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir,
            f"reward_differences/elbow/elbow_iteration_{current_iteration:04d}.png"
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
            optimal_k = num_pairs

        self.logger.info(f"Optimal k suggested by KneeLocator: {optimal_k}")
        return optimal_k

class RewardDifferenceDiversityFragmenter(Fragmenter):
    """Selects diverse queries by clustering in the space of predicted reward differences (DUO/ξD)."""
    def __init__(
        self,
        preference_model: PreferenceModel,
        base_fragmenter: Fragmenter,
        max_k: int = 10,
        min_k: int = 2,
        random_state: int = 42,
        fragment_sample_factor: float = 10.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        clustering_method: str = "kmeans",
    ):
        """
        Args:
            preference_model: The reward model used to compute predicted rewards.
            base_fragmenter: Fragmenter to generate candidate fragment pairs (e.g., UncertaintyFragmenter).
            max_k: Maximum number of clusters to consider for elbow method.
            min_k: Minimum number of clusters to consider for elbow method.
            elbow_tol: Tolerance for elbow detection (fractional drop in inertia).
            random_state: Random seed for KMeans/AgglomerativeClustering.
            custom_logger: Logger.
            visualize_elbow: Whether to visualize the elbow method.
            clustering_method: 'kmeans' (default) or 'agglomerative'. Determines which clustering algorithm to use.
        """
        super().__init__(custom_logger=custom_logger)
        self.preference_model = preference_model
        self.base_fragmenter = base_fragmenter
        self.max_k = max_k
        self.min_k = min_k
        self.fragment_sample_factor = fragment_sample_factor
        self.random_state = random_state
        self.visualize_elbow = visualize_elbow
        self.elbow_visualizer = ElbowVisualizer(logger=custom_logger)
        self.current_iteration = 0
        self.clustering_method = clustering_method.lower()
        if self.clustering_method not in ("kmeans", "agglomerative"):
            raise ValueError(f"clustering_method must be 'kmeans' or 'agglomerative', got {self.clustering_method}")

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        # Step 1: Get candidate queries from base fragmenter
        fragments_to_sample = int(self.fragment_sample_factor * num_pairs)
        candidate_pairs = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=max(self.max_k * num_pairs, num_pairs * 2),  # oversample
            # num_pairs=fragments_to_sample,  # oversample
        )
        if len(candidate_pairs) == 0:
            return []

        # Step 2: Compute reward difference vectors for each pair
        diff_vecs = []
        for frag1, frag2 in candidate_pairs:
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            with th.no_grad():
                r1 = self.preference_model.rewards(trans1)
                r2 = self.preference_model.rewards(trans2)
                # Pad to same length if needed
                minlen = min(len(r1), len(r2))
                r1, r2 = r1[:minlen], r2[:minlen]
                diff = r2 - r1
                # Flatten the difference vector for each pair
                diff_vecs.append(diff.flatten())
        
        # Stack all differences into a single tensor
        diff_vecs = th.stack(diff_vecs)  # Shape: (num_pairs, flattened_diff_length)
        
        # Normalize on GPU
        mean = diff_vecs.mean(dim=0)
        std = diff_vecs.std(dim=0) + 1e-8
        diff_vecs_normalized = (diff_vecs - mean) / std
        diff_vecs_normalized = diff_vecs_normalized.cpu().numpy()

        # Step 3: Find K using elbow method (only for kmeans)
        if self.clustering_method == "kmeans":
            inertias = []
            K_range = range(self.min_k, min(self.max_k, len(diff_vecs_normalized)) + 1)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300)
                kmeans.fit(diff_vecs_normalized)
                inertias.append(kmeans.inertia_)
            if self.visualize_elbow:
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
                max_k = min(self.max_k, len(diff_vecs_normalized))
                optimal_k = max(self.min_k + (max_k - self.min_k) // 2 - 1, 1)
            self.logger.info(f"optimal k suggested by KneeLocator: {optimal_k}")
            
            clustering = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10, max_iter=300)
            clustering.fit(diff_vecs_normalized)
            labels = clustering.labels_
            centers = clustering.cluster_centers_
        else:  # AgglomerativeClustering
            # For Agglomerative, user must specify n_clusters (use min(self.max_k, num_pairs, len(diff_vecs_normalized)))
            n_clusters = min(self.max_k, num_pairs, len(diff_vecs_normalized))
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(diff_vecs_normalized)
            centers = np.array([
                diff_vecs_normalized[labels == i].mean(axis=0) for i in range(n_clusters)
            ])
            self.logger.info(f"AgglomerativeClustering used with n_clusters={n_clusters}")

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
        cluster_buckets = {i: np.where(labels == i)[0].tolist() for i in range(len(centers))}
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
