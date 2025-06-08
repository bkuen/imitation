import os
from typing import Sequence, List, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from sklearn.cluster import KMeans

from imitation.algorithms.preference_comparisons import Fragmenter, PreferenceModel
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew, TrajectoryWithRewPair
from imitation.util import logger as imit_logger
from imitation.util.logger import HierarchicalLogger


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
        min_k: int = 2,
        max_k: int = 10,
        use_elbow: bool = False,
        use_consensual_filtering: bool = False,
        random_state: int = 42,
        fragment_sample_factor: float = 20.0,
        total_timesteps: int = 1000000,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
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
        """
        super().__init__(custom_logger=custom_logger)
        self.preference_model = preference_model
        self.base_fragmenter = base_fragmenter
        self.fragment_sample_factor = fragment_sample_factor
        self.use_consensual_filtering = use_consensual_filtering
        self.reward_diff_selector = RewardDifferenceSelector(
            preference_model=preference_model,
            logger=custom_logger,
            min_k=min_k,
            max_k=max_k,
            random_state=random_state,
            use_elbow=use_elbow,
        )
        self.consensual_filtering = ConsensualFiltering(
            preference_model=preference_model,
            logger=self.logger,
            threshold=0.5,
            train_acc_threshold=0.8,
            confidence_threshold=0.5,
            total_timesteps=total_timesteps,
        )

        self.current_iteration = 0

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
        metrics: Dict[str, float],
    ) -> Sequence[TrajectoryWithRewPair]:
        # Step 1: Get candidate queries from base fragmenter
        fragments_to_sample = int(self.fragment_sample_factor * num_pairs)
        candidate_pairs = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=fragments_to_sample,  # oversample
            metrics=metrics,
        )

        if self.use_consensual_filtering:
            candidate_pairs = self.consensual_filtering.filter_pairs(candidate_pairs, metrics)

        ranked_pairs = self._rank_by_ensemble_variance(candidate_pairs, uncertainty_on="probs_interval")
        top_m_pairs = ranked_pairs[:(int(num_pairs * self.fragment_sample_factor) // 2)]
        return self.reward_diff_selector.select_pairs(
            candidate_pairs=top_m_pairs,
            num_pairs=num_pairs,
            current_iteration=self.current_iteration,
        )

    def _rank_by_ensemble_variance(
            self,
            pairs: Sequence[TrajectoryWithRewPair],
            uncertainty_on: str = "logit",
    ) -> List[TrajectoryWithRewPair]:
        """Rank pairs based on the disagreement of the ensemble members"""
        uncertainties = []
        for pair in pairs:
            first = pair[0]
            second = pair[1]
            trans1 = rollout.flatten_trajectories([first])
            trans2 = rollout.flatten_trajectories([second])

            with th.no_grad():
                rews1 = self.preference_model.rewards(trans1)
                rews2 = self.preference_model.rewards(trans2)

            # Ensure rewards are in the correct sha
            uncertainties.append(self.uncertainty_estimate(
                rews1=rews1,
                rews2=rews2,
                uncertainty_on=uncertainty_on,
            ))

        # Sort the pairs by the variance of the rewards
        sorted_pairs = [x for _, x in sorted(zip(uncertainties, pairs), key=lambda pair: pair[0], reverse=True)]
        return sorted_pairs

    def uncertainty_estimate(self, rews1: th.Tensor, rews2: th.Tensor, uncertainty_on: str = "logit") -> float:
        """Gets the uncertainty estimate from the rewards of a fragment pair.

        Args:
            rews1: rewards obtained by all the ensemble models for the first fragment.
                Shape - (fragment_length, num_ensemble_members)
            rews2: rewards obtained by all the ensemble models for the second fragment.
                Shape - (fragment_length, num_ensemble_members)
            uncertainty_on: the type of uncertainty estimate to use. Options are: logit, probability, label, probs_interval.

        Returns:
            the uncertainty estimate based on the `uncertainty_on` flag.
        """
        if uncertainty_on == "logit":
            returns1, returns2 = rews1.sum(0), rews2.sum(0)
            return (returns1 - returns2).var().item()
        elif uncertainty_on == "probability":
            probs = self.preference_model.probability(rews1, rews2)
            probs_np = probs.cpu().numpy()
            assert probs_np.shape == (self.preference_model.model.num_members,)
            return probs_np.var()
        elif uncertainty_on == "label":
            probs = self.preference_model.probability(rews1, rews2)
            probs_np = probs.cpu().numpy()
            assert probs_np.shape == (self.preference_model.model.num_members,)
            preds = (probs_np > 0.5).astype(np.float32)
            # probability estimate of Bernoulli random variable
            prob_estimate = preds.mean()
            # variance estimate of Bernoulli random variable
            return prob_estimate * (1 - prob_estimate)
        elif uncertainty_on == "probs_interval":
            probs = self.preference_model.probability(rews1, rews2)
            probs_np = probs.cpu().numpy()
            if not hasattr(self.preference_model, 'ensemble_model') or self.preference_model.ensemble_model is None:
                raise ValueError("'probs_interval' uncertainty_on requires an ensemble model.")
            return np.max(probs_np) - np.min(probs_np)
        else:
            raise ValueError(f"Unknown uncertainty_on type: {uncertainty_on}. "
                             "Options are: logit, probability, label, probs_interval.")

class ConsensualFiltering:
    """Filters pairs based on the consensus of the ensemble members."""
    def __init__(
        self, 
        preference_model: PreferenceModel, 
        logger: Optional[HierarchicalLogger],
        total_timesteps: int,  # m
        threshold: float = 0.5,
        train_acc_threshold: float = 0.8,  # α_train_acc
        confidence_threshold: float = 0.5,  # β_conf
    ):
        """
        Args:
            preference_model: The reward model used to compute predicted rewards.
            threshold: Minimum probability for a pair to be considered consensual.
            train_acc_threshold: Training accuracy threshold (α_train_acc) that must be exceeded.
            confidence_threshold: Confidence threshold (β_conf) for model reliability.
        """
        self.preference_model = preference_model
        self.logger = logger or imit_logger.configure()
        self.threshold = threshold
        self.train_acc_threshold = train_acc_threshold
        self.confidence_threshold = confidence_threshold
        self.total_timesteps = total_timesteps

    def should_apply_filtering(self, metrics: Dict[str, Any]) -> bool:
        """Determine if we should apply the 0.5 filtering condition.
        
        Returns:
            bool: True if α > α_train_acc and β ≤ β_conf
        """
        current_accuracy = metrics.get("reward_accuracy")
        current_timestep = metrics.get("current_timestep")
            
        # Calculate confidence β = m_thre/m
        confidence = current_timestep / (self.total_timesteps + 1e-8)  # Avoid division by zero
        
        should_filter = (current_accuracy > self.train_acc_threshold and
                        confidence <= self.confidence_threshold)
        
        if should_filter:
            self.logger.info(f"Applying consensual filtering: accuracy={current_accuracy:.3f}, "
                           f"confidence={confidence:.3f}")
        else:
            self.logger.info(f"Skipping consensual filtering: accuracy={current_accuracy:.3f}, "
                           f"confidence={confidence:.3f}")
        
        return should_filter

    def filter_pairs(self, pairs: Sequence[TrajectoryWithRewPair], metrics: Dict[str, Any]) -> List[TrajectoryWithRewPair]:
        filtered_pairs = []

        # Only apply filtering if conditions are met
        if not self.should_apply_filtering(metrics):
            return list(pairs)

        """Filters pairs based on the consensus of the ensemble members."""
        for pair in pairs:
            first = pair[0]
            second = pair[1]
            trans1 = rollout.flatten_trajectories([first])
            trans2 = rollout.flatten_trajectories([second])

            with th.no_grad():
                rews1 = self.preference_model.rewards(trans1)
                rews2 = self.preference_model.rewards(trans2)
                probs = self.preference_model.probability(rews1, rews2)

            if self._keep_pair(probs):
                filtered_pairs.append(pair)

        self.logger.info(f"Dismiss {len(pairs) - len(filtered_pairs)} out of {len(pairs)} pairs based on consensus.")

        return filtered_pairs

    def _keep_pair(self, probs: th.Tensor) -> bool:
        min_p = probs.min()  # scalar tensor
        max_p = probs.max()  # scalar tensor

        # Keep pair if 0.5 lies between min and max probability
        keep = (min_p <= self.threshold) & (max_p >= self.threshold)  # scalar BoolTensor
        keep_flag = bool(keep)

        return keep_flag