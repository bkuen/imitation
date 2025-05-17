import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from imitation.algorithms.preference_comparisons import Fragmenter, PreferenceModel, RandomFragmenter
from imitation.data import rollout
from imitation.regularization import regularizers
from imitation.rewards import reward_nets
from imitation.util import logger as imit_logger, util
from imitation.data.types import (
    TrajectoryWithRew,
    TrajectoryWithRewPair,
)
from torch import nn
from torch.utils import data as data_th
from tqdm.auto import tqdm
from typing import (
    List,
    Sequence,
    Tuple,
    Optional,
    Dict,
)

import numpy as np
import torch as th
import torch.nn.functional as F

class StateSegmentDataset(data_th.Dataset):
    """Dataset for the VARIQuery algorithm that handles pre-made fragments"""

    def __init__(
        self,
        fragments: Sequence[TrajectoryWithRew],
        fragment_length: int,
    ):
        # Store fragments directly
        self.fragments = list(fragments)
        self.fragment_length = fragment_length

        # Add this check
        if len(self.fragments) == 0:
            raise ValueError("No fragments provided. The fragment sequence is empty.")

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx: int) -> TrajectoryWithRew:
        """Return the fragment as a TrajectoryWithRew object"""
        return self.fragments[idx]

    def as_tensor(self) -> th.Tensor:
        """Convert the fragments to a tensor for VAE training"""
        # Stack all fragments into a single tensor
        return th.stack([self.get_tensor(i) for i in range(len(self))], dim=0)

    def get_tensor(self, idx: int) -> th.Tensor:
        """Convert a fragment to a tensor for VAE training
        
        Note: We only use the first fragment_length observations for VAE training,
        excluding the final observation which is only needed for RL.
        """
        fragment = self.fragments[idx]
        return th.from_numpy(fragment.obs[:self.fragment_length]).float()

class ClusterVisualizer:
    """Visualizer for clusters and selected pairs in the latent space."""

    def __init__(self, logger: Optional[imit_logger.HierarchicalLogger] = None):
        """Initialize the cluster visualizer.
        
        Args:
            logger: Optional logger for tracking visualization progress
        """
        self.logger = logger

    def visualize_clusters_and_pairs(
        self,
        encoded_segments: th.Tensor,
        clusters: List[List[int]],
        selected_pairs: List[TrajectoryWithRewPair],
        fragments_to_indices: Dict,
        save_path: str,
        title: str = 'Latent Space Clusters and Selected Pairs'
    ):
        """Visualize clusters and selected pairs in 2D using t-SNE.
        
        Args:
            encoded_segments: Encoded segments in latent space
            clusters: List of lists containing indices for each cluster
            selected_pairs: List of selected trajectory pairs
            fragments_to_indices: Mapping from fragment ID to index
            save_path: Path to save the plot
            title: Title for the plot
        """
        # Convert to numpy for t-SNE
        latent_vectors = encoded_segments.cpu().numpy()
        n_samples = latent_vectors.shape[0]
        
        # Normalize vectors
        norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
        normalized_vectors = latent_vectors / (norms + 1e-8)

        # from sklearn.decomposition import PCA
        #
        # … right before your TSNE call …
        # pca = PCA(n_components=min(20, normalized_vectors.shape[1]), random_state=42)
        # vectors_for_tsne = pca.fit_transform(normalized_vectors)

        # Adjust t-SNE parameters based on dataset size
        perplexity = min(30, max(5, n_samples // 5))  # Scale perplexity with dataset size
        
        if n_samples < 4:
            if self.logger:
                self.logger.log(f"Warning: Too few samples ({n_samples}) for meaningful t-SNE visualization")
            return
            
        # # Create t-SNE with adjusted parameters
        # tsne = TSNE(
        #     n_components=2,
        #     random_state=42,
        #     perplexity=perplexity,
        #     n_iter=5000,  # Increase iterations for better convergence
        #     init='pca',   # Use PCA initialization for better global structure
        #     learning_rate='auto',
        #     early_exaggeration=12.0,  # Increase for better cluster separation
        #     metric='cosine',
        # )

        import umap
        umap_mapper = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        
        try:
            embedded = umap_mapper.fit_transform(normalized_vectors)
            
            # Normalize the embedding to improve visualization
            embedded = (embedded - embedded.min(axis=0)) / (embedded.max(axis=0) - embedded.min(axis=0))
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"UMAP visualization failed: {str(e)}")
            return
        
        # Create plot with improved styling
        plt.style.use('default')  # Reset to default style
        plt.figure(figsize=(12, 8))
        
        # Set background color and grid
        plt.gca().set_facecolor('#f0f0f0')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot each cluster with different colors and improved visibility
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Better color palette
        
        # First plot all points with lower alpha for context
        plt.scatter(embedded[:, 0], embedded[:, 1], c='gray', alpha=0.1, s=50)
        
        # Then plot clusters
        for cluster_idx, cluster in enumerate(clusters):
            if not cluster:  # Skip empty clusters
                continue
            cluster_points = embedded[cluster]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                alpha=0.6,
                c=[colors[cluster_idx % len(colors)]],
                label=f'Cluster {cluster_idx} (n={len(cluster)})',
                s=100,  # Larger point size
                edgecolors='white',  # White edges for better visibility
                linewidth=0.5
            )
        
        # Draw lines between selected pairs with curved arrows
        for pair in selected_pairs:
            try:
                idx1 = fragments_to_indices[id(pair[0])]
                idx2 = fragments_to_indices[id(pair[1])]
                
                # Create curved arrow between pairs
                from matplotlib.patches import ConnectionPatch
                con = ConnectionPatch(
                    xyA=(embedded[idx1, 0], embedded[idx1, 1]),
                    xyB=(embedded[idx2, 0], embedded[idx2, 1]),
                    coordsA="data", coordsB="data",
                    axesA=plt.gca(), axesB=plt.gca(),
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.2",
                    edgecolor='red',
                    alpha=0.5,
                    linewidth=1.5
                )
                plt.gca().add_patch(con)
                
                # Highlight selected points
                plt.scatter(
                    [embedded[idx1, 0], embedded[idx2, 0]],
                    [embedded[idx1, 1], embedded[idx2, 1]],
                    c='red',
                    s=150,
                    alpha=0.05,
                    zorder=5,
                    edgecolors='white',
                    linewidth=0.5
                )
            except (KeyError, IndexError) as e:
                if self.logger:
                    self.logger.log(f"Warning: Could not plot pair due to missing index: {str(e)}")
                continue
        
        # Improve title and labels
        plt.title(f"{title}\n(n_samples={n_samples}, perplexity={perplexity})", 
                 pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Component 1', fontsize=10)
        plt.ylabel('Component 2', fontsize=10)
        
        # Improve legend
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          borderaxespad=0., frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)
        
        # Set figure background color
        plt.gcf().patch.set_facecolor('white')
        
        # Add a border around the plot
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        
        plt.tight_layout()  # Adjust layout to prevent label clipping
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')  # Higher DPI for better quality
        plt.close()

        if self.logger:
            self.logger.log(f"Saved cluster visualization to {save_path}")

class VARIQueryFragmenter(Fragmenter):
    """
    Fragmenter for the VARIQuery algorithm. This fragmenter is used to sample trajectories
    from the replay buffer based on the disagreement of the ensemble members.
    """

    def __init__(
        self,
        state_dim: int,
        sequence_length: int,
        vae_latent_dim: int,
        preference_model: PreferenceModel,
        rng: np.random.Generator,
        base_fragmenter: Optional[Fragmenter] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        warning_threshold: int = 10,
        allow_variable_horizon: bool = False,
        variquery_num_clusters: int = 3,
        # VAE training parameters
        vae_hidden_dims: List[int] = [128, 64, 32],
        vae_epochs: int = 10,
        vae_batch_size: int = 32,
        vae_lr: float = 1e-3,
        vae_kl_weight: float = 1.0,
        vae_early_stopping_patience: Optional[int] = None,
        fragment_sample_factor: float = 2.0,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
        visualization_interval: int = 10,
        vae_mode: str = "state",  # New parameter: "state" or "state_reward"
    ):
        super().__init__(custom_logger)
        self.allow_variable_horizon = allow_variable_horizon
        self.preference_model = preference_model
        self.fragment_sample_factor = fragment_sample_factor
        self.visualization_interval = visualization_interval
        self.visualizer = ClusterVisualizer(custom_logger)
        self.vae_mode = vae_mode
        
        # Use provided base_fragmenter or create default RandomFragmenter
        self.base_fragmenter = base_fragmenter or RandomFragmenter(
            rng=rng,
            warning_threshold=warning_threshold,
            custom_logger=custom_logger
        )

        # Store VARIQuery parameters
        self.variquery_num_clusters = variquery_num_clusters
        self.current_iteration = 0

        # Store VAE training parameters
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_lr = vae_lr
        self.vae_kl_weight = vae_kl_weight
        self.vae_early_stopping_patience = vae_early_stopping_patience
        self.device = device
        
        # Create appropriate VAE based on mode
        if vae_mode == "state":
            self.vae = MLPStateVAE(
                state_dim=state_dim,
                sequence_length=sequence_length,
                latent_dim=vae_latent_dim,
                hidden_dims=vae_hidden_dims,
                custom_logger=self.logger,
            )
        elif vae_mode == "state_reward":
            self.vae = MLPStateRewardCVAE(
                state_dim=state_dim,
                sequence_length=sequence_length,
                latent_dim=vae_latent_dim,
                hidden_dims=vae_hidden_dims,
                custom_logger=self.logger,
            )
        else:
            raise ValueError(f"Invalid VAE mode: {vae_mode}. Must be either 'state' or 'state_reward'")

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        # Step 1: Sample more fragments than needed using base_fragmenter (RandomFragmenter)
        fragments_to_sample = int(self.fragment_sample_factor * num_pairs)
        initial_fragments = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=fragments_to_sample
        )
        
        # Convert to dataset for VAE training
        # Extract all fragments from the pairs into a single list
        all_fragments = []
        for f1, f2 in initial_fragments:
            all_fragments.append(f1)
            all_fragments.append(f2)
            
        D_un = StateSegmentDataset(all_fragments, fragment_length)
        
        # Create mapping from fragment to index for visualization
        self.fragments_to_indices = {
            id(fragment): idx 
            for idx, fragment in enumerate(D_un.fragments)
        }
        
        # Step 2: Train VAE and encode segments
        self._train_vae(D_un)
        D_z = self._encode_segments(D_un)
        
        # Step 3: Cluster and sample final pairs
        clusters = self._cluster_latent_space(D_z, num_clusters=self.variquery_num_clusters)

        # Step 4: Sample and rank pairs
        pairs = self._sample_random_pairs(clusters, D_un, num_pairs)
        ranked_pairs = self._rank_by_ensemble_variance(pairs)

        # Always create visualization directory
        output_dir = self.logger.get_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize if it's time to do so or if it's the first iteration
        should_visualize = self.current_iteration == 0 or self.current_iteration % self.visualization_interval == 0
        
        if should_visualize:
            self.logger.log(f"Creating visualization for iteration {self.current_iteration}")
            viz_path = os.path.join(
                output_dir, 
                f"clusters_iteration_{self.current_iteration:04d}.png"
            )
            self.visualizer.visualize_clusters_and_pairs(
                D_z,
                clusters,
                ranked_pairs[:num_pairs],
                self.fragments_to_indices,
                save_path=viz_path,
                title=f'Clusters and Selected Pairs (Iteration {self.current_iteration})'
            )

        # Increment the number of iterations
        self.current_iteration += 1
        
        # Step 5: Return top N pairs
        return ranked_pairs[:num_pairs]
        

    @th.no_grad()
    def _encode_segments(
        self,
        segments: StateSegmentDataset,
    ) -> th.Tensor:
        self.logger.info("inside __encdoe_segments")
        self.logger.info("fragments, len: {}".format(len(segments.fragments)))

        x = segments.as_tensor().to(self.device)
        self.logger.info("stacked segments, shape: {}".format(x.shape))
        
        if self.vae_mode == "state":
            _, _, z = self.vae.encode(x)
        else:  # state_reward mode
            # Get predicted rewards using preference model
            trans = rollout.flatten_trajectories(segments.fragments)
            predicted_rewards = self.preference_model.rewards(trans)
            
            # Reshape rewards to match sequence length
            batch_size = len(segments.fragments)
            rewards = predicted_rewards.view(batch_size, -1)
            
            self.logger.info("predicted rewards shape: {}".format(rewards.shape))
            _, _, z = self.vae.encode(x, rewards)
            
        return z

        # return [self.vae.encode(segment) for segment in segments]
    
    def _cluster_latent_space(
        self,
        encoded_segments: th.Tensor,
        num_clusters: int,
    ) -> List[List[int]]:
        """Cluster the latent space using k-means clustering
        
        Args:
            encoded_segments: List of encoded segments
            num_clusters: The number of clusters to use

        Returns:
            List of lists containing indices for each cluster
        """
        # Convert to numpy and normalize
        latent_vectors = encoded_segments.cpu().numpy()
        
        # Normalize the vectors to unit length
        norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
        normalized_vectors = latent_vectors / (norms + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Fit k-means with multiple initializations
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10,  # Try multiple initializations
            max_iter=300  # Increase max iterations
        )
        cluster_labels = kmeans.fit_predict(normalized_vectors)
        
        # Calculate silhouette score to evaluate clustering quality
        from sklearn.metrics import silhouette_score
        if len(normalized_vectors) > 1:
            score = silhouette_score(normalized_vectors, cluster_labels)
            self.logger.log(f"Clustering silhouette score: {score:.3f}")
        
        # Group indices by cluster
        clusters = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
            
        # Log cluster sizes
        for i, cluster in enumerate(clusters):
            self.logger.log(f"Cluster {i} size: {len(cluster)}")
            
        return clusters
    
    def _sample_random_pairs(
        self,
        clusters: List[List[int]],
        dataset: StateSegmentDataset,
        num_pairs: int,
    ) -> List[TrajectoryWithRewPair]:
        """Sample pairs at random from the clusters where the two trajectories are in different clusters"""
        pairs = []
        for _ in range(num_pairs):
            # Sample two random indices from the clusters
            c1, c2 = np.random.choice(len(clusters), size=2, replace=False)
            self.logger.info("sampled clusters: {} {}".format(c1, c2))
            idx1 = np.random.choice(len(clusters[c1]), replace=False)
            idx2 = np.random.choice(len(clusters[c2]), replace=False)
            self.logger.info("sampled indices: {} {}".format(idx1, idx2))
            index1 = clusters[c1][idx1]
            index2 = clusters[c2][idx2]
            self.logger.info("sampled final indices: {} {}".format(index1, index2))
            traj1 = dataset.fragments[index1]
            traj2 = dataset.fragments[index2]

            pair: TrajectoryWithRewPair = (traj1, traj2)

            pairs.append(pair)
            
        return pairs
        
    def _rank_by_ensemble_variance(
        self,
        pairs: List[TrajectoryWithRewPair],
    ) -> List[TrajectoryWithRewPair]:
        print("pairs type:", type(pairs))  # Actual type of the pair
        print("is list:", isinstance(pairs, list))  # Check if it's actually a tuple

        """Rank pairs based on the disagreement of the ensemble members"""
        variances = []
        for pair in pairs:
            first = pair[0]
            second = pair[1]
            trans1 = rollout.flatten_trajectories([first])
            trans2 = rollout.flatten_trajectories([second])

            with th.no_grad():
                rews1 = self.preference_model.rewards(trans1)
                rews2 = self.preference_model.rewards(trans2)

            returns1 = rews1.sum(dim=0)
            returns2 = rews2.sum(dim=0)


            var = th.var(returns1 - returns2, dim=0)
            variances.append(var)
        
        # Sort the pairs by the variance of the rewards
        sorted_pairs = [x for _, x in sorted(zip(variances, pairs), key=lambda pair: pair[0], reverse=True)]
        return sorted_pairs

    def _train_vae(
        self,
        dataset: StateSegmentDataset,
    ):
        """Train the VAE on the dataset"""
        trainer = VAETrainer(
            vae=self.vae,
            epochs=self.vae_epochs,
            device=self.device,
            lr=self.vae_lr,
            kl_weight_beta=self.vae_kl_weight,
            batch_size=self.vae_batch_size,
            early_stopping_patience=self.vae_early_stopping_patience,
            optimizer=th.optim.Adam(self.vae.parameters(), lr=self.vae_lr),
            custom_logger=self.logger,
        )
        
        # Move VAE to appropriate device
        self.vae = self.vae.to(self.device)
        
        # Train the VAE
        trainer.train(dataset)

class MLPVae(nn.Module):
    """
    Base class for MLP-based Variational Autoencoders.
    Contains common functionality shared between different VAE implementations.
    """

    def __init__(
        self,
        state_dim: int,
        sequence_length: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.flat_dim = state_dim * sequence_length
        self.logger = custom_logger or imit_logger.configure()

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        """Reparameterization trick to sample from the latent space
        
        Args:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

class MLPStateVAE(MLPVae):
    """
    State VAE for the VARIQuery algorithm.
    """

    def __init__(
        self,
        state_dim: int,
        sequence_length: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        super().__init__(
            state_dim=state_dim,
            sequence_length=sequence_length,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            custom_logger=custom_logger,
        )

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

    def _create_encoder(self):
        encoder_layers = []
        in_dim = self.flat_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        return nn.Sequential(*encoder_layers)
    
    def _create_decoder(self):
        decoder_layers = []
        in_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        decoder_layers.extend([
            nn.Linear(in_dim, self.flat_dim),
            # No activation - raw outputs for MSE loss
        ])
        return nn.Sequential(*decoder_layers)
    
    def encode(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Encode state segments into latent space
        
        Args:
            x: (batch_size, fragment_length, state_dim)

        Returns:
            Tuple of (mu, logvar, z) each of shape (batch_size, latent_dim)
        """
        self.logger.info("encode state segments, shape: {}".format(x.shape))

        # Flatten input: (batch, seq_len, state_dim)
        x_flat = x.view(x.shape[0], -1)
        self.logger.info("flattened state segments, shape: {}".format(x_flat.shape))

        # Encode
        hidden = self.encoder(x_flat)
        self.logger.info("encoded state segments, shape: {}".format(hidden.shape))

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        z = self.reparameterize(mu, logvar)

        self.logger.info("latent space, shape: {}".format(z.shape))

        return mu, logvar, z
    
    def decode(self, z: th.Tensor) -> th.Tensor:
        """Decode latent space samples into state segments
        
        Args:
            z: Tensor of shape (batch_size, latent_dim)

        Returns:
            Tensor of shape (batch_size, sequence_length, state_dim)
        """
        # Decode to flattened state segments
        x_flat = self.decoder(z)
        
        # Reshape to (batch_size, sequence_length, state_dim)
        x = x_flat.view(-1, self.sequence_length, self.state_dim)

        return x
    
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass through VAE
        
        Args:
            x: Tensor of shape (batch_size, sequence_length, state_dim)
            
        Returns:
            Tuple of (x_reconstructed, mu, logvar)
        """
        mu, logvar, z = self.encode(x)
        x_reconstructed = self.decode(z)

        return x_reconstructed, mu, logvar

class MLPStateRewardCVAE(MLPVae):
    """
    Conditional VAE for encoding states based on rewards.
    This model takes both state sequences and their corresponding reward sequences as input.
    """

    def __init__(
        self,
        state_dim: int,
        sequence_length: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        super().__init__(
            state_dim=state_dim,
            sequence_length=sequence_length,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            custom_logger=custom_logger,
        )

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

    def _create_encoder(self):
        encoder_layers = []
        # Input dimension includes both flattened states and flattened rewards
        in_dim = self.flat_dim + self.sequence_length  # +sequence_length for the reward sequence
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        return nn.Sequential(*encoder_layers)
    
    def _create_decoder(self):
        decoder_layers = []
        # Input dimension includes both latent vector and flattened rewards
        in_dim = self.latent_dim + self.sequence_length  # +sequence_length for the reward sequence
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        decoder_layers.extend([
            nn.Linear(in_dim, self.flat_dim),
            # No activation - raw outputs for MSE loss
        ])
        return nn.Sequential(*decoder_layers)
    
    def encode(self, x: th.Tensor, rewards: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Encode state segments into latent space conditioned on reward sequence
        
        Args:
            x: (batch_size, fragment_length, state_dim)
            rewards: (batch_size, fragment_length) reward sequence for each trajectory

        Returns:
            Tuple of (mu, logvar, z) each of shape (batch_size, latent_dim)
        """
        self.logger.info("encode state segments, shape: {}".format(x.shape))
        self.logger.info("encode reward sequences, shape: {}".format(rewards.shape))

        # Flatten input: (batch, seq_len, state_dim)
        x_flat = x.view(x.shape[0], -1)
        self.logger.info("flattened state segments, shape: {}".format(x_flat.shape))

        # Ensure rewards are properly shaped
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)  # Add sequence dimension if missing
        if rewards.shape[1] != self.sequence_length:
            # If rewards are not the right length, pad or truncate
            if rewards.shape[1] > self.sequence_length:
                rewards = rewards[:, :self.sequence_length]
            else:
                padding = th.zeros(rewards.shape[0], self.sequence_length - rewards.shape[1], device=rewards.device)
                rewards = th.cat([rewards, padding], dim=1)

        # Concatenate flattened states with flattened rewards
        x_with_rewards = th.cat([x_flat, rewards], dim=1)
        self.logger.info("concatenated with rewards, shape: {}".format(x_with_rewards.shape))

        # Encode
        hidden = self.encoder(x_with_rewards)
        self.logger.info("encoded state segments, shape: {}".format(hidden.shape))

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        z = self.reparameterize(mu, logvar)

        self.logger.info("latent space, shape: {}".format(z.shape))

        return mu, logvar, z
    
    def decode(self, z: th.Tensor, rewards: th.Tensor) -> th.Tensor:
        """Decode latent space samples into state segments conditioned on reward sequence
        
        Args:
            z: Tensor of shape (batch_size, latent_dim)
            rewards: Tensor of shape (batch_size, sequence_length) reward sequence

        Returns:
            Tensor of shape (batch_size, sequence_length, state_dim)
        """
        # Ensure rewards are properly shaped
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)  # Add sequence dimension if missing
        if rewards.shape[1] != self.sequence_length:
            # If rewards are not the right length, pad or truncate
            if rewards.shape[1] > self.sequence_length:
                rewards = rewards[:, :self.sequence_length]
            else:
                padding = th.zeros(rewards.shape[0], self.sequence_length - rewards.shape[1], device=rewards.device)
                rewards = th.cat([rewards, padding], dim=1)

        # Concatenate latent vector with flattened rewards
        z_with_rewards = th.cat([z, rewards], dim=1)
        
        # Decode to flattened state segments
        x_flat = self.decoder(z_with_rewards)
        
        # Reshape to (batch_size, sequence_length, state_dim)
        x = x_flat.view(-1, self.sequence_length, self.state_dim)

        return x
    
    def forward(self, x: th.Tensor, rewards: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass through CVAE
        
        Args:
            x: Tensor of shape (batch_size, sequence_length, state_dim)
            rewards: Tensor of shape (batch_size, sequence_length) reward sequence
            
        Returns:
            Tuple of (x_reconstructed, mu, logvar)
        """
        mu, logvar, z = self.encode(x, rewards)
        x_reconstructed = self.decode(z, rewards)

        return x_reconstructed, mu, logvar

class VAETrainer:
    """Trainer for the VAE"""

    regularizer: Optional[regularizers.Regularizer]

    def __init__(
        self,
        vae: MLPVae,
        epochs: int,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
        lr: float = 1e-3,
        kl_weight_beta: float = 1.0,
        batch_size: int = 32,
        early_stopping_patience: Optional[int] = None,
        optimizer: Optional[th.optim.Optimizer] = None,
        regularizer_factory: Optional[regularizers.RegularizerFactory] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Initialize the VAETrainer
        
        Args:
            vae: The VAE to train (either MLPStateVAE or MLPStateRewardCVAE)
            epochs: The number of epochs to train for
            device: The device to run the training on
            lr: The learning rate
            kl_weight_beta: The weight for the KL divergence term in the loss function (β-VAE parameter)
            optimizer: The optimizer to use. If None, a default Adam optimizer is used.
            regularizer_factory: The regularizer factory to use
            custom_logger: The logger to use. If None, a default logger is created.
        """
        self.vae = vae.to(device)
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer or th.optim.Adam(vae.parameters(), lr=lr)
        self.kl_weight_beta = kl_weight_beta
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.logger = custom_logger or imit_logger.configure()
        self.regularizer = (
            regularizer_factory(optimizer=self.optimizer, logger=self.logger)
            if regularizer_factory is not None
            else None
        )
        
        # Check if we're using a CVAE
        self.is_cvae = isinstance(vae, MLPStateRewardCVAE)

    def train(
        self,
        dataset: StateSegmentDataset    
    ):
        dataloader, val_dataloader = self._create_data_loaders(dataset)

        assert self.epochs > 0, "Must train for at least one epoch."

        best_val_loss = float("inf")
        patience_counter = 0

        with self.logger.accumulate_means("variquery"):
            for epoch in tqdm(range(self.epochs), desc="Training VAE"):
                with self.logger.add_key_prefix(f"epoch-{epoch}"):
                    avg_loss, val_loss = self._train_epoch(dataloader, val_dataloader)

                    if self.early_stopping_patience is not None and val_loss is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= self.early_stopping_patience:
                            self.logger.log("Early stopping triggered at epoch {}".format(epoch), step=epoch)
                            break

    def _make_data_loader(
        self,
        dataset: StateSegmentDataset
    ) -> data_th.DataLoader:
        """Create a DataLoader from a dataset.
        
        Args:
            dataset: The dataset to create a loader for
            
        Returns:
            DataLoader for the dataset
        """
        class TensorDataset(data_th.Dataset):
            def __init__(self, state_dataset: StateSegmentDataset, is_cvae: bool):
                self.state_dataset = state_dataset
                self.is_cvae = is_cvae

            def __len__(self):
                return len(self.state_dataset)

            def __getitem__(self, idx):
                if self.is_cvae:
                    # For CVAE, return both states and rewards
                    states = self.state_dataset.get_tensor(idx)
                    # Convert rewards to float32
                    rewards = th.tensor(self.state_dataset.fragments[idx].rews, dtype=th.float32)
                    return states, rewards
                else:
                    # For regular VAE, just return states
                    return self.state_dataset.get_tensor(idx)

        tensor_dataset = TensorDataset(dataset, self.is_cvae)

        if len(tensor_dataset) < self.batch_size:
            raise ValueError(f"Dataset size ({len(tensor_dataset)}) is smaller than batch_size ({self.batch_size})")

        return data_th.DataLoader(
            tensor_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.device == "cuda",
            num_workers=0,
        )

    def _create_data_loaders(
        self,
        dataset: StateSegmentDataset,
    ) -> Tuple[data_th.DataLoader, Optional[data_th.DataLoader]]:
        if self.regularizer is not None and self.regularizer.val_split is not None:
            val_length = int(len(dataset) * self.regularizer.val_split)
            train_length = len(dataset) - val_length
            if val_length < 1 or train_length < 1:
                raise ValueError(
                    "Not enough data samples to split into training and validation, "
                    "or the validation split is too large/small. "
                    "Make sure you've generated enough initial preference data. "
                    "You can adjust this through initial_comparison_frac in "
                    "PreferenceComparisons.",
                )
            train_dataset, val_dataset = data_th.random_split(
                dataset,
                lengths=[train_length, val_length],
                # we convert the numpy generator to the pytorch generator.
                generator=th.Generator().manual_seed(util.make_seeds(self.rng)),
            )
            dataloader = self._make_data_loader(train_dataset)
            val_dataloader = self._make_data_loader(val_dataset)
        else:
            dataloader = self._make_data_loader(dataset)
            val_dataloader = None

        return dataloader, val_dataloader
    
    def _train_epoch(
        self,
        train_loader: data_th.DataLoader,
        val_loader: Optional[data_th.DataLoader] = None,
    ) -> Tuple[float, Optional[float]]:
        """Train the VAE for one epoch
        
        Args:
            train_loader: The training data loader
            val_loader: Optional validation data loader, if None, no validation is done

        Returns:
            Tuple of (avg_loss, val_loss)
        """
        # Training loop
        self.vae.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if self.is_cvae:
                x, rewards = batch
                x = x.to(self.device).float()  # Ensure float32
                rewards = rewards.to(self.device).float()  # Ensure float32
                # Forward pass
                x_reconstructed, mu, logvar = self.vae(x, rewards)
            else:
                x = batch.to(self.device).float()  # Ensure float32
                # Forward pass
                x_reconstructed, mu, logvar = self.vae(x)

            # Compute loss
            loss, recon_loss, kl_loss = self._vae_loss(
                x=x,
                mu=mu,
                logvar=logvar,
                x_reconstructed=x_reconstructed,
                kl_weight_beta=self.kl_weight_beta,
                reduction="mean",
            )

            # Backward pass
            self.optimizer.zero_grad()
            
            if self.regularizer:
                self.regularizer.regularize_and_backward(loss)
            else:
                loss.backward()

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

        # Log training metrics
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        with self.logger.add_key_prefix("train"):
            self.logger.log("loss", avg_loss)
            self.logger.log("recon_loss", avg_recon_loss)
            self.logger.log("kl_loss", avg_kl_loss)
            
        # Validation loop
        val_loss = None
        if val_loader is not None:
            val_loss = self._validate(val_loader)

        if self.regularizer is not None:
            self.regularizer.update_params(avg_loss, val_loss)

        return avg_loss, val_loss

    @th.no_grad()
    def _validate(
        self,
        val_loader: data_th.DataLoader,
    ):
        """Validate the VAE on the validation set

        Args:
            val_loader: The validation data loader

        Returns:
            The average loss on the validation set
        """
        self.vae.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        num_val_batches = 0

        for batch in val_loader:
            if self.is_cvae:
                x, rewards = batch
                x = x.to(self.device).float()  # Ensure float32
                rewards = rewards.to(self.device).float()  # Ensure float32
                # Forward pass
                x_reconstructed, mu, logvar = self.vae(x, rewards)
            else:
                x = batch.to(self.device).float()  # Ensure float32
                # Forward pass
                x_reconstructed, mu, logvar = self.vae(x)
            
            # Compute loss
            loss, recon_loss, kl_loss = self._vae_loss(
                x=x,
                mu=mu,
                logvar=logvar,
                x_reconstructed=x_reconstructed,
                kl_weight_beta=self.kl_weight_beta,
                reduction="mean",
            )

            # Accumulate losses
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            num_val_batches += 1

        # Log validation metrics
        avg_loss = val_loss / num_val_batches
        avg_recon_loss = val_recon_loss / num_val_batches
        avg_kl_loss = val_kl_loss / num_val_batches

        with self.logger.add_key_prefix("val"):
            self.logger.log("loss", avg_loss)
            self.logger.log("recon_loss", avg_recon_loss)
            self.logger.log("kl_loss", avg_kl_loss)

        return avg_loss

    def _vae_loss(
        self, 
        x: th.Tensor, 
        mu: th.Tensor, 
        logvar: th.Tensor,
        x_reconstructed: th.Tensor,
        kl_weight_beta: float = 1.0,
        reduction: str = "mean",
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute VAE loss (reconstruction + KL divergence)
        
        Args:
            x: Tensor of shape (batch_size, sequence_length, state_dim)
            mu: Tensor of shape (batch_size, latent_dim)
            logvar: Tensor of shape (batch_size, latent_dim)
            x_reconstructed: Tensor of shape (batch_size, sequence_length, state_dim)
            kl_weight_beta: Weight for KL divergence (β-VAE parameter)
            reduction: Reduction method ("mean", "sum", "none")

        Returns:
            Tuple of (loss, recon_loss, kl_loss)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, x, reduction=reduction)

        # KL divergence
        kl_loss = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        if reduction == "mean":
            kl_loss = kl_loss.mean()
        elif reduction == "sum":
            kl_loss = kl_loss.sum()

        # Total loss
        loss = recon_loss + kl_weight_beta * kl_loss

        return loss, recon_loss, kl_loss

        
    