import torch
from sklearn.cluster import KMeans
from imitation.algorithms.preference_comparisons import Fragmenter, PreferenceModel
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
)

import numpy as np
import torch as th
import torch.nn.functional as F

class StateSegmentDataset(data_th.Dataset):
    """Dataset for the VARIQuery algorithm"""

    def __init__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
    ):
        # Store fragments as TrajectoryWithRew objects
        self.fragments: List[TrajectoryWithRew] = []
        self.fragment_length = fragment_length
        
        # Create fixed-length fragments from each trajectory
        for traj in trajectories:
            # Number of possible fragments in this trajectory
            # Subtract 1 to ensure we have fragment_length actions and fragment_length + 1 observations
            num_fragments = len(traj) - fragment_length
            
            for start_idx in range(num_fragments):
                # For observations, include one more timestep
                obs_end_idx = start_idx + fragment_length + 1
                # For actions and rewards, use one less timestep
                act_end_idx = start_idx + fragment_length
                
                # Create a new TrajectoryWithRew for this fragment
                fragment = TrajectoryWithRew(
                    obs=traj.obs[start_idx:obs_end_idx],  # fragment_length + 1 observations
                    acts=traj.acts[start_idx:act_end_idx] if traj.acts is not None else None,  # fragment_length actions
                    infos=traj.infos[start_idx:act_end_idx] if traj.infos is not None else None,  # fragment_length infos
                    terminal=False,  # Since this is a fragment, it's not terminal
                    rews=traj.rews[start_idx:act_end_idx] if traj.rews is not None else None,  # fragment_length rewards
                )
                self.fragments.append(fragment)

        # Add this check
        if len(self.fragments) == 0:
            raise ValueError(
                f"No fragments were created. Check that trajectories are longer than "
                f"fragment_length ({fragment_length}) and that trajectories is not empty."
            )

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
        vae_hidden_dims: List[int] = [128, 64, 32],
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        warning_threshold: int = 10,
        allow_variable_horizon: bool = False,
        # VAE training parameters
        vae_epochs: int = 10,
        vae_batch_size: int = 32,
        vae_lr: float = 1e-3,
        vae_kl_weight: float = 1.0,
        vae_early_stopping_patience: Optional[int] = 10,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
    ):
        super().__init__(custom_logger=custom_logger)
        self.warning_threshold = warning_threshold
        self.allow_variable_horizon = allow_variable_horizon
        self.preference_model = preference_model
        
        # Store VAE training parameters
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_lr = vae_lr
        self.vae_kl_weight = vae_kl_weight
        self.vae_batch_size = vae_batch_size
        self.vae_early_stopping_patience = vae_early_stopping_patience
        self.device = device
        
        self.vae = MLPStateVAE(
            state_dim=state_dim,
            sequence_length=sequence_length,
            latent_dim=vae_latent_dim,
            hidden_dims=vae_hidden_dims,
            custom_logger=self.logger,
        )

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        """The VARIQuery algorithm implementation"""
        # Check for variable horizon if not allowed
        if not self.allow_variable_horizon:
            trajectory_lengths = {len(traj) for traj in trajectories}
            if len(trajectory_lengths) > 1:
                raise ValueError(
                    f"Episodes of different length detected: {trajectory_lengths}. "
                    "Variable horizon environments are discouraged -- termination "
                    "conditions leak information about reward. See "
                    "https://imitation.readthedocs.io/en/latest/getting-started/"
                    "variable-horizon.html for more information. If you are SURE "
                    "you want to run imitation on a variable horizon task, then "
                    "please pass in the flag: `allow_variable_horizon=True`."
                )

        # Filter out trajectories that are too short
        trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
        
        if not trajectories:
            raise ValueError(
                f"No trajectories are long enough to create fragments of length "
                f"{fragment_length}. All trajectories must be at least {fragment_length} "
                "steps long."
            )

        if self.warning_threshold > 0:
            num_transitions = sum(len(traj) - fragment_length + 1 for traj in trajectories)
            if num_transitions < self.warning_threshold:
                self.logger.warn(
                    f"Fewer transitions ({num_transitions}) than the warning threshold "
                    f"of {self.warning_threshold} in the fragmenter. This may lead to "
                    "too little variance in the sampled fragments."
                )

        # Step 1: Create the dataset of fixed-length state segments
        self.logger.info("creating dataset of fixed-length state segments")
        D_un = StateSegmentDataset(trajectories, fragment_length)

        # Step 2: Train the VAE on the dataset and encode the segments
        self.logger.info("training VAE on dataset")
        self._train_vae(D_un)
        self.logger.info("encoding segments")
        D_z = self._encode_segments(D_un)

        self.logger.info("encoded segments shape: {}".format(D_z.shape))

        # Step 3: Cluster the latent space using k-NN
        self.logger.info("clustering latent space")
        clusters = self._cluster_latent_space(D_z, num_clusters=10)
        self.logger.info("clusters: {}".format(clusters))

        # Step 4: Sample and rank pairs
        self.logger.info("sampling random pairs from clusters")
        self.D_q = self._sample_random_pairs(clusters, D_un, num_pairs)

        self.logger.info("D_q type: {}".format(type(self.D_q)))
        self.logger.info("D_q inner type: {}".format(type(self.D_q[0])))

        self.logger.info("ranking pairs by ensemble variance")
        ranked_pairs = self._rank_by_ensemble_variance(self.D_q)

        # Step 5: Return top N pairs
        self.logger.info("returning top {} pairs".format(num_pairs))
        return ranked_pairs[:num_pairs]
        

    @th.no_grad()
    def _encode_segments(
        self,
        segments: StateSegmentDataset,
    ) -> th.Tensor:
        self.logger.info("inside __encdoe_segments")
        self.logger.info("fragments, len: {}".format(len(segments.fragments)))

        x = segments.as_tensor()
        self.logger.info("stacked segments, shape: {}".format(x.shape))
        _, _, z = self.vae.encode(x)
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
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(encoded_segments.cpu().numpy())
       # Group indices by cluster
        clusters = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
            
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

class MLPStateVAE(nn.Module):
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
        super().__init__()

        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.flat_dim = state_dim * sequence_length
        self.logger = custom_logger or imit_logger.configure()

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

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
    
    def encode(self, x: th.Tensor) -> th.Tensor:
        """Encode state segments into latent space
        
        Args:
            x: (batch_size, fragment_length, state_dim)

        Returns:
            Tuple of (mu, logvar) each of shape (batch_size, latent_dim)
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
    
    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        """Reparameterization trick to sample from the latent space
        
        Args:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std
    
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
    
class VAETrainer:
    """Trainer for the VAE"""

    regularizer: Optional[regularizers.Regularizer]

    def __init__(
        self,
        vae: MLPStateVAE,
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
            vae: The VAE to train
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
          def __init__(self, state_dataset: StateSegmentDataset):
              self.state_dataset = state_dataset

          def __len__(self):
              return len(self.state_dataset)

          def __getitem__(self, idx):
              return self.state_dataset.get_tensor(idx)

        tensor_dataset = TensorDataset(dataset)

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
            batch = batch.to(self.device)

            # Forward pass
            x_reconstructed, mu, logvar = self.vae(batch)

            # Compute loss
            loss, recon_loss, kl_loss = self._vae_loss(
                x=batch,
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
            batch = batch.to(self.device)
            
            # Forward pass
            x_reconstructed, mu, logvar = self.vae(batch)

            # Compute loss
            loss, recon_loss, kl_loss = self._vae_loss(
                x=batch,
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

        
    