from sklearn.neighbors import NearestNeighbors
from imitation.algorithms.preference_comparisons import Fragmenter
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
        segments: Sequence[TrajectoryWithRew],
    ):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> th.Tensor:
        segment = self.segments[idx]
        # Convert segment to tensor of shape (sequence_length, state_dim)
        return th.from_numpy(segment.obs[:-1]).float()

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
        reward_ensemble: reward_nets.RewardEnsemble,
        vae_hidden_dims: List[int] = [128, 64, 32],
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        super().__init__(custom_logger=custom_logger)

        self.reward_ensemble = reward_ensemble
        self.vae = MLPStateVAE(
            state_dim=state_dim, 
            sequence_length=sequence_length, 
            latent_dim=vae_latent_dim, 
            hidden_dims=vae_hidden_dims,
        )

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        """The VARIQuery alogrithm implementation"""

        # Step 1: Create the dataset of state segments from sampled trajectories
        D_un = StateSegmentDataset(trajectories)

        # Step 2: Train the VAE on the dataset and encode the segments
        self._train_vae(D_un)
        D_z = self._encode_segments(trajectories)

        # Step 3: Cluster the latent space using k-NN
        clusters = self._cluster_latent_space(D_z, num_clusters=10)

        # Step 4: Sample and rank pairs
        self.D_q = self._sample_random_pairs(clusters, num_pairs)
        ranked_pairs = self._rank_pairs(self.D_q)

        # Step 5: Return top N pairs
        self._sample_from_top_pairs(ranked_pairs, num_pairs)
        

    @th.no_grad()
    def _encode_segments(
        self,
        segments: Sequence[TrajectoryWithRew],
    ) -> List[th.Tensor]:
        return [self.vae.encode(self._segment_to_tensor(segment)) for segment in segments]
    
    def _cluster_latent_space(
        self,
        encoded_segments: List[th.Tensor],
        num_clusters: int,
    ) -> List[List[int]]:
        """Cluster the latent space using k-NN
        
        Args:
            encoded_segments: List of encoded segments
            num_clusters: The number of clusters to use

        Returns:
        """
        knn = NearestNeighbors(n_neighbors=num_clusters)
        knn.fit(th.stack(encoded_segments).numpy())
        # Return indices grouped by cluster
        return knn.kneighbors(th.stack(encoded_segments).numpy(), return_distance=False)

    def _sample_random_pairs(
        self,
        clusters: List[List[int]],
        trajectories: Sequence[TrajectoryWithRew],
        num_pairs: int,
    ) -> List[TrajectoryWithRewPair]:
        """Sample pairs at random from the clusters"""
        pairs = []
        for cluster in clusters:
            # Sample two random indices from the cluster
            idx1, idx2 = np.random.choice(cluster, size=2, replace=False)
            pairs.append(TrajectoryWithRewPair(trajectory1=trajectories[idx1], trajectory2=trajectories[idx2]))
        return pairs
        
    def _rank_by_ensemble_variance(
        self,
        pairs: List[TrajectoryWithRewPair],
    ) -> List[TrajectoryWithRewPair]:
        """Rank pairs based on the disagreement of the ensemble members"""
        rews1 = self.reward_ensemble.predict_processed(pairs[0].trajectory1.obs[:-1], pairs[0].trajectory1.action, pairs[0].trajectory1.next_obs[:-1], pairs[0].trajectory1.done)
        rews2 = self.reward_ensemble.predict_processed(pairs[1].trajectory1.obs[:-1], pairs[1].trajectory1.action, pairs[1].trajectory1.next_obs[:-1], pairs[1].trajectory1.done)
        # Compute the variance of the rewards
        var = th.var(rews1 - rews2, dim=0)
        # Sort the pairs by the variance of the rewards
        sorted_pairs = [x for _, x in sorted(zip(var, pairs), key=lambda pair: pair[0])]
        return sorted_pairs
        

    def _sample_from_top_pairs(
        self,
        pairs: List[TrajectoryWithRewPair],
        num_pairs: int,
    ) -> List[TrajectoryWithRew]:
        """Sample from the top pairs"""
        return [pairs[i] for i in range(num_pairs)]
            

    def _segment_to_tensor(
        self,
        segment: TrajectoryWithRew,
    ) -> th.Tensor:
        """Convert a segment to a tensor
        
        Args:
            segment: The segment to convert
        """
        return th.from_numpy(segment.obs[:-1]).float()

    def _train_vae(
        self,
        dataset: StateSegmentDataset
    ):
        """Train the VAE on the dataset"""
        trainer = VAETrainer(
            vae=self.vae,
            # TOOD: Add remaining arguments
        )
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
    ):
        super().__init__()

        self.state_dim = state_dim
        self.fragment_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.flat_dim = state_dim * sequence_length

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
            x: (batch_size, sequence_length, state_dim)

        Returns:
            Tuple of (mu, logvar) each of shape (batch_size, latent_dim)
        """
        # Flatten input: (batch, seq_len, state_dim)
        x_flat = x.view(x.shape[0], -1)

        # Encode
        hidden = self.encoder(x_flat)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar
    
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
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
        self.optimizer = optimizer or th.optim.Adam(vae.parameters(), lr=lr)
        self.kl_weight_beta = kl_weight_beta
        self.early_stopping_patience = early_stopping_patience
        self.logger = custom_logger or imit_logger.configure()
        self.regularizer = (
            regularizer_factory(optimizer=self.optimizer, logger=self.logger)
            if regularizer_factory is not None
            else None
        )

    def train(
        self,
        dataset: data_th.Dataset,    
    ):
        dataloader, val_dataloader = self._create_data_loaders(dataset)

        assert self.epochs > 0, "Must train for at least one epoch."

        best_val_loss = float("inf")
        patience_counter = 0

        with self.logger.accumulate_means("variquery"):
            for epoch in tqdm(range(self.epochs), desc="Training VAE"):
                with self.logger.add_key_prefix(f"epoch-{epoch}"):
                    avg_loss, val_loss = self._train_epoch(dataloader, val_dataloader)

                    if self.early_stopping_patience is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= self.early_stopping_patience:
                            self.logger.log("Early stopping triggered at epoch {}".format(epoch), step=epoch)
                            break

    def _create_data_loaders(
        self,
        dataset: data_th.Dataset,
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
          self.logger.log("loss", avg_loss, step=self.epoch)
          self.logger.log("recon_loss", avg_recon_loss, step=self.epoch)
          self.logger.log("kl_loss", avg_kl_loss, step=self.epoch)
            
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
          self.logger.log("loss", avg_loss, step=self.epoch)
          self.logger.log("recon_loss", avg_recon_loss, step=self.epoch)
          self.logger.log("kl_loss", avg_kl_loss, step=self.epoch)

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

        
    