"""Train a reward model using preference comparisons.

Can be used as a CLI script, or the `train_preference_comparisons` function
can be called directly.
"""

import functools
import pathlib
from typing import Any, Mapping, Optional, Type, Union

import os
import numpy as np
import torch as th
from imitation.util.util import make_seeds
from sacred.observers import FileStorageObserver
from stable_baselines3.common import type_aliases

import imitation.data.serialize as data_serialize
import imitation.policies.serialize as policies_serialize
from imitation.algorithms import preference_comparisons
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)
from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward
from imitation.scripts.ingredients import rl as rl_common
import imitation.algorithms.variquery.variquery as variquery
from imitation.algorithms.duo.duo import RewardDifferenceDiversityFragmenter


def save_model(
    agent_trainer: preference_comparisons.AgentTrainer,
    save_path: pathlib.Path,
):
    """Save the model as `model.zip`."""
    policies_serialize.save_stable_model(
        output_dir=save_path / "policy",
        model=agent_trainer.algorithm,
    )


def save_checkpoint(
    trainer: preference_comparisons.PreferenceComparisons,
    save_path: pathlib.Path,
    allow_save_policy: Optional[bool],
):
    """Save reward model and optionally policy."""
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.model, save_path / "reward_net.pt")
    if allow_save_policy:
        # Note: We should only save the model as model.zip if `trajectory_generator`
        # contains one. Currently we are slightly over-conservative, by requiring
        # that an AgentTrainer be used if we're saving the policy.
        assert isinstance(
            trainer.trajectory_generator,
            preference_comparisons.AgentTrainer,
        )
        save_model(trainer.trajectory_generator, save_path)
    else:
        trainer.logger.warn(
            "trainer.trajectory_generator doesn't contain a policy to save.",
        )


@train_preference_comparisons_ex.main
def train_preference_comparisons(
    total_timesteps: int,
    total_comparisons: int,
    num_iterations: int,
    comparison_queue_size: Optional[int],
    fragment_length: int,
    transition_oversampling: float,
    initial_comparison_frac: float,
    exploration_frac: float,
    trajectory_path: Optional[str],
    trajectory_generator_kwargs: Mapping[str, Any],
    save_preferences: bool,
    agent_path: Optional[str],
    preference_model_kwargs: Mapping[str, Any],
    reward_trainer_kwargs: Mapping[str, Any],
    gatherer_cls: Type[preference_comparisons.PreferenceGatherer],
    gatherer_kwargs: Mapping[str, Any],
    active_selection: bool,
    active_selection_oversampling: int,
    variquery_enabled: bool,
    variquery_oversampling: int,
    variquery_num_clusters: int,
    vae_epochs: int,
    vae_latent_dim: int,
    vae_hidden_dims: list[int],
    vae_batch_size: int,
    vae_lr: float,
    vae_kl_weight: float,
    vae_early_stopping_patience: Optional[int],
    vae_mode: str,
    uncertainty_on: str,
    uncertainty_consensual_filtering: bool,
    fragmenter_kwargs: Mapping[str, Any],
    allow_variable_horizon: bool,
    checkpoint_interval: int,
    query_schedule: Union[str, type_aliases.Schedule],
    _rnd: np.random.Generator,
    sampling_strategy: str = 'random',
    diversity_filtering: Optional[str] = None,
    diversity_filtering_clustering_method: str = "kmeans",
    replay_buffer_size: int = 1000000,
) -> Mapping[str, Any]:
    """Train a reward model using preference comparisons.

    Args:
        total_timesteps: number of environment interaction steps
        total_comparisons: number of preferences to gather in total
        num_iterations: number of times to train the agent against the reward model
            and then train the reward model against newly gathered preferences.
        comparison_queue_size: the maximum number of comparisons to keep in the
            queue for training the reward model. If None, the queue will grow
            without bound as new comparisons are added.
        fragment_length: number of timesteps per fragment that is used to elicit
            preferences
        transition_oversampling: factor by which to oversample transitions before
            creating fragments. Since fragments are sampled with replacement,
            this is usually chosen > 1 to avoid having the same transition
            in too many fragments.
        initial_comparison_frac: fraction of total_comparisons that will be
            sampled before the rest of training begins (using the randomly initialized
            agent). This can be used to pretrain the reward model before the agent
            is trained on the learned reward.
        exploration_frac: fraction of trajectory samples that will be created using
            partially random actions, rather than the current policy. Might be helpful
            if the learned policy explores too little and gets stuck with a wrong
            reward.
        trajectory_path: either None, in which case an agent will be trained
            and used to sample trajectories on the fly, or a path to a pickled
            sequence of TrajectoryWithRew to be trained on.
        trajectory_generator_kwargs: kwargs to pass to the trajectory generator.
        save_preferences: if True, store the final dataset of preferences to disk.
        agent_path: if given, initialize the agent using this stored policy
            rather than randomly.
        preference_model_kwargs: passed to PreferenceModel
        reward_trainer_kwargs: passed to BasicRewardTrainer or EnsembleRewardTrainer
        gatherer_cls: type of PreferenceGatherer to use (defaults to SyntheticGatherer)
        gatherer_kwargs: passed to the PreferenceGatherer specified by gatherer_cls
        active_selection: use active selection fragmenter instead of random fragmenter
        active_selection_oversampling: factor by which to oversample random fragments
            from the base fragmenter of active selection.
            this is usually chosen > 1 to allow the active selection algorithm to pick
            fragment pairs with highest uncertainty. = 1 implies no active selection.
        uncertainty_on: passed to ActiveSelectionFragmenter
        fragmenter_kwargs: passed to RandomFragmenter
        allow_variable_horizon: If False (default), algorithm will raise an
            exception if it detects trajectories of different length during
            training. If True, overrides this safety check. WARNING: variable
            horizon episodes leak information about the reward via termination
            condition, and can seriously confound evaluation. Read
            https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
            before overriding this.
        checkpoint_interval: Save the reward model and policy models (if
            trajectory_generator contains a policy) every `checkpoint_interval`
            iterations and after training is complete. If 0, then only save weights
            after training is complete. If <0, then don't save weights at all.
        query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic").
            A function indicating how the total number of preference queries should
            be allocated to each iteration. "hyperbolic" and "inverse_quadratic"
            apportion fewer queries to later iterations when the policy is assumed
            to be better and more stable.
        _rnd: Random number generator provided by Sacred.
        sampling_strategy: Which fragment sampling strategy to use. 'random' (default) uses RandomFragmenter, 'priority' uses DUO-style PriorityFragmenter.
        diversity_filtering: Which diversity filtering method to use. 'reward_difference' uses RewardDifferenceDiversityFragmenter.
        diversity_filtering_clustering_method: Which clustering method to use for diversity filtering. 'kmeans' (default) uses KMeans, 'agglomerative' uses AgglomerativeClustering.

    Returns:
        Rollout statistics from trained policy.

    Raises:
        ValueError: Inconsistency between config and deserialized policy normalization.
    """
    # seed = make_seeds(_rnd)
    # th.manual_seed(seed)
    # th.cuda.manual_seed_all(seed)
    #
    # th.backends.cudnn.deterministic = True
    # th.backends.cudnn.benchmark = False
    # th.backends.cuda.matmul.allow_tf32 = False
    # th.backends.cudnn.allow_tf32 = False
    #
    # th.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # cuBLAS deterministic

    # This allows to specify total_timesteps, total_comparisons etc. in scientific
    # notation, which is interpreted as a float by python.
    total_timesteps = int(total_timesteps)
    total_comparisons = int(total_comparisons)
    num_iterations = int(num_iterations)
    comparison_queue_size = (
        int(comparison_queue_size) if comparison_queue_size is not None else None
    )
    fragment_length = int(fragment_length)
    active_selection_oversampling = int(active_selection_oversampling)
    checkpoint_interval = int(checkpoint_interval)

    custom_logger, log_dir = logging_ingredient.setup_logging()

    with environment.make_venv() as venv:  # type: ignore[wrong-arg-count]
        reward_net = reward.make_reward_net(venv)
        relabel_reward_fn = functools.partial(
            reward_net.predict_processed,
            update_stats=False,
        )
        if agent_path is None:
            agent = rl_common.make_rl_algo(venv, relabel_reward_fn=relabel_reward_fn)
        else:
            agent = rl_common.load_rl_algo_from_path(
                agent_path=agent_path,
                venv=venv,
                relabel_reward_fn=relabel_reward_fn,
            )

        if trajectory_path is None:
            # Setting the logger here is not necessary (PreferenceComparisons takes care
            # of it automatically) but it avoids creating unnecessary loggers.
            agent_trainer = preference_comparisons.AgentTrainer(
                algorithm=agent,
                reward_fn=reward_net,
                venv=venv,
                exploration_frac=exploration_frac,
                rng=_rnd,
                custom_logger=custom_logger,
                **trajectory_generator_kwargs,
            )
            # Stable Baselines will automatically occupy GPU 0 if it is available.
            # Let's use the same device as the SB3 agent for the reward model.
            reward_net = reward_net.to(agent_trainer.algorithm.device)
            trajectory_generator: preference_comparisons.TrajectoryGenerator = (
                agent_trainer
            )
        else:
            if exploration_frac > 0:
                raise ValueError(
                    "exploration_frac can't be set when a trajectory dataset is used",
                )
            trajectory_generator = preference_comparisons.TrajectoryDataset(
                trajectories=data_serialize.load_with_rewards(
                    trajectory_path,
                ),
                rng=_rnd,
                custom_logger=custom_logger,
                **trajectory_generator_kwargs,
            )

        fragmenter = preference_comparisons.RandomFragmenter(
            **fragmenter_kwargs,
            rng=_rnd,
            custom_logger=custom_logger,
        )

        preference_model = preference_comparisons.PreferenceModel(
            **preference_model_kwargs,
            model=reward_net,
        )
        if active_selection:
            fragmenter = preference_comparisons.UncertaintyFragmenter(
                preference_model=preference_model,
                base_fragmenter=fragmenter,
                fragment_sample_factor=active_selection_oversampling,
                uncertainty_on=uncertainty_on,
                consensual_filter=uncertainty_consensual_filtering,
                custom_logger=custom_logger,
            )
        if variquery_enabled:
            # Create VARIQuery fragmenter
            fragmenter = variquery.VARIQueryFragmenter(
                state_dim=venv.observation_space.shape[0],
                allow_variable_horizon=allow_variable_horizon,
                sequence_length=fragment_length,
                fragment_sample_factor=variquery_oversampling,
                preference_model=preference_model,
                base_fragmenter=fragmenter,
                rng=_rnd,
                variquery_num_clusters=variquery_num_clusters,
                vae_epochs=vae_epochs,
                vae_latent_dim=vae_latent_dim,
                vae_hidden_dims=vae_hidden_dims,
                vae_lr=vae_lr,
                vae_kl_weight=vae_kl_weight,
                vae_batch_size=vae_batch_size,
                vae_early_stopping_patience=vae_early_stopping_patience,
                vae_mode=vae_mode,
                custom_logger=custom_logger,
                device=agent_trainer.algorithm.device,
            )

        if diversity_filtering is not None:
            if diversity_filtering == "reward_difference":
                fragmenter = RewardDifferenceDiversityFragmenter(
                    preference_model=preference_model,
                    base_fragmenter=fragmenter,
                    custom_logger=custom_logger,
                    clustering_method=diversity_filtering_clustering_method,
                )
            else:
                raise ValueError(f"Invalid diversity filtering: {diversity_filtering}")

        gatherer = gatherer_cls(
            **gatherer_kwargs,
            rng=_rnd,
            custom_logger=custom_logger,
        )

        loss = preference_comparisons.CrossEntropyRewardLoss()

        reward_trainer = preference_comparisons._make_reward_trainer(
            preference_model,
            loss,
            _rnd,
            reward_trainer_kwargs,
        )

        main_trainer = preference_comparisons.PreferenceComparisons(
            trajectory_generator,
            reward_net,
            rng=_rnd,
            num_iterations=num_iterations,
            fragmenter=fragmenter,
            preference_gatherer=gatherer,
            base_algorithm=agent,
            reward_trainer=reward_trainer,
            comparison_queue_size=comparison_queue_size,
            fragment_length=fragment_length,
            transition_oversampling=transition_oversampling,
            initial_comparison_frac=initial_comparison_frac,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
            query_schedule=query_schedule,
            sampling_strategy=sampling_strategy,
            replay_buffer_size=replay_buffer_size,
        )

        def save_callback(iteration_num):
            if checkpoint_interval > 0 and iteration_num % checkpoint_interval == 0:
                save_checkpoint(
                    trainer=main_trainer,
                    save_path=log_dir / "checkpoints" / f"{iteration_num:04d}",
                    allow_save_policy=bool(trajectory_path is None),
                )

        results = main_trainer.train(
            total_timesteps,
            total_comparisons,
            callback=save_callback,
        )

        # Storing and evaluating policy only useful if we generated trajectory data
        if bool(trajectory_path is None):
            results = dict(results)
            results["imit_stats"] = policy_evaluation.eval_policy(agent, venv)

    if save_preferences:
        main_trainer.dataset.save(log_dir / "preferences.pkl")

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save_checkpoint(
            trainer=main_trainer,
            save_path=log_dir / "checkpoints" / "final",
            allow_save_policy=bool(trajectory_path is None),
        )

    return results


def main_console():
    observer_path = (
        pathlib.Path.cwd() / "output" / "sacred" / "train_preference_comparisons"
    )
    observer = FileStorageObserver(observer_path)
    train_preference_comparisons_ex.observers.append(observer)
    train_preference_comparisons_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
