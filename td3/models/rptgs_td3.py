from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy, Actor
from typing import Any, ClassVar, Optional, TypeVar, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update


SelfTD3 = TypeVar("SelfTD3", bound="TD3")


def _flat_gradients(
    parameters: Tuple[nn.Parameter, ...],
    gradients: Tuple[Optional[th.Tensor], ...],
) -> th.Tensor:
    """Flatten a task gradient, treating unused parameters as exact zeros."""
    return th.cat(
        [
            th.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def _set_two_task_pcgrad(
    parameters: Tuple[nn.Parameter, ...],
    return_gradients: Tuple[Optional[th.Tensor], ...],
    temporal_gradients: Tuple[Optional[th.Tensor], ...],
) -> Tuple[float, float, float]:
    """Assign return-priority, norm-balanced two-task PCGrad to ``parameter.grad``.

    Active task gradients are first normalized so the temporal objective
    cannot disappear solely because its raw gradient has a smaller scale than
    the critic-derived return gradient. When their global dot product is
    negative, only the unit temporal gradient is projected off the unit return
    gradient, leaving return progress as the priority. Their mean is finally
    restored to the return gradient's norm, keeping the standard TD3 actor step
    as the scale anchor without a loss weight. An absent or all-zero objective
    contributes nothing rather than diluting the valid task gradient.

    Returns (cosine, return_norm, temporal_norm) for logging.
    """
    return_flat = _flat_gradients(parameters, return_gradients)
    temporal_flat = _flat_gradients(parameters, temporal_gradients)
    return_norm_sq = th.dot(return_flat, return_flat)
    temporal_norm_sq = th.dot(temporal_flat, temporal_flat)
    epsilon = th.finfo(return_flat.dtype).eps
    has_return = bool(return_norm_sq.item() > 0.0)
    has_temporal = bool(temporal_norm_sq.item() > 0.0)

    cosine = 0.0
    return_norm_val = 0.0
    temporal_norm_val = 0.0

    if has_return and has_temporal:
        return_norm = return_norm_sq.sqrt().clamp_min(epsilon)
        temporal_norm = temporal_norm_sq.sqrt().clamp_min(epsilon)
        balanced_return = return_flat / return_norm
        balanced_temporal = temporal_flat / temporal_norm
        dot_product = th.dot(balanced_return, balanced_temporal)
        cosine = float(dot_product.item())
        return_norm_val = float(return_norm.item())
        temporal_norm_val = float(temporal_norm.item())
        if dot_product.item() < 0.0:
            projected_temporal = balanced_temporal - dot_product * balanced_return
        else:
            projected_temporal = balanced_temporal
        merged = 0.5 * return_norm * (balanced_return + projected_temporal)
    elif has_return:
        merged = return_flat
        return_norm_val = float(return_norm_sq.sqrt().item())
    elif has_temporal:
        merged = temporal_flat
        temporal_norm_val = float(temporal_norm_sq.sqrt().item())
    else:
        merged = th.zeros_like(return_flat)

    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = merged[offset : offset + count].view_as(parameter).clone()
        offset += count

    return cosine, return_norm_val, temporal_norm_val


class RPTGSTD3(TD3):
    """Return-Priority Two-task Gradient Surgery TD3 (RPTGS).

    The actor is trained on exactly two objectives with **no** weighting
    coefficient: the standard deterministic return objective ``-Q1(s, pi(s))``
    and a direct temporal action-smoothness objective ``||pi(s') - pi(s)||^2``.
    Their gradients are computed on separate graphs and reconciled solely by
    norm-balanced, return-priority two-task PCGrad geometry (see
    ``_set_two_task_pcgrad``). Unlike CAPS/PAVE, there is no ``lambda`` scalar
    trading off the two terms -- the return gradient is the scale anchor and the
    temporal gradient is only projected off it when the two conflict.

    The critic update is identical to standard TD3.
    """

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # === [RPTGS Hyperparameters] ===
        # RPTGS is deliberately weight-free: the temporal objective is balanced
        # against the return objective by gradient geometry, not by a scalar.
        # Only the activation choice is exposed here for parity with PAVE/CAPS.
    ):
        if policy_kwargs is None:
            policy_kwargs = {}

        # Parity with PAVE/CAPS baselines: SiLU keeps the smoothness geometry
        # well-behaved (twice-differentiable) even though RPTGS itself only
        # needs first-order actor gradients.
        if "activation_fn" not in policy_kwargs:
            policy_kwargs["activation_fn"] = nn.SiLU

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        # Actor(Policy)의 활성화 함수 확인
        actor_activations = [m for m in self.actor.modules() if isinstance(m, nn.SiLU)]
        # Critic의 활성화 함수 확인
        critic_activations = [m for m in self.critic.modules() if isinstance(m, nn.SiLU)]

        print(f"[*] Actor Activation: {'SiLU' if actor_activations else 'Other'}")
        print(f"[*] Critic Activation: {'SiLU' if critic_activations else 'Other'}")

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        # RPTGS diagnostics
        temporal_losses, cosines, ret_norms, tmp_norms = [], [], [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # -----------------------------------------------------------
            # [Standard TD3 Critic Update]
            # -----------------------------------------------------------
            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # -----------------------------------------------------------
            # [RPTGS Actor Update (Delayed) -- two-task gradient surgery]
            # -----------------------------------------------------------
            if self._n_updates % self.policy_delay == 0:
                actor_parameters = tuple(self.actor.parameters())

                # Freeze the critic so return-objective backprop does not touch
                # critic gradients (parity with the standalone reference impl).
                for parameter in self.critic.parameters():
                    parameter.requires_grad_(False)
                try:
                    # --- Task 1: standard deterministic return objective ---
                    return_loss = -self.critic.q1_forward(
                        replay_data.observations, self.actor(replay_data.observations)
                    ).mean()
                    return_gradients = th.autograd.grad(
                        return_loss,
                        actor_parameters,
                        allow_unused=True,
                    )

                    # --- Task 2: temporal action-smoothness objective ---
                    # ||pi(s') - pi(s)||^2 over transitions that do not cross an
                    # episode boundary. dones marks a real terminal / truncation
                    # boundary; those next_observations are resets, so exclude
                    # them from the smoothness penalty.
                    nonterminal = replay_data.dones.squeeze(-1) < 0.5
                    if bool(nonterminal.any().item()):
                        obs_now = replay_data.observations[nonterminal]
                        obs_next = replay_data.next_observations[nonterminal]
                        current_policy_action = self.actor(obs_now)
                        next_policy_action = self.actor(obs_next)
                        temporal_loss = (
                            (next_policy_action - current_policy_action).square().sum(dim=-1)
                        ).mean()
                        temporal_gradients = th.autograd.grad(
                            temporal_loss,
                            actor_parameters,
                            allow_unused=True,
                        )
                        temporal_losses.append(temporal_loss.item())
                    else:
                        temporal_gradients = tuple(None for _ in actor_parameters)

                    # Return-priority, norm-balanced two-task PCGrad.
                    self.actor.optimizer.zero_grad()
                    cosine, ret_norm, tmp_norm = _set_two_task_pcgrad(
                        actor_parameters,
                        return_gradients,
                        temporal_gradients,
                    )
                    self.actor.optimizer.step()

                    actor_losses.append(return_loss.item())
                    cosines.append(cosine)
                    ret_norms.append(ret_norm)
                    tmp_norms.append(tmp_norm)
                finally:
                    for parameter in self.critic.parameters():
                        parameter.requires_grad_(True)

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

        # RPTGS diagnostics
        if len(temporal_losses) > 0:
            self.logger.record("train/rptgs_temporal_loss", np.mean(temporal_losses))
        if len(cosines) > 0:
            self.logger.record("train/rptgs_grad_cosine", np.mean(cosines))
            self.logger.record("train/rptgs_return_grad_norm", np.mean(ret_norms))
            self.logger.record("train/rptgs_temporal_grad_norm", np.mean(tmp_norms))
