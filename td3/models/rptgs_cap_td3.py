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


def _set_two_task_capgrad(
    parameters: Tuple[nn.Parameter, ...],
    return_gradients: Tuple[Optional[th.Tensor], ...],
    temporal_gradients: Tuple[Optional[th.Tensor], ...],
) -> Tuple[float, float, float]:
    """Assign return-priority, cap-only two-task surgery to ``parameter.grad``.

    Cap-only variant of the RPTGS norm-balanced merge. The temporal gradient
    keeps its RAW magnitude and is only ever scaled DOWN: its norm is capped at
    the return gradient's norm so smoothness can never outvote the return
    objective, but -- unlike the norm-balanced variant -- a small temporal
    gradient is never amplified up to the return scale. When the two gradients
    conflict (negative dot product), the capped temporal gradient is projected
    off the return direction, leaving return progress untouched. The return
    gradient itself is used as-is (no halving), so the standard TD3 actor step
    is fully preserved whenever the policy is already smooth.

    This targets the locomotion failure mode of norm balancing (e.g. Ant,
    Hopper), where amplifying a tiny smoothness gradient to a fixed ~50% share
    of every update suppresses the periodic gait and collapses the policy.
    There is still no tunable weight: the temporal contribution is set entirely
    by the two gradients' relative raw magnitudes.

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
        return_norm_val = float(return_norm.item())
        temporal_norm_val = float(temporal_norm.item())
        cosine = float(
            (th.dot(return_flat, temporal_flat) / (return_norm * temporal_norm)).item()
        )

        # Cap only: never let the temporal gradient exceed the return norm,
        # never scale it up.
        cap_scale = th.clamp(return_norm / temporal_norm, max=1.0)
        capped_temporal = temporal_flat * cap_scale

        # Return-priority conflict resolution: project the capped temporal
        # gradient off the return direction when they disagree.
        dot_product = th.dot(return_flat, capped_temporal)
        if dot_product.item() < 0.0:
            capped_temporal = capped_temporal - (dot_product / return_norm_sq) * return_flat

        merged = return_flat + capped_temporal
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


class RPTGSCapTD3(TD3):
    """Cap-only Return-Priority Two-task Gradient Surgery TD3 (RPTGS-Cap).

    Identical to RPTGSTD3 except for the gradient-merge geometry: instead of
    norm-balancing both task gradients (which grants the temporal objective a
    fixed ~50% share of every actor update regardless of its raw scale), the
    temporal gradient keeps its raw magnitude and is only capped at the return
    gradient's norm -- smoothness may never outvote return, but is also never
    amplified. See ``_set_two_task_capgrad``. Still weight-free and
    return-priority; the critic update is identical to standard TD3.
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
        # RPTGS-Cap diagnostics
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
            # [RPTGS-Cap Actor Update (Delayed) -- cap-only two-task surgery]
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

                    # Return-priority, cap-only two-task surgery.
                    self.actor.optimizer.zero_grad()
                    cosine, ret_norm, tmp_norm = _set_two_task_capgrad(
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

        # RPTGS-Cap diagnostics
        if len(temporal_losses) > 0:
            self.logger.record("train/rptgs_temporal_loss", np.mean(temporal_losses))
        if len(cosines) > 0:
            self.logger.record("train/rptgs_grad_cosine", np.mean(cosines))
            self.logger.record("train/rptgs_return_grad_norm", np.mean(ret_norms))
            self.logger.record("train/rptgs_temporal_grad_norm", np.mean(tmp_norms))
