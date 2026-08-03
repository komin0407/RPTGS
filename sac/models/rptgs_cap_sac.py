from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from typing import Any, ClassVar, Optional, TypeVar, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update


SelfSAC = TypeVar("SelfSAC", bound="SAC")


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
    gradient itself is used as-is (no halving), so the standard SAC actor step
    is fully preserved whenever the policy is already smooth.

    This targets the locomotion failure mode of norm balancing (e.g. Ant),
    where amplifying a tiny smoothness gradient to a fixed ~50% share of every
    update suppresses the periodic gait and collapses the policy to standing
    still. There is still no tunable weight: the temporal contribution is set
    entirely by the two gradients' relative raw magnitudes.

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


class RPTGSCapSAC(SAC):
    """Cap-only Return-Priority Two-task Gradient Surgery SAC (RPTGS-Cap).

    Identical to RPTGSSAC except for the gradient-merge geometry: instead of
    norm-balancing both task gradients (which grants the temporal objective a
    fixed ~50% share of every actor update regardless of its raw scale), the
    temporal gradient keeps its raw magnitude and is only capped at the return
    gradient's norm -- smoothness may never outvote return, but is also never
    amplified. See ``_set_two_task_capgrad``. Still weight-free and
    return-priority; the critic and entropy updates are standard SAC.
    """

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
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
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # === [RPTGS-Cap Hyperparameters] ===
        # Still deliberately weight-free: the temporal contribution is set by
        # the two gradients' relative raw magnitudes (capped, never amplified).
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
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
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

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        # RPTGS-Cap diagnostics
        temporal_losses, cosines, ret_norms, tmp_norms = [], [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # -----------------------------------------------------------
            # [RPTGS-Cap Actor Update -- cap-only two-task surgery]
            # -----------------------------------------------------------
            actor_parameters = tuple(self.actor.parameters())

            # Freeze the critic so return-objective backprop does not touch
            # critic gradients (parity with the other RPTGS impls).
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            try:
                # --- Task 1: standard SAC actor objective (entropy included) ---
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                return_loss = (ent_coef * log_prob - min_qf_pi).mean()
                return_gradients = th.autograd.grad(
                    return_loss,
                    actor_parameters,
                    allow_unused=True,
                )

                # --- Task 2: temporal action-smoothness objective ---
                # ||mu(s') - mu(s)||^2 on the deterministic mean action (same
                # choice as CAPS-SAC), over transitions that do not cross an
                # episode boundary; those next_observations are resets, so
                # exclude them from the smoothness penalty.
                nonterminal = replay_data.dones.squeeze(-1) < 0.5
                if bool(nonterminal.any().item()):
                    obs_now = replay_data.observations[nonterminal]
                    obs_next = replay_data.next_observations[nonterminal]
                    current_policy_action = self.policy._predict(obs_now, deterministic=True).type(th.float32)
                    next_policy_action = self.policy._predict(obs_next, deterministic=True).type(th.float32)
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

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        # RPTGS-Cap diagnostics
        if len(temporal_losses) > 0:
            self.logger.record("train/rptgs_temporal_loss", np.mean(temporal_losses))
        if len(cosines) > 0:
            self.logger.record("train/rptgs_grad_cosine", np.mean(cosines))
            self.logger.record("train/rptgs_return_grad_norm", np.mean(ret_norms))
            self.logger.record("train/rptgs_temporal_grad_norm", np.mean(tmp_norms))
