import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from custom_td3 import CustomTD3
from stable_baselines3.td3.policies import TD3Policy, Actor
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import math

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class L2C2TD3(CustomTD3):
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
        l2c2_sigma = 1.0,
        l2c2_lamD = 0.01,
        l2c2_lamU = 1.0,
        l2c2_beta = 0.1
    ):
        self.l2c2_sigma = l2c2_sigma
        self.l2c2_lamD = l2c2_lamD
        self.l2c2_lamU = l2c2_lamU
        self.l2c2_beta = l2c2_beta
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

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # L2C2 Loss
            observations = replay_data.observations.clone()
            next_observations = replay_data.next_observations.clone()

            sigma = self.l2c2_sigma
            ulam = self.l2c2_lamU
            dlam = self.l2c2_lamD
            beta = self.l2c2_beta
            epsilon = sigma * dlam / (ulam - sigma * dlam)
            width = sigma + (sigma - 1) * epsilon
            u = th.rand_like(observations) * 2 * width - width
            s_bar = observations + (next_observations - observations) * u

            diff = (s_bar - observations) / (next_observations - observations + 1e-8)
            d_us = th.norm(diff, p=float("inf"), dim=-1).detach() + epsilon

            # 가치 함수의 출력 계산
            v_s_c = th.cat(self.critic(observations, replay_data.actions), dim=1)
            v_s_bar_c = th.cat(self.critic(s_bar, replay_data.actions), dim=1)
            v_s, _ = th.min(v_s_c, dim=1, keepdim=True)
            v_s_bar, _ = th.min(v_s_bar_c, dim=1, keepdim=True)
            value_distance = th.norm(v_s - v_s_bar, 2.0, dim=-1) / 2

            # L2C2 정규화 손실
            lambda_pi = epsilon * ulam / d_us  # 정책 정규화 가중치
            lambda_v = beta * lambda_pi / d_us  # 가치 정규화 가중치

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values) + th.mean(lambda_v * value_distance)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # 결정론적 정책이므로 SAC의 Hellinger 거리 대신
                # 행동 간 L2 거리로 정책 정규화 항 계산
                a_s = self.actor(replay_data.observations)
                a_s_bar = self.actor(s_bar)
                d_pi_per_sample = th.norm(a_s - a_s_bar, p=2, dim=-1) / 2

                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, a_s).mean() + th.mean(lambda_pi * d_pi_per_sample)
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
