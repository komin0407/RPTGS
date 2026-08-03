#!/usr/bin/env python3
"""RPTGS-TD3 (PAVE/td3/models/rptgs_td3.py) on Reacher-v5.

Identical hyperparameters to the Pendulum runs (train_rptgs_sb3.py): same
RPTGSTD3 class (return-priority, norm-balanced two-task PCGrad; SiLU forced;
net_arch [400,300]), same learning_rate=3e-4, same SB3 TD3 defaults for
buffer_size/learning_starts/gamma/train_freq/gradient_steps. Only the
environment and total_timesteps change (500_000 for Reacher vs 100_000
for Pendulum, matching PAVE's own Reacher schedule).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from td3.models.rptgs_td3 import RPTGSTD3  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ENV_ID = "Hopper-v5"
TRAIN_SEED = 20260718
FULL_TRAIN_STEPS = 1_000_000


def make_env():
    def _init():
        return gym.make(ENV_ID)
    return _init


def read_validation_seeds() -> list[int]:
    seed_lines = Path("data/validation_seeds.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    validation_seeds = [int(line.strip()) for line in seed_lines if line.strip()]
    if not validation_seeds:
        raise RuntimeError("data/validation_seeds.txt contains no seeds")
    return validation_seeds


@th.no_grad()
def evaluate_policy(model: RPTGSTD3, validation_seeds: list[int]) -> list[dict[str, object]]:
    env = gym.make(ENV_ID)
    episodes: list[dict[str, object]] = []
    try:
        for seed in validation_seeds:
            observation, _ = env.reset(seed=seed)
            episode_return = 0.0
            actions: list[list[float]] = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                action = np.clip(
                    action, env.action_space.low, env.action_space.high
                ).astype(np.float32)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                actions.append([float(v) for v in np.asarray(action).reshape(-1)])
                episode_return += float(reward)
                observation = next_observation
            episodes.append(
                {"seed": int(seed), "return": float(episode_return), "actions": actions}
            )
    finally:
        env.close()
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_minutes", type=float, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--train_seed", type=int, default=TRAIN_SEED)
    args = parser.parse_args()
    if not np.isfinite(args.max_minutes) or args.max_minutes <= 0.0:
        parser.error("--max_minutes must be a positive finite number")
    run_path = Path(args.run_name)
    if not args.run_name or run_path.name != args.run_name or args.run_name in {".", ".."}:
        parser.error("--run_name must be a single safe path component")
    return args


def main() -> None:
    args = parse_args()
    train_seed = args.train_seed
    started = time.monotonic()
    total_seconds = args.max_minutes * 60.0
    reserve_seconds = min(60.0, max(15.0, total_seconds * 0.05))
    train_deadline = started + max(0.0, total_seconds - reserve_seconds)
    smoke = args.max_minutes <= 2.01
    max_steps = 1_500 if smoke else FULL_TRAIN_STEPS

    validation_seeds = read_validation_seeds()

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecMonitor(vec_env)
    action_dim = vec_env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = RPTGSTD3(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        verbose=0,
        seed=train_seed,
        device=device,
        action_noise=action_noise,
    )

    model.learn(total_timesteps=max_steps)
    vec_env.close()

    if time.monotonic() >= train_deadline:
        print("warning: training ran past the reserved deadline", file=sys.stderr)

    seeds_to_evaluate = validation_seeds[:1] if smoke else validation_seeds
    episodes = evaluate_policy(model, seeds_to_evaluate)

    artifact_path = Path("runs") / args.run_name / "results.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "env_id": ENV_ID,
        "backbone": "TD3",
        "method": "RPTGS-SB3",
        "train_seed": train_seed,
        "episodes": episodes,
    }
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, allow_nan=False, separators=(",", ":"))

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = th.cuda.max_memory_allocated(device) / (1024.0**2)
    print(f"completed_steps:  {max_steps}")
    print("---")
    print(f"artifact:         {artifact_path.as_posix()}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")


if __name__ == "__main__":
    main()
