#!/usr/bin/env python3
"""Reward + spectral smoothness for a results.json, no PAVE baseline comparison.

Ant has no PAVE reward0/smooth0 reference in this workspace (only
Pendulum was measured), so this reports raw reward and smoothness only.
"""
import json
import sys
from pathlib import Path

import numpy as np


def spectral_smoothness(actions) -> float:
    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.shape[0] < 2:
        return 0.0
    if action_array.ndim == 1:
        action_array = action_array[:, None]
    n = action_array.shape[0]
    frequencies = np.fft.fftfreq(n)[: n // 2, None]
    magnitudes = np.abs(np.fft.fft(action_array, axis=0))[: n // 2]
    spectral_per_action = 2.0 / n * np.sum(frequencies * magnitudes, axis=0)
    return float(np.mean(spectral_per_action))


def main() -> None:
    results_path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else results_path.as_posix()
    output = json.loads(results_path.read_text(encoding="utf-8"))
    episodes = output.get("episodes", [])
    returns = [float(e["return"]) for e in episodes if "return" in e]
    smooths = [spectral_smoothness(e["actions"]) for e in episodes if "actions" in e]

    print(f"=== {label} ===")
    print(f"episodes: {len(episodes)}")
    for e in episodes:
        print(f"  seed={e['seed']:>7}  return={e['return']:>10.3f}  steps={len(e['actions'])}")
    print(f"reward_mean: {np.mean(returns):.4f}")
    print(f"reward_std:  {np.std(returns):.4f}")
    print(f"smoothness_mean: {np.mean(smooths):.6f}")
    print(f"smoothness_std:  {np.std(smooths):.6f}")


if __name__ == "__main__":
    main()
