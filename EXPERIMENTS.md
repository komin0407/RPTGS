# RPTGS Experiment Specification

Full reproduction spec for the RPTGS experiments (return-priority temporal
gradient surgery on TD3 / SAC). Everything below is exactly what produced the
reference numbers in the results tables.

## 1. Software environment

- Python 3.11 (3.11.13 used)
- `torch==2.5.1+cu121` (GPU runs, CUDA 12.x driver) — earlier CPU-only runs
  used `torch 2.13.0`; the algorithm is unaffected by the torch version
- `stable-baselines3==2.5.0`
- `gymnasium[mujoco]==1.0.0` (mujoco 3.11.0)
- LunarLander additionally needs: `pip install swig && pip install "gymnasium[box2d]"`
  (box2d-py 2.3.8)
- numpy, scipy

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "stable-baselines3==2.5.0" "gymnasium[mujoco]==1.0.0" numpy scipy
pip install swig && pip install "gymnasium[box2d]"   # for LunarLander only
```

## 2. Method variants

All models live in `td3/models/` and `sac/models/`:

| Variant | TD3 class | SAC class | Merge geometry |
|---|---|---|---|
| norm-balanced (original RPTGS) | `RPTGSTD3` (`rptgs_td3.py`) | `RPTGSSAC` (`rptgs_sac.py`) | both gradients normalized, merged at equal weight |
| cap-only | `RPTGSCapTD3` (`rptgs_cap_td3.py`) | `RPTGSCapSAC` (`rptgs_cap_sac.py`) | temporal keeps raw norm, capped at return norm, never amplified |
| alignment-adaptive (exploratory) | — | `RPTGSAdaSAC` (`rptgs_ada_sac.py`) | amplification gated by gradient cosine |
| **unified (interpolated)** | `RPTGSUnifiedTD3` (`rptgs_unified_td3.py`) | `RPTGSUnifiedSAC` (`rptgs_unified_sac.py`) | single knob `eta` in [0,1]; `eta=0` == cap-only, `eta=1` == norm-balanced (verified numerically) |

## 3. Hyperparameters (identical across ALL environments and backbones)

| Parameter | Value | Note |
|---|---|---|
| learning_rate | **3e-4** (actor & critic) | passed explicitly (SB3 TD3 default is 1e-3) |
| batch_size | 256 | SB3 default |
| buffer_size | 1,000,000 | SB3 default |
| learning_starts | 100 | SB3 default |
| tau | 0.005 | |
| gamma | 0.99 | |
| train_freq / gradient_steps | 1 / 1 | |
| activation | **SiLU** | forced inside every RPTGS class |
| net_arch | SB3 default — TD3 `[400,300]`, SAC `[256,256]` | never overridden |
| **train seed** | **20260718** | multi-seed runs add: 410580, 922852, 787576, 660993 |

TD3-only: `policy_delay=2`, `target_policy_noise=0.2`, `target_noise_clip=0.5`,
exploration noise `NormalActionNoise(mean=0, sigma=0.1)` per action dim.

SAC-only: `ent_coef="auto"`, no external action noise.

Per-environment hyperparameter tuning was deliberately NOT used (PAVE's
per-env values, e.g. lunar gamma=0.98, were not applied) — the same settings
run everywhere.

## 4. Environments and training steps

| Environment | env id | Total steps |
|---|---|---|
| Pendulum | `Pendulum-v1` | 100,000 |
| LunarLander | `gym.make("LunarLander-v3", continuous=True)` | 500,000 |
| Reacher | `Reacher-v5` | 500,000 |
| Hopper | `Hopper-v5` | 1,000,000 |
| Walker | `Walker2d-v5` | 1,000,000 |
| Ant | `Ant-v5` | 1,000,000 |

Env wrapper: `DummyVecEnv([...])` then `VecMonitor`, `n_envs=1`.

## 5. Evaluation protocol

1. After training, evaluate on the **10 fixed validation seeds**, one episode
   each (in order): `178132, 6021, 40415, 92510, 13377, 58024, 71190, 3298,
   84461, 26753` (also stored in each `experiments/<env>/data/validation_seeds.txt`).
2. `env.reset(seed=<validation seed>)`, then roll out the **deterministic**
   policy (`model.predict(obs, deterministic=True)`), no noise, no filtering.
3. Clip actions to `env.action_space.low/high` before stepping.
4. Record the episode return and the full action sequence `(T, act_dim)`
   until `terminated or truncated`.

## 6. Metrics

- **reward** = mean episode return over the 10 evaluation episodes
- **smoothness** = mean spectral smoothness (lower = smoother; PAVE definition):

```python
def spectral_smoothness(actions):          # actions: (T, act_dim)
    n = len(actions)
    freqs = np.fft.fftfreq(n)[: n // 2, None]
    mags  = np.abs(np.fft.fft(actions, axis=0))[: n // 2]
    return float(np.mean(2.0 / n * np.sum(freqs * mags, axis=0)))
```

- Pendulum only: relative score vs the PAVE-TD3 reference
  (`reward0 = -156.1981`, `smooth0 = 0.340948`):
  `score = min(reward_gain%, smooth_gain%)` where
  `reward_gain% = (reward - reward0)/|reward0|*100`,
  `smooth_gain% = (smooth0 - smooth)/smooth0*100`.

## 7. Running the experiments

Ready-to-run trainers live under `experiments/<env>/`. They read
`data/validation_seeds.txt` relative to the current directory, so **run from
inside the environment directory**:

```bash
cd experiments/hopper
python train_rptgs_sac_hopper.py      --max_minutes 420 --run_name sac_nb   --train_seed 20260718
python train_rptgs_cap_sac_hopper.py  --max_minutes 420 --run_name sac_cap  --train_seed 20260718
python train_rptgs_td3_hopper.py      --max_minutes 420 --run_name td3_nb   --train_seed 20260718
python train_rptgs_cap_td3_hopper.py  --max_minutes 420 --run_name td3_cap  --train_seed 20260718
python score_hopper.py runs/sac_nb/results.json "label"
```

Notes:
- `--max_minutes` is a wall-clock budget for the artifact contract; training
  runs the full step count regardless (SB3 `learn()` is not interruptible).
  Use a generous value.
- Output artifact: `runs/<run_name>/results.json` with per-episode returns and
  full action sequences; the score scripts recompute reward/smoothness from it.
- Smoke test: `--max_minutes 2` trains only 1,500 steps and evaluates 1 seed.
- The unified variant is used the same way in your own script:

```python
from sac.models.rptgs_unified_sac import RPTGSUnifiedSAC
model = RPTGSUnifiedSAC("MlpPolicy", env, learning_rate=3e-4,
                        seed=20260718, eta=0.5)   # eta in [0,1]
```

Cluster submission (Slurm sbatch files) is intentionally not included — the
originals contained machine-specific paths/partitions. Any scheduler works;
one GPU (or CPU-only) per run is enough. Peak VRAM is ~25 MB; these workloads
are environment-simulation-bound, so GPU speedup is modest.

## 8. Reference results (train seed 20260718, mean over the 10 eval seeds)

### SAC backbone

| Env | norm-balanced reward / smooth | cap-only reward / smooth |
|---|---|---|
| Pendulum | -155.77 / 0.3060 (score +0.28 vs PAVE) | — |
| LunarLander | 287.79 / 0.1357 | 262.37 / 0.4516 |
| Reacher | -5.64 / 0.0278 | -4.88 / 0.0500 |
| Hopper | 1544.52 / 0.1919 (falls ~400 steps) | 3232.63 / 0.5055 (10/10 full) |
| Walker2d | 5557.91 / 0.2420 | 4950.72 / 0.5117 |
| Ant | 939.10 / 0.0015 (standing collapse) | 5361.89 / 1.4118 |

Ant, alignment-adaptive (`RPTGSAdaSAC`): 3097.25 / 0.8852.

### TD3 backbone

| Env | norm-balanced reward / smooth | cap-only reward / smooth |
|---|---|---|
| Pendulum | -155.05 / 0.3448 | — |
| LunarLander | 236.86 / 0.1467 | — |
| Reacher | -5.00 / 0.0444 | -4.84 / 0.0480 |
| Hopper | 3270.96 / 0.1414 | 3494.72 / 0.7190 |
| Walker2d | 4090.95 / 0.3889 | 4320.66 / 0.6985 |
| Ant | 4646.98 / 0.8279 | 4712.27 / 1.2712 |

Pendulum multi-seed (5 seeds, TD3): RPTGS mean reward -158.09 / smooth 0.3956
(score mean -17.05) vs measured PAVE-TD3 -162.27 / 0.4010 (score mean -17.62).

Key observations:
- SAC + norm-balanced collapses on Ant (standing still) and Hopper (early
  falls); cap-only fixes both. This failure does NOT reproduce on TD3.
- On TD3 all four compared envs show the same mild trade-off: cap-only is
  slightly better on reward, norm-balanced clearly better on smoothness.
- Walker2d refutes "locomotion always breaks norm-balancing": SAC
  norm-balanced wins both axes there.

Expect matching trends (not bit-identical numbers) when reproducing on
different hardware / torch builds.
