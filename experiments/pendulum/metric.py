"""
Pure metrics for the Pendulum-v1 online-RL task.
Path-free: loads nothing, reads nothing. Public — the agent may read this.

Two quantities are measured, then combined into ONE ranking scalar that keeps
them CO-EQUAL:

  reward     — mean raw episode return (PAVE convention; higher is better).
  smoothness — mean FFT-based spectral smoothness of the action sequence
               (lower is smoother / better).

The spectral-smoothness definition is copied verbatim from the PAVE experiment
code (stage2_q_projection.action_metrics): for an action sequence of length T,

    freqs = fftfreq(T)[:T//2]
    mags  = |fft(actions)|[:T//2]
    spectral = (2/T) * sum(freqs * mags)   # per action dim, then averaged

i.e. a frequency-weighted average of the one-sided FFT magnitude. High-frequency
action content raises the value, so lower means a smoother control signal.

RANKING — min of relative improvement over the PAVE baseline
------------------------------------------------------------
The bar to beat is **PAVE**, not vanilla TD3. The goal of this campaign is a
method that matches (or beats) PAVE's reward/smoothness while being simpler and
more robust than PAVE's three-auxiliary-loss recipe. So the ranking anchor is
PAVE's own measured (reward, smoothness) on this exact eval protocol.

A `reference` dict {"reward0": float, "smooth0": float} carries PAVE-TD3's mean
return and mean smoothness (measured on the task's validation seeds with this
same metric; a public, reproducible constant supplied by the scorer). Each
metric is turned into a percent improvement over PAVE:

    reward_gain%  = (reward   - reward0) / |reward0| * 100     # higher return
    smooth_gain%  = (smooth0  - smooth)  / smooth0    * 100     # lower smoothness

    score = min(reward_gain%, smooth_gain%)

Taking the MIN means a node scores as well as its WEAKER axis: you cannot buy
reward by sacrificing smoothness (or vice-versa). Matching PAVE on both scores
~0 (a simpler method that merely ties PAVE is already a win on the research
goal); beating PAVE on both raises the floor above 0. This is what "always
judged by comparative advantage over PAVE" means, made robust — no hand-tuned
trade-off weight.
"""
import numpy as np

# Fallback baseline used only if the scorer passes no reference (e.g. a smoke
# run). The real, tamper-proof PAVE baseline is held out and passed in by the
# scorer. These placeholders are the paper's Pendulum PAVE-TD3 range; the
# measured values are supplied by the scorer at grading time.
_DEFAULT_REWARD0 = -156.2
_DEFAULT_SMOOTH0 = 0.341


def spectral_smoothness(actions):
    """FFT frequency-weighted magnitude of an action sequence (PAVE definition).

    Args:
        actions: array-like of shape (T,) or (T, act_dim).

    Returns:
        float: spectral smoothness (>= 0; lower is smoother). Sequences shorter
        than 2 steps return 0.0.
    """
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


def _episodes(output):
    """Extract the episode list from a results.json dict; [] if malformed."""
    if not isinstance(output, dict):
        return []
    eps = output.get("episodes", [])
    return eps if isinstance(eps, list) else []


def reward(output):
    """Mean raw episode return across evaluation episodes."""
    eps = _episodes(output)
    returns = [float(e["return"]) for e in eps if isinstance(e, dict) and "return" in e]
    if not returns:
        return float("nan")
    return float(np.mean(returns))


def smoothness(output):
    """Mean spectral smoothness across evaluation episodes (lower is better)."""
    eps = _episodes(output)
    vals = [
        spectral_smoothness(e["actions"])
        for e in eps
        if isinstance(e, dict) and "actions" in e
    ]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _baseline(reference):
    """Read (reward0, smooth0) out of the baseline dict, with safe fallbacks."""
    if isinstance(reference, dict):
        r0 = float(reference.get("reward0", _DEFAULT_REWARD0))
        s0 = float(reference.get("smooth0", _DEFAULT_SMOOTH0))
    else:
        r0, s0 = _DEFAULT_REWARD0, _DEFAULT_SMOOTH0
    return r0, s0


def reward_gain(output, reference=None):
    """Percent improvement in mean return over the baseline (higher is better)."""
    r = reward(output)
    r0, _ = _baseline(reference)
    if not np.isfinite(r) or r0 == 0:
        return float("nan")
    return (r - r0) / abs(r0) * 100.0


def smoothness_gain(output, reference=None):
    """Percent reduction in spectral smoothness over the baseline (higher better)."""
    s = smoothness(output)
    _, s0 = _baseline(reference)
    if not np.isfinite(s) or s0 == 0:
        return float("nan")
    return (s0 - s) / s0 * 100.0


def score(output, reference=None):
    """Ranking scalar: min of the two relative improvements (co-equal, higher better).

    A node scores as well as its weaker axis, so reward and smoothness stay
    co-equal — neither can be traded for the other. If either metric is missing
    (nan), the score is nan (a [BUGGY] node).
    """
    rg = reward_gain(output, reference)
    sg = smoothness_gain(output, reference)
    if not (np.isfinite(rg) and np.isfinite(sg)):
        return float("nan")
    return float(min(rg, sg))


def metrics(output, reference=None):
    """Raw metrics + relative gains, for logging."""
    return {
        "reward": reward(output),
        "smoothness": smoothness(output),
        "reward_gain_pct": reward_gain(output, reference),
        "smoothness_gain_pct": smoothness_gain(output, reference),
    }


def diagnostics(output, reference=None):
    """Human-readable per-metric breakdown string."""
    eps = _episodes(output)
    r = reward(output)
    s = smoothness(output)
    r0, s0 = _baseline(reference)
    rg = reward_gain(output, reference)
    sg = smoothness_gain(output, reference)
    lines = [
        f"  episodes:      {len(eps)}",
        f"  baseline:      reward0={r0:.3f}  smooth0={s0:.6f}",
        f"  reward:        {r:.3f}   -> {rg:+.2f}%  (mean return, higher better)",
        f"  smoothness:    {s:.6f} -> {sg:+.2f}%  (spectral, lower better)",
        f"  score = min(gains): {min(rg, sg):+.2f}%" if np.isfinite(rg) and np.isfinite(sg) else "  score = nan",
    ]
    return "\n".join(lines)
