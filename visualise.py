"""
Visualisations for the Mesa assessment.

Produces four figures:
  fig1_scaling.png         — tick time vs population size
  fig2_feedback_loop.png   — outcome distribution evolving over ticks
  fig3_policy_sweep.png    — retraining rate vs policy lever strength
  fig4_writeback_cost.png  — Mesa write-back overhead vs pure NumPy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from labour_model import LabourModel, OUTCOMES

plt.rcParams.update({
    "font.family"  : "sans-serif",
    "font.size"    : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"   : 150,
})

SEED = 42
OUT  = os.path.dirname(__file__)

COLORS = {
    "seek_similar" : "#4C72B0",
    "retrain"      : "#55A868",
    "underemployed": "#C44E52",
    "exit"         : "#8172B2",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def mean_tick_ms(n_agents, n_ticks=15, seed=SEED):
    rng   = np.random.default_rng(seed)
    model = LabourModel(n_agents=n_agents, rng=rng)
    times = []
    for _ in range(n_ticks):
        t = time.perf_counter()
        model.step()
        times.append(time.perf_counter() - t)
    return np.mean(times) * 1000   # ms


def run_ticks(n_agents, n_ticks, policy_lever=0.30, seed=SEED):
    """Return per-tick outcome fractions as dict of lists."""
    rng   = np.random.default_rng(seed)
    model = LabourModel(n_agents=n_agents, policy_lever=policy_lever, rng=rng)
    history = {o: [] for o in OUTCOMES}
    for _ in range(n_ticks):
        model.step()
        dist = model.outcome_distribution()
        for o in OUTCOMES:
            history[o].append(dist[o])
    return history


def mean_tick_no_writeback_ms(n_agents, n_ticks=15, seed=SEED):
    """Tick time if we skip writing outcomes back to Mesa agent objects."""
    rng   = np.random.default_rng(seed)
    model = LabourModel(n_agents=n_agents, rng=rng)
    times = []
    for _ in range(n_ticks):
        t = time.perf_counter()
        # replicate step() but skip the write-back loop
        n = model.n_agents
        model._X[:, 5] = model.policy_lever
        model._X[:, 6] = model.peer_retrain_lag
        model._X[:, 7] = model.sector_sat_lag
        Z = model._X @ __import__('labour_model').BETAS.T
        # softmax
        Z2 = Z - Z.max(axis=1, keepdims=True)
        E  = np.exp(Z2)
        probs = E / E.sum(axis=1, keepdims=True)
        cumprobs = probs.cumsum(axis=1)
        u = model.rng.random(n)
        model._outcomes = (u[:, None] > cumprobs).sum(axis=1).astype(np.int32)
        model.peer_retrain_lag = float((model._outcomes == 1).mean())
        model.sector_sat_lag   = float((model._outcomes == 0).mean())
        times.append(time.perf_counter() - t)
    return np.mean(times) * 1000


# ── Figure 1 — Scaling ───────────────────────────────────────────────────────

def fig_scaling():
    print("Running scaling benchmark...")
    sizes = [1_000, 5_000, 10_000, 30_000, 50_000, 100_000, 300_000]
    times = [mean_tick_ms(n) for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([n/1000 for n in sizes], times,
            marker="o", color="#4C72B0", linewidth=2, markersize=6)

    # linear reference line
    scale = times[0] / (sizes[0] / 1000)
    ref   = [scale * (n/1000) for n in sizes]
    ax.plot([n/1000 for n in sizes], ref,
            linestyle="--", color="#aaa", linewidth=1.2, label="linear O(N) reference")

    ax.set_xlabel("Population size (thousands)")
    ax.set_ylabel("Mean tick time (ms)")
    ax.set_title("Fig 1 — Tick time scales near-linearly with population")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}K"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig1_scaling.png"))
    plt.close()
    print("  Saved fig1_scaling.png")


# ── Figure 2 — Feedback loop ─────────────────────────────────────────────────

def fig_feedback_loop():
    print("Running feedback loop simulation...")
    history = run_ticks(n_agents=50_000, n_ticks=25)
    ticks   = range(1, 26)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for outcome in OUTCOMES:
        ax.plot(ticks, [v * 100 for v in history[outcome]],
                label=outcome.replace("_", " "),
                color=COLORS[outcome], linewidth=2)

    ax.set_xlabel("Tick")
    ax.set_ylabel("Share of population (%)")
    ax.set_title("Fig 2 — Feedback loop: peer retraining rate reinforces itself over time\n"
                 "(50K agents, policy lever = 0.30)")
    ax.legend(frameon=False, loc="right")
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_feedback_loop.png"))
    plt.close()
    print("  Saved fig2_feedback_loop.png")


# ── Figure 3 — Policy lever sweep ────────────────────────────────────────────

def fig_policy_sweep():
    print("Running policy lever sweep...")
    levers     = np.linspace(0.0, 1.0, 20)
    retrain_at = []

    for lev in levers:
        rng   = np.random.default_rng(SEED)
        model = LabourModel(n_agents=20_000, policy_lever=float(lev), rng=rng)
        for _ in range(10):
            model.step()
        retrain_at.append(model.peer_retrain_lag * 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(levers, retrain_at, color="#55A868", linewidth=2.5, marker="o", markersize=4)
    ax.set_xlabel("Policy lever (0 = no subsidy, 1 = full subsidy)")
    ax.set_ylabel("% population retraining after 10 ticks")
    ax.set_title("Fig 3 — Model is sensitive to the policy lever\n"
                 "(20K agents, 10 ticks, seed fixed)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig3_policy_sweep.png"))
    plt.close()
    print("  Saved fig3_policy_sweep.png")


# ── Figure 4 — Write-back overhead ───────────────────────────────────────────

def fig_writeback_cost():
    print("Running write-back overhead comparison...")
    sizes      = [10_000, 50_000, 100_000, 300_000]
    with_wb    = [mean_tick_ms(n) for n in sizes]
    without_wb = [mean_tick_no_writeback_ms(n) for n in sizes]

    x     = np.arange(len(sizes))
    width = 0.35
    labels = [f"{n//1000}K" for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, with_wb,    width, label="With Mesa write-back",    color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, without_wb, width, label="Pure NumPy (no write-back)", color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Population size")
    ax.set_ylabel("Mean tick time (ms)")
    ax.set_title("Fig 4 — Mesa write-back loop vs pure NumPy\n"
                 "The gap widens at scale")
    ax.legend(frameon=False)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig4_writeback_cost.png"))
    plt.close()
    print("  Saved fig4_writeback_cost.png")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_scaling()
    fig_feedback_loop()
    fig_policy_sweep()
    fig_writeback_cost()
    print("\nAll figures saved.")
