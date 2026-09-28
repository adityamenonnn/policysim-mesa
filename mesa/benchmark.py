"""
Benchmark + reproducibility check for the LabourModel prototype.

Run:
    python benchmark.py
"""

import time
import numpy as np
from labour_model import LabourModel, OUTCOMES


# ── Reproducibility ──────────────────────────────────────────────────────────

def check_reproducibility(n_agents: int = 5_000, n_ticks: int = 5,
                           seed: int = 42) -> bool:
    """
    Run the same model twice with the same seed; verify identical outcomes.
    Uses numpy.random.default_rng — the only source of randomness in the model.
    """
    def run(s):
        m = LabourModel(n_agents=n_agents,
                        rng=np.random.default_rng(s))
        for _ in range(n_ticks):
            m.step()
        return m._outcomes.copy()

    out_a = run(seed)
    out_b = run(seed)
    return np.array_equal(out_a, out_b)


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(n_agents: int, n_ticks: int = 10, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    t0    = time.perf_counter()
    model = LabourModel(n_agents=n_agents, rng=rng)
    t_init = time.perf_counter() - t0

    tick_times = []
    for _ in range(n_ticks):
        t1 = time.perf_counter()
        model.step()
        tick_times.append(time.perf_counter() - t1)

    total = time.perf_counter() - t0
    return {
        "n_agents"       : n_agents,
        "n_ticks"        : n_ticks,
        "init_s"         : round(t_init, 3),
        "mean_tick_ms"   : round(float(np.mean(tick_times)) * 1000, 2),
        "p95_tick_ms"    : round(float(np.percentile(tick_times, 95)) * 1000, 2),
        "total_s"        : round(total, 3),
        "distribution"   : model.outcome_distribution(),
        "feedback_check" : {               # shows feedback is live
            "peer_retrain_lag" : round(model.peer_retrain_lag, 4),
            "sector_sat_lag"   : round(model.sector_sat_lag,   4),
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Reproducibility check")
    print("=" * 60)
    ok = check_reproducibility()
    print(f"  5 000 agents × 5 ticks, seed=42 run twice: "
          f"{'IDENTICAL' if ok else 'DIFFER — BUG!'}\n")

    print("=" * 60)
    print("Benchmarks (10 ticks each)")
    print("=" * 60)
    for n in [10_000, 100_000]:
        r = benchmark(n_agents=n)
        print(f"\n  {n:>7,} agents")
        print(f"    Initialisation  : {r['init_s']} s")
        print(f"    Mean tick       : {r['mean_tick_ms']} ms")
        print(f"    p95  tick       : {r['p95_tick_ms']} ms")
        print(f"    Total (init+run): {r['total_s']} s")
        print(f"    Distribution    : {r['distribution']}")
        print(f"    Feedback (lag)  : {r['feedback_check']}")

    print()
