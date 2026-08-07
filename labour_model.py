"""
PolicySim — Mesa Labour-Market Displacement Prototype
======================================================
Multinomial logit ABM: workers choose among four outcomes each tick.
Feedback: previous-tick peer-retraining rate enters the current choice.

Vectorisation strategy
----------------------
Mesa's per-agent step() is bypassed for the heavy computation.
The choice step (matrix multiply + softmax + draw) runs as a single
NumPy operation over the entire population.  Individual agents are
thin state containers; results are written back after the batch step.

This is a deliberate architectural choice — and directly relevant to
the "does Mesa fit our design?" question the assessment asks.
"""

import numpy as np
import mesa
from collections import Counter


# ── Outcome labels ──────────────────────────────────────────────────────────
OUTCOMES = ["seek_similar", "retrain", "underemployed", "exit"]
N_OUTCOMES = len(OUTCOMES)      # 4
N_FEATURES = 8                  # see feature vector below


# ── Multinomial logit coefficients (invented) ───────────────────────────────
# Rows = outcomes (0 = reference category, coefficients all zero)
# Cols = [intercept, age_norm, skill, is_services, is_tech,
#         policy_lever, peer_retrain_lag, sector_sat_lag]
#
# Interpretation (rough):
#   Higher skill  → more likely to retrain, less likely to exit
#   Older workers → more likely to exit, less likely to retrain
#   Strong policy → pulls workers toward retraining
#   High peer retraining rate → social contagion pushes more toward retrain
BETAS = np.array([
    # intc   age   skill  svc   tech  policy peer_r  sec_sat
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # seek_similar (ref)
    [ 0.50, -0.30,  0.80,  0.20,  0.40,  1.20,  0.60, -0.50],  # retrain
    [-0.50,  0.20, -0.60,  0.10, -0.30, -0.40, -0.20,  0.80],  # underemployed
    [-1.00,  0.50, -1.00, -0.20, -0.50, -0.80, -0.30,  0.30],  # exit
], dtype=np.float64)            # shape: (4, 8)


# ── Agent ────────────────────────────────────────────────────────────────────

class WorkerAgent(mesa.Agent):
    """
    A displaced worker.

    Attributes are set by the model at creation; outcome is updated
    by the model each tick after the batch choice step.
    """
    __slots__ = ("age_norm", "skill", "sector", "outcome")

    def __init__(self, model: "LabourModel",
                 age_norm: float, skill: float, sector: int):
        super().__init__(model)
        self.age_norm = float(age_norm)   # (age − 40) / 15
        self.skill    = float(skill)      # 0–1  (Beta(2,2))
        self.sector   = int(sector)       # 0=manufacturing 1=services 2=tech
        self.outcome  = 0                 # starts as seek_similar

    # step() is intentionally empty — the model drives the choice step.
    def step(self):
        pass


# ── Model ────────────────────────────────────────────────────────────────────

class LabourModel(mesa.Model):
    """
    Labour-market displacement model.

    Parameters
    ----------
    n_agents     : population size (try 10_000 or 100_000)
    policy_lever : retraining-subsidy strength, 0–1
    rng          : numpy Generator (controls all randomness)
    """

    def __init__(self,
                 n_agents: int = 10_000,
                 policy_lever: float = 0.30,
                 rng: np.random.Generator | None = None):

        # Mesa 3.x: pass rng directly; seed kwarg is deprecated
        super().__init__(rng=rng)

        self.policy_lever = policy_lever
        self.n_agents     = n_agents

        # ── lagged feedback aggregates ──────────────────────────────────────
        # These are the "social terms" — previous-tick population shares
        # that feed back into each agent's choice index next tick.
        self.peer_retrain_lag : float = 0.0   # share retraining last tick
        self.sector_sat_lag   : float = 0.0   # share in seek_similar last tick

        # ── sample static agent attributes ─────────────────────────────────
        age_norms = self.rng.normal(0.0, 1.0, n_agents)          # ~N(40,15) normalised
        skills    = self.rng.beta(2.0, 2.0, n_agents)            # peaked around 0.5
        sectors   = self.rng.integers(0, 3, n_agents)            # 0/1/2

        # ── create agents ───────────────────────────────────────────────────
        for i in range(n_agents):
            WorkerAgent(self, age_norms[i], skills[i], sectors[i])

        # ── pre-build static columns of the feature matrix ─────────────────
        # Dynamic columns (policy, feedback) are updated each tick.
        self._X = np.zeros((n_agents, N_FEATURES), dtype=np.float64)
        self._X[:, 0] = 1.0                           # intercept
        self._X[:, 1] = age_norms                     # age
        self._X[:, 2] = skills                        # skill
        self._X[:, 3] = (sectors == 1).astype(float)  # is_services
        self._X[:, 4] = (sectors == 2).astype(float)  # is_tech

        # ── outcome array (mirrors agent.outcome for fast access) ───────────
        self._outcomes = np.zeros(n_agents, dtype=np.int32)

        self.tick_num = 0

    # ── core numerical step ─────────────────────────────────────────────────

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        """
        Row-wise numerically stable softmax.
        Z : (N, K) → returns (N, K) probability matrix.
        """
        Z = Z - Z.max(axis=1, keepdims=True)   # subtract row max for stability
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def step(self):
        """
        One simulation tick:
        1. Fill dynamic feature columns.
        2. Compute Z = X @ BETAS.T  (vectorised over all agents).
        3. Softmax Z → probability matrix.
        4. Draw one outcome per agent.
        5. Write outcomes back to agents.
        6. Update lagged feedback aggregates for next tick.
        """
        n = self.n_agents

        # 1. Dynamic columns
        self._X[:, 5] = self.policy_lever       # same for all agents this tick
        self._X[:, 6] = self.peer_retrain_lag   # broadcast scalar → all rows
        self._X[:, 7] = self.sector_sat_lag

        # 2. Linear index  Z[i, k] = X[i] · beta[k]
        Z = self._X @ BETAS.T                   # (N, 4)

        # 3. Probabilities
        probs = self._softmax(Z)                # (N, 4)

        # 4. Draw outcomes — inverse-CDF trick; single RNG call
        cumprobs = probs.cumsum(axis=1)         # (N, 4)
        u        = self.rng.random(n)           # (N,)
        self._outcomes = (u[:, None] > cumprobs).sum(axis=1).astype(np.int32)

        # 5. Write back to Mesa agents (preserves Mesa compatibility)
        for agent, outcome in zip(self.agents, self._outcomes):
            agent.outcome = int(outcome)

        # 6. Update feedback for next tick
        self.peer_retrain_lag = float((self._outcomes == 1).mean())
        self.sector_sat_lag   = float((self._outcomes == 0).mean())

        self.tick_num += 1

    # ── convenience ─────────────────────────────────────────────────────────

    def outcome_distribution(self) -> dict[str, float]:
        """Fractional distribution over outcomes after the last tick."""
        counts = Counter(int(o) for o in self._outcomes)
        total  = self.n_agents
        return {OUTCOMES[k]: round(counts.get(k, 0) / total, 4)
                for k in range(N_OUTCOMES)}
