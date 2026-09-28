# PolicySim Mesa: Labour-Market Displacement ABM

An agent-based model simulating labour-market outcomes for displaced workers, built on the [Mesa](https://mesa.readthedocs.io/) framework. Workers choose among four outcomes each tick using a multinomial logit choice model, with social feedback loops where peer behaviour influences future decisions.

## How It Works

### Multinomial Logit Choice Model

Each displaced worker chooses one of four outcomes per simulation tick:

| Outcome | Description |
|---------|-------------|
| **Seek similar** | Look for a comparable job in the same sector (reference category) |
| **Retrain** | Invest in new skills to switch sectors |
| **Underemployed** | Accept a lower-quality job |
| **Exit** | Leave the labour market entirely |

Choice probabilities are computed via a multinomial logit over an 8-dimensional feature vector per agent:

```
[intercept, age_norm, skill, is_services, is_tech, policy_lever, peer_retrain_lag, sector_sat_lag]
```

Key dynamics:
- Higher-skilled workers are more likely to retrain and less likely to exit
- Older workers lean toward exit, younger workers toward retraining
- A stronger policy lever (retraining subsidy) pushes workers toward retraining
- **Feedback loop**: the previous tick's peer retraining rate enters the current choice, creating social contagion where retraining begets more retraining

### Vectorised Computation

Mesa's per-agent `step()` is deliberately bypassed for the heavy computation. The entire choice step (matrix multiply, softmax, random draw) runs as a single NumPy operation over the full population. Individual agents are thin state containers; results are written back after the batch step. This achieves near-linear O(N) scaling and handles 100K+ agents efficiently.

## Figures

The `mesa/visualise.py` script generates four figures:

| Figure | What it shows |
|--------|--------------|
| `fig1_scaling.png` | Tick time vs population size (near-linear scaling) |
| `fig2_feedback_loop.png` | Outcome distribution evolving over 25 ticks (feedback loop in action) |
| `fig3_policy_sweep.png` | Retraining rate vs policy lever strength (model sensitivity) |
| `fig4_writeback_cost.png` | Mesa write-back overhead vs pure NumPy at scale |

## Project Structure

```
mesa/                        -- ABM model and assessment
  labour_model.py            -- Core ABM: WorkerAgent, LabourModel, multinomial logit
  benchmark.py               -- Reproducibility check and performance benchmarks
  visualise.py               -- Generates all four assessment figures
  assessment.tex             -- Written assessment document
  fig*.png                   -- Generated figures

research/                    -- Supporting data and processing scripts
  ons_data/                  -- ONS labour market datasets
  redcar/                    -- Redcar SSI case study panel and pipeline
  data.py                    -- Data loading utilities
  displacement_evidence.csv  -- Extracted displacement evidence

extract_evidence.py          -- Extracts displacement evidence from literature
displacement_evidence.csv    -- Evidence dataset
```

## Running

```bash
# Install dependencies
pip install mesa numpy matplotlib

# Run the benchmark (reproducibility + timing)
cd mesa
python benchmark.py

# Generate all figures
python visualise.py
```

## Tech Stack

- **Python 3.10+**
- **Mesa 3.x** for the ABM framework
- **NumPy** for vectorised computation
- **Matplotlib** for visualisation
