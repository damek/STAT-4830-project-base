# Week 4 Self-Critique

## OBSERVE

After re-reading the report and re-running the notebook end-to-end, the core pipeline works: data ingestion, differentiable battery simulation, gradient-based optimization, and a three-case validation suite all execute without errors. The model does learn to buy low and sell high on real NYISO price data. However, the training curve is unstable (profit oscillates wildly across epochs), the report still lacks a convex baseline for comparison, and several implementation details diverge from the mathematical formulation described in the report.

---

## ORIENT

### Strengths (Max 3)
- **Working end-to-end pipeline.** The `DifferentiableBattery` model ingests live NYISO data, optimizes via Adam, and produces interpretable charge/discharge schedules with zero constraint violations.
- **Clear mathematical formulation.** The report lays out the objective, dynamics, and constraints precisely enough to reproduce the approach.
- **Thoughtful validation design.** The three synthetic test cases (flat, negative, spike) directly probe whether the optimizer exhibits the correct qualitative behavior, and all three pass.

### Areas for Improvement (Max 3)
1. **Convergence instability.** Profit swings from +$132 to -$23 across epochs. The learning rate (0.1) is likely too high, and the fixed penalty coefficient creates a non-smooth loss landscape. Without stable convergence, the reported profit is not trustworthy.
2. **Missing QP baseline.** The entire project thesis depends on comparing differentiable physics against convex relaxation, but the CVXPY implementation does not yet exist. Without it, the report cannot answer its own research question.
3. **Formulation–code mismatch.** The report describes a degradation penalty $\lambda(c_t + d_t)^2$, but the notebook objective does not include it. Additionally, the revenue calculation differs between the main loop and the validation runner. These inconsistencies undermine reproducibility.

### Critical Risks / Assumptions
We are currently assuming that the soft-penalty approach will scale to 8,760 steps without gradient issues. The current experiment only covers 259 steps (one day at 5-min resolution). Scaling to a full year may expose vanishing gradients through the cumulative-sum dynamics, which would require architectural changes (e.g., truncated BPTT or a recurrent formulation). We also assume that a single penalty coefficient will work across different price regimes; this has not been tested.

---

## DECIDE

### Concrete Next Actions (Max 3)
1. **Implement the CVXPY baseline** on the same NYISO data to produce a convex-optimal profit number and SoC trajectory. This directly addresses Area #2 and is the highest priority.
2. **Add a learning-rate scheduler and the degradation penalty** to the PyTorch loop, unify the revenue formula across training and validation, and re-run to check whether convergence stabilizes. This addresses Areas #1 and #3.
3. **Scale the experiment to one week of hourly data** (~168 steps) as an intermediate step toward the full-year horizon, and log gradient norms to detect early signs of vanishing gradients.

---

## ACT

### Resource Needs
- Need to learn CVXPY's QP interface for time-series problems with coupling constraints (SoC dynamics). The [CVXPY tutorial on portfolio optimization](https://www.cvxpy.org/examples/index.html) is the closest analog and will be the starting reference.
- Will use `torch.optim.lr_scheduler.CosineAnnealingLR` for the learning-rate schedule; no new dependencies required.
- May need `gridstatus` historical data access for multi-day experiments; need to verify that the free tier supports date-range queries.
