# Development Log

## Week 4 — Initial Implementation (Jan 27 – Feb 6, 2026)

### Goals
- Define the optimization problem and mathematical formulation.
- Build a working differentiable battery model in PyTorch.
- Run initial experiments on real electricity price data.
- Set up a validation suite to sanity-check optimizer behavior.

### Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Optimization framework | PyTorch (Adam) | Enables differentiable physics simulation with autograd; Adam handles sparse gradients well. |
| Constraint handling | Soft quadratic penalty on SoC bounds | Simpler to implement than Lagrangian relaxation; penalty coefficient (2000) chosen empirically to eliminate violations. |
| Power-limit enforcement | `tanh` parameterization | Squashes raw actions to $[-1, 1]$ and scales by max power, guaranteeing power limits without projection. |
| Data source | NYISO real-time 5-min LMP (N.Y.C. zone) via `gridstatus` | Freely available, high-resolution, and representative of volatile urban pricing. |
| Battery parameters | 1 MWh / 1 MW / 90% round-trip efficiency | Standard utility-scale BESS assumptions for a first pass. |

### Progress

1. **Problem formulation (Days 1–2):**
   - Wrote out the objective function, SoC dynamics, and constraints in LaTeX.
   - Identified the core research question: Optimality Gap (convex relaxation) vs. Convergence Error (gradient descent).
   - Documented the formulation in the notebook's Problem Setup section and in `report.md`.

2. **Implementation (Days 2–3):**
   - Built the `DifferentiableBattery` class (`nn.Module`) with learnable per-step actions.
   - Implemented the forward pass: `tanh` → charge/discharge split → efficiency-adjusted energy delta → `cumsum` for SoC.
   - Set up the training loop with Adam, revenue calculation, and soft SoC penalty.
   - Fetched live NYISO data via `gridstatus` and ran the first optimization (259 intervals, 1000 epochs).

3. **Validation (Days 3–4):**
   - Designed three synthetic test cases: flat prices, negative prices, and a single price spike.
   - Wrapped the validation runner in a `measure_resources` decorator to track execution time and memory.
   - All three tests produce qualitatively correct behavior with zero constraint violations.

4. **Report & documentation (Days 4–5):**
   - Completed Problem Statement and Technical Approach sections of `report.md`.
   - Wrote Initial Results and Next Steps sections based on observed training behavior.
   - Created self-critique following the OODA framework.

### Challenges & Failed Attempts

- **Learning rate sensitivity:** Initial experiments with lr=0.01 converged very slowly (profit barely moved after 1000 epochs). Switching to lr=0.1 accelerated learning but introduced large oscillations. A scheduler is needed.
- **Revenue sign convention confusion:** The main training loop uses `−control × price` while the validation loop explicitly separates charge cost and discharge revenue. This led to initially inconsistent profit numbers. Needs to be unified.
- **Penalty coefficient tuning:** Tried penalty_coeff=100 first; SoC violations appeared. Increased to 2000, which eliminated violations but may be contributing to the rugged loss landscape.

### Open Questions
- Will `torch.cumsum` propagate gradients cleanly over 8,760 steps, or will we hit vanishing gradients?
- Is there a principled way to set the penalty coefficient, or should we switch to an augmented Lagrangian scheme?
- How do we fairly compare wall-clock time between CVXPY (which uses compiled solvers) and PyTorch (which uses Python-level autograd)?

---

*Next entry: Week 5 — QP baseline implementation and convergence stabilization.*
