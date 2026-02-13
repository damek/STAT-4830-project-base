# Project Report: Differentiable Physics vs. Convex Relaxation for Battery Arbitrage

## Problem Statement

### What are you optimizing?
We are rigorously benchmarking **Convex Relaxation (QP)** against **Non-Convex Differentiable Physics (PyTorch)** for battery energy arbitrage. The goal is to maximize financial returns while minimizing battery degradation.

### Why does this problem matter?
The core challenge is to determine if the "Optimality Gap" caused by simplifying physics (to achieve convexity) is greater than the "Convergence Error" introduced by the instability of gradient descent in a constrained, non-convex landscape.

### How will you measure success?
Success is defined by the model's ability to "learn" to buy at troughs and sell at peaks purely via gradient descent. We will measure success by comparing the objective value (profit) and constraint violation rates against the QP baseline.

### Constraints & Data
* **Constraints:** The system must strictly respect $0 < SoC < 100$ limits[cite: 14].
* **Data:** The optimization covers an 8,760-step time series (one year), creating an extremely deep computational graph[cite: 9].

### What could go wrong?
* **Constraint Satisfaction:** Unlike CVXPY, PyTorch does not natively handle hard constraints[cite: 7].
* **Vanishing Gradients:** The deep computational graph (8,760 steps) risks vanishing gradients, making convergence difficult[cite: 9].

---

## Technical Approach

### Mathematical Formulation
[cite_start]We maximize an objective function $J$ composed of Arbitrage Profit minus a Degradation Penalty[cite: 17, 19, 22]:

$$
\text{Maximize } J = \sum_{t=1}^{T} \left( P_t (d_t - c_t) - \lambda (c_t + d_t)^2 \right)
$$

Where:
* $P_t$: Price at time $t$
* $d_t, c_t$: Discharge and Charge amounts
* $\lambda$: Degradation penalty coefficient. [cite_start]We derive this such that the quadratic penalty roughly equals the average cost per MWh ($k$) at nominal power ($P_{nom}$): $\lambda \approx \frac{k}{P_{nom}}$[cite: 27, 29].

### Algorithm & Implementation
* **Architecture:** We are implementing a custom `DifferentiableBattery` class in PyTorch to simulate charge/discharge dynamics[cite: 12].
* **Optimization:** We use the Adam optimizer to update control variables[cite: 13].
* **Constraint Strategy:** To handle the lack of native constraints, we are implementing stable **Projected Gradient Descent** or **Lagrangian Relaxations** to enforce limits without causing gradient explosion[cite: 8].

### Validation Methods
We utilize a "Proof of Life" experiment: training the model on a synthetic "Sine Wave" price curve to verify gradient propagation before introducing real-world data noise.

---

## Initial Results

### Evidence of a Working Implementation
We ran our differentiable battery on **live NYISO real-time 5-minute LMP data** for the N.Y.C. zone (259 intervals, average price $214.58/MWh). The optimizer was configured with Adam (lr=0.1), 1,000 epochs, and a soft SoC penalty coefficient of 2,000. Key observations:

* **Constraint Satisfaction:** The soft-penalty approach achieved **zero SoC violations** across all 1,000 epochs, confirming that the penalty coefficient is large enough to keep the battery within its $[0, 1]$ MWh capacity bounds.
* **Profit Trajectory:** The model showed clear learning, rising from \$0 at epoch 0 to a peak of roughly \$132 by epoch 700. However, convergence was **highly oscillatory** — profit dropped to -\$23 at epoch 400, recovered to \$115, dipped to \$24 at epoch 600, and ended at roughly \$125 at epoch 900. This instability is a central finding: gradient descent on this non-convex landscape is noisy, confirming the "Convergence Error" concern from our problem statement.

### Validation Suite Results
We designed three synthetic test cases to verify qualitative behavior:

| Test Case | Expected Behavior | Profit | Max SoC Violation | Runtime |
|---|---|---|---|---|
| Flat Prices ($50/MWh) | Idle (action ≈ 0) | $9.30 | 0.0000 | 0.08 s |
| Negative Prices (-$5 to -$10) | Charge (action > 0) | $3.02 | 0.0000 | 0.07 s |
| Single Price Spike ($500 at t=50) | Discharge at spike | $32.16 | 0.0000 | 0.08 s |

All three tests pass the "eye test": the model charges during negative prices, discharges into the spike, and remains mostly idle when prices are flat (the small \$9.30 profit in the flat case suggests minor residual cycling, which is expected given the lack of a degradation penalty in the validation loop).

### Current Limitations
* **Convergence Instability:** The profit oscillation during training suggests that lr=0.1 may be too aggressive, or that the penalty-based constraint strategy creates a rugged loss landscape. A learning-rate schedule or Lagrangian relaxation may help.
* **No QP Baseline Yet:** We cannot quantify the "Optimality Gap" until the CVXPY benchmark is implemented.
* **Short Horizon:** The current experiment uses a single day of 5-minute data (259 steps), far short of the planned 8,760-step annual horizon. Scaling up may exacerbate vanishing-gradient issues.
* **Revenue Sign Convention:** The main optimization loop and the validation loop use slightly different revenue formulations (one uses `−control × price`, the other explicitly separates charge cost and discharge revenue). These should be unified to ensure consistency.

### Resource Usage
Each validation case ran in under 0.1 seconds and used less than 0.02 MB of peak memory on a 100-step horizon. The full 259-step optimization (1,000 epochs) completed in seconds on a CPU, indicating that compute is not a bottleneck at this scale.

---

## Next Steps

### Immediate Improvements
* **Stabilize Convergence:** Implement a learning-rate scheduler (e.g., cosine annealing or reduce-on-plateau) and experiment with lower initial learning rates. Evaluate whether Lagrangian relaxation produces smoother convergence than the current quadratic penalty.
* **Unify Revenue Formulation:** Reconcile the revenue calculation between the main training loop and the validation runner so that reported profits are directly comparable.
* **Add the Degradation Penalty:** The current objective omits the $\lambda(c_t + d_t)^2$ degradation term described in the formulation. Adding it will reduce unnecessary cycling and better reflect real-world battery economics.

### Technical Challenges to Address
* **Implement the QP Baseline (CVXPY):** This is critical for the core research question. Without the convex benchmark, we cannot measure the optimality gap.
* **Scale to a Full Year:** Move from a single day (259 steps) to a full year (8,760 hourly steps). Monitor for vanishing gradients and memory growth, and consider chunked / truncated backpropagation through time if needed.
* **Projected Gradient Descent:** Explore hard projection of SoC after each optimizer step (clamp SoC to $[0, SoC_{max}]$) as an alternative to the soft penalty, and compare feasibility and profit.

### Questions for Course Staff
* Is there a recommended approach for benchmarking non-convex PyTorch solutions against CVXPY on problems of this size (8,760 variables)?
* Are there best practices for balancing penalty coefficients vs. Lagrangian multiplier updates in differentiable physics settings?

### Alternative Approaches to Try
* **Augmented Lagrangian Method:** Replace the fixed penalty with adaptive dual-variable updates for tighter constraint satisfaction.
* **Neural Network Policy:** Instead of directly optimizing per-step actions, train a small network that maps price features to actions, enabling generalization to unseen price trajectories.

### What We've Learned So Far
Gradient descent *can* learn basic arbitrage behavior (buy low, sell high) from price signals alone, and the `tanh` parameterization effectively enforces power limits without projection. However, the convergence path is far noisier than expected, reinforcing the value of the planned comparison with a convex solver.
