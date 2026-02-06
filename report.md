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



---

## Next Steps

