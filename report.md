# Week 4 Report — Learning a Prompt-Level “Slop” Predictor

## 1) Problem Statement

### What are we optimizing? (Be specific)
We optimize parameters **θ** of a prompt-only predictor $ f_\theta(\text{prompt}) $ that estimates a scalar slop score computed from the observed response paired with that prompt in the dataset (not from a newly generated response). Concretely, we define a response-level slop metric $ s(\text{response}) $ using surface statistics (repetition, diversity, entropy/compressibility proxies, etc.), and then train $ f_\theta $ to **minimize prediction error** of that slop score from **prompt text alone**.

This is an optimization problem over model parameters:
- **Inputs:** prompts $x$
- **Targets:** computed slop scores $y = s(\text{response})$
- **Model:** $f_\theta(x)$
- **Goal:** learn $\theta$ to best predict $y$

### Why does this matter?
Low-quality outputs (“slop”) waste user time, harm trust, and increase moderation / support load. If we can **predict slop risk from a prompt**, we can:
- proactively **rewrite or constrain prompts**,
- route risky prompts to stronger models or additional checks,
- and ultimately support **optimization over prompts** (e.g., prompt search / RL) to reduce slop while preserving usefulness.

### How will we measure success?
We evaluate success at two levels:

1. **Metric validity (response-level):**
   - Does the slop score behave sensibly across examples?
   - Does it align (even weakly) with preference signals such as “chosen vs rejected” responses?

2. **Predictive performance (prompt → slop):**
   - Regression metrics: **MAE, RMSE, $R^2$**
   - Rank correlation: **Spearman ρ** (do we rank prompts by slop risk correctly?)

### Constraints
- **No generation-time optimization** in the initial deliverable: we do not alter a base LLM or run RL on a generator yet.
- **Compute limits:** features should be cheap and batchable; model should train on a laptop/Colab GPU quickly.
- **Interpretability:** features and the composite slop score should be explainable and debuggable.

### Data needed
We need prompt–response pairs with some quality signal. For the initial experiment we use **paired preference data** with:
- a prompt $x$,
- a preferred response $r^{+}$ (“chosen”),
- a less preferred response $r^{-}$ (“rejected”).

We use the Anthropic/hh-rlhf dataset (train split), which provides the (prompt, chosen, rejected) pairs.

We compute $s(r)$ on each response to study whether the metric tracks quality and to create training targets for $f_\theta$.

### What could go wrong?
- **Metric mis-specification:** surface heuristics might not capture what humans mean by “slop” (could confound with length, topic, style).
- **Shortcut learning:** the prompt model might learn dataset artifacts rather than genuine slop risk.
- **Weak supervision mismatch:** “chosen vs rejected” reflects many factors (helpfulness, safety, tone), not just slop.
- **Generalization:** a predictor trained on one dataset/model family may not transfer.


## 2) Technical Approach

### Mathematical formulation
We define a response-level slop function:
$$
y = s(r) \in \mathbb{R},
$$
constructed from standardized surface metrics (examples: n-gram repetition rate, distinct-n, entropy proxy, compression ratio).

We then train a prompt-only regressor $f_\theta(x)$ by minimizing mean squared error:
$$
\min_{\theta} \; \mathbb{E}_{(x,r)\sim \mathcal{D}} \left[ \left(f_\theta(x) - s(r)\right)^2 \right].
$$
In practice, we approximate the expectation with an empirical average over a sampled dataset:
$$
\min_{\theta} \; \frac{1}{N}\sum_{i=1}^{N}\left(f_\theta(x_i) - y_i\right)^2.
$$

#### Slop score definition
We compute response features:
- trigram repetition `ngram_repetition_3`
- distinct-2 `distinct_2`
- character entropy `char_entropy`
- compression ratio `compression_ratio`
- punctuation density `punct_density`
- caps ratio `caps_ratio`

We z-score each feature across the dataset and define:
$$
s(r)=
1.0\,z_{\text{rep3}}
-1.0\,z_{\text{distinct2}}
-0.7\,z_{\text{entropy}}
-0.7\,z_{\text{compression}}
+0.2\,z_{\text{punct}}
+0.2\,z_{\text{caps}}.
$$

#### Prompt representation
We featurize prompts using TF–IDF with a maximum vocabulary size of 12,000 features:
$$
x \mapsto X \in \mathbb{R}^{N\times d}, \quad d \le 12000.
$$
The TF–IDF matrix is used as input to PyTorch models (linear and MLP regressors).

### Algorithm / approach choice and justification
- **Slop metric:** heuristics-based composite score enables fast iteration and interpretability. It is a pragmatic proxy objective suitable for an initial deliverable.
- **Prompt featurization:** TF–IDF over prompt text provides a strong sparse baseline for text regression without requiring a large encoder.
- **Models:**
  - Linear regression in PyTorch as a baseline (fast, interpretable).
  - Small MLP regressor in PyTorch as a higher-capacity model (still lightweight).

### PyTorch implementation strategy
1. Build dataset of prompts and computed targets $y=s(r)$.
2. Convert prompts to TF–IDF vectors $X \in \mathbb{R}^{N\times d}$.
3. Train:
   - **Linear**: $ \hat{y}=XW+b $
   - **MLP**: $ \hat{y}=\text{MLP}(X) $
4. Optimize with AdamW on MSE loss; mini-batch training.
5. Track training/validation loss and evaluation metrics each epoch.

### Validation methods
- Report MAE/RMSE/$R^2$, Spearman ρ on held-out test split of prompt–response rows.
- Sanity checks for metric behavior:
  - Inspect high-slop vs low-slop examples.
  - Compare slop distributions for chosen vs rejected responses.

### Resource requirements and constraints
- CPU: feature computation and TF–IDF fitting.
- Memory: TF–IDF matrix (sparse) and batch tensors.
- Optional GPU: speeds up MLP training but not required.
- Time: minutes per run on Colab for moderate sample sizes.


## 3) Initial Results

### Evidence the implementation works
- End-to-end pipeline executes:
  1) loads preference dataset,
  2) computes response-level slop features,
  3) aggregates them into a scalar slop score,
  4) trains prompt-only regressors in PyTorch,
  5) evaluates on held-out test split of prompt–response rows with standard regression metrics.

### Basic performance metrics
We compute and report:
- **MAE** (mean absolute error)
- **RMSE** (root mean squared error)
- **$R^2$** (variance explained)
- **Spearman ρ** (rank correlation)

These metrics quantify how well prompt text predicts the slop score.

### Quantitative results
Using 25,000 prompt–response rows total (20,000 train / 5,000 test) and TF–IDF with 12,000 features:

**Linear:**
- MAE ≈ 1.20
- RMSE ≈ 2.06
- $R^2$ ≈ 0.02
- Spearman ρ ≈ 0.24

**MLP:**
- MAE ≈ 1.30
- RMSE ≈ 2.26
- $R^2$ < 0
- Spearman ρ ≈ 0.16

### Current limitations
- Slop score is **heuristic** and may conflate:
  - response length,
  - refusal/safety boilerplate,
  - specific stylistic patterns.
- TF–IDF prompt features ignore deeper semantics; may miss long-range structure.
- Dataset preference signal is not “pure slop,” so alignment is imperfect.

### Resource usage measurements
- Feature extraction is linear in response length and fast per example.
- TF–IDF + linear model training is efficient on CPU.
- MLP adds modest compute but remains lightweight.

### Unexpected challenges
- Metric scaling / combining heterogeneous features requires careful standardization.
- Some metrics are sensitive to tokenization choices (word vs char vs n-gram boundaries).
- Evaluation requires care to avoid leakage across prompt duplicates or near-duplicates.


## 4) Next Steps

### Immediate improvements needed
1. **Ablation study** on slop metric components:
   - Remove one feature at a time and measure (a) chosen–rejected separation and (b) predictability from prompts.
2. **Robust normalization**:
   - Ensure score is stable across response lengths and styles (e.g., length-controlled variants).
3. **Stronger baselines**:
   - Add ridge regression / elastic net; compare to MLP.

### Technical challenges to address
- **Metric validity:** quantify whether slop score matches human judgments (or at least preference ordering) beyond anecdotal examples.
- **Generalization:** does a predictor trained on one dataset transfer to other prompt distributions?
- **Confounds:** disentangle slop from refusal templates, politeness, verbosity, and safety-related text.

### Questions we need help with
- What is the best **ground-truth** signal for “slop” (human ratings vs pairwise preference vs curated test set)?
- How should we define constraints that preserve usefulness while reducing slop (e.g., factuality, on-topic, safety)?
- What is the most appropriate optimization framing for later stages (direct prompt optimization vs learning a reward model)?

### Alternative approaches to try
- Pairwise response-level modeling to learn a response “slop/quality” scorer that separates chosen vs rejected, then use it as a target/reward.
- Swap TF–IDF for a small frozen text encoder (e.g., sentence embeddings) with a shallow PyTorch head.
- Learn the slop metric itself (weights or a small model over features) using preference supervision.

### What we’ve learned so far
- A practical slop proxy can be built from cheap surface statistics and used as an optimization target.
- Prompt-only prediction is feasible as a first step and provides a foundation for future **prompt optimization** loops.
- The biggest risk is **objective validity**; the next phase should focus on verifying and refining the slop definition before optimizing prompts against it.
