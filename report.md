# Predicting “Slop” in LLM Outputs  
**STAT 4830 – Numerical Optimization for Data Science and Machine Learning**  
Milestone 1 Report  

---

# 1. Problem Statement

## What Are We Optimizing?

The long-term objective of this project is to reduce low-quality (“sloppy”) outputs from large language models by treating prompt design as a numerical optimization problem.

Let:

-p= prompt  
-y = f(p)= response from a frozen language model  
-S(y)= scalar slop score computed from response-level surface statistics  

Ultimately, we aim to solve:

$$
\min_{\theta} \; \mathbb{E}_{p \sim \mathcal{D}} \left[ S(f(p_\theta)) \right]
$$

wherep_\thetadenotes a parameterized prompt.

For this milestone, we focus on a necessary subproblem:

> Can we learn a function that predicts a measurable slop score using only the prompt text?

We train a predictor:

$$
g_\phi(p) \approx S(y)
$$

and optimize:

$$
\min_{\phi} \ \frac{1}{N} \sum_{i=1}^{N} \left(g_\phi(p_i) - S(y_i)\right)^2
$$

This is a supervised regression problem where the prompt is the input and the response-derived slop score is the target.

---

## Why This Problem Matters

Large language models often generate responses that are repetitive, low-diversity, mechanically structured, or information-light. While “quality” is subjective, many undesirable outputs share measurable structural characteristics.

If:

1. Slop can be quantified numerically,
2. Slop varies systematically with prompt structure,
3. Slop is predictable from prompts,

then prompt engineering can be reframed as a principled optimization problem rather than heuristic trial-and-error.

Milestone 1 evaluates whether slop is measurable and whether prompt text contains predictive signal.

---

## How We Measure Success

We evaluate predictive performance using:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
-R^2
- Spearman rank correlation  

Success criteria for this milestone:

- Linear model achieves positiveR^2relative to predicting the mean  
- Non-zero Spearman correlation  
- Stable training under MSE minimization  

Because identical prompts can produce multiple responses, we expect inherent noise that limits achievableR^2.

---

## Constraints

- The language model is frozen (no fine-tuning).
- Only prompt text is used as input features.
- Slop is defined using surface-level statistics only.
- Dataset size limited to ~20,000 sampled examples.
- CPU-based PyTorch training (Google Colab).

---

## Data Used

We use the `Anthropic/hh-rlhf` dataset (train split).

Procedure:

- Randomly sample ~20,000 examples.
- Extract the prompt.
- Include both chosen and rejected responses.
- Compute a separate slop score for each response.

Each flattened row consists of:

$$
(\text{prompt}, \text{response}, S(\text{response}))
$$

This results in approximately 40,000 prompt-response pairs.

---

## What Could Go Wrong

- Surface metrics may not align with human-perceived quality.
- Identical prompts with different responses introduce target noise.
- Slop may depend more on model randomness than prompt structure.
- TF–IDF features may miss semantic meaning.
- Heuristic slop weights may bias the metric.

---

# 2. Technical Approach

## Slop Score Definition

For each responsey, we compute:

- Trigram repetition rate  
- Distinct-2 score (unique bigrams / total bigrams)  
- Character-level Shannon entropy  
- Compression ratio (via zlib)  
- Punctuation density  
- Capital letter ratio  

Each feature is standardized:

$$
z_i = \frac{x_i - \mu_i}{\sigma_i}
$$

The scalar slop score is defined as a weighted linear combination:

$$
S(y) =
+ w_1 z_{\text{repetition}}
- w_2 z_{\text{distinct2}}
- w_3 z_{\text{entropy}}
- w_4 z_{\text{compression}}
+ w_5 z_{\text{punctuation}}
+ w_6 z_{\text{caps}}
$$

Weights are fixed heuristically and are not learned.

HigherS(y)corresponds to more repetitive, lower-diversity, lower-entropy responses.

---

## Prompt Representation

Prompts are vectorized using TF–IDF:

$$
p \rightarrow x \in \mathbb{R}^d
$$

- Word-level features  
- Maximum feature size capped at 12,000  
- Sparse matrix converted to dense PyTorch tensors  

---

## Models Implemented

Two PyTorch regression models were trained.

### Linear Model

$$
g_\phi(x) = Wx + b
$$

Implemented using `torch.nn.Linear`.

---

### Multi-Layer Perceptron (MLP)

Architecture:

- Input layer  
- One hidden layer with ReLU activation  
- Output layer  

This introduces nonlinearity in mapping prompt features to slop score.

---

## Objective Function

Both models minimize mean squared error:

$$
\mathcal{L}(\phi) =
\frac{1}{N}
\sum_{i=1}^{N}
\left(g_\phi(p_i) - S(y_i)\right)^2
$$

Optimizer: Adam  
Training performed over multiple epochs using mini-batches.

---

## Validation Methods

- 80/20 train-test split  
- Evaluation on held-out test set  
- Metrics: MAE, RMSE (computed as\sqrt{\text{MSE}},R^2, Spearman correlation  
- Scatter plots of predicted vs true slop  

---

## Resource Requirements and Constraints

- ~40k flattened samples  
- TF–IDF dimension ≤ 12k  
- CPU-only PyTorch training  
- Runtime: minutes per experiment  
- Memory dominated by TF–IDF matrix  

---

# 3. Initial Results

## Evidence Implementation Works

- All slop metrics compute successfully.
- Z-score normalization stabilizes feature scaling.
- Both models converge under MSE training.
- No numerical instability observed.

---

## Performance Metrics

### Linear Model

- MAE ≈ 1.20  
- RMSE ≈ 2.06  
-R^2 \approx 0.02
- Spearman\rho \approx 0.23

### MLP

- MAE ≈ 1.20
- RMSE ≈ 2.26
-R^2 < 0 
- Spearman\rho \approx 0.16

---

## Interpretation

- The **linear model achieves a small but positiveR^2**, indicating modest predictive signal.
- The **MLP performs worse**, with negativeR^2, suggesting additional nonlinearity does not improve performance.
- Spearman correlations indicate weak but non-zero rank predictability.
- Overall predictive power is limited.

These results suggest that prompt text contains some signal about response-level slop, but the signal is weak.

---

## Current Limitations

- Identical prompts map to multiple slop targets, limiting achievableR^2.
- Slop score is heuristic and not human-validated.
- TF–IDF ignores semantic structure.
- No regularization or hyperparameter tuning performed.

---

## Resource Usage Measurements

- Dataset: ~20k examples (~40k flattened rows)  
- Feature dimension: ≤ 12k  
- CPU training  
- Training time: several minutes per run  

---

## Unexpected Challenges

- RMSE required manual computation via\sqrt{\text{MSE}}due to sklearn version differences.
- Target noise from response stochasticity reduced achievable performance.
- MLP overfitting occurred without performance gains.

---

# 4. Next Steps

## Immediate Improvements Needed

- Aggregate slop per prompt (e.g., mean over responses) to reduce noise.
- Add L2 regularization to linear model.
- Tune hyperparameters.
- Compare against stronger baselines.

---

## Technical Challenges to Address

- Designing a principled, human-aligned slop metric.
- Improving semantic representation of prompts.
- Determining whether ranking formulation is more appropriate than regression.
- Quantifying irreducible noise from response randomness.

---

## Questions for Instructor

- Is weak but positive rank correlation sufficient to justify optimization framing?
- Should we incorporate human evaluation of slop?
- Would contrastive modeling (chosen vs rejected) be more appropriate?

---

## Alternative Approaches to Try

- Transformer-based prompt embeddings.
- Pairwise ranking loss instead of MSE.
- Binary classification (high vs low slop).
- Contrastive learning between chosen and rejected responses.

---

## What We Have Learned So Far

- Slop can be formalized numerically using structural metrics.
- Slop exhibits measurable variation across responses.
- Prompt text contains weak but detectable predictive signal.
- Linear models outperform nonlinear MLPs in this setting.
- Response stochasticity is a central bottleneck.

This milestone establishes that slop is measurable and weakly predictable from prompts, providing preliminary justification for framing prompt design as a numerical optimization problem.
