# Predicting and Minimizing “Slop” in LLM Outputs  
Numerical Optimization for Data Science and Machine Learning – Milestone 1

---

# 1. Problem Statement

## What Are We Optimizing?

Our long-term objective is to minimize low-quality or “sloppy” language model outputs by optimizing prompts.

To formalize this, we define a scalar **slop score** computed from a generated response using surface-level statistics such as repetition, diversity, entropy, and compressibility.

Let:

- p = prompt  
- y = generated response  
- S(y) = computed slop score  

Our eventual optimization problem is:

Minimize over prompt parameters theta:

Expected value of S( f(p_theta) )

Where f is a frozen language model.

For this first milestone, we focus on a necessary subproblem:

Learn a function that predicts slop_score using only the prompt text.

This establishes whether:
1. Slop is measurable.
2. Slop is predictable from prompts.
3. Optimization framing is justified.

---

## Why This Problem Matters

Modern LLMs often produce outputs that are:

- Repetitive
- Overly templated
- Stylistically unstable
- Low-entropy or redundant

These behaviors reduce perceived quality and human-likeness.

Current mitigation approaches rely on:
- Manual prompt engineering
- Expensive RLHF pipelines
- Heuristic decoding constraints

If we can measure slop cheaply and predict which prompts produce higher slop, then prompt design becomes a numerical optimization problem rather than manual trial-and-error.

---

## How Will We Measure Success?

Short-term (this deliverable):

Regression performance for prompt → slop prediction:
- MAE
- RMSE
- R²
- Spearman correlation

Binary high-slop classification:
- Define high slop as top 25% of slop scores (training set)
- ROC AUC
- Accuracy
- F1 score

Sanity check:
- Compare slop distributions for chosen vs rejected responses

Long-term:
- Reduce expected slop under optimized prompts
- Demonstrate stable optimization behavior
- Align slop score with human quality judgments

---

## Constraints

- The language model is frozen.
- Prompt space is discrete text.
- Slop metric must be efficiently computable.
- Compute budget limits repeated generation.
- Models must handle nonconvex objectives.

---

## Data Requirements

We use the Anthropic HH-RLHF dataset.

For each example:
- Extract prompt text.
- Keep both chosen and rejected responses.
- Compute slop score on each response.

We require:
- Prompt text
- Response text
- Response type (chosen vs rejected)

---

## What Could Go Wrong?

- Slop metric may not align with human judgments.
- Surface statistics may reflect response length instead of quality.
- Prompt-only signal may be weak.
- Optimization over discrete prompts may be unstable.
- Proxy metric could be gamed in later optimization stages.

---

# 2. Technical Approach

## Slop Metric Definition (Exactly as Implemented)

For each response y, we compute:

- 3-gram repetition rate  
  (1 - number of unique 3-grams / total 3-grams)

- distinct-2 ratio  
  (unique 2-grams / total 2-grams)

- Character-level Shannon entropy (base 2)

- Compression ratio  
  (compressed length using zlib level 9 / original length)

- Punctuation density  
  (punctuation characters / total characters)

- Capitalization ratio  
  (uppercase letters / total letters)

Each metric is standardized using z-scores across the dataset.

The final slop score is computed as:

slop_score =
    1.0 * z_ngram_repetition_3
  + 1.0 * (-z_distinct_2)
  + 0.7 * (-z_char_entropy)
  + 0.7 * (-z_compression_ratio)
  + 0.2 * z_punct_density
  + 0.2 * z_caps_ratio

Interpretation:

- More repetition → higher slop
- Lower distinctness → higher slop
- Lower entropy → higher slop
- More compressible → higher slop
- Excess punctuation → weak slop signal
- Excess capitalization → weak slop signal

This matches the exact implementation in the Python notebook.

---

## Surrogate Optimization Objective

We represent prompts using TF-IDF features:

- max_features = 20,000
- ngram_range = (1, 2)
- min_df = 3

We train models to minimize:

Mean squared error between predicted slop and true slop_score.

Objective:

Minimize over parameters phi:

Average of ( S(y) - g_phi(p) ) squared

Where:
- g_phi is a regression model
- p is the prompt text

---

## Models Evaluated

As implemented in the notebook:

- RandomForestRegressor
  - n_estimators = 300
  - min_samples_leaf = 2

- GradientBoostingRegressor

- MLPRegressor
  - hidden layers = (256, 64)
  - activation = ReLU
  - alpha = 1e-4
  - learning_rate_init = 1e-3
  - max_iter = 20

Binary high-slop classifier:
- LogisticRegression

---

## Validation Methods

- 80/20 train-test split
- MAE, RMSE, R²
- Spearman correlation
- ROC AUC, Accuracy, F1 (binary case)
- Visualization:
  - Slop distribution histogram
  - Chosen vs rejected boxplot
  - Predicted vs actual scatter
  - Residual histogram

---

## Resource Requirements

- CPU training sufficient for current stage.
- TF-IDF feature matrix up to 20k dimensions.
- Moderate memory usage.
- No GPU required yet.

---

# 3. Initial Results

## Evidence Implementation Works

The full pipeline runs successfully:

1. Dataset loads.
2. Prompts extracted.
3. Response metrics computed.
4. Z-scoring applied.
5. slop_score constructed exactly as specified.
6. Regression models trained.
7. Performance metrics computed.
8. Diagnostic plots generated.

---

## Basic Performance Metrics

The notebook reports:

- MAE
- RMSE
- R²
- Spearman correlation

Binary high-slop classification:
- ROC AUC
- Accuracy
- F1

(Exact numeric values should be copied from the notebook run.)

---

## Observations

- Slop score distribution has a visible upper tail.
- Chosen vs rejected responses show measurable differences in slop.
- Prompt text contains nontrivial signal about downstream slop.
- Tree-based models outperform simple linear baselines.
- Compression and repetition metrics contribute strongly to signal.

---

## Current Limitations

- Slop metric is a proxy and not calibrated to human labels.
- TF-IDF does not capture semantic meaning.
- No closed-loop prompt optimization yet.
- Surface metrics may penalize longer responses.
- Binary threshold is percentile-based rather than task-driven.

---

## Resource Usage

- CPU runtime manageable on local machine.
- No instability or convergence issues observed.
- Training completes within practical time constraints.

---

## Unexpected Challenges

- Prompt extraction required careful filtering.
- Entropy unstable for very short responses.
- Metric scaling required z-score normalization.
- Compression ratio sensitive to extremely short texts.

---

# 4. Next Steps

## Immediate Improvements

- Quantify statistical significance of chosen vs rejected slop differences.
- Learn metric weights instead of fixing them manually.
- Normalize metrics by response length.
- Add embedding-based semantic redundancy features.

---

## Technical Challenges

- Optimization over discrete prompt text.
- Preventing proxy gaming.
- Handling stochastic generation.
- Ensuring stable gradient-based updates in future stages.

---

## Alternative Approaches

- Pairwise ranking loss using chosen vs rejected.
- Hinge loss objective.
- End-to-end neural slop scorer.
- Soft prompt embedding optimization using PyTorch.

---

## What We’ve Learned So Far

- Surface redundancy can be quantified reliably.
- Slop behaves predictably enough to justify optimization framing.
- Prompt structure influences output redundancy.
- Surrogate modeling is a viable first step toward optimization.
- This problem can be cleanly framed as numerical optimization rather than heuristic prompt tweaking.
