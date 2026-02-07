# Project Report: Predicting “Slop” from Prompts (Proof of Life)

**STAT 4830 — Week 4 Draft**

---

## 1. Problem Statement (~½ page)

**What are you optimizing?**  
We are optimizing a **regression model** that predicts a scalar **slop score** from **prompt text only**. The slop score is a proxy for “low-quality” or generic AI text, defined on the *response* using surface-level metrics (repetition, entropy, style). The model sees only the prompt; the goal is to learn how much “slop-like” the *response* tends to be, before the response is generated. This is a proof-of-life formulation: we do not yet optimize an LLM directly; we optimize a predictor of a hand-crafted slop proxy.

**Why does this problem matter?**  
As LLM output floods the web, detecting or anticipating low-quality (“slop”) content matters for filtering, ranking, and for future reward models. Showing that prompt-only signal correlates with a simple slop proxy would support (1) pre-generation quality cues and (2) iterating toward a learnable, human-aligned slop metric.

**How will you measure success?**  
- **Primary:** Spearman correlation between predicted and actual slop score (rank correlation).  
- **Secondary:** MAE, RMSE, R² on held-out test data.  
- **Validation of the proxy:** Correlation of slop score with chosen vs. rejected response labels in the dataset (to be added).

**What are your constraints?**  
- Models may use only **prompt** text (no response at inference).  
- Compute: CPU/Colab-friendly; no large LLM fine-tuning in this phase.  
- Slop is defined by cheap, interpretable metrics so we can iterate quickly.

**What data do you need?**  
Prompt–response pairs with optional preference labels. Current source: **Anthropic HH-RLHF** (`Anthropic/hh-rlhf`), flattened to rows (prompt, response, chosen/rejected). We use a subsample (e.g., 20k prompts → 40k rows with chosen/rejected) for speed.

**What could go wrong?**  
- **Information bottleneck:** The response is not observed at prediction time; performance has a low ceiling.  
- **Proxy validity:** Our slop score may not align with human notions of quality; we need to validate against chosen/rejected.  
- **Data/scale:** Subsampling and single-dataset choice may limit generality; full-dataset and memory use need checking as we scale.

---

## 2. Technical Approach (~½ page)

**Mathematical formulation**  
- **Input:** Prompt text \(x\).  
- **Target:** Scalar slop score \(y \in \mathbb{R}\), computed from the response (not seen at inference).  
- **Objective:** Minimize expected loss on a held-out set. We use **MSE** as the training loss: \(\mathcal{L}(\theta) = \frac{1}{n}\sum_i (y_i - f_\theta(x_i))^2\), with \(f_\theta\) the model (linear or MLP). No explicit constraints; the slop score is unbounded (z-score combination in practice).  
- **Slop score definition:** For each response we compute six metrics (ngram repetition, distinct-2, char entropy, punctuation density, caps ratio, compression ratio), standardize to z-scores, then \(y = 1\cdot z_{\text{rep}} - 1\cdot z_{\text{distinct}} - 0.7\cdot z_{\text{entropy}} - 0.7\cdot z_{\text{compression}} + 0.2\cdot z_{\text{punct}} + 0.2\cdot z_{\text{caps}}\). Higher \(y\) = more “slop-like.”

**Algorithm/approach choice**  
- **Features:** TF-IDF on prompt text (e.g., 12k features, subsampled for speed). Chosen for simplicity and no GPU requirement for feature extraction.  
- **Models:** (1) **Linear regression** (PyTorch, MSE loss) as baseline; (2) **MLP** (hidden layers, ReLU) to capture nonlinearities. Linear is justified as a strong baseline for high-dimensional sparse TF-IDF; MLP to test if rank correlation improves.

**PyTorch implementation strategy**  
- TF-IDF is computed with sklearn; sparse matrix is converted to dense (or batched) for PyTorch.  
- Custom `Dataset`/`DataLoader` over (X, y) for training.  
- Optimizer: Adam; fixed epoch count and optional time budget per run.  
- Inference: forward pass over test loader; predictions collected and evaluated with sklearn/scipy (MAE, RMSE, R², Spearman).

**Validation methods**  
- Train/test split (e.g., 80/20) with fixed seed; same split for all models.  
- Metrics on test set only; no cross-validation in the current draft.  
- Planned: correlation of slop score with chosen vs. rejected; ablation on slop weights.

**Resource requirements and constraints**  
- RAM: dataset + TF-IDF matrix in memory (subsample ~25k rows in current runs).  
- CPU sufficient for TF-IDF and small PyTorch models; GPU optional for MLP.  
- Runtime: on the order of minutes per model (e.g., ~12 s linear, ~14 s MLP in reported runs).

---

## 3. Initial Results (~½ page)

**Evidence the implementation works**  
- End-to-end pipeline runs: load HH-RLHF → flatten to prompt/response rows → compute six metrics per response → z-score and combine into slop_score → train/test split → TF-IDF fit on train → PyTorch linear and MLP trained → predictions on test set.  
- EDA in the notebook: slop_score distribution, boxplot by chosen/rejected, correlation heatmap of raw metrics, and high/low slop examples (subject to fixing stale output scale in one cell).

**Basic performance metrics (test set)**  
- **TorchLinear:** MAE ≈ 1.20, RMSE ≈ 2.06, R² ≈ 0.02, **Spearman ≈ 0.23**.  
- **PyTorch MLP:** MAE ≈ 1.27, RMSE ≈ 2.23, R² ≈ −0.15, Spearman ≈ 0.19.  
- Linear baseline slightly outperforms the MLP on both R² and Spearman, consistent with limited signal and high-dimensional sparse input.

**Test case results**  
- Predictions are real-valued and correlate weakly but positively with actual slop (Spearman ~0.23). No formal unit tests yet; “test” here means held-out test split.

**Current limitations**  
- Only two models (linear + MLP); no Random Forest or Gradient Boosting in the current run (notebook intro mentions them but they are not executed).  
- Slop score is heuristic; not yet validated against chosen/rejected.  
- Possible bug: high/low example cell output shows slop_score on a different scale than the rest of the notebook; needs a full re-run to ensure consistency.  
- Single split, no confidence intervals or multiple seeds.

**Resource usage**  
- Training time ~12–14 s per model on subsampled data; memory fit in Colab/laptop.

**Unexpected challenges**  
- MLP degrades (negative R²) vs linear, suggesting overfitting or need for stronger regularization/simpler architecture.  
- Prompt-only prediction ceiling is low; Spearman ~0.23 is a reasonable “proof of life” but leaves room for better features (e.g., embeddings) or a different task formulation.

---

## 4. Next Steps (~½ page)

**Immediate improvements**  
- **Validate slop proxy:** Report mean/median slop for chosen vs rejected (or point-biserial correlation); if the proxy does not separate chosen/rejected, revise metrics or weights.  
- **Fix notebook consistency:** Re-run all cells from top; ensure slop_score scale is single and that high/low example outputs match the current formula.  
- Either **add sklearn RF/GB** baselines (same train/test, same eval) or update the notebook intro to state only linear + MLP are used in this draft.

**Technical challenges**  
- Improving rank correlation beyond ~0.23 may require better prompt representations (e.g., pretrained sentence embeddings or small language models) or learning slop weights from human preferences.  
- Scaling to full HH-RLHF or larger data without OOM; consider chunked TF-IDF or streaming.

**Questions you need help with**  
- Whether to frame the next milestone as (a) better prompt-only predictors, (b) learning slop weights from chosen/rejected, or (c) using the slop score inside an LLM training loop (e.g., as a reward signal).  
- Best practice for validating a scalar proxy against binary chosen/rejected (metrics and visualization).

**Alternative approaches**  
- Learn slop weights via regression or ranking loss on chosen/rejected instead of fixed heuristic weights.  
- Use pretrained encoders (e.g., sentence-transformers) for prompt encoding instead of TF-IDF.  
- Predict chosen vs rejected directly (classification) and compare with slop-based ranking.

**What you’ve learned so far**  
- A simple linear model on TF-IDF of the prompt captures weak but nonzero signal (Spearman ~0.23) for a hand-crafted slop proxy.  
- The task is inherently limited by not seeing the response; the proxy itself needs validation against human preferences.  
- PyTorch linear + MLP pipeline is in place and can be extended with more baselines and better features.
