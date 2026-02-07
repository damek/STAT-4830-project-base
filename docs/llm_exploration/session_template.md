# LLM Exploration Summary

## Session Focus
I was trying to stand up a quick proof of concept notebook that predicts a “slop” score from prompts. The goal was to get a full pipeline running end to end (data ingestion, slop-metric construction, modeling, EDA), then iterate through real-world environment issues until everything was stable.

## Surprising Insights

### Conversation: HH-RLHF dataset parsing and schema mismatch
**Prompt That Worked:**
- Debug why `KeyError: 'prompt'` occurs and replace the prompt extraction logic.

**Key Insights:**
- The dataset instance only exposed `chosen` and `rejected`, so “prompt” was embedded in the conversation text and had to be parsed out.
- Robust parsing via splitting on the last `"\n\nAssistant:"` made the pipeline resilient to schema variation.

### Conversation: Progress bars and pandas integration
**Prompt That Worked:**
- Diagnose why `Series.progress_apply` failed and fix it with minimal changes.

**Key Insights:**
- `progress_apply` is not native pandas; it only exists after calling `tqdm.pandas()`.
- The cleanest fix was either enabling tqdm integration once or swapping to plain `.apply()`.

### Conversation: “Use GPU” request and toolchain reality
**Prompt That Worked:**
- Rewrite training to use GPU, plus progress and ETA.

**Key Insights:**
- Scikit-learn models do not run on GPU, so “GPU training” required switching libraries (XGBoost CUDA builds, PyTorch).
- Even when a runtime has CUDA (PyTorch shows `cuda`), XGBoost can still be CPU-only depending on installation.

### Conversation: Version compatibility failures and minimal patches
**Prompt That Worked:**
- Fix errors like `mean_squared_error(..., squared=False)` and XGBoost `early_stopping_rounds` incompatibility.

**Key Insights:**
- Small API differences across versions were the dominant cause of failures, not the modeling logic.
- The smallest reliable fixes were:
  - RMSE via `sqrt(mean_squared_error(...))`
  - removing unsupported XGBoost fit kwargs
  - gating evaluation so `.predict()` is only called if training completed

### Conversation: Make models lighter and time-bounded
**Prompt That Worked:**
- Add a single parameter to control “heaviness” and keep each model under ~10 minutes.

**Key Insights:**
- Runtime was dominated by feature dimensionality and number of boosting rounds.
- A single “heaviness” knob tied to subsampling, TF-IDF max features, and epochs made the proof of concept controllable.

### Conversation: EDA for trained models
**Prompt That Worked:**
- Add EDA for both models that trains successfully and skips missing ones.

**Key Insights:**
- EDA needs to be robust to partial success: one model failing should not break downstream plots.
- Residual plots and error vs prompt length were the fastest ways to see whether the model was learning anything meaningful.

### Conversation: Replacing sklearn training with PyTorch while staying compatible
**Prompt That Worked:**
- Rewrite training portion to PyTorch-only but preserve variable names used by the rest of the notebook.

**Key Insights:**
- Keeping the same interface variables (`vectorizer`, `X_train`, `X_test`, `mlp`, `predict_loader`, `test_loader`, `results`) prevents breaking later cells.
- A TorchLinear baseline plus an MLP gives the same “two model types” spirit with fewer dependency and version pitfalls.

## Techniques That Worked
- Ask for the smallest possible patch when an error is version-related.
- Build guards around model evaluation so failed training does not cascade into unrelated cells.
- Prioritize “runs everywhere” defaults over maximum performance for early proof of concept work.
- Use a single scaling knob for model heaviness that affects data size, feature count, and training schedule together.

## Dead Ends Worth Noting
- Attempting GPU training via scikit-learn models is not feasible.
- XGBoost GPU settings (`gpu_hist`) failed because the installed XGBoost build did not support it.
- XGBoost early stopping and callbacks were incompatible with the installed version, leading to failures unless carefully gated.
- Drop-in replacement of sklearn pipelines broke downstream cells that expected `fit_pipelines`.

## Next Steps
- Replace per-sample densification with a sparse-friendly PyTorch approach or further reduce TF-IDF dimensionality.
- Tighten the slop definition by adding sentence-level repetition, readability, and burstiness metrics.
- Add a small calibration set with human labels or pairwise preferences to learn slop weights instead of fixing them heuristically.
- Store and reuse train/test splits so all models compare on identical data across runs.
