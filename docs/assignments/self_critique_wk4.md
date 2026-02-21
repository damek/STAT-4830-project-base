# Self-Critique: Proof-of-Life Draft (Week 4)

*Focused, actionable critique of the current ML project draft. Max 1 page.*

---

## OBSERVE

- **Artifacts reviewed:** Notebook `slop_prelim_experiment (2).ipynb` (end-to-end pipeline: load HH-RLHF → slop metrics → EDA → predict slop from prompt). No `report.md` yet; README is template.
- **Code re-check:** Pipeline runs: 6 response-level metrics → z-score combined `slop_score` → TF-IDF + Torch linear/MLP. Best result: TorchLinear MAE ≈ 1.20, R² ≈ 0.02, Spearman ≈ 0.23. MLP degrades (R² &lt; 0, Spearman ≈ 0.19).
- **Reactions:** Goal is clear and the task is hard (predict response quality from prompt only). Notebook intro promises "Random Forest, Gradient Boosting, MLP" but only PyTorch linear + MLP are actually run. High/low slop example output shows `slop_score` values ~15–18 and ~-2, while `describe()` shows roughly [-1.2, 0.36]—suggests stale outputs or inconsistent scaling.

---

## ORIENT

**Strengths (max 3)**

- **Clear proof-of-life goal:** Predict a single, interpretable “slop” proxy from prompt-only input; end-to-end pipeline (data → metrics → EDA → models → MAE/RMSE/R²/Spearman) is in place and runnable.
- **Interpretable slop proxy:** Six surface-level metrics (n-gram repetition, distinct-2, char entropy, punctuation/caps density, compression ratio) are defined and combined; good base for iteration and ablation.
- **Honest framing:** Notebook states this is “not a perfect definition of slop” and that prompt-only prediction is expected to be modest—sets reader expectations.

**Areas for improvement (max 3)**

- **No written report:** Problem statement, objective, and next steps live only in the notebook. Week 4 expects a 2-page `report.md` (problem, technical approach, initial results, next steps); without it, the project is hard to evaluate and the optimization story is under-specified.
- **Intro vs. implementation mismatch:** Intro says “Random Forest, Gradient Boosting, MLP”; only Torch linear and PyTorch MLP are trained. Either add sklearn RF/GB (and optionally sklearn MLP) for baselines or update the intro so it matches what is run.
- **Slop proxy not validated against labels:** `slop_score` is a heuristic combination; there’s no check that it correlates with “chosen vs. rejected” (or any human notion of quality). Weights are fixed; no ablation or justification.

**Critical risks/assumptions (2–3 sentences)**

We assume that prompt-only prediction is a useful task despite the information bottleneck; that the current weighted combination of metrics is a reasonable slop proxy; and that the dataset subsample and TF-IDF setup are representative. There is also a likely bug or stale output: high/low example cells show `slop_score` on a different scale than the rest of the notebook, which can confuse readers and interpretation.

---

## DECIDE

**Concrete next actions (max 3)**

1. **Add `report.md` (≤2 pages):** Problem statement (what we optimize, why it matters, success metrics, data); technical approach (regression objective, TF-IDF + models, validation); initial results (current metrics, limitations); next steps (validate proxy, add baselines, fix notebook consistency).
2. **Align notebook with narrative:** Either add sklearn RF and GB (and optionally MLP) in a single pipeline cell (e.g. `Pipeline(TfidfVectorizer(), Regressor)`), record MAE/R²/Spearman, and add to a results table; or remove RF/GB/MLP from the intro and state that only Torch linear + MLP are used for this draft.
3. **Validate slop proxy and fix scale:** (a) Report correlation of `slop_score` with `resp_type` (e.g. mean/median slop for chosen vs rejected; or point-biserial). (b) Re-run the notebook from top to bottom and ensure high/low example cells use the same `slop_score` as in the rest of the notebook; fix or remove any duplicate or legacy scaling.

---

## ACT

**Resource needs (2–3 sentences)**

- **Report:** Use the Week 4 instructions and existing notebook content; no new tools. Blockers: carving out time to write problem/approach/results/next steps clearly.
- **Baselines:** sklearn is already imported; RF/GB need one extra cell with the same `X_train`/`X_test` and `eval_regression`. No new dependencies.
- **Proxy validation and scaling:** Only pandas/matplotlib; no new libraries. Blocker: confirming one canonical definition of `slop_score` (single place where z-scores are computed and combined) and re-running the full notebook so all outputs are consistent.

---

*Keep this document handy when drafting Report Draft 2 and when updating the notebook so each “Area for improvement” is addressed by a “Concrete next action.”*
