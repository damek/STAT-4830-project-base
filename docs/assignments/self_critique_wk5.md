# Self-Critique: Week 5 — Review of Original Code (STAT4830_KaggleEssays_Verifier.ipynb)

*OODA format. Review of the original Kaggle Essays Verifier notebook that was copied/adapted.*

---

## OBSERVE

- **Artifact reviewed:** `STAT4830_KaggleEssays_Verifier.ipynb` — the original notebook implementing a Human vs Slop essay verifier.
- **Code re-check:** Notebook includes setup, human essay loading (Kaggle Aeon), two slop strategies (quick corruption + LLM from titles), stratified splits, DistilBERT training with early stopping, evaluation, checkpoint saving, and visualization. Uses 256-token truncation, AdamW, patience=2.
- **Reactions:** Well-structured pipeline with clear cell summaries. Good choice of quick corruption for sanity-check vs LLM for harder negatives. Relies on Colab/Kaggle for `essays.csv`.

---

## ORIENT

**Strengths (max 3)**

- **Complete end-to-end pipeline:** From raw CSV through slop generation, training, eval, and `score_humanlikeness` for downstream reward use. Easy to follow and run.
- **Dual slop strategy:** Quick corruption (sentence shuffle + fillers) for fast iteration; LLM generation from titles to avoid leaking human text. Sensible design.
- **Strong baseline:** DistilBERT + early stopping on val ROC-AUC. Practical for Colab GPU.

**Areas for improvement (max 3)**

- **Length control:** Human essays filtered to 400–20k chars; LLM slop to 1500–6000. No explicit length-matching between classes in the same split—risk of length leakage.
- **No modular src:** Logic lives in notebook cells; harder to test or reuse. Extraction to `src/model.py` and `src/utils.py` improves testability.
- **Limited error handling:** Assumes `essays.csv` exists and has `essay`/`title` columns; no fallback for missing data or schema issues.

**Critical risks/assumptions (2–3 sentences)**

Assumes Aeon Kaggle schema and that DistilGPT-2 slop is a reasonable proxy for “real” AI slop. 256-token truncation may lose signal in long essays. Colab-specific paths (e.g. `/content/verifier_ckpt`) may not transfer to local runs.

---

## DECIDE

**Concrete next actions (max 3)**

1. **Add length-matching step:** Subset human and slop to overlapping length bins before concatenating, to reduce length-based shortcuts.
2. **Refactor into src:** Move `load_verifier`, `run_eval`, `score_humanlikeness`, and slop utilities into `src/`; keep notebook as a thin driver.
3. **Add data validation:** Check CSV schema and optional fallback (e.g. synthetic sample) when `essays.csv` is missing.

---

## ACT

**Resource needs (2–3 sentences)**

- **Length-matching:** Pandas only; no new deps. Need to define overlap window (e.g. 1000–6000 chars).
- **Refactor:** Already partially done in `src/`; align notebook imports with extracted functions.
- **Data validation:** Add a small validation cell; synthetic fallback requires template design.
