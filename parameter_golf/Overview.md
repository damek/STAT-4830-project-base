# Parameter Golf (OpenAI challenge): approach overview

This note is a **medium-level** map of the problem: what you ship, how you are scored, and how the **16MB budget** shapes design choices. For technique deep-dives and leaderboard context, see `FINDINGS.md`.

---

## What you are building

**Goal**: A small language model that predicts the next token well on held-out text, under strict size and (timed track) training-time limits.

**Deliverables** (must fit together under **16MB** total):

1. **Code** that trains the model (e.g. a reproducible `train_gpt.py`–style entry point).
2. **Model weights** in a **compressed** form (often a single packed blob—e.g. quantized weights plus metadata, then **zlib** or stronger compression; teams sometimes refer to the shipped artifact as a compressed checkpoint or `.ptz`-style file).

The eval harness does not need your training loop at runtime; it needs **inference code + the compressed weights** so the bundle stays under the cap.

---

## End-to-end pipeline (conceptual)

1. **Train** a next-token model on the competition training split (e.g. FineWeb-style data).
2. **Export** weights: typically **quantized** (int6/int8, sometimes ternary) so more parameters fit after compression.
3. **Compress** the weight blob (and any small side tables) so **code + compressed weights ≤ 16MB**.
4. **Validate** locally with the same **BPB** (bits per byte) logic the leaderboard uses (see **Loss metric and BPB** below).

Most of the **artifact size** is usually the **compressed weight file**, not the Python. That means the core tradeoff is: **how big a model can you pack** (width, depth, MLP expansion, vocab) **versus** quantization quality, compression, and whether training still finishes in time.

---

## How evaluation works (next token → score)

You are given text as a token sequence. For each position, the model sees the **prefix** and must produce a **probability distribution over the next token** (a full PMF over the vocabulary, or whatever the host API standardizes).

**Example**: For something like `I love cats.` in tokens, at some step the true next token might be `cats` (or a subword). The model outputs a **full distribution** over the vocabulary: probability of each possible next token given the prefix. The grader takes the **true** next token and scores how much probability you put on it.

So the objective is not “guess one token” in isolation—it is **calibrated next-token likelihood** aggregated over the validation corpus, under the competition’s tokenization and eval protocol.

---

## Loss metric and BPB (what actually gets optimized)

**Per position (conceptual loss)**  
For the true next token `y` and your predicted distribution `p(· | context)`, the grader cares about **how much probability you assigned to `y`**. The standard per-step quantity is the **negative log-probability** of the correct token:

- **`-log p(y | context)`** (natural log in training is fine; scores are reported in **bits** after conversion).

Intuition:

- If `p(y | context)` is **close to 1**, then **`-log p`** is **close to 0** — you are barely “surprised” by the truth.
- If `p(y | context)` is **tiny**, **`-log p`** is **large** — heavy penalty.

That is the same object as **cross-entropy** between a one-hot truth and your softmax: training with **cross-entropy** directly pushes down this per-step loss.

**From per-step loss to the leaderboard number**  
The competition does not ask for argmax accuracy alone; it measures **compression quality** of the model on text. Results are reported as **bits per byte (BPB)** on the validation set. **Lower BPB is better** (less average “surprise” per byte of data). `FINDINGS.md` records the metric as:

**`BPB = negative log-likelihood / log(2) / 8`**

Use the **official eval / reference implementation** for the exact aggregation (which tokens count, how bytes are measured). For intuition: **negative log-likelihood** is built from the same **`-log p(correct token)`** terms; dividing by **`log(2)`** converts from **natural log (nats)** to **bits**; the competition’s reported BPB is that **bit-total** scaled to **per byte** of text under their definition.

**Why this matters for your loop**

- **Training**: minimizing **cross-entropy** (sum or mean of **`-log p(correct token)`**) is aligned with what BPB rewards, modulo quantization and any train–eval mismatch.
- **Debugging**: a small change that barely moves **validation cross-entropy** may still move **BPB** if it changes calibration or if eval uses a different windowing/packing rule—match the reference eval when possible.
- **Reading the leaderboard**: numbers like **~1.12 BPB** vs **~1.22 BPB** baseline are **small-looking gaps** but reflect real modeling headroom; top entries improved on the order of **~0.1 BPB** over the baseline in the `FINDINGS.md` timeline.

---

## Design tension: size vs quality vs speed

- **Larger** models (more layers, wider MLP, bigger vocab) can improve BPB **if** they still fit after **quantization + compression**.
- **Aggressive** quantization and packing shrink the file but can hurt BPB unless you use **QAT**, post-training calibration (e.g. GPTQ-style methods where rules allow), or architectural choices that tolerate low bit-width weights.
- **Training and eval speed** matter on the **timed** track: you have a fixed wall-clock budget on prescribed hardware, so you cannot only optimize for “best model in theory”—you need a recipe that **converges** within the limit.

Use `FINDINGS.md` for concrete techniques (int6, EMA, compression presets, etc.) that top submissions used to navigate this triangle.

---

## Practical takeaway

Treat the submission as **one bundle**: **reproducible training script + compressed quantized weights**, scored by **next-token negative log-likelihood → BPB**, optimized under **≤16MB** and your track’s **time** constraints. Iterate on architecture and quantization **together**, because the **compressed weight size** is usually what binds first.
