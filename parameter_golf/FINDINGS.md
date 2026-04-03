# Parameter Golf: Techniques Analysis

## Competition Overview

**Challenge**: Train the best language model that fits in **16MB** (code + compressed weights) and trains in **under 10 minutes** on 8×H100s, evaluated by compression on the FineWeb validation set.

**Metric**: Bits per byte (BPB) — lower is better. BPB = negative log-likelihood / log(2) / 8.

**Hardware**: 8×H100 SXM (80GB HBM3 each), PyTorch 2.9.1+cu128.

**Baseline**: 1.2244 BPB — 9-layer, 512-dim GPT with int8 quantization + zlib compression.

**Timeline**: March 17 – April 30, 2026. Active SOTA improvements occurred over ~8 days.

**Rules**: 3+ training seeds required; artifact must be reproducible from the submission's `train_gpt.py`; no external data during quantization (AR self-generation is allowed).

---

## Leaderboard (Top 10, Timed Track)

| Rank | BPB | Submission | Author | Key Innovation |
|------|-----|------------|--------|----------------|
| 1 | **1.1147** | AR Self-Gen GPTQ + XSA-all | abaybektursun | Full Hessian GPTQ with AR calibration + XSA on all 11 layers |
| 2 | 1.1194 | LeakyReLU² + Legal TTT + Parallel Muon | abaybektursun | LeakyReLU(0.5)², backward-looking TTT, Parameter Banking |
| 3 | 1.1228 | GPTQ-lite + EMA + warmdown | signalrush | Clip-percentile GPTQ search, EMA averaging, 3500-iter warmdown |
| 4 | 1.1248 | Partial RoPE + LN Scale + EMA | jfprincz | Partial RoPE (16/64 dims), layer-wise LN scaling (1/√layer), EMA |
| 5 | 1.1271 | XSA4 + EMA + Int6 MLP3x | jfprincz | XSA on last 4 layers, EMA, 3× MLP, int6 quantization |
| 6 | 1.1307 | Efficient Partial XSA | unnir | GQA-aware XSA reshape (zero-alloc), FA3, applied to 3 deepest layers |
| 7 | 1.1428 | Int5-MLP + BigramHash | thwu1 | Mixed int5/int6, BigramHash (10240 buckets), SWA |
| 8 | 1.1458 | Int6 MLP3x + SmearGate + BigramHash | Raahil Shah | int6, 3× MLP, SmearGate, BigramHash (4096), OrthoInit, SWA |
| 9 | 1.1502 | MLP3x + Int6 QAT | aruniyer | 11 layers, 3× MLP, int6 QAT, sliding window eval |
| 10 | 1.1570 | Ternary Quantization | Ciprian Ifrim | BitNet b1.58 ternary (1.6 bits/param), 73.7M params in 15.92MB |

**Baseline** (rank 27): 1.2244 BPB. **Total improvement: 0.1097 BPB** over 8 days.

---

## Techniques by Category

### 1. Quantization

Quantization is the highest-leverage technique in this competition — it determines how many model parameters fit in 16MB, which then enables every other architectural improvement.

#### Int6 Per-Row Quantization
- Reduces weights to 6 bits (64 levels) per row, versus 8 bits (256 levels) in the baseline.
- Embeddings stay int8 (quantization-sensitive); all other weights use int6.
- Saves ~2MB of artifact space vs. int8, which can be reinvested into a wider MLP (2× → 3×) or an additional transformer layer.
- **BPB gap vs. fp16**: ~0.002 (vs. ~0.007 for uniform int8).
- Pioneered by aquariouseworkman; adopted by the majority of top-10 submissions.

#### Late STE QAT (Straight-Through Estimator)
- During training, simulate quantization in the forward pass; use full-precision gradients in the backward pass.
- "Late" variant: apply STE only in the final ~10% of training steps to avoid noisy early gradients.
- Reduces quantization gap by ~50%, contributing −0.0001 to −0.0005 BPB.
- Used by signalrush (#3), jfprincz (#4, #5), abaybektursun (#1, #2).

#### GPTQ-Lite (Post-Training)
- After training ends, search over 5 clip percentiles per row (0.999, 0.9995, 0.9999, 0.99999, 1.0).
- Pick the clip level that minimizes reconstruction MSE for each row independently.
- Zero training cost; applied in ~seconds at eval time.
- **Improvement**: −0.0006 BPB.
- Developed by signalrush (PR #374); used in top-3 submissions.

#### Full Hessian GPTQ (SOTA technique)
- Upgrades GPTQ from diagonal Hessian approximation to full Hessian with Cholesky error compensation.
- **Calibration data problem**: standard GPTQ uses external reference data, which is forbidden by rules. Solution: use the model itself to auto-regressively generate 64 sequences × 2048 tokens (temp=0.8) as calibration data.
- Strictly better than clip-search GPTQ; estimated −0.001 BPB over GPTQ-lite.
- Developed by abaybektursun; used in SOTA submission (#1) — enabled dropping TTT while still improving score.

#### Ternary / Binary Quantization (Extreme)
- **Ternary** (Ciprian Ifrim): BitNet b1.58 — weights ∈ {−1, 0, +1}, ~1.6 bits/param.
  - Enables 73.7M parameters in 15.92MB; uses 8192-token BPE, 768d, 15 layers.
  - Requires NeoMuon (3 Newton-Schulz steps); per-group scaling (group_size=128).
  - LZMA preset=9 compression (39% reduction over int8+zlib for ternary-packed data).
  - Score: 1.1570 BPB (timed track, rank 10).
- **Binary** (unlimited compute track): weights ∈ {−1, +1}, 1 bit/param.
  - 106M parameters; 2.15 hours training; score 1.1239 BPB.

#### Compression Algorithm
- Baseline: zlib level 9 (deflate).
- **zstd level 22**: ~5% better compression on int6 data; ~22 seconds slower evaluation. Used by signalrush and others.
- **LZMA**: Best for ternary/binary packed data; used exclusively in extreme quantization submissions.

---

### 2. Architecture

#### MLP Width Expansion (2× → 3×)
- Widening the feedforward hidden dimension from 1024 to 1536 (3×) is the most consistent free improvement once int6 quantization frees up ~2MB of space.
- **Cost**: ~1.5M extra parameters (fits within 16MB when quantized to int6).
- **Improvement**: −0.029 BPB.
- Used by rank #5, #8, #9, and others.

#### Model Depth (9 → 10 → 11 Layers)
- Adding one layer at int6 precision costs ~1.6MB uncompressed, which fits in budget.
- 11 layers fits with 3× MLP + int6, though deeper models need lower LR and careful init.
- Tradeoff: deeper → slower per-step, fewer total steps in 10 minutes. Empirically 11L with 3× MLP wins at 10-minute budget.

#### Exclusive Self Attention (XSA)
- Modification of standard attention: subtract the self-token contribution from each output.
  - Formula: `y' = y − (y · normalized(v)) · normalized(v)`
- Encourages tokens to attend to *other* positions rather than trivially copying themselves.
- arXiv: 2603.09078 (unnir).
- **Efficient implementation**: GQA-aware reshape + broadcast instead of `repeat_interleave` (reduces overhead from 7ms → 2ms/step).
- **Partial XSA** (last 3–4 layers): −0.002 BPB at <2ms cost.
- **Full XSA** (all 11 layers): −0.005 BPB additional (SOTA).

#### BigramHash Embedding
- Hash adjacent token pairs: `(prev_token × 31 + curr_token) % hash_size`.
- Lookup 128-dim embeddings for the hash → project to model dim (512d).
- Sizes: 1536 → 2048 → 3072 → 4096 → 10240 buckets tried.
- **Improvement**: −0.005 BPB. Adds ~500K–800K parameters.
- Used by rank #7, #8, and higher.

#### SmearGate
- Learned gate that blends each token's embedding with the previous token's embedding.
- Formula: `embed_t = gate × embed_t + (1 − gate) × embed_{t−1}`
- Lightweight bigram-level context; adds ~512 parameters total.
- PR #65 (aquariouseworkman). Used by rank #2, #8, ternary submissions.

#### Partial RoPE
- Apply rotational position encoding to only 16 of 64 head dimensions.
- Remaining 48 dims attend without positional bias → can learn position-invariant patterns.
- Zero new parameters; **improvement**: −0.0023 BPB.
- Developed by jfprincz (rank #4).

#### LayerNorm Depth Scaling
- Scale each RMSNorm output by `1/sqrt(layer_idx + 1)`.
- Dampens deeper layers' contributions; stabilizes training of 11-layer models.
- Zero new parameters; **improvement**: −0.0023 BPB.
- Developed by jfprincz (rank #4).

#### U-Net Skip Connections
- Encoder–decoder structure: 5 encoder layers + 6 decoder layers.
- Learned skip weight matrices (ones-initialized) from corresponding encoder to decoder layers.
- Per-block residual mixing from the input embedding.
- Used in ternary/binary submissions; concept explored in non-record track.

#### Activation Function
- **relu²** (baseline for most submissions): squaring the ReLU output — simple, fast, better than standard ReLU (−0.024 BPB).
- **LeakyReLU(0.5)²** (abaybektursun, PR #493): `leaky_relu(x, 0.5).square()`
  - Preserves negative gradient flow through MLP; eliminates dead neurons.
  - Additional **−0.003 BPB** over relu².
  - Used in rank #1, #2.

#### Factored Tied Embedding (Ternary/Binary)
- 8192×254 bottleneck with learned 254→768 and 768→254 projections.
- Frees ~4MB for wider MLP vs. standard full-rank tied embedding.
- Only used when vocab size is 8192 (ternary/binary submissions).

---

### 3. Training

#### Muon Optimizer
- Matrix parameter updates use Newton-Schulz orthogonalization (NeoMuon: 3 NS steps).
- Momentum: 0.99, warmup from 0.92 over 1500 steps; weight decay: 0.04.
- Scalar/embedding parameters use AdamW (lr=0.025–0.035, wd=0.01–0.04).
- Used universally across all competitive submissions.

#### Parameter Banking + Parallel Muon (PR #399)
- Pack all 66 weight matrices into 4 contiguous 3D `nn.Parameter` banks.
- Enables batched Newton-Schulz via `torch.bmm` — avoids per-matrix Python overhead.
- DDP communication: async reduce-scatter → local NS → async all-gather (overlaps compute).
- **Speedup**: −1.7ms/step (~70 free training steps in 10 minutes).
- Developed by abaybektursun; used in rank #1, #2.

#### Orthogonal Initialization
- All large weight matrices initialized with `orthogonal_(gain=1.0)`.
- Output projections scaled by `1/sqrt(2 × num_layers)` (muP convention).
- Accelerates early training convergence.
- Used by rank #8 and most competitive submissions from rank #5 onward.

#### EMA (Exponential Moving Average)
- Maintain a running EMA of weights with decay=0.997, applied every step.
- At eval time, use the EMA weights instead of the live weights.
- **Improvement**: ~−0.0006 BPB.
- Used by rank #3, #4, #5.

#### Tight SWA (Stochastic Weight Averaging)
- During warmdown (when LR scale < 0.2–0.5), save a checkpoint every 50–120 steps.
- Uniformly average 13–30 checkpoints.
- **Improvement**: −0.0005 to −0.0010 BPB.
- Combined with EMA (they stack orthogonally): −0.0011 BPB total.
- Used by rank #7, #8.

#### Warmdown Length
- Cosine LR decay from peak to ~10% over final N iterations.
- Longer warmdown → better convergence: each +500 iters ≈ −0.0002 to −0.0005 BPB.
- Range across submissions: 1500 → 2000 → 3000 → 3500 → 4000 iters.
- signalrush (rank #3) used 3500-iter warmdown as a key improvement.

#### Gradient Clipping
- Norm clip at 0.3 (prevents divergence in deep/wide models).
- Consistent across all submissions.

#### FlashAttention-3
- Hopper-optimized FA3 kernel via `flash_attn_func` (requires H100/H200 + PyTorch 2.9.1).
- **Speedup**: −9% step time (~380 free training steps in 10 minutes).
- Used by unnir (rank #6) and ternary/binary submissions.

---

### 4. Evaluation

#### Sliding Window Evaluation (stride=64)
- **Problem with naive chunking**: first token in each 1024-token chunk has ~0 context; average effective context is ~512 tokens.
- **Fix**: overlapping 1024-token windows advancing by 64 tokens; only the rightmost 64 tokens in each window are scored.
- Every token is scored exactly once, with 960+ tokens of prior context.
- **Improvement**: −0.032 BPB (the single largest jump in the competition).
- Developed by Matthew Li (PR #461); adopted universally from rank #6 onward.
- Cost: 4.3× slower evaluation (~70s vs. ~16s); acceptable within 10-minute budget.

#### Test-Time Training (TTT)

**LoRA TTT** (samacqua, PR #548):
- Per-document LoRA adaptation (rank-8 on `lm_head`, `c_q`, `c_v`).
- Split documents into overlapping chunks (chunk_size=256 within 1024 context).
- One Adam step per chunk (lr=0.01); reset LoRA between documents.
- **Improvement**: −0.0368 BPB (large, but slow eval).

**Score-First Legal TTT** (abaybektursun, PR #461):
- Backward-looking protocol to avoid data leakage: score chunk N under inference mode, then adapt on chunks 0..N−1.
- SGD (lr=0.002, momentum=0.9, 3 epochs per chunk); cosine LR decay.
- Timing: 600s train + 120s standard eval + 410s TTT ≈ 530s total (< 10 min).
- **Improvement**: −0.0025 BPB.
- Used in rank #2 (1.1194). Dropped in SOTA rank #1 because Full Hessian GPTQ compensates without TTT overhead.

#### Temperature Scaling
- Grid search over 5 temperatures {0.80, 0.85, 0.90, 0.95, 1.00} on a held-out calibration set.
- Optimal: 0.90 for ternary/binary models (model is slightly underconfident due to extreme quantization).
- Adds ~0.0002–0.0005 BPB improvement for models far from fp16 quality.

---

## Frequency Analysis

How many of the **top 10 submissions** used each technique:

| Technique | Top-10 Count | Notes |
|-----------|:---:|-------|
| Int6 quantization | 9/10 | Universal except ternary submission |
| Sliding window eval (stride=64) | 9/10 | Adopted after PR #461; baseline didn't use it |
| Muon optimizer | 10/10 | Universal |
| relu² or LeakyReLU² activation | 10/10 | All use squared activation; top-2 use Leaky variant |
| 10–11 layer depth | 8/10 | Baseline is 9L; extra layers enabled by int6 savings |
| 3× MLP width | 7/10 | Int6 frees space; most top submissions use it |
| OrthoInit | 7/10 | Adopted by rank #5 onward |
| EMA or SWA weight averaging | 7/10 | EMA preferred; SWA also used |
| XSA (Exclusive Self Attention) | 6/10 | Partial or full; top-6 all use it |
| BigramHash embedding | 5/10 | 1536–10240 buckets; top-7 and #8 use it |
| SmearGate | 5/10 | Lightweight, often paired with BigramHash |
| Longer warmdown (3000+ iters) | 5/10 | Low-cost improvement, widely adopted |
| GPTQ (lite or full Hessian) | 4/10 | Top-3 plus ternary PTQ |
| zstd compression | 4/10 | Replaces zlib for ~5% smaller artifacts |
| Partial RoPE | 2/10 | jfprincz submissions (#4, #5) |
| LayerNorm depth scaling | 2/10 | jfprincz submissions (#4, #5) |
| LeakyReLU(0.5)² | 2/10 | abaybektursun (#1, #2) |
| Parameter Banking + Parallel Muon | 2/10 | abaybektursun (#1, #2) |
| FlashAttention-3 | 3/10 | unnir + ternary/binary |
| Test-Time Training | 2/10 | rank #2 (Legal TTT), LoRA TTT earlier |
| Temperature calibration | 2/10 | Ternary + binary |

**Most widely adopted** (6+ of top 10): int6 quantization, sliding window eval, Muon, squared activation, 3× MLP, deeper models.

**Highest leverage per unit effort**:
1. Sliding window eval (−0.032 BPB, ~50 lines of code, zero training cost)
2. Int6 quantization (−0.007 BPB + unlocks MLP/depth expansion)
3. 3× MLP expansion (−0.029 BPB, 1-line change once int6 is in)
4. XSA all layers (−0.005 BPB, efficient implementation, no new params)

---

## Technique Progression (SOTA Timeline)

| Date | Score | Innovation That Drove Jump |
|------|-------|---------------------------|
| 2026-03-17 | 1.2244 | **Baseline**: 9L, 512d, int8, zlib, non-overlapping eval |
| 2026-03-18 | 1.2197 | FP16 embedding passthrough (−0.007 quantization gap) |
| 2026-03-18 | 1.2060 | 2048 sequence length (+context at cost of fewer steps) |
| 2026-03-19 | 1.1925 | **Sliding window eval** (stride=64) → −0.032 BPB jump |
| 2026-03-19 | 1.1748 | Muon + 10 layers + SmearGate |
| 2026-03-20 | 1.1458 | Int6 + 3× MLP + SmearGate + BigramHash + SWA |
| 2026-03-20 | 1.1428 | Mixed int5/int6 |
| 2026-03-21 | 1.1307 | Efficient Partial XSA (GQA-aware reshape) |
| 2026-03-21 | 1.1271 | XSA4 + EMA + 3× MLP combined |
| 2026-03-21 | 1.1248 | Partial RoPE + LayerNorm depth scaling + EMA |
| 2026-03-22 | 1.1228 | GPTQ-lite clip search + EMA + 3500-iter warmdown |
| 2026-03-23 | 1.1194 | LeakyReLU(0.5)² + Legal TTT + Parallel Muon |
| 2026-03-25 | **1.1147** | **Full Hessian GPTQ** (AR self-gen calibration) + XSA all layers |

Key observations:
- **Sliding window eval** was the single biggest jump (−0.032 BPB) and came from a pure evaluation change, not a model change.
- **Int6 quantization** was an enabler more than a direct improvement — it freed budget for MLP expansion and additional layers.
- After rank #6, improvements became incremental (0.001–0.003 BPB each), requiring stacking of many small gains.
- The SOTA (#1) achieved its score by (a) replacing TTT with a better PTQ method and (b) applying XSA to all layers.

---

## Unexplored Techniques (Opportunity Areas)

The competition README listed these as suggested but not yet implemented as of the last SOTA:

| Technique | Why It Might Help |
|-----------|-------------------|
| JEPA (Joint Embedding Predictive Architecture) | Predicting in latent space could capture richer representations |
| Text diffusion models | Bidirectional context could improve compression on the forward pass |
| H-net tokenization | Hierarchical tokenization allows variable granularity |
| Universal Transformer (depth recurrence) | Parameter sharing across layers → more capacity per MB |
| E2E Test-Time Training | End-to-end TTT rather than LoRA adaptation |
| State-space models (Mamba, etc.) | Linear-time sequence modeling; may fit better in 16MB |
| Megakernels (fused ops) | More training steps via reduced CUDA overhead |
| Learning adapters on random linear maps | Novel parameterization for low-bit efficiency |

---

## Implementation Notes

**Key files**:
- [parameter_golf/train_gpt.py](train_gpt.py) — baseline training script; all submissions modify this
- [parameter_golf/README.md](README.md) — full competition rules and submission format
- `parameter_golf/records/track_10min_16mb/*/README.md` — per-submission technique write-ups
- `parameter_golf/records/track_10min_16mb/*/submission.json` — scores and metadata

**Submitting**: each submission needs `train_gpt.py`, `README.md`, `submission.json`, and 3+ training logs (`train_v1.txt`, `train_v2.txt`, `train_v3.txt`). The `submission.json` must include `bpb`, `author`, `date`, and `description`.

**Reproducibility constraint**: must achieve within p < 0.01 statistical significance across 3 seeds.
