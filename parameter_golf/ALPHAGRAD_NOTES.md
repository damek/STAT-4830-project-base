# AlphaGrad — Parameter Golf Notes

Team exploration of OpenAI's [Parameter Golf](https://openai.com/index/parameter-golf/) challenge (STAT 4830 project).

---

## Challenge Summary

Train the best language model (GPT-style) that:
- Fits inside **16 MB** (code + zlib-compressed weights combined)
- Trains in **under 10 minutes** on **8×H100 SXM GPUs**
- Evaluated by **Bits Per Byte (BPB)** on the FineWeb validation set — lower is better

BPB = `negative_log_likelihood / log(2) / 8`; equivalently, BPB = cross-entropy loss × log₂(e) / 8.

---

## Leaderboard (public, from `/records`)

| BPB | Date | Submission | Key techniques |
|-----|------|------------|----------------|
| **1.1147** | 2026-03-25 | AR Self-Gen GPTQ + XSA-all + BigramHash | Full Hessian GPTQ (AR calibration), XSA all 11L, BigramHash 3072×112, LZMA-9, Parallel Muon |
| 1.1194 | 2026-03-23 | LeakyReLU² + Legal TTT + Parallel Muon | LeakyReLU(0.5)², backward-looking TTT (SGD lr=0.002), Parallel Muon |
| 1.1228 | 2026-03-22 | GPTQ-lite + EMA + warmdown=3500 | Clip-percentile GPTQ, EMA decay=0.997, 3500-iter warmdown, QAT 15% |
| 1.1248 | 2026-03-21 | XSA4 + EMA + Partial RoPE + LateQAT | XSA on last 4L, EMA, Partial RoPE (16/64 dims), LN depth scaling |
| 1.1271 | 2026-03-21 | XSA4 + EMA + Int6 MLP3x | XSA 4L, EMA, int6, 3×MLP, zstd-22, FA3, SmearGate, BigramHash 2048, 11L |
| 1.1307 | 2026-03-20 | Efficient Partial XSA | GQA-aware XSA (3 deepest layers), FA3, SWA-120 |
| 1.1458 | 2026-03-20 | Int6 MLP3x + SmearGate + BigramHash | int6, 3×MLP, SmearGate, BigramHash 4096, OrthoInit, SWA |
| 1.1502 | 2026-03-20 | MLP3x + Int6 QAT + Sliding Window | 11L, 3×MLP, int6 QAT, sliding window eval, zstd-22 |
| 1.1574 | 2026-03-19 | Warmdown Quantization | Warmdown + quantization refinement |
| 1.1929 | 2026-03-19 | LoRA TTT | Baseline + per-document LoRA test-time training at eval |
| **1.2244** | 2026-03-17 | **Naive Baseline** | 9L, 512d, int8+zlib, standard chunked eval |

Full leaderboard in `/records/track_10min_16mb/*/submission.json`.

---

## Our Runs

### Run 1 — Short smoke test (A10G, 120s, 500 iterations)
- **Hardware**: 1×A10G via Modal (`modal run modal_exploration.py --iterations 500 --max-wallclock 120`)
- **val_bpb**: ~4.1 (expected; only ~2.5% of full training completed)
- **Artifact**: not produced (run crashed mid-way due to missing C compiler in image)
- **Status**: debugging run; confirmed Modal pipeline works

### Run 2 — Full baseline (8×H100, 600s)
- **Hardware**: 8×H100 SXM via Modal (`modal run modal_exploration.py`)
- **Architecture**: 9 layers, 512 dim, 2×MLP, 8 heads, 4 KV-heads, vocab 1024, tied embeddings
- **Quantization**: int8 per-row + zlib level 9
- **val_bpb at roundtrip**: not logged locally (Modal streams to terminal; logs not saved)
- **Artifact**: `output/final_model_latest.int8.ptz` — **10.888 MB** (well under 16 MB)
- **Expected val_bpb**: ~1.22–1.23 (matches public baseline; ~1.2244 is the published score for identical config)

### What We've Implemented (not yet run end-to-end)

Changes in `train_gpt_exploration.py` relative to the baseline:

| Change | Default | Notes |
|--------|---------|-------|
| `NUM_LAYERS` | 9 → **11** | Deeper model enabled by int6 space savings |
| `MLP_MULT` | 2 → **3** | 1024 → 1536 hidden; −0.029 BPB gain |
| Int6 quantization | int8 only | Non-embedding 2D weights → 6-bit packed (25% smaller) |
| Sliding window eval | chunked 1024 | Stride=64, score last 64 tokens; −0.032 BPB |
| XSA (partial) | none | Last 4 of 11 layers; −0.002 BPB |
| EMA | none | Decay=0.997; −0.0006 BPB |

**Projected model size** (11L, 3×MLP, int6): ~20 MB raw payload → ~14 MB after 30% zlib → under 16 MB.

**Projected val_bpb** if all changes work as in literature: ~1.19–1.21 BPB (vs. 1.2244 baseline).

---

## File Map

```
parameter_golf/
├── train_gpt_exploration.py   ← Our modified training script (all changes live here)
├── modal_exploration.py       ← Modal app: dispatches train_gpt_exploration.py to 8×H100
├── output/
│   └── final_model_latest.int8.ptz  ← Artifact from Run 2 (10.888 MB)
├── FINDINGS.md                ← Technique analysis: BPB estimates, leaderboard context
├── Overview.md                ← Medium-level challenge overview
├── alphagrad/
│   └── modal_runner.py        ← Downloads FineWeb data into Modal volume (run once)
└── records/
    └── track_10min_16mb/      ← Public submission records with scores + technique write-ups
```

---

## Rerunning Everything from Scratch

### Prerequisites

**Python 3.11** (required; 3.14 has asyncio deprecation issues with Modal CLI)

```powershell
# Windows — install Python 3.11 via winget
winget install Python.Python.3.11

# Verify
py -3.11 --version
```

**Create a virtual environment**

```powershell
cd parameter_golf
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install modal sentencepiece numpy
```

**Authenticate with Modal** (one-time; tokens persist in `~/.modal/`)

```powershell
modal token set --token-id <your-modal-token-id> --token-secret <your-modal-token-secret>
```

Get tokens from: https://modal.com/settings → API tokens.

### Step 1 — Download FineWeb data into Modal volume (one-time)

This fills the `pg-fineweb-data` Modal Volume with the tokenised dataset.  
Takes ~10–20 minutes and ~15 GB of Modal storage; only needed once per Modal account.

```powershell
cd parameter_golf
modal run alphagrad/modal_runner.py
```

### Step 2 — Run training

**Smoke test (30 seconds, cheap)**
```powershell
modal run modal_exploration.py --max-wallclock 30
```

**Full baseline run (10 minutes, 8×H100)**
```powershell
modal run modal_exploration.py
```

**Full run with explicit settings**
```powershell
modal run modal_exploration.py --iterations 20000 --max-wallclock 600
```

The script streams all training logs to your terminal and saves the artifact to `output/final_model_<run_id>.int8.ptz`.

### Step 3 — Interpreting output

Key log lines to look for:

```
step:1000/20000 val_loss:X.XXXX val_bpb:X.XXXX train_time:XXXXXms ...
stopping_early: wallclock_cap train_time:600000ms step:NNNN/20000
Serialized model int8+zlib: XXXXXXX bytes (payload:... raw_torch:... payload_ratio:Xx)
Total submission size int8+zlib: XXXXXXX bytes
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
```

The **`final_int8_zlib_roundtrip val_bpb`** line is the number that counts — it measures BPB after quantize → compress → decompress → eval, matching what the competition judges.

---

## Environment Variables (Hyperparameter Overrides)

All hyperparameters can be overridden via environment variables. Set them in the `config` dict inside `modal_exploration.py::main()`, or pass them via `os.environ` in `run_exploration`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `NUM_LAYERS` | 11 | Transformer depth |
| `MODEL_DIM` | 512 | Hidden dimension |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_KV_HEADS` | 4 | Key/Value heads (GQA) |
| `MLP_MULT` | 3 | MLP hidden = MLP_MULT × MODEL_DIM |
| `VOCAB_SIZE` | 1024 | Tokenizer vocabulary size |
| `ITERATIONS` | 20000 | Training steps |
| `MAX_WALLCLOCK_SECONDS` | 600 | Hard stop for training (seconds) |
| `WARMDOWN_ITERS` | 1200 | LR warmdown steps |
| `WARMUP_STEPS` | 20 | Compile-warmup steps before training timer starts |
| `TRAIN_BATCH_TOKENS` | 524288 | Global tokens per step (512K) |
| `TRAIN_SEQ_LEN` | 1024 | Sequence length |
| `SWA_STRIDE` | 64 | Sliding-window eval stride (64 = standard) |
| `XSA_START_LAYER` | 7 | Apply XSA to layers ≥ this index (7 = last 4 of 11) |
| `EMA_DECAY` | 0.997 | EMA shadow weight decay (0 = disabled) |
| `MATRIX_LR` | 0.04 | Muon LR for weight matrices |
| `SCALAR_LR` | 0.04 | Adam LR for scalars/biases |
| `EMBED_LR` | 0.6 | Adam LR for embeddings |
| `MUON_MOMENTUM` | 0.95 | Muon final momentum |
| `GRAD_CLIP_NORM` | 0.0 | Gradient clip norm (0 = disabled) |
| `SEED` | 1337 | RNG seed |
| `VAL_LOSS_EVERY` | 1000 | Validate every N steps |

**Example: run with full XSA on all layers**
```powershell
# Set XSA_START_LAYER=0 to apply to all 11 layers
# Set in modal_exploration.py's config dict, or modify the script's env passthrough
```

---

## Size Budget Math

With the current config (11L, 512d, 3×MLP, int6 non-embedding, int8 embedding):

| Component | Params | Int6 bytes | Int8 bytes |
|-----------|--------|------------|------------|
| `tok_emb` [1024, 512] | 524,288 | — (int8) | 526,336 |
| Skip weights [5, 512] | 2,560 | — (fp32) | 10,240 |
| Per block: attn (c_q/k/v/proj) | 786,432 | 589,824 + scales | — |
| Per block: MLP 3× (fc + proj) | 1,572,864 | 1,179,648 + scales | — |
| Per block: scalars | 2,056 | fp32 passthrough | — |
| **11 blocks total** | **25,971,072** | — | — |

**Total raw int6 payload ≈ 20 MB → after zlib-9 (~30% compression) → ~14 MB → under 16 MB ✓**

Published benchmark (jfprincz, rank #5, same basic config): **15.53 MB** total artifact.

---

## Key Technique Reference

| Technique | BPB gain | Effort | Notes |
|-----------|----------|--------|-------|
| Sliding window eval (stride=64) | −0.032 | Low | Zero retraining; biggest single jump |
| 3× MLP width | −0.029 | 1 line | Requires int6 for budget |
| Int6 quantization | −0.007 + unlocks above | Medium | Frees ~2 MB vs int8 |
| XSA all layers | −0.005 | Low | GQA-aware broadcast |
| XSA partial (last 3–4L) | −0.002 | Low | Cheaper version |
| EMA (decay=0.997) | −0.0006 | Low | Stackable |
| Longer warmdown (3000+ iters) | −0.0002–0.0005 | Zero | Set `WARMDOWN_ITERS` |
| GPTQ-lite (clip-search) | −0.0006 | Medium | Post-training; no retraining |
| LeakyReLU(0.5)² | −0.003 | 1 line | Replace `relu` in `MLP.forward` |
| BigramHash embedding | −0.005 | Medium | 500K extra params |

See `FINDINGS.md` for full analysis with citations.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `modal.gpu has no attribute H100` | Old Modal version uses string GPU spec | Use `gpu="h100:8"` not `modal.gpu.H100(count=8)` |
| `Failed to find C compiler (CC)` | `pytorch:...-runtime` image lacks gcc | `exploration_image.apt_install("gcc")` in Modal image |
| `No artifact produced` | `subprocess.TimeoutExpired` killed before serialization | Increase `proc_timeout` (now: wallclock + 600s) |
| `val_bpb: 4.1+` | Too few training steps | Normal for short tests; need ≥5000 steps for meaningful BPB |
| `asyncio deprecation warning` | Python ≥ 3.12 + Modal CLI | Use Python 3.11 venv |
| `Got unexpected extra arguments` | Old Modal `local_entrypoint` style | Use typed params in `main()`, no `argparse` |
