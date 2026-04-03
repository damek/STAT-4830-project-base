"""
Base configuration from the current #1 leaderboard entry (1.1147 BPB).
Record: 2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072

All keys correspond to env vars read by train_gpt.py.

Two configs here:
  SOTA_CONFIG  — full config for the SOTA script (full eval, 8xH100)
  PROXY_CONFIG — stripped config for the BASE script (proxy eval, A10G)
                 Base script doesn't have BigramHash/XSA/VE, so those are omitted.

TUNABLE_PARAMS defines the search space for Ax BO.
Only params present in BOTH scripts are tunable (so proxy evals are meaningful).
"""
from typing import Any

# ── SOTA config (used for full eval) ────────────────────────────────────────
SOTA_CONFIG: dict[str, Any] = {
    # Model shape
    "NUM_LAYERS": 11,
    "NUM_KV_HEADS": 4,
    "MODEL_DIM": 512,
    "NUM_HEADS": 8,
    "MLP_MULT": 3.0,
    "ROPE_BASE": 10000.0,
    "LOGIT_SOFTCAP": 30.0,
    "ROPE_DIMS": 16,
    "LN_SCALE": 1,
    "TIE_EMBEDDINGS": 1,
    # BigramHash (SOTA-only)
    "BIGRAM_VOCAB_SIZE": 3072,
    "BIGRAM_DIM": 112,
    # XSA on all layers (SOTA-only)
    "XSA_LAST_N": 11,
    # Value embeddings on layers 9-10 (SOTA-only)
    "VE_ENABLED": 1,
    "VE_DIM": 128,
    "VE_LAYERS": "9,10",
    # Training schedule
    "ITERATIONS": 20000,
    "WARMDOWN_ITERS": 4000,
    "WARMUP_STEPS": 20,
    "TRAIN_BATCH_TOKENS": 786432,
    "TRAIN_SEQ_LEN": 2048,
    "EVAL_SEQ_LEN": 2048,
    "EVAL_STRIDE": 64,
    "MAX_WALLCLOCK_SECONDS": 600.0,
    # Optimizer
    "MATRIX_LR": 0.025,
    "SCALAR_LR": 0.025,
    "EMBED_LR": 0.6,
    "HEAD_LR": 0.008,
    "TIED_EMBED_LR": 0.035,
    "TIED_EMBED_INIT_STD": 0.005,
    "MUON_MOMENTUM": 0.99,
    "MUON_BACKEND_STEPS": 5,
    "MUON_MOMENTUM_WARMUP_START": 0.92,
    "MUON_MOMENTUM_WARMUP_STEPS": 1500,
    "MUON_WD": 0.04,
    "BETA1": 0.9,
    "BETA2": 0.95,
    "ADAM_EPS": 1e-8,
    "ADAM_WD": 0.04,
    "GRAD_CLIP_NORM": 0.3,
    "QK_GAIN_INIT": 1.5,
    # QAT
    "LATE_QAT_THRESHOLD": 0.15,
    # Model averaging
    "SWA_ENABLED": 1,
    "SWA_EVERY": 50,
    # GPTQ calibration
    "GPTQ_CALIB_BATCHES": 64,
    "GPTQ_BLOCK_SIZE": 128,
}

# ── Proxy config (used for cheap A10G evals with base train_gpt.py) ──────────
# Only includes params the base script understands.
# Smaller batch/seq for faster steps on 1 GPU.
PROXY_CONFIG: dict[str, Any] = {
    "NUM_LAYERS": 11,
    "NUM_KV_HEADS": 4,
    "MODEL_DIM": 512,
    "NUM_HEADS": 8,
    "MLP_MULT": 2,           # base script uses int MLP_MULT
    "ROPE_BASE": 10000.0,
    "LOGIT_SOFTCAP": 30.0,
    "TIE_EMBEDDINGS": 1,
    "VOCAB_SIZE": 1024,
    # Training — smaller for proxy speed
    "ITERATIONS": 20000,
    "WARMDOWN_ITERS": 4000,
    "WARMUP_STEPS": 20,
    "TRAIN_BATCH_TOKENS": 131072,
    "TRAIN_SEQ_LEN": 1024,
    "MAX_WALLCLOCK_SECONDS": 300.0,
    "VAL_LOSS_EVERY": 10,
    # Optimizer (same as SOTA — these are what we tune)
    "MATRIX_LR": 0.025,
    "SCALAR_LR": 0.025,
    "EMBED_LR": 0.6,
    "HEAD_LR": 0.008,
    "TIED_EMBED_LR": 0.035,
    "TIED_EMBED_INIT_STD": 0.005,
    "MUON_MOMENTUM": 0.99,
    "MUON_BACKEND_STEPS": 5,
    "MUON_MOMENTUM_WARMUP_START": 0.92,
    "MUON_MOMENTUM_WARMUP_STEPS": 1500,
    "MUON_WD": 0.04,
    "BETA1": 0.9,
    "BETA2": 0.95,
    "ADAM_EPS": 1e-8,
    "ADAM_WD": 0.04,
    "GRAD_CLIP_NORM": 0.3,
    "QK_GAIN_INIT": 1.5,
}

# ── Tunable parameter definitions for Ax ────────────────────────────────────
# Only params present in PROXY_CONFIG (so proxy evals are valid signals).
# Ranges are deliberately tight — we're doing local search around SOTA.
TUNABLE_PARAMS: dict[str, dict] = {
    "MATRIX_LR":    {"type": "range", "bounds": [0.015, 0.04],   "value_type": "float"},
    "SCALAR_LR":    {"type": "range", "bounds": [0.015, 0.04],   "value_type": "float"},
    "MUON_MOMENTUM":{"type": "range", "bounds": [0.97,  0.995],  "value_type": "float"},
    "MUON_WD":      {"type": "range", "bounds": [0.02,  0.08],   "value_type": "float"},
    "ADAM_WD":      {"type": "range", "bounds": [0.02,  0.08],   "value_type": "float"},
    "WARMDOWN_ITERS":{"type": "range","bounds": [3000,  5500],   "value_type": "int"},
    "ROPE_BASE":    {"type": "range", "bounds": [5000.0, 50000.0],"value_type": "float", "log_scale": True},
    "LOGIT_SOFTCAP":{"type": "range", "bounds": [20.0,  50.0],   "value_type": "float"},
    "QK_GAIN_INIT": {"type": "range", "bounds": [1.0,   2.5],    "value_type": "float"},
    "GRAD_CLIP_NORM":{"type": "range","bounds": [0.1,   0.5],    "value_type": "float"},
    "EMBED_LR":     {"type": "range", "bounds": [0.3,   1.2],    "value_type": "float"},
    "HEAD_LR":      {"type": "range", "bounds": [0.004, 0.02],   "value_type": "float"},
    "MUON_MOMENTUM_WARMUP_STEPS": {"type": "range", "bounds": [500, 3000], "value_type": "int"},
}

# Starting BPB from the current #1 record (3-seed mean)
SOTA_BPB = 1.1147
