#!/usr/bin/env python3
"""Modal SFT training for Qwen2.5-Coder on vector-db-bench rollouts.

Trains a QLoRA adapter on top of Qwen2.5-Coder-{7B,32B}-Instruct using
compile-clean worker (prompt, output) pairs produced by extract_sft_data.py.

Prerequisites:
  pip install modal
  modal token set ...

Usage:
  # First extract training data:
  python scripts/vector_db_bench/extract_sft_data.py --only-valid

  # Train 7B on A10G (~$1.10/hr, fits in ~16GB VRAM with QLoRA):
  modal run scripts/vector_db_bench/modal_sft_train.py \\
      --dataset-path data/vector_db_bench/sft_data.jsonl

  # Train 32B on A100-80GB (~$3.70/hr):
  modal run scripts/vector_db_bench/modal_sft_train.py \\
      --dataset-path data/vector_db_bench/sft_data.jsonl \\
      --model-id Qwen/Qwen2.5-Coder-32B-Instruct \\
      --gpu A100-80GB

  # Retrieve saved adapter:
  modal volume get vdb-sft-checkpoints <remote-path> <local-dest>

Checkpoints are saved to Modal Volume "vdb-sft-checkpoints" at:
  /<output-name>/   (LoRA adapter weights + tokenizer)
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "alpha-vdb-sft"
VOLUME_NAME = "vdb-sft-checkpoints"
CHECKPOINT_MOUNT = "/checkpoints"
DATASET_MOUNT = "/data/sft_data.jsonl"

# ---------------------------------------------------------------------------
# Image — PyTorch + Hugging Face training stack
# ---------------------------------------------------------------------------

sft_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.46.0",
        "trl>=0.12.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.1",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "scipy",
    )
)

app = modal.App(APP_NAME)
checkpoints_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=sft_image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={CHECKPOINT_MOUNT: checkpoints_volume},
    memory=32768,
)
def train_sft(
    dataset_jsonl: str,
    *,
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    max_seq_length: int = 8192,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    output_name: str = "qwen25-coder-vdb-sft",
) -> dict:
    import json
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # ---- dataset -----------------------------------------------------------
    records = [
        json.loads(line)
        for line in dataset_jsonl.splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("dataset_jsonl is empty — run extract_sft_data.py first")

    # Keep only the messages list; drop metadata (not needed during training).
    train_records = [{"messages": r["messages"]} for r in records]
    dataset = Dataset.from_list(train_records)
    print(f"[sft] loaded {len(dataset)} training examples", flush=True)

    # ---- tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_length  # TRL 1.x reads max length from tokenizer

    # ---- model (4-bit QLoRA) -----------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # swap to flash_attention_2 if installed
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ---- LoRA --------------------------------------------------------------
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- training args -----------------------------------------------------
    output_dir = f"{CHECKPOINT_MOUNT}/{output_name}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        dataset_num_proc=2,
        report_to="none",
        # Use the tokenizer's built-in chat template to format messages.
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"[sft] trainable params: ", flush=True)
    trainer.model.print_trainable_parameters()

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Flush volume so the adapter is visible outside this container.
    checkpoints_volume.commit()

    result = {
        "output_dir": output_dir,
        "model_id": model_id,
        "num_examples": len(dataset),
        "lora_rank": lora_rank,
        "epochs": epochs,
    }
    print(f"[sft] done: {result}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    dataset_path: str,
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    gpu: str = "A10G",
    lora_rank: int = 64,
    epochs: int = 3,
    max_seq_length: int = 8192,
    output_name: str = "qwen25-coder-vdb-sft",
) -> None:
    """
    Args:
        dataset_path:    Local path to sft_data.jsonl from extract_sft_data.py.
        model_id:        HuggingFace model ID (Qwen/Qwen2.5-Coder-{7B,32B}-Instruct).
        gpu:             Modal GPU type: A10G (7B default) or A100-80GB (for 32B).
        lora_rank:       LoRA rank; 64 is a good default.
        epochs:          Number of training epochs.
        max_seq_length:  Max token length per example (worker prompts can be long).
        output_name:     Subdirectory name inside the Modal volume for this run.
    """
    import json

    path = Path(dataset_path)
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Run: python scripts/vector_db_bench/extract_sft_data.py --only-valid"
        )

    dataset_jsonl = path.read_text(encoding="utf-8")
    num_lines = sum(1 for l in dataset_jsonl.splitlines() if l.strip())
    print(f"[main] uploading {num_lines} examples from {path}")

    valid_gpus = {"A10G", "A100-40GB", "A100-80GB", "H100"}
    if gpu not in valid_gpus:
        raise SystemExit(f"Unknown GPU type: {gpu}. Choose from: {sorted(valid_gpus)}")
    if gpu != "A10G":
        print(f"[main] NOTE: GPU override '{gpu}' requested but function is decorated with A10G. "
              "Edit GPU_TYPE in the decorator to change GPU at deploy time.")

    result = train_sft.remote(
        dataset_jsonl,
        model_id=model_id,
        lora_rank=lora_rank,
        epochs=epochs,
        max_seq_length=max_seq_length,
        output_name=output_name,
    )
    print(f"[main] training complete: {json.dumps(result, indent=2)}")
    print(f"\nRetrieve adapter with:")
    print(f"  modal volume get {VOLUME_NAME} {result['output_dir'].lstrip('/')} ./local-adapter/")
