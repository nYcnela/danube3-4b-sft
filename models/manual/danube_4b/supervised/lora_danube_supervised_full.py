#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O-Danube3-4B - SUPERVISED (SFT) Training with LoRA (NO quantization)

Model: h2oai/h2o-danube3-4b-chat (Instruct version)
Data:  ../model_v12.0/10_2training_prompts (shared with Mistral v12.0)
Output: outputs/model_danube_supervised_full/

Identical to qlora_danube_supervised.py but WITHOUT 4-bit quantization.
Model is loaded in full precision (bf16/fp16) - no BitsAndBytesConfig.
Requires more VRAM (~8-10 GB for 4B in bf16) but avoids quantization noise.
"""

import sys
import warnings
from pathlib import Path

import torch
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

warnings.filterwarnings("ignore", message=".*use_reentrant.*")

SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = SCRIPT_DIR.parents[2]
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from common.training_utils import (  # noqa: E402
    DEVICE, DTYPE, IS_DARWIN,
    safe_tokenizer, load_model_for_training,
    load_bucket_dataset_supervised,
)

MODEL_ID = "h2oai/h2o-danube3-4b-chat"  # Instruct version

DATA_ROOT = SCRIPT_DIR.parents[3] / "data" / "json" / "manual" / "supervised" / "training_prompts"
OUT_DIR = SCRIPT_DIR.parents[3] / "outputs" / "manual" / "danube_4b" / "model_danube_supervised_full"

SAFE_CTX = 8192

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Training hyperparams - identical to QLoRA version
NUM_EPOCHS = 3
BATCH_SIZE = 32
GRAD_ACC_STEPS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 1536

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# ==================================


def formatting_func_for_danube(tokenizer):
    """
    Create formatting function for Danube chat template.
    Danube uses Mistral-like format with [INST] tags or chat template.
    """
    def _format(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            user_content = f"{inst}\n\n{inp}" if inp else inst
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": out},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                text = f"[INST] {user_content} [/INST] {out}"
            texts.append(text)
        return texts
    return _format


def main():
    print("=" * 70)
    print("H2O-DANUBE3-4B - SUPERVISED TRAINING (LoRA, NO QUANTIZATION)")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Data:  {DATA_ROOT}")
    print(f"Output: {OUT_DIR}")
    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print("[INFO] Quantization: DISABLED (full precision)")
    print("=" * 70)

    # Tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = safe_tokenizer(MODEL_ID, SAFE_CTX)
    print(f"[OK] Tokenizer loaded. pad_token={tokenizer.pad_token}")

    # Dataset
    print("\n[2/5] Loading datasets...")
    train_folder = DATA_ROOT / "train"
    val_folder = DATA_ROOT / "val"

    train_ds = load_bucket_dataset_supervised(train_folder)
    val_ds = load_bucket_dataset_supervised(val_folder)

    if train_ds is None:
        print(f"[ERROR] No training data found in {train_folder}")
        sys.exit(1)

    print(f"[OK] Train: {len(train_ds)} samples")
    if val_ds:
        print(f"[OK] Val: {len(val_ds)} samples")
    else:
        print("[WARN] No validation data, will skip eval")

    # Model (no quantization)
    print("\n[3/5] Loading model (full precision, no quantization)...")
    model = load_model_for_training(MODEL_ID, use_4bit=False)
    print(f"[OK] Base model loaded")

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    print("\n[4/5] Setting up training...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        save_total_limit=2,
        fp16=(DTYPE == torch.float16 and DEVICE.type == "cuda"),
        bf16=(DTYPE == torch.bfloat16 and DEVICE.type == "cuda"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=0 if IS_DARWIN else 2,
        remove_unused_columns=True,
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func_for_danube(tokenizer),
        max_seq_length=MAX_SEQ_LEN,
        tokenizer=tokenizer,
        packing=False,
    )

    # Train
    print("\n[5/5] Starting training...")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACC_STEPS}")
    trainer.train()

    # Save adapter
    adapter_path = OUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n[DONE] Adapter saved to {adapter_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
