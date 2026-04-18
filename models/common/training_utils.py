#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for QLoRA/LoRA fine-tuning scripts.

Import in each training script:
    MODELS_DIR = Path(__file__).resolve().parents[2]
    import sys; sys.path.insert(0, str(MODELS_DIR))
    from common.training_utils import (
        DEVICE, DTYPE, IS_DARWIN,
        safe_tokenizer, load_model_for_training,
        load_bucket_dataset_supervised, load_bucket_dataset_unsupervised,
    )
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Generator

import torch
from datasets import Dataset
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device | None = None) -> torch.dtype:
    if device is None:
        device = pick_device()
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


DEVICE: torch.device = pick_device()
IS_DARWIN: bool = platform.system() == "Darwin"
DTYPE: torch.dtype = pick_dtype(DEVICE)


def safe_tokenizer(model_id: str, safe_ctx: int, trust_remote_code: bool = False):
    """Load tokenizer with fast→slow fallback, sets pad token and max length."""
    kwargs: dict = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)
    except Exception as e:
        print(f"[info] Fast tokenizer failed ({e}), using slow", file=sys.stderr)
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = safe_ctx
    tok.init_kwargs["model_max_length"] = safe_ctx
    return tok


def make_bnb_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    """Build 4-bit NF4 BitsAndBytesConfig."""
    compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_for_training(
    model_id: str,
    use_4bit: bool = True,
    trust_remote_code: bool = False,
    extra_kwargs: dict | None = None,
):
    """Load a causal LM for QLoRA/LoRA training.

    Uses 4-bit quantization when CUDA is available, falls back to full
    precision on MPS/CPU.

    Args:
        model_id: HuggingFace model ID.
        use_4bit: Whether to use 4-bit quantization (requires CUDA).
        trust_remote_code: Pass trust_remote_code=True to from_pretrained (needed for Qwen).
        extra_kwargs: Additional kwargs forwarded to from_pretrained
                      (e.g. {"attn_implementation": "eager"} for Gemma 2).
    """
    kwargs: dict = extra_kwargs.copy() if extra_kwargs else {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    can_4bit = use_4bit and DEVICE.type == "cuda"

    if can_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=make_bnb_config(DTYPE),
            torch_dtype=DTYPE,
            **kwargs,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            device_map=None,
            **kwargs,
        ).to(DEVICE)

    model.config.use_cache = False
    return model


def _read_jsonl(path: Path) -> Generator:
    """Generator reading a JSONL file line by line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_bucket_dataset_supervised(folder: Path) -> Dataset | None:
    """Load instruction/input/output examples from good/medium/bad bucket folders."""
    items = []
    for bucket in ("good", "medium", "bad"):
        p = folder / bucket / "data.jsonl"
        if not p.exists():
            print(f"[warn] Missing {p}")
            continue
        for obj in _read_jsonl(p):
            items.append({
                "instruction": (obj.get("instruction") or "").strip(),
                "input": (obj.get("input") or "").strip(),
                "output": (obj.get("output") or "").strip(),
            })
    return Dataset.from_list(items) if items else None


def load_bucket_dataset_unsupervised(folder: Path) -> Dataset | None:
    """Load plain text examples from good/medium/bad bucket folders."""
    items = []
    for bucket in ("good", "medium", "bad"):
        p = folder / bucket / "data.jsonl"
        if not p.exists():
            print(f"[warn] Missing {p}")
            continue
        for obj in _read_jsonl(p):
            text = obj.get("text", "").strip()
            if text:
                items.append({"text": text})
    return Dataset.from_list(items) if items else None
