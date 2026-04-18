#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for model evaluation and test scripts.

Import in each test script:
    MODELS_DIR = Path(__file__).resolve().parents[2]
    import sys; sys.path.insert(0, str(MODELS_DIR))
    from common.eval_utils import (
        DEVICE, DTYPE,
        safe_tokenizer, load_model_and_tokenizer,
        generate_text, compute_perplexity,
        normalize_output, extract_score, detect_hallucination,
        compute_rmse, compute_mean_std, prepare_metric_pairs,
        load_val_examples, load_val_examples_by_bucket,
        sample_fewshot_examples, FewShotPromptBuilder,
    )
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Re-export device/tokenizer utilities from training_utils so test scripts
# only need one import source.
from common.training_utils import (  # noqa: F401
    DEVICE,
    DTYPE,
    IS_DARWIN,
    make_bnb_config,
    pick_device,
    pick_dtype,
    safe_tokenizer,
)

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id: str,
    adapter_dir: Path,
    safe_ctx: int,
    use_4bit: bool = True,
    trust_remote_code: bool = False,
):
    """Load a base model + LoRA adapter for inference.

    Returns (model, tokenizer). The model is in eval mode.
    """
    tok = safe_tokenizer(model_id, safe_ctx, trust_remote_code=trust_remote_code)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    can_4bit = use_4bit and _HAS_BNB and DEVICE.type == "cuda"
    if can_4bit:
        try:
            base = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=make_bnb_config(DTYPE),
                torch_dtype=DTYPE,
            )
            model = PeftModel.from_pretrained(base, adapter_dir, device_map="auto")
            model.eval()
            return model, tok
        except Exception as e:
            print(f"[warn] 4-bit load failed ({e}), falling back to full precision.", file=sys.stderr)

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if DEVICE.type == "cuda" else None,
        torch_dtype=DTYPE,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    if DEVICE.type != "cuda":
        model = model.to(DEVICE)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    safe_ctx: int = 8192,
    repetition_penalty: float = 1.0,
    first_line_only: bool = False,
) -> str:
    """Run greedy decoding and return the generated text (prompt stripped).

    Args:
        repetition_penalty: Values > 1.0 discourage repetition (useful for CLM models).
        first_line_only: If True, return only the first non-empty output line
                         (useful for few-shot prompting where the model may continue
                         with more examples).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=safe_ctx)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    out_ids = outputs[0][prompt_len:] if outputs.shape[1] > prompt_len else outputs[0]
    text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    if first_line_only:
        text = text.split("\n")[0].strip()

    return text


def compute_perplexity(model, tokenizer, text: str, safe_ctx: int = 8192) -> float:
    """Compute perplexity of *text* under the model (lower = better domain fit)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=safe_ctx)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


# ---------------------------------------------------------------------------
# Text post-processing
# ---------------------------------------------------------------------------

def normalize_output(text: str) -> str:
    """Normalise whitespace and canonicalise the 'Score: X' suffix."""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(
        r"(You should .+?)(\s*)Score:\s*([1-5])\s*$",
        lambda m: m.group(1).rstrip(".") + f". Score: {m.group(3)}",
        t,
        flags=re.I,
    )
    return t


def extract_score(text: str) -> float:
    """Extract a 1-5 score from generated text. Returns -1.0 if not found."""
    match = re.search(r"Score:\s*([1-5])", text, re.IGNORECASE)
    return float(match.group(1)) if match else -1.0


def detect_hallucination(generated: str, input_text: str) -> bool:
    """Heuristic hallucination check: body parts not mentioned in input, or contradictions."""
    generated_lower = generated.lower()
    input_lower = input_text.lower()

    body_parts = [
        "left arm", "right arm", "left elbow", "right elbow",
        "left leg", "right leg", "left knee", "right knee",
        "left hand", "right hand", "left foot", "right foot",
        "left side", "right side", "bow", "step", "accent",
    ]
    hallucinated = [
        p for p in body_parts
        if p in generated_lower
        and p not in input_lower
        and p.split()[-1] not in input_lower
    ]

    contradictions = [
        ("too fast", "too slow"), ("too slow", "too fast"),
        ("too high", "too low"), ("too low", "too high"),
        ("too long", "too short"), ("too short", "too long"),
        ("too deep", "too shallow"), ("too shallow", "too deep"),
    ]
    for input_term, output_term in contradictions:
        if input_term in input_lower and output_term in generated_lower:
            return True

    return len(hallucinated) >= 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(predicted_scores: list[float], reference_scores: list[float]) -> float:
    """RMSE over valid (>=1) score pairs. Returns inf if no valid pairs."""
    valid = [(p, r) for p, r in zip(predicted_scores, reference_scores) if p >= 1.0 and r >= 1.0]
    if not valid:
        return float("inf")
    mse = sum((p - r) ** 2 for p, r in valid) / len(valid)
    return math.sqrt(mse)


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, stdev). Returns (0.0, 0.0) for empty lists."""
    if not values:
        return 0.0, 0.0
    return sum(values) / len(values), statistics.stdev(values) if len(values) > 1 else 0.0


def prepare_metric_pairs(
    predictions: list[str],
    references: list[str],
) -> dict[str, list[str]]:
    """Filter out pairs where either prediction or reference is empty."""
    pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    preds, refs = zip(*pairs) if pairs else ([], [])
    return {"predictions": list(preds), "references": list(refs)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                pass


def load_val_examples(val_root: Path) -> list[dict[str, str]]:
    """Load all examples from good/medium/bad bucket folders as a flat list."""
    items = []
    for bucket in ("good", "medium", "bad"):
        p = val_root / bucket / "data.jsonl"
        if not p.exists():
            continue
        for ex in _read_jsonl(p):
            items.append({
                "instruction": (ex.get("instruction") or "").strip(),
                "input": (ex.get("input") or "").strip(),
                "output": (ex.get("output") or "").strip(),
            })
    return items


def load_val_examples_by_bucket(val_root: Path) -> dict[str, list[dict[str, str]]]:
    """Load examples split by bucket (good/medium/bad)."""
    data: dict[str, list] = {"good": [], "medium": [], "bad": []}
    for bucket in ("good", "medium", "bad"):
        p = val_root / bucket / "data.jsonl"
        if not p.exists():
            continue
        for ex in _read_jsonl(p):
            data[bucket].append({
                "instruction": (ex.get("instruction") or "").strip(),
                "input": (ex.get("input") or "").strip(),
                "output": (ex.get("output") or "").strip(),
            })
    return data


def load_fewshot_fixed(path: Path) -> list[dict[str, str]]:
    """Load fixed few-shot examples from a JSONL file."""
    items = []
    for ex in _read_jsonl(path):
        items.append({
            "instruction": (ex.get("instruction") or "").strip(),
            "input": (ex.get("input") or "").strip(),
            "output": (ex.get("output") or "").strip(),
        })
    return items


# ---------------------------------------------------------------------------
# Few-shot helpers
# ---------------------------------------------------------------------------

def sample_fewshot_examples(
    data_by_bucket: dict[str, list[dict]],
    num_good: int = 1,
    num_medium: int = 3,
    num_bad: int = 6,
    rng: random.Random | None = None,
) -> list[dict]:
    """Sample few-shot examples from bucketed data with the given distribution."""
    if rng is None:
        rng = random.Random()
    examples = []
    for bucket, needed in [("good", num_good), ("medium", num_medium), ("bad", num_bad)]:
        available = data_by_bucket.get(bucket, [])
        if len(available) < needed:
            print(f"[WARN] Bucket '{bucket}' has only {len(available)} samples, need {needed}")
            needed = len(available)
        if needed > 0:
            examples.extend(rng.sample(available, needed))
    rng.shuffle(examples)
    return examples


class FewShotPromptBuilder:
    """Builds and caches a few-shot prompt prefix."""

    def __init__(self, examples: list[dict[str, str]]):
        self.examples = examples
        self.prefix = self._build_prefix()

    def _build_prefix(self) -> str:
        prompt = (
            "You are a Polonaise dance teacher. Based on the movement description, "
            "provide one short corrective feedback sentence, then give a score from 1 to 5 (5 is best).\n\n"
        )
        for i, ex in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {ex['input']}\n"
            prompt += f"Output: {ex['output']}\n\n"
        prompt += "Now evaluate:\n"
        return prompt

    def build_prompt(self, user_input: str) -> str:
        return self.prefix + f"Input: {user_input}\nOutput:"

    def get_num_examples(self) -> int:
        return len(self.examples)

    def print_examples(self):
        print(f"\n[INFO] Few-shot examples ({len(self.examples)} total):")
        for i, ex in enumerate(self.examples, 1):
            inp = ex["input"][:80] + "..." if len(ex["input"]) > 80 else ex["input"]
            out = ex["output"][:80] + "..." if len(ex["output"]) > 80 else ex["output"]
            print(f"  {i}. Input:  {inp}")
            print(f"     Output: {out}")
