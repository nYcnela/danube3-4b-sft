#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O-Danube3-4B - SUPERVISED Test Script

Model: h2oai/h2o-danube3-4b-chat (Instruct version)
Adapter: outputs/model_danube_supervised/lora_adapter
Test Data: ../model_v12.0/test_data (shared test set)

Based on test_llama3_supervised.py
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import evaluate


# ======= DOMYŚLNA KONFIG =======
MODEL_ID_DEFAULT = "h2oai/h2o-danube3-4b-chat"

SCRIPT_DIR = Path(__file__).resolve().parent

# Adapter path
DEFAULT_ADAPTER_DIR = SCRIPT_DIR.parents[3] / "outputs" / "manual" / "danube_4b" / "model_danube_supervised" / "lora_adapter"
# Test data from manual training prompts
DEFAULT_VAL_ROOT = SCRIPT_DIR.parents[3] / "data" / "json" / "manual" / "supervised" / "training_prompts" / "test_unique"

MAX_NEW_TOKENS = 64
SAFE_CTX = 8192
SEED_DEFAULT = 123
# ================================

_MODELS_DIR = SCRIPT_DIR.parents[2]
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from common.eval_utils import (  # noqa: E402
    DEVICE, DTYPE,
    safe_tokenizer, load_model_and_tokenizer,
    generate_text, compute_perplexity,
    normalize_output, extract_score, detect_hallucination,
    compute_rmse, compute_mean_std, prepare_metric_pairs,
    load_val_examples,
)


def build_prompt(tokenizer, instruction: str, user_input: str) -> str:
    """Build prompt using Danube chat template."""
    default_instr = (
        "You are a supportive expert coach. Provide one concise corrective sentence, "
        "then 'Score: X' where X is from 1 to 5 (5 is best)."
    )
    sys_msg = instruction.strip() or default_instr
    user_content = f"{sys_msg}\n\n{user_input.strip()}" if user_input else sys_msg

    msgs = [
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[INST] {user_content} [/INST]"


def main():
    ap = argparse.ArgumentParser(description="Test H2O-Danube3-4B SUPERVISED")
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--adapter-dir", type=str, default=str(DEFAULT_ADAPTER_DIR))
    ap.add_argument("--val-root", type=str, default=str(DEFAULT_VAL_ROOT))
    ap.add_argument("--n", type=int, default=200, help="Ile przykładów z walidacji zsamplować")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("H2O-DANUBE3-4B - SUPERVISED TEST")
    print("=" * 70)
    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print(f"Model: {args.model_id}")
    print(f"Adapter: {args.adapter_dir}")
    print("=" * 70)

    adapter_dir = Path(args.adapter_dir)
    val_root = Path(args.val_root)

    model, tokenizer = load_model_and_tokenizer(args.model_id, adapter_dir, SAFE_CTX, use_4bit=not args.no_4bit)

    all_val = load_val_examples(val_root)
    if not all_val:
        print(f"[ERR] Brak przykładów w {val_root}/(good|medium|bad)/data.jsonl", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    sample = rng.sample(all_val, k=min(args.n, len(all_val)))

    # Metryki
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    all_model_outputs = []
    all_expected_outputs = []
    predicted_scores = []
    reference_scores = []
    perplexities = []
    hallucination_count = 0
    total_samples = 0
    times_sec = []

    for i, ex in enumerate(sample, 1):
        instr = ex["instruction"]
        inp = ex["input"]
        expected = normalize_output(ex["output"])

        t0 = time.perf_counter()
        prompt = build_prompt(tokenizer, instr, inp)
        model_out_raw = generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, safe_ctx=SAFE_CTX)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times_sec.append(elapsed)

        model_out = normalize_output(model_out_raw)

        all_model_outputs.append(model_out)
        all_expected_outputs.append(expected)

        pred_score = extract_score(model_out)
        ref_score = extract_score(expected)
        predicted_scores.append(pred_score)
        reference_scores.append(ref_score)

        ppl = compute_perplexity(model, tokenizer, build_prompt(tokenizer, instr, inp), safe_ctx=SAFE_CTX)
        perplexities.append(ppl)

        total_samples += 1
        if detect_hallucination(model_out, inp):
            hallucination_count += 1

        print(f"\n{'=' * 70}")
        print(f"[Example {i} / {len(sample)}]")
        print(f"{'=' * 70}")
        print(f"\nINSTRUCTION (full):")
        print(f"{instr}")
        print(f"\nINPUT (movement description):")
        print(f"{inp}")
        print(f"\n{'- ' * 35}")
        print(f"\nTARGET (expected output):")
        print(f"{expected}")
        print(f"\nGENERATED (model output):")
        print(f"{model_out}")
        print(f"\nRESPONSE TIME: {elapsed:.2f}s")
        print(f"{'=' * 70}\n")

    if times_sec:
        avg_t, std_t = compute_mean_std(times_sec)
        print(f"\n===== LATENCY =====")
        print(f"Average response time: {avg_t:.2f}s")
        print(f"Std response time: {std_t:.2f}s")

    print("\n\n===== AUTOMATED EVALUATION =====")

    # ROUGE
    rouge_results = rouge.compute(predictions=all_model_outputs, references=all_expected_outputs)
    rouge_per_sample = rouge.compute(
        predictions=all_model_outputs,
        references=all_expected_outputs,
        use_aggregator=False,
    )
    rouge1_mean, rouge1_std = compute_mean_std(rouge_per_sample.get("rouge1", []))
    rouge2_mean, rouge2_std = compute_mean_std(rouge_per_sample.get("rouge2", []))
    rougeL_mean, rougeL_std = compute_mean_std(rouge_per_sample.get("rougeL", []))
    rougeLsum_mean, rougeLsum_std = compute_mean_std(rouge_per_sample.get("rougeLsum", []))
    print("\n--- ROUGE ---")
    print(f"ROUGE-1: {rouge_results.get('rouge1', 0.0):.4f} (std: {rouge1_std:.4f})")
    print(f"ROUGE-2: {rouge_results.get('rouge2', 0.0):.4f} (std: {rouge2_std:.4f})")
    print(f"ROUGE-L: {rouge_results.get('rougeL', 0.0):.4f} (std: {rougeL_std:.4f})")
    print(f"ROUGE-Lsum: {rouge_results.get('rougeLsum', 0.0):.4f} (std: {rougeLsum_std:.4f})")

    # BLEU
    bleu_results = bleu.compute(predictions=all_model_outputs, references=[[ref] for ref in all_expected_outputs])
    bleu_per_sample = []
    for pred, ref in zip(all_model_outputs, all_expected_outputs):
        per = bleu.compute(predictions=[pred], references=[[ref]])
        bleu_per_sample.append(per.get("bleu", 0.0))
    bleu_mean, bleu_std = compute_mean_std(bleu_per_sample)
    print(f"\n--- BLEU ---")
    print(f"BLEU: {bleu_results.get('bleu', 0.0):.4f} (std: {bleu_std:.4f})")
    print(f"BLEU precisions: {bleu_results.get('precisions', [])}")

    # METEOR
    meteor_results = meteor.compute(predictions=all_model_outputs, references=all_expected_outputs)
    meteor_per_sample = []
    for pred, ref in zip(all_model_outputs, all_expected_outputs):
        per = meteor.compute(predictions=[pred], references=[ref])
        meteor_per_sample.append(per.get("meteor", 0.0))
    meteor_mean, meteor_std = compute_mean_std(meteor_per_sample)
    print(f"\n--- METEOR ---")
    print(f"METEOR: {meteor_results.get('meteor', 0.0):.4f} (std: {meteor_std:.4f})")

    # BERTScore
    bert_results = bertscore.compute(predictions=all_model_outputs, references=all_expected_outputs, lang="en")
    avg_bert_f1, std_bert_f1 = compute_mean_std(bert_results.get("f1", []))
    print(f"\n--- BERTScore ---")
    print(f"BERTScore (avg F1): {avg_bert_f1:.4f} (std: {std_bert_f1:.4f})")

    # Summary H1
    print(f"\n{'=' * 70}")
    print("SUMMARY (for Hypothesis H1)")
    print(f"{'=' * 70}")
    print(f"BLEU:      {bleu_results.get('bleu', 0.0):.4f} (std: {bleu_std:.4f})")
    print(f"ROUGE-L:   {rouge_results.get('rougeL', 0.0):.4f} (std: {rougeL_std:.4f})")
    print(f"METEOR:    {meteor_results.get('meteor', 0.0):.4f} (std: {meteor_std:.4f})")
    print(f"BERTScore: {avg_bert_f1:.4f} (std: {std_bert_f1:.4f})")
    print(f"{'=' * 70}")

    # H2: RMSE
    rmse = compute_rmse(predicted_scores, reference_scores)
    valid_scores = sum(1 for p, r in zip(predicted_scores, reference_scores) if p >= 1.0 and r >= 1.0)
    print(f"\n{'=' * 70}")
    print("HYPOTHESIS H2: Score Prediction (RMSE)")
    print(f"{'=' * 70}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Valid predictions: {valid_scores}/{len(predicted_scores)}")
    print(f"Success rate: {(valid_scores/len(predicted_scores)*100):.2f}%")
    print(f"Target: < 1.0 (hypothesis confirmed if RMSE < 1.0)")
    print(f"Result: {'✓ CONFIRMED' if rmse < 1.0 else '✗ NOT CONFIRMED'}")
    print(f"{'=' * 70}")

    # H3: Perplexity
    avg_ppl, std_ppl = compute_mean_std(perplexities) if perplexities else (float('inf'), 0.0)
    print(f"\n{'=' * 70}")
    print("HYPOTHESIS H3: Perplexity")
    print(f"{'=' * 70}")
    print(f"Average Perplexity: {avg_ppl:.2f}")
    print(f"Std Perplexity: {std_ppl:.2f}")
    print(f"(Lower is better - measures model's prediction quality)")
    print(f"{'=' * 70}")

    # H4: Hallucinations
    hallucination_rate = (hallucination_count / total_samples * 100) if total_samples > 0 else 0.0
    print(f"\n{'=' * 70}")
    print("HYPOTHESIS H4: Hallucination Rate")
    print(f"{'=' * 70}")
    print(f"Hallucinations detected: {hallucination_count}/{total_samples}")
    print(f"Hallucination Rate: {hallucination_rate:.2f}%")
    print(f"{'=' * 70}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
