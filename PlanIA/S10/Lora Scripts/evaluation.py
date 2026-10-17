"""
Evaluation Module
=================
This module defines the evaluation pipeline for comparing the base model 
and the LoRA-fine-tuned model on a set of unseen test prompts. 
The evaluation focuses on two key metrics:
1. JSON-parse rate: The percentage of model outputs that can be successfully 
parsed as JSON.
2. Schema-compliance rate: The percentage of outputs that not only parse as JSON 
but also conform to the expected schema 
(i.e., contain the required fields with valid values).

Steps performed by this module:
- Defines unseen test prompts (regular + stress cases)
- Runs generation with base vs. LoRA model
- Computes JSON-parse rate & schema-compliance rate
- Prints a comparison report
"""

from typing import Dict, List, Tuple

import torch

from dataset_generator import SYSTEM_INSTRUCTION
from validator import validate_json_output

# ── Test prompts (unseen during training) ──────────────────────────────────

REGULAR_CASES = [
    "I bought a vacuum cleaner last month and it stopped working. I want a full refund!",
    "Can you help me set up my new smart thermostat? I'm having trouble.",
    "Where's my order? It's been two weeks since I ordered the running shoes.",
    "The gaming mouse I received has a broken scroll wheel. Send me a new one please.",
    "What's your policy on international shipping?",
    "I need to cancel my subscription immediately. This is urgent!",
    "The wireless earbuds have terrible sound quality. Not happy at all.",
    "How do I reset my password for the mobile app?",
    "My blender arrived broken. I need a replacement before the weekend!",
    "Just wanted to say thank you for the excellent customer service!",
    "The software license key isn't working. Can someone help?",
    "I'd like to return the dress I ordered. Wrong size.",
    "My order still shows as processing after 5 days. What's going on?",
    "The instructions for the desk assembly are missing. Can you email them?",
    "I received the wrong color phone case. Need to exchange it.",
]

STRESS_CASES = [
    "This is the WORST experience I've ever had! Your company is terrible! "
    "I want a refund for the printer NOW and I never want to shop here again!",

    "Hi! Hope you're having a great day! I was wondering if you could help "
    "me with a small issue? My camera lens has a scratch and I think maybe "
    "I should get a replacement? What do you think? Thanks so much!",

    "Quick question about the guitar strings I ordered - they haven't "
    "arrived yet and I'm getting worried because I have a concert coming "
    "up soon. Can you check on this for me please? Really appreciate your help!",

    "The USB drive is defective but honestly I'm not even sure if I want to "
    "deal with returns right now because last time it took forever. Maybe just a refund?",

    "I NEED HELP ASAP!!! The laptop charger sparked when I plugged it in! "
    "This is dangerous! Someone needs to respond immediately!",
]


def create_evaluation_prompts() -> List[Dict[str, str]]:
    """Build a list of eval dicts with message, full_prompt, and is_stress flag."""
    stress_set = set(STRESS_CASES)
    prompts = []
    for msg in REGULAR_CASES + STRESS_CASES:
        prompts.append({
            "message": msg,
            "full_prompt": f"{SYSTEM_INSTRUCTION}\n\nCustomer message: {msg}",
            "is_stress": msg in stress_set,
        })
    return prompts


# ── Generation helper ──────────────────────────────────────────────────────

def _extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from text via brace matching."""
    # This is a simple heuristic that looks for the first '{' and then finds the 
    # matching '}'.
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text  # unclosed — return as-is


def generate_response(prompt: str, model, tokenizer, config) -> Tuple[str, str]:
    """
    Generate a response using the Gemma chat format.

    Returns:
        (raw, post_processed) — raw is the cleaned-but-uncut output,
        post_processed has the first JSON object extracted.
    """
    formatted = (
        f"<start_of_turn>user\n"
        f"{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    # "pt" means "PyTorch tensors" — this converts the tokenized input into the 
    # format needed for the model.
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Build stop token list: EOS + <end_of_turn>
    stop_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    # If the tokenizer has a special token for <end_of_turn>, include it as a 
    # stop token.
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        stop_ids.append(eot_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )

    # Decode the generated tokens back into text, skipping special tokens. Then
    # clean residual chat-template markers. The result is the "raw" output from 
    # the model, which may contain extra text
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    # Clean residual chat-template markers
    generated = generated.replace("<end_of_turn>", "").replace("<start_of_turn>", "")
    raw = generated.strip() # removes white space (Tabs, spaces, newlines all removed)

    # Post-process: extract the first complete JSON object
    post = _extract_json_object(raw)

    return raw, post


# ── Evaluation loop ────────────────────────────────────────────────────────

def evaluate_model(
    model, tokenizer, config, model_name: str
) -> Tuple[float, float, float, float, List[Dict]]:
    """
    Run all eval prompts through `model` and return:
        (raw_json_rate, raw_schema_rate,
         post_json_rate, post_schema_rate,
         per-example results)
    """
    prompts = create_evaluation_prompts()
    results: List[Dict] = []
    raw_json_ok = 0
    raw_schema_ok = 0
    post_json_ok = 0
    post_schema_ok = 0

    print(f"\nEvaluating {model_name}...")
    print(f"Processing {len(prompts)} test cases...\n")

    for i, p in enumerate(prompts):
        raw, post = generate_response(p["full_prompt"], model, tokenizer, config)

        # Validate RAW output (no post-processing)
        v_raw = validate_json_output(raw)
        raw_is_json = v_raw.error is None or "JSON parse error" not in v_raw.error
        raw_json_ok += int(raw_is_json)
        raw_schema_ok += int(v_raw.valid)

        # Validate POST-PROCESSED output (JSON extraction)
        v_post = validate_json_output(post)
        post_is_json = v_post.error is None or "JSON parse error" not in v_post.error
        post_json_ok += int(post_is_json)
        post_schema_ok += int(v_post.valid)

        results.append({
            "prompt": p["message"],
            "raw_response": raw,
            "post_response": post,
            "is_stress": p["is_stress"],
            # RAW metrics
            "raw_is_valid_json": raw_is_json,
            "raw_is_schema_compliant": v_raw.valid,
            "raw_error": v_raw.error,
            # POST metrics
            "post_is_valid_json": post_is_json,
            "post_is_schema_compliant": v_post.valid,
            "post_error": v_post.error,
        })

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} examples")

    n = len(prompts)
    return (raw_json_ok / n, raw_schema_ok / n,
            post_json_ok / n, post_schema_ok / n, results)


# ── Report printer ─────────────────────────────────────────────────────────

def print_evaluation_report(
    base_results: Tuple[float, float, float, float, List[Dict]],
    lora_results: Tuple[float, float, float, float, List[Dict]],
):
    """Print a side-by-side comparison of base vs LoRA (RAW + POST metrics)."""
    # b_det means "base model details" and l_det means "LoRA model details" — 
    # these are the per-example results returned by `evaluate_model`, which we 
    # can use to show specific examples in the report.
    b_raw_json, b_raw_schema, b_post_json, b_post_schema, b_det = base_results
    l_raw_json, l_raw_schema, l_post_json, l_post_schema, l_det = lora_results

    print(f"\n{'='*70}")
    print("EVALUATION REPORT")
    print(f"{'='*70}\n")

    print(f"{'Metric':<40} {'Base Model':<15} {'LoRA Model':<15}")
    print("-" * 70)
    print(f"  RAW (model output as-is, no cleanup):")
    # >13.1 means "format as percentage with 1 decimal, and pad to at least 13 
    # characters wide, right-aligned"
    # :<40 means "pad to at least 40 characters wide, left-aligned"
    print(f"{'    Parseable JSON Rate':<40} {b_raw_json:>13.1%} {l_raw_json:>14.1%}")
    print(f"{'    Schema Compliance Rate':<40} {b_raw_schema:>13.1%} {l_raw_schema:>14.1%}")
    print(f"  POST (after JSON extraction):")
    print(f"{'    Parseable JSON Rate':<40} {b_post_json:>13.1%} {l_post_json:>14.1%}")
    print(f"{'    Schema Compliance Rate':<40} {b_post_schema:>13.1%} {l_post_schema:>14.1%}")
    print("-" * 70)

    # imp_raw means "improvement in raw compliance" and imp_post means "improvement 
    # in post compliance" — these are the differences between the LoRA model and
    # the base model, expressed in percentage points (pp).
    imp_raw = (l_raw_schema - b_raw_schema) * 100
    imp_post = (l_post_schema - b_post_schema) * 100
    print(f"\nImprovement (schema compliance):")
    print(f"  RAW:  {imp_raw:+.1f} pp")
    print(f"  POST: {imp_post:+.1f} pp\n")

    # Examples where base failed but LoRA succeeded (using POST metrics)
    print(f"{'='*70}")
    print("Examples: Base FAILED → LoRA SUCCEEDED (post-processed)")
    print(f"{'='*70}\n")
    shown = 0
    for b, l in zip(b_det, l_det):
        if not b["post_is_schema_compliant"] and l["post_is_schema_compliant"]:
            shown += 1
            if shown <= 3:
                print(f"Prompt: {b['prompt'][:80]}...")
                print(f"  Base  → {b['raw_response'][:200]}…")
                print(f"          ✗ {b['post_error'] or b['raw_error']}")
                print(f"  LoRA  → {l['post_response']}")
                print(f"          ✓ Valid\n")
    if shown == 0:
        print("  (none)\n")

    # Stress test results
    print(f"{'='*70}")
    print("Stress-Test Cases")
    print(f"{'='*70}\n")
    shown = 0
    # Show some examples of stress cases, with their raw outputs and whether they
    # were compliant or not. This gives a qualitative sense of how the models
    # handle the most difficult prompts.
    for b, l in zip(b_det, l_det):
        if b["is_stress"]:
            shown += 1
            if shown <= 2:
                tag_b = "✓" if b["post_is_schema_compliant"] else "✗"
                tag_l = "✓" if l["post_is_schema_compliant"] else "✗"
                print(f"Prompt: {b['prompt'][:80]}...")
                print(f"  Base [{tag_b}]: {b['raw_response'][:150]}…")
                print(f"  LoRA [{tag_l}]: {l['post_response'][:150]}…\n")

    # Final summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"  RAW  strict compliance:  Base {b_raw_schema:.1%} → LoRA {l_raw_schema:.1%}  ({imp_raw:+.1f} pp)")
    print(f"  POST strict compliance:  Base {b_post_schema:.1%} → LoRA {l_post_schema:.1%}  ({imp_post:+.1f} pp)")
    print()
    if l_raw_schema >= 0.9:
        print(f"  ✓ EXCELLENT raw compliance ({l_raw_schema:.1%}) — model outputs clean JSON")
    elif l_raw_schema >= 0.7:
        print(f"  ✓ GOOD raw compliance ({l_raw_schema:.1%}) — model mostly outputs clean JSON")
    else:
        print(f"  ⚠ Raw compliance is moderate ({l_raw_schema:.1%}) — model needs post-processing")
    if l_post_schema > l_raw_schema:
        print(f"  ℹ Post-processing rescues {(l_post_schema - l_raw_schema):.1%} more cases")
    print()
