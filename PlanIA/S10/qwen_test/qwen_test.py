#!/usr/bin/env python3
"""Qwen3.5-4B: una consulta y sus métricas.

En Khipu: sbatch run_qwen.sbatch
Dentro de una asignación Slurm con GPU: .venv/bin/python -u qwen_test.py
"""
# 1. Preparar el modelo local
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
    raise RuntimeError("En Khipu ejecuta este script dentro de un job Slurm con GPU: sbatch run_qwen.sbatch")

script_dir = Path(__file__).resolve().parent
model_roots = [script_dir, *script_dir.parents, script_dir.parent / "Lora Scripts"]
project_dir = next((p for p in model_roots
                    if (p / "models/model_manifest.json").is_file()), None)

if project_dir is None:
    raise FileNotFoundError("No se encontró models/model_manifest.json del proyecto.")

manifest = json.loads((project_dir / "models/model_manifest.json").read_text())
assert manifest["model_id"] == "Qwen/Qwen3.5-4B"

model_path = Path(manifest["snapshot_path"])
if not model_path.is_dir():
    raise FileNotFoundError(f"Checkpoint local no encontrado: {model_path}")

precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    device_map={"": 0},
    dtype=precision,
    attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=precision,
    ),
).eval()

model.requires_grad_(False)
torch.cuda.synchronize()
load_seconds = time.perf_counter() - load_start

print(f"Modelo: {manifest['model_id']} | GPU: {torch.cuda.get_device_name(0)}")
print(f"Carga del tokenizer y modelo: {load_seconds:.2f} s")

# 2. Hacer un prompt sencillo y medir la inferencia
prompt = "Explica en dos frases qué es un modelo de lenguaje."
max_new_tokens = 128

gpu_samples = []
stop_sampling = threading.Event()
gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
sampling_errors = []

def sample_gpu():
    while not stop_sampling.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            gpu_samples.append(float(result.stdout.strip()))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            sampling_errors.append(getattr(exc, "stderr", None) or str(exc))
        stop_sampling.wait(0.2)

sampler = threading.Thread(target=sample_gpu, daemon=True)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

free_before, total_gpu_memory = torch.cuda.mem_get_info()
inference_start = time.perf_counter()

sampler.start()

try:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, 
        add_generation_prompt=True, 
        enable_thinking=False,
        return_dict=True, 
        return_tensors="pt",
    ).to("cuda:0")

    torch.cuda.synchronize()

    generation_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=sorted({tokenizer.eos_token_id,
                                tokenizer.convert_tokens_to_ids("<|endoftext|>")}),
        )

    torch.cuda.synchronize()

    generation_seconds = time.perf_counter() - generation_start
    input_tokens = inputs.input_ids.shape[1]
    generated_ids = outputs[0, input_tokens:]
    output_tokens = len(generated_ids)
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    inference_seconds = time.perf_counter() - inference_start
finally:
    stop_sampling.set()
    sampler.join(timeout=3)

free_after, _ = torch.cuda.mem_get_info()
metrics = {
    "model": manifest["model_id"],
    "model_revision": manifest["revision"],
    "gpu": torch.cuda.get_device_name(0),
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "total_tokens": input_tokens + output_tokens,
    "model_load_seconds": round(load_seconds, 3),
    "generation_seconds": round(generation_seconds, 3),
    "inference_seconds": round(inference_seconds, 3),
    "output_tokens_per_second": round(output_tokens / generation_seconds, 2),
    "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
    "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    "device_total_memory_gib": round(total_gpu_memory / 2**30, 3),
    "device_free_before_gib": round(free_before / 2**30, 3),
    "device_free_after_gib": round(free_after / 2**30, 3),
    "device_gpu_utilization_mean_percent": round(statistics.mean(gpu_samples), 1) if gpu_samples else None,
    "device_gpu_utilization_max_percent": max(gpu_samples) if gpu_samples else None,
    "gpu_utilization_samples": len(gpu_samples),
    "gpu_utilization_error": sampling_errors[-1].strip() if not gpu_samples and sampling_errors else None,
}

# 3. Mostrar la respuesta y las métricas
print("PROMPT:", prompt)
print("\nRESPUESTA:\n", response)
print("\nMÉTRICAS:\n", json.dumps(metrics, ensure_ascii=False, indent=2))
