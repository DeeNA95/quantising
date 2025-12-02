import time
import torch
import wandb
from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_vram_and_latency(model_path):
    """
    Measure VRAM usage and inference latency.
    """
    print("📊 Measuring VRAM and latency...")

    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Measure VRAM after loading
    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  💾 VRAM Usage: {vram_gb:.2f} GB")

    # Benchmark latency with a sample prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    # Measure latency
    torch.cuda.synchronize()
    num_runs = 5
    total_tokens = 0
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            total_tokens += output.shape[1] - inputs['input_ids'].shape[1]

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    latency_tok_per_sec = total_tokens / elapsed_time
    print(f"  ⚡ Latency: {latency_tok_per_sec:.2f} tokens/sec")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return vram_gb, latency_tok_per_sec


def evaluate_model(model_path):
    """
    Run evaluation using transformers backend and measure performance metrics.
    """
    results = {}

    print(f"🔍 Running Zero-Shot Evaluation on {model_path}...")

    # Measure VRAM and latency first
    vram_gb, latency = measure_vram_and_latency(model_path)
    results["vram_gb"] = round(vram_gb, 2)
    results["latency"] = round(latency, 2)

    # Select tasks - HellaSwag + a subset of MMLU for speed
    task_names = [
        "hellaswag",
        "mmlu_abstract_algebra",
        "mmlu_anatomy",
        "mmlu_astronomy",
        "mmlu_business_ethics",
        "mmlu_clinical_knowledge",
    ]

    # Use HF transformers backend (simpler, no server needed)
    eval_results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=auto,trust_remote_code=true",
        tasks=task_names,
        num_fewshot=0,
        batch_size="auto",
    )

    # Extract metrics
    mmlu_scores = []
    for task in task_names:
        acc = eval_results["results"][task].get("acc,none")
        if acc is None:
            acc = eval_results["results"][task].get("acc")
        results[f"{task}_acc"] = acc
        print(f"  {task}: {acc:.4f}")

        # Collect MMLU scores for averaging
        if task.startswith("mmlu_"):
            mmlu_scores.append(acc)

    # Calculate average MMLU score
    if mmlu_scores:
        results["mmlu_avg"] = sum(mmlu_scores) / len(mmlu_scores)
        print(f"📊 MMLU Average: {results['mmlu_avg']:.4f}")

    # Log to WandB
    wandb.log(results)

    return results
