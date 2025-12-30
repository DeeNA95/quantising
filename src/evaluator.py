import time
import torch
import wandb
import gc
from lm_eval import evaluator
from vllm import LLM, SamplingParams

def cleanup_vllm():
    """Force garbage collection to free VLLM memory."""
    gc.collect()
    torch.cuda.empty_cache()

def measure_vram_and_latency(model_path, quantization=None):
    """
    Measure inference latency using vllm.
    VRAM usage in vllm is typically determined by gpu_memory_utilization,
    so we focus on latency here.
    """
    print(f"📊 Measuring Latency with vllm (quantization={quantization})...")
    cleanup_vllm()

    try:
        # distinct "vllm" loads the weights.
        llm = LLM(
            model=model_path,
            quantization=quantization,
            trust_remote_code=True,
            gpu_memory_utilization=0.8, # Reserve some space
            tensor_parallel_size=1
        )

        # Sampling params for greedy generation
        sampling_params = SamplingParams(temperature=0, max_tokens=50)

        # Warmup
        prompt = "The quick brown fox jumps over the lazy dog. " * 10
        llm.generate([prompt], sampling_params)

        # Measurement
        num_runs = 5
        total_tokens = 0
        start_time = time.time()

        # Prepare batch for better throughput measurement or single for latency?
        # User asked for speed, likely throughput/latency.
        # We'll stick to single stream latency for now as per previous logic.
        prompts = [prompt] * num_runs

        outputs = llm.generate(prompts, sampling_params)

        # Calculate tokens
        for output in outputs:
            total_tokens += len(output.outputs[0].token_ids)

        elapsed_time = time.time() - start_time
        latency_tok_per_sec = total_tokens / elapsed_time

        print(f"  ⚡ Latency (Throughput): {latency_tok_per_sec:.2f} tokens/sec")

        # VRAM metric is hard to get exact "usage" vs "reserved" with vllm.
        # encoding torch.cuda.memory_allocated() might be misleading due to vllm pre-alloc.
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  💾 Peak VRAM (Approx): {vram_gb:.2f} GB")

        del llm
        cleanup_vllm()

        return vram_gb, latency_tok_per_sec

    except Exception as e:
        print(f"Error during latency measurement: {e}")
        cleanup_vllm()
        return 0.0, 0.0

def evaluate_model(model_path, quantization=None):
    """
    Run evaluation using vllm backend via lm_eval.
    """
    results = {}
    print(f"🔍 Running Zero-Shot Evaluation on {model_path} with vllm...")

    # Measure Latency First
    # Note: Loading vllm twice (here and inside simple_evaluate) is heavy.
    # But simple_evaluate manages its own model.
    vram_gb, latency = measure_vram_and_latency(model_path, quantization)
    results["vram_gb"] = round(vram_gb, 2)
    results["latency"] = round(latency, 2)

    # Select tasks
    task_names = [
        "hellaswag",
        "mmlu_abstract_algebra",
        "mmlu_anatomy",
        "mmlu_astronomy",
        "mmlu_business_ethics",
        "mmlu_clinical_knowledge",
    ]

    # Construct model args for lm_eval
    # vllm args: pretrained, trust_remote_code, quantization, etc.
    model_args = f"pretrained={model_path},trust_remote_code=True,gpu_memory_utilization=0.8"
    if quantization:
        model_args += f",quantization={quantization}"

    print(f"  Running lm_eval with model_args: {model_args}")

    eval_results = evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=0,
        batch_size="auto",
    )

    # Extract metrics
    mmlu_scores = []
    for task in task_names:
        # lm_eval v0.4+ struct might vary, but assuming similar to before
        # 'acc,none' is standard for recent versions
        acc = eval_results["results"][task].get("acc,none")
        if acc is None:
            acc = eval_results["results"][task].get("acc")

        results[f"{task}_acc"] = acc
        print(f"  {task}: {acc:.4f}")

        if task.startswith("mmlu_"):
            mmlu_scores.append(acc)

    if mmlu_scores:
        results["mmlu_avg"] = sum(mmlu_scores) / len(mmlu_scores)
        print(f"📊 MMLU Average: {results['mmlu_avg']:.4f}")

    wandb.log(results)
    return results
