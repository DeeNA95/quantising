# Mixed-Precision INT8 Quantization Experiments

This project explores mixed-precision quantization strategies to reduce VRAM usage of large LLMs (like Llama-3-8B and Mistral-7B) while maintaining performance. It leverages **native `bitsandbytes` INT8 quantization** with selective layer skipping to implement various quantization configurations.

## 🚀 Key Features

- **Native INT8 Quantization**: Uses `bitsandbytes` optimized kernels for actual compute speedup (not just storage savings).
- **Selective Quantization**: Implements complex strategies like "The Sandwich Method" by dynamically calculating `llm_int8_skip_modules` to keep specific layers in BF16.
- **Modal H100 Integration**: Runs experiments on high-performance H100 GPUs via Modal.
- **Parallel Execution**: Can spawn multiple H100 containers to run all experiments simultaneously.
- **Comprehensive Evaluation**: Tracks VRAM, Latency (tokens/sec), HellaSwag Accuracy, and MMLU (Average) via Weights & Biases.

## 🧪 Experiment Configurations

We compare 5 distinct configurations defined in `config.yaml`:

| ID | Name | Description | Precision |
|----|------|-------------|-----------|
| **C-1** | Full BF16 Baseline | Standard baseline, no quantization. | BF16 |
| **C-2** | Uniform INT8 Baseline | All linear layers quantized to INT8. | INT8 |
| **E-A** | MLP-Only Quant | MLP blocks in INT8, Attention blocks in BF16. | Mixed |
| **E-B** | Attention-Only Quant | Attention blocks in INT8, MLP blocks in BF16. | Mixed |
| **E-C** | The Sandwich Method | Middle layers in INT8, first 5 & last 5 layers in BF16. | Mixed |

## 🛠️ Setup

### Prerequisites

- Python 3.10+
- [Modal](https://modal.com/) account
- [Weights & Biases](https://wandb.ai/) account
- Hugging Face Token (for gated models like Llama-3)

### Installation

This project uses `uv` for dependency management, but standard pip works too.

```bash
pip install torch transformers accelerate lm-eval scipy wandb modal bitsandbytes
```

### Secrets

Ensure you have the following secrets set in Modal:

- `huggingface-secret`: Contains `HF_TOKEN`
- `wandb-secret`: Contains `WANDB_API_KEY`

## 🏃‍♂️ Usage

### Run All Experiments (Parallel)

The fastest way to gather results. Spawns 5 concurrent H100 containers.

```bash
modal run run_all_experiments.py
```

### Run Single Experiment

Run a specific configuration (e.g., C-2).

```bash
modal run modal_app.py --config-id C-2
```

### Local Development

You can run the logic locally (requires GPU for quantization, or CPU for testing loading).

```bash
python main.py --config C-1
```

## 📊 Results

Results are logged to Weights & Biases.
**Project Dashboard**: [https://wandb.ai/dadadee02-n-a/mixed-precision-quant](https://wandb.ai/dadadee02-n-a/mixed-precision-quant)

## 📂 Project Structure

- `config.yaml`: Experiment definitions and model settings.
- `src/quantizer.py`: Core logic for applying INT8 quantization and calculating skip modules.
- `src/evaluator.py`: Evaluation pipeline (VRAM, Latency, HellaSwag, MMLU).
- `modal_app.py`: Modal infrastructure definition.
- `run_all_experiments.py`: Orchestrator for parallel execution.
