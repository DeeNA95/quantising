import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands(
        'pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128',
        'pip install "sglang[all]"',
        'pip install lm-eval transformers accelerate datasets bitsandbytes',
        'pip install wandb pyyaml',
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("scripts/main.py", remote_path="/root/main.py")
)

app = modal.App("llama3-quant-experiment", image=image)


@app.function(
    gpu="H100",
    timeout=3600 * 3,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_experiment(config_id: str):
    import subprocess

    print(f"Running experiment: {config_id} on H100")

    cmd = ["python", "main.py", "--config", config_id]
    subprocess.check_call(cmd)


@app.local_entrypoint()
def main(config_id: str = "C-1"):
    run_experiment.remote(config_id)
