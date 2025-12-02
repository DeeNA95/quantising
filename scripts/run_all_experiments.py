#!/usr/bin/env python3
"""
Run all quantization experiments in parallel on Modal.
This will spawn 5 H100 containers simultaneously to run all experiments.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modal_app import app, run_experiment


@app.local_entrypoint()
def run_all():
    """
    Run all 5 experiments in parallel.
    """
    experiments = ["C-1", "C-2", "E-A", "E-B", "E-C"]

    print(f"🚀 Launching {len(experiments)} experiments in parallel on Modal H100s...")
    print(f"Experiments: {', '.join(experiments)}")
    print()

    # This will spawn 5 separate H100 containers
    list(run_experiment.map(experiments))

    print()
    print("✅ All experiments completed!")
    print("Check WandB dashboard: https://wandb.ai/dadadee02-n-a/mixed-precision-quant")


if __name__ == "__main__":
    run_all()
