import argparse
import os
import shutil
from src.utils import load_config, setup_wandb
from src.quantizer import calibrate_and_save, load_quantized_model
from src.evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Config ID (e.g., C-1, E-A)"
    )
    args = parser.parse_args()

    config = load_config()
    if args.config not in config["experiments"]:
        raise ValueError(f"Config {args.config} not found in config.yaml")

    exp_config = config["experiments"][args.config]
    model_name = config["model"]["name"]

    setup_wandb(config, args.config)

    # Define path for quantized model
    save_path = f"./quantized_models/{args.config}"

    print(f"🚀 Running experiment {args.config}...")

    calibrate_and_save(model_name, args.config, exp_config, save_path)

    # Debug: Check if files exist
    print(f"🧐 Checking files in {save_path}:")
    for root, dirs, files in os.walk(save_path):
        for file in files:
            print(os.path.join(root, file))

    model_path = load_quantized_model(save_path)

    evaluate_model(model_path)

    print(f"🧹 Cleaning up {save_path}...")
    try:
        shutil.rmtree(save_path)
    except Exception as e:
        print(f"Warning: Failed to cleanup {save_path}: {e}")


if __name__ == "__main__":
    main()
