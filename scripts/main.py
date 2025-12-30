import argparse
import os
import shutil
from src.utils import load_config, setup_wandb
from src.quantizer import calibrate_and_save
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

    # Define path for quantized model (might not be used if native)
    save_path = f"./quantized_models/{args.config}"

    print(f"🚀 Running experiment {args.config}...")

    # calibrate_and_save now returns the path to load (could be model_name or save_path)
    # and the quantization argument for vllm.
    model_path, quantization_arg = calibrate_and_save(model_name, args.config, exp_config, save_path)

    # Debug: Check if files exist if save_path returned
    if model_path == save_path:
        print(f"🧐 Checking files in {save_path}:")
        if os.path.exists(save_path):
            for root, dirs, files in os.walk(save_path):
                for file in files:
                    print(os.path.join(root, file))
        else:
            print(f"⚠️ {save_path} does not exist (unexpected if returned).")

    # Evaluate using vllm
    evaluate_model(model_path, quantization=quantization_arg)

    # Cleanup if we created a local model
    if os.path.exists(save_path) and model_path == save_path:
        print(f"🧹 Cleaning up {save_path}...")
        try:
            shutil.rmtree(save_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup {save_path}: {e}")


if __name__ == "__main__":
    main()
