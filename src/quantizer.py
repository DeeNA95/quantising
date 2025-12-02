import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_quantization_config(config_id, config_data):
    """
    Create quantization configuration based on experiment config.
    Returns BitsAndBytesConfig or None for baseline.
    """
    if config_id == "C-1":
        return None


    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=False,
    )

    return quantization_config


def calibrate_and_save(model_name, config_id, config_data, save_path):
    """
    Load model with quantization config and save it.
    For SGLang, we save model in a format it can load.
    """
    print(f"🚀 Starting quantization for {config_id}...")

    quant_config = get_quantization_config(config_id, config_data)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if config_id == "C-1":
        print("📦 Loading baseline model (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Quantized: Load with quantization config
        print(f"📦 Loading model with 8-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

    print(f"💾 Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ Model saved to {save_path}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()


def load_quantized_model(path):
    """
    Return the path to the saved model.
    SGLang will handle loading.
    """
    print(f"📂 Model path: {path}")
    return path
