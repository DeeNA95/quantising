import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def get_layer_names(num_layers, target_modules=None, skip_modules=None):
    """
    Helper to generate list of module names to skip or target.
    Llama 3 structure: model.layers.{i}.self_attn.{q,k,v,o}_proj, model.layers.{i}.mlp.{gate,up,down}_proj
    """
    # This is a heuristic. For robust usage we might check model structure, but hardcoding for Llama3 is faster.
    layers = []

    # Common Llama-3 module suffixes
    attn_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    mlp_modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    all_submodules = attn_modules + mlp_modules

    skips = []

    # Sandwich logic handled by caller passing skip ranges

    return skips

def get_quantization_info(config_id, config_data, model_name):
    """
    Determine if we can use native vllm quantization or need to save a custom model.
    Returns: (needs_save, quant_arg, bnb_config)
    """

    strategy = config_data.get("strategy")

    if config_id == "C-1":
        # Baseline: No quantization
        return False, None, None

    if config_id == "C-2":
        # Uniform W8A16: Standard bitsandbytes 8bit
        # vllm supports "bitsandbytes" load format, but checking if it supports
        # on-the-fly quantization of a BF16 model -> bitsandbytes isn't standard in vllm args like "fp8" is.
        # But for 8-bit, usually we load a saved 8-bit model.
        # However, vllm can load with `load_format="bitsandbytes"` IF the model config says so.
        # If the base model is BF16, we can pass `quantization="bitsandbytes"`?
        # vllm docs say: `quantization="bitsandbytes"` is supported.
        # So we can try native.
        return False, "bitsandbytes", None

    # For E-A, E-B, E-C (Selective/Sandwich), we need custom skipping.
    # vllm args don't support "skip these layers" easily.
    # So we MUST save the model with the config config.

    needs_save = True
    quant_arg = "bitsandbytes" # The saved model will be a BnB model

    # Build BnB config with skips
    # We need to know num_layers. Llama-3-8B has 32 layers.
    num_layers = 32 # Hardcoding for Llama-3-8B as per config

    modules_to_skip = ["lm_head", "model.norm"] # Always skip head/norm usually for INT8
    # Config defaults also mention these.

    if strategy == "sandwich":
        sandwich_cfg = config_data.get("sandwich", {})
        keep_first = sandwich_cfg.get("keep_first_layers", 0)
        keep_last = sandwich_cfg.get("keep_last_layers", 0)

        # Skip quantization for these layers
        for i in range(keep_first):
            modules_to_skip.append(f"model.layers.{i}") # simplistic prefix match?
            # Transformers check: if module name *contains* or *endswith*?
            # BnB implementation usually checks if name is in list.
            # We might need full names for all submodules if it requires exact match.
            # Usually strict match or recursion.
            # Safest is to list the main layer block if transformers supports it, or extensive list.
            # Let's assume we need to be specific if recursive matching isn't guaranteed.
            pass

    # For simplicity, we will instantiate the config.
    # Note: Implementing complex selective logic dynamically is error-prone.
    # We will try to map high-level intent.

    # Re-reading config:
    # E-A: MLP-only (Target MLP) -> Skip Attn
    # E-B: Attention-only (Target Attn) -> Skip MLP

    # We will do a generic pass to generate skip list.
    skip_list = list(modules_to_skip)

    # Helper to expand 'model.layers.X' into submodules
    def expand_layer(layer_idx, submodules):
        return [f"model.layers.{layer_idx}.{sub}" for sub in submodules]

    attn_subs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    mlp_subs = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    if strategy == "selective":
        targets = config_data.get("targets", [])
        if "mlp" in targets and "attention" not in targets:
            # Target MLP -> Skip Attention
            for i in range(num_layers):
                skip_list.extend(expand_layer(i, attn_subs))
        elif "attention" in targets and "mlp" not in targets:
            # Target Attention -> Skip MLP
            for i in range(num_layers):
                skip_list.extend(expand_layer(i, mlp_subs))

    elif strategy == "sandwich":
        keep_first = config_data.get("sandwich", {}).get("keep_first_layers", 0)
        keep_last = config_data.get("sandwich", {}).get("keep_last_layers", 0)

        # Skip first N
        for i in range(keep_first):
             skip_list.extend(expand_layer(i, attn_subs + mlp_subs))

        # Skip last N
        start_last = num_layers - keep_last
        for i in range(start_last, num_layers):
             skip_list.extend(expand_layer(i, attn_subs + mlp_subs))

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=skip_list,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    return True, "bitsandbytes", bnb_config


def calibrate_and_save(model_name, config_id, config_data, save_path):
    """
    Orchestrate quantization.
    Returns: (path_to_load_in_vllm, quantization_arg)
    """
    print(f"🚀 Preparing model for {config_id}...")

    needs_save, quant_arg, bnb_config = get_quantization_info(config_id, config_data, model_name)

    if not needs_save:
        print(f"⏩ functionality natively supported by vllm (arg='{quant_arg}'). Skipping save.")
        return model_name, quant_arg

    # Check if already saved?
    if os.path.exists(save_path) and os.path.isdir(save_path):
         # basic check
         if os.path.exists(os.path.join(save_path, "config.json")):
             print(f"Found existing saved model at {save_path}, using it.")
             return save_path, None # Config.json should have the quantization info

    print(f"📦 Loading and Quantizing for {config_id} (Strategy: {config_data.get('strategy')})...")

    # Load with bnb config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"💾 Saving quantized model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    del model
    torch.cuda.empty_cache()

    # When loading a saved bitsandbytes model in vllm, we usually let it detect from config.json
    # So we pass quantization=None (or rely on auto detection).
    # But explicitly, if it's a BnB model, vllm might want `quantization='bitsandbytes'`.
    # Usually 'bitsandbytes' is safe to pass if we know it is one.
    return save_path, "bitsandbytes"

def load_quantized_model(path):
    # This function is now deprecated in favor of calibrate_and_save returning the path directly
    return path
