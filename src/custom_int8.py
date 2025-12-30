import torch
import torch.nn as nn


class INT8Linear(nn.Module):
    """
    A simple wrapper that stores weights in INT8 and dequantizes to BF16 for compute.
    This is a 'storage-only' quantization for VRAM savings.
    """

    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Quantize weights to INT8 with per-tensor scaling
        w_data = original_linear.weight.data.to(torch.bfloat16)
        self.scale = w_data.abs().max() / 127.0  # INT8 range is -128 to 127

        # Store as INT8
        self.weight = nn.Parameter(
            (w_data / self.scale).round().to(torch.int8), requires_grad=False
        )

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.to(torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # Dequantize to BF16 for computation
        w_bf16 = self.weight.to(torch.bfloat16) * self.scale
        return nn.functional.linear(x, w_bf16, self.bias)
