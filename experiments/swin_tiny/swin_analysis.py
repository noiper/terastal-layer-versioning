# Support both `python -m experiments.swin_tiny.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.swin_tiny"

import torch
import torch.nn as nn
import torch.optim as optim
import functools
from transformers import AutoModelForImageClassification, AutoImageProcessor
import importlib
from torch.utils.data import DataLoader

# Import the focused conversion utility from our separate module
from utils.layer_versioning.models import D2SConvS2D, LinearToConv, replace_linear_with_conv
model_analysis = importlib.import_module("utils.layer_versioning.model_analysis")

class SwinD2SWrapper(nn.Module):
    """
    A wrapper for D2SConvS2D to handle the 3D tensor shapes found in
    Swin Transformer models. This wrapper treats each token independently.
    """
    def __init__(self, d2s_module):
        super().__init__()
        self.d2s_module = d2s_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return self.d2s_module(x)
        B, N, C = x.shape
        x_reshaped = x.reshape(B * N, C).unsqueeze(-1).unsqueeze(-1)
        output_reshaped = self.d2s_module(x_reshaped)
        _B_N, C_out, _H, _W = output_reshaped.shape
        output = output_reshaped.view(B, N, C_out)
        return output

def replace_with_d2sconv_s2d(model: nn.Module, layer_path: str, r=2):
    """
    Replaces a target layer (Conv2d or Linear) with a D2SConvS2D module.
    """
    parts = layer_path.split(".")
    parent = functools.reduce(getattr, [model] + parts[:-1])
    name = parts[-1]
    orig = getattr(parent, name)
    was_linear = isinstance(orig, nn.Linear)
    
    if was_linear:
        print(f"Layer {layer_path} is Linear. Converting to Conv2d first.")
        replace_linear_with_conv(model, layer_path)
        orig = getattr(parent, name)

    conv_layer = orig.conv if isinstance(orig, LinearToConv) else orig
    
    if not isinstance(conv_layer, nn.Conv2d):
        raise TypeError(f"Target layer {layer_path} could not be resolved to a Conv2d layer.")

    k = conv_layer.kernel_size[0] if isinstance(conv_layer.kernel_size, tuple) else conv_layer.kernel_size
    p = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
    
    assert conv_layer.in_channels % (r*r) == 0, f"in_channels {conv_layer.in_channels} not divisible by r^2 ({r*r})"
    assert conv_layer.out_channels % (r*r) == 0, f"out_channels {conv_layer.out_channels} not divisible by r^2 ({r*r})"
    
    new_mod = D2SConvS2D(conv_layer.in_channels, conv_layer.out_channels,
                         kernel_size=k, stride=conv_layer.stride, padding=p, r=r,
                         groups=conv_layer.groups, bias=(conv_layer.bias is not None))
    
    if was_linear:
        setattr(parent, name, SwinD2SWrapper(new_mod))
    else:
        setattr(parent, name, new_mod)
        
    print(f"✅ Replaced {layer_path} with D2SConvS2D")
    return model

def main():
    example_input = torch.randn(1, 3, 224, 224)
    network_name = "swin-tiny"
    RETRAIN_LAYERS = [
        "swin.encoder.layers.3.blocks.0.intermediate.dense",
        "swin.encoder.layers.3.blocks.0.output.dense",
        "swin.encoder.layers.3.blocks.1.intermediate.dense",
        "swin.encoder.layers.3.blocks.1.output.dense"
    ]

    for idx, layer_path in enumerate(RETRAIN_LAYERS):
        
        model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        model = replace_with_d2sconv_s2d(model, layer_path, r=4)

        file_name = f"{network_name}_{idx}"
        model_analysis.export_maestro_conv_layers(
            model, example_input, network_name=network_name,
            fps=30, filename=f"{file_name}.m"
        )
        model_analysis.export_torch_layers(model, f"{file_name}_layers.txt")

if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description="Export the configured Swin-Tiny model; see README.").parse_args()
    main()
