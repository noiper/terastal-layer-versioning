# Support both `python -m experiments.vgg11.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.vgg11"

import importlib
import torch
import torch.nn as nn
import torchvision.models as models
import functools
from utils.layer_versioning.models import D2SConvS2D, LinearToConv, replace_linear_with_conv

model_analysis = importlib.import_module("utils.layer_versioning.model_analysis")

class VGGD2SWrapper(nn.Module):
    """
    A wrapper for D2SConvS2D to handle the 3D tensor shapes found in
    VGG models. This wrapper treats each token independently.
    """
    def __init__(self, d2s_module):
        super().__init__()
        self.d2s_module = d2s_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.shape
        x_reshaped = x.reshape(N, C).unsqueeze(-1).unsqueeze(-1)
        output_reshaped = self.d2s_module(x_reshaped)
        _B_N, C_out, _H, _W = output_reshaped.shape
        output = output_reshaped.view(N, C_out)
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
        setattr(parent, name, VGGD2SWrapper(new_mod))
    else:
        setattr(parent, name, new_mod)
        
    print(f"✅ Replaced {layer_path} with D2SConvS2D")
    return model

def profile(model, input_size):
    from thop import profile, clever_format
    shape = (1, 3, input_size, input_size)

    model.eval()
    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    macs, params = profile(model, inputs=(torch.zeros(shape),), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    print(f'MACs: {macs}')
    print(f'Parameters: {params}')

RETRAIN_LAYERS = [
    "features.8",
    "features.16",
    "features.18",
    "classifier.0",
    "classifier.3",
]

# Historical profiling ratio; training/evaluation use r=2.
R = [2, 2, 2, 4, 2]

def main(classifier_r=4):
    weights_vgg11 = models.VGG11_Weights.DEFAULT
    example_input = torch.randn(1, 3, 224, 224)
    network_name = "vgg11"

    model = models.vgg11(weights=weights_vgg11)
    model.eval()
    profile(model, 224)

    for idx, layer_path in enumerate(RETRAIN_LAYERS):
        model = models.vgg11(weights=weights_vgg11)
        model.eval()
        layers_to_replace = [layer_path]
        model = replace_with_d2sconv_s2d(model, layer_path, r=classifier_r if idx == 3 else R[idx])

        profile(model, 224)
        print("===============================================")

        # Convert to MAESTRO format
        # maestro_model_string = model_analysis.export_maestro_conv_layers(
        #     model,
        #     example_input,
        #     network_name=network_name,
        #     fps=30,
        #     filename=f"{network_name}_{idx}.m"
        # )

        # model_analysis.export_torch_layers(model, f"{network_name}_{idx}layers.txt")
        # print(f"finish {layer_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile VGG11 variants")
    parser.add_argument("--classifier-r", type=int, choices=[2, 4], default=4, help="Use 2 to match training/evaluation; 4 preserves historical profiling")
    main(parser.parse_args().classifier_r)
