# Support both `python -m experiments.inceptionv3.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.inceptionv3"

import importlib
import torch
import torch.nn as nn
import torchvision.models as models
import functools
from utils.layer_versioning.models import D2SConvS2D

model_analysis = importlib.import_module("utils.layer_versioning.model_analysis")

def replace_with_d2sconv_s2d(model, layer_path: str, r=2):
    parts = layer_path.split(".")
    parent = functools.reduce(getattr, [model] + parts[:-1])
    name = parts[-1]
    orig = getattr(parent, name)
    
    assert isinstance(orig, nn.Conv2d), "Target must be nn.Conv2d"
    k = orig.kernel_size[0] if isinstance(orig.kernel_size, tuple) else orig.kernel_size
    p = orig.padding[0] if isinstance(orig.padding, tuple) else orig.padding
    
    # Check divisibility
    assert orig.in_channels % (r*r) == 0, f"in_channels {orig.in_channels} not divisible by r^2 ({r*r})"
    assert orig.out_channels % (r*r) == 0, f"out_channels {orig.out_channels} not divisible by r^2 ({r*r})"
    
    new_mod = D2SConvS2D(orig.in_channels, orig.out_channels,
                         kernel_size=k, stride=orig.stride, padding=p, r=r,
                         groups=orig.groups, bias=(orig.bias is not None))
    
    setattr(parent, name, new_mod)
    print(f"✅ Replaced {layer_path} with D2SConvS2D")
    return model

RETRAIN_LAYERS = [
    "Mixed_6a.branch3x3.conv"
]

def main():
    weights = models.Inception_V3_Weights.DEFAULT
    example_input = torch.randn(1, 3, 299, 299)
    network_name = "inceptionv3"

    for idx, layer_path in enumerate(RETRAIN_LAYERS):
        model = models.inception_v3(weights=weights)
        model.eval()
        layers_to_replace = [layer_path]
        model = replace_with_d2sconv_s2d(model, layer_path, r=2)

        # Convert to MAESTRO format
        maestro_model_string = model_analysis.export_maestro_conv_layers(
            model,
            example_input,
            network_name=network_name,
            fps=30,
            filename=f"{network_name}_{idx}.m"
        )

        model_analysis.export_torch_layers(model, f"{network_name}_{idx}layers.txt")
        print(f"finish {layer_path}")

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Export/profile the configured model; see this experiment's README.").parse_args()
    main()
