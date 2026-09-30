# Support both `python -m experiments.mobilenetv2_ssd.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.mobilenetv2_ssd"

import importlib
import torch
import torch.nn as nn
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
import functools
from .profiling_layer import D2SConvS2D

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

def main():
    num_classes = 21
    model_path = "models/mb2-ssd-lite.pth"

    cpu_device = torch.device("cpu")

    example_input = torch.randn(1, 3, 300, 300)

    RETRAIN_LAYERS = [
        # "base_net.2.conv.3",
        "classification_headers.0.3", # r%4 != 0; choose r=3
        "base_net.17.conv.6",
        "base_net.18.0",
        "extras.0.conv.0",
    ]

    R = [3, 2, 2, 2]

    for idx, layer_path in enumerate(RETRAIN_LAYERS):
        model = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
        model.load(model_path)
        model.eval()
        model.to(cpu_device)
        if hasattr(model, 'priors'):
            model.priors = model.priors.to(cpu_device)

        layers_to_replace = [layer_path]
        model = replace_with_d2sconv_s2d(model, layer_path, r=R[idx])

        # Convert to MAESTRO format
        network_name = f"mb2-ssd-lite_{idx}"

        maestro_model_string = model_analysis.export_maestro_conv_layers(
            model,
            example_input,
            network_name=network_name,
            fps=30,
            filename=f"{network_name}.m"
        )

        model_analysis.export_torch_layers(model, f"{network_name}_layers.txt")

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Export/profile the configured model; see this experiment's README.").parse_args()
    main()
