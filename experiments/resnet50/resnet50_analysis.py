# Support both `python -m experiments.resnet50.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.resnet50"

import importlib
import torch
import torch.nn as nn
import functools
from utils.layer_versioning.models.D2SConvS2D import D2SConvS2D
from torchvision import transforms, datasets, models

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
    example_input = torch.randn(1, 3, 224, 224)

    model = models.resnet50(weights="IMAGENET1K_V1")
    model.eval()
    network_name = f"resnet50"

    maestro_model_string = model_analysis.export_maestro_conv_layers(
        model,
        example_input,
        network_name=network_name,
        fps=30,
        filename=f"{network_name}.m"
    )

    model_analysis.export_torch_layers(model, f"{network_name}_layers.txt")

    RETRAIN_LAYERS = [
        "layer2.0.conv1",  # Layer id 13 CONV3_1_1
        "layer3.0.conv1",  # Layer id 29 CONV4_1_1
        "layer3.0.conv2",  # Layer id 30 CONV4_1_2
        "layer4.0.conv1",  # Layer id 53 CONV5_1_1
        "layer4.0.conv2",  # Layer id 54 CONV5_1_2
        "layer4.1.conv2",  # Layer id 58 CONV5_2_2
        "layer4.2.conv2",  # Layer id 63 CONV5_3_2
    ]

    for idx, layer_path in enumerate(RETRAIN_LAYERS):
        model_name = f"resnet50_{idx}"
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.eval()
        model = replace_with_d2sconv_s2d(model, layer_path, r=2)

        # Convert to MAESTRO format
        network_name = f"resnet50_{idx}"

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
