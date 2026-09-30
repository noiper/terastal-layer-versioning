# Support both `python -m experiments.mobilenetv2_ssd.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.mobilenetv2_ssd"

import importlib
import torch
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

model_analysis = importlib.import_module("utils.layer_versioning.model_analysis")

def main():
    num_classes = 21

    net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)

    model_path = "models/mb2-ssd-lite.pth"
    net.load(model_path)
    net.eval()

    cpu_device = torch.device("cpu")

    net.to(cpu_device)
    if hasattr(net, 'priors'):
        net.priors = net.priors.to(cpu_device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    # example_input = torch.randn(1, 3, 300, 300)
    # network_name = "mb2-ssd-lite"

    # maestro_model_string = model_analysis.export_maestro_conv_layers(
    #     net,
    #     example_input,
    #     network_name=network_name,
    #     fps=30,
    #     filename=f"{network_name}.m"
    # )

    # model_analysis.export_torch_layers(net, f"{network_name}_layers.txt")

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Export/profile the configured model; see this experiment's README.").parse_args()
    main()
