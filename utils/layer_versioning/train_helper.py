import torch
import torch.nn as nn
import functools
from contextlib import nullcontext
from .models.D2SConvS2D import D2SConvS2D

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

def setup_custom_layer_model(model, layers_to_replace=None):
    if layers_to_replace is not None:
        # Replace specified layers with D2SConvS2D
        for layer_path in layers_to_replace:
            try:
                model = replace_with_d2sconv_s2d(model, layer_path, r=2)
            except Exception as e:
                raise RuntimeError(f"Failed to replace {layer_path}") from e
    return model

def print_trainable_params(model):
    """Print information about trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print(f"📊 MODEL PARAMETERS SUMMARY")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Show which layers are trainable
    print("\n🔓 Trainable layers:")
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):
            print(f" - {name} ({module.__class__.__name__})")
    print("="*50)

def partial_train(model, train_layers):
    set_train_layers = set(train_layers)
    for name, module in model.named_modules():
        if name in set_train_layers:
            module.train()
            set_train_layers.remove(name)
        else:
            module.eval()
    if len(set_train_layers) > 0:
        assert False, f"Layers {set_train_layers} not found in model"

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_specific_layers(model, layer_names):
    set_layer_names = set(layer_names)
    for name, module in model.named_modules():
        if name in set_layer_names:
            set_layer_names.remove(name)
            print(f"🔓 Unfreezing layer: {name}")
            for param in module.parameters():
                param.requires_grad = True
    if len(set_layer_names) > 0:
        assert False, f"Layers {set_layer_names} not found in model"

class _StopForward(Exception):
    pass

def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    if hasattr(root, "module"):  # unwrap DP/DDP
        root = root.module
    mod = root
    for p in path.split("."):
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod

def get_partial_model(model: nn.Module, layer: str):
    """
    usage:
        get_feat = get_partial_model(model, "layer1.0.conv1")
        y = get_feat(x, stop_early=True, detach=True, no_grad=True)
    """
    target = _get_submodule(model, layer)

    def run(x, *args, stop_early: bool = False, detach: bool = False,
            no_grad: bool = True, **kwargs):
        captured = {}

        def hook(_m, _inp, out):
            captured["interm"] = out
            if stop_early:
                raise _StopForward

        handle = target.register_forward_hook(hook)

        was_training = model.training
        try:
            if no_grad:
                ctx = torch.no_grad()
            else:
                ctx = nullcontext()
            model.eval()
            with ctx:
                try:
                    captured["out"] = model(x, *args, **kwargs)
                except _StopForward:
                    captured["out"] = None
        finally:
            handle.remove()
            model.train(was_training)

        interm = captured["interm"]
        captured["interm"] = interm.detach() if detach else interm
        return captured

    return run

def _unit_test():
    from torchvision import models
    model = models.resnet50(pretrained=True)
    print(model)
    layers_to_unfreeze = ['layer4', 'fc']
    freeze_all_layers(model)
    unfreeze_specific_layers(model, layers_to_unfreeze)
    partial_train(model, layers_to_unfreeze)

    class test_cnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.resnet50(pretrained=True)
            self.head = nn.Linear(1000, 10)

        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            return x

    test_model = test_cnn()
    get_head = get_partial_model(test_model, "backbone")
    x = torch.randn(1, 3, 224, 224)
    y = get_head(x, stop_early=False, detach=True, no_grad=True)
    print(y["interm"].shape, y["out"].shape)

if __name__ == "__main__":
    _unit_test()