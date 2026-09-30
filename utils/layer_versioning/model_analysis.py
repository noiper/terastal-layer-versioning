import re
import torch
from torch import nn
from collections import OrderedDict

def _sanitize_layer_name(name: str, prefix: str = None) -> str:
    """Turn a module path like 'layer1.0.conv1' into a MAESTRO-friendly name.
    Example: prefix='pretrained' -> 'pretrained_layer1_0_conv1'
    """
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name.replace('.', '_'))
    return f"{prefix}_{safe}" if prefix else safe

@torch.no_grad()
def export_torch_layers(model, filepath):
    with open(filepath, 'w') as f:
        f.write(repr(model))

@torch.no_grad()
def export_maestro_conv_layers(model: nn.Module,
                               example_input,
                               network_name: str = "Network",
                               fps: int | None = None,
                               filename: str | None = None,
                               name_prefix: str | None = None,
                               include_residual: bool = True) -> str:
    """
    Run one forward pass, capture every nn.Conv2d, nn.ConvTranspose2d, and nn.Linear (as 1x1 CONV) layer's I/O to emit MAESTRO model text. Optionally emit residual shortcuts and Unpool ops (as TRCONV).

    Args:
        model: PyTorch model.
        example_input: A tensor `x` or a tuple `(x, *args)` forwarded to `model`.
        network_name: Name to put in the MAESTRO header.
        fps: Optional FPS line; if None, it's omitted.
        filename: If set, writes the MAESTRO text to this path.
        name_prefix: Optional prefix added to each layer name (e.g., 'pretrained').
        include_residual: If True, emit residual shortcuts as DSCONV layers. Downsample convs inside residual blocks are skipped to avoid double counting.

    Returns:
        The MAESTRO model description string.
    """
    device = next(model.parameters(), torch.empty(0)).device

    # Map module object -> qualified name, and register hooks in forward order
    name_by_module = {m: n for n, m in model.named_modules()}

    # Known residual block class names (torchvision ResNet, etc.)
    _RESIDUAL_BLOCK_NAMES = {"Bottleneck", "BasicBlock"}

    records: list[dict] = []
    handles = []

    def _as_2tuple(x):
        if isinstance(x, tuple):
            return x
        return (x, x)

    def make_hook(mod: nn.Conv2d):
        def _hook(m: nn.Conv2d, inp, out):
            mod_name = name_by_module[m]
            # Skip explicit downsample convs; the residual connection will be represented as a DSCONV layer.
            if "downsample" in mod_name.split('.'):
                return
            try:
                # Input shape (N, C_in, H_in, W_in)
                i = inp[0] if isinstance(inp, (list, tuple)) else inp
                n, c_in, h_in, w_in = i.shape
                # Output (to ensure the module actually ran)
                o = out if isinstance(out, torch.Tensor) else out[0]
                _ = o.shape  # not used now except to validate
            except Exception:
                return

            in_ch = m.in_channels
            out_ch = m.out_channels
            groups = m.groups
            kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size))
            sy, sx = _as_2tuple(m.stride)

            is_depthwise = (groups == in_ch)
            if is_depthwise:
                multiplier = max(1, out_ch // max(1, in_ch))
                K = multiplier
                C = in_ch
                typ = 'DSCONV'
            else:
                K = out_ch
                C = in_ch
                typ = 'CONV'

            records.append({
                'name': _sanitize_layer_name(name_by_module[m], prefix=name_prefix),
                'Type': typ,
                'N': int(n),
                'K': int(K),
                'C': int(C),
                'Y': int(h_in),
                'X': int(w_in),
                'R': int(kh),
                'S': int(kw),
                'stride_y': int(sy),
                'stride_x': int(sx),
            })
        return _hook

    def make_tconv_hook(mod: nn.ConvTranspose2d):
        def _hook(m: nn.ConvTranspose2d, inp, out):
            try:
                i = inp[0] if isinstance(inp, (list, tuple)) else inp
                n, c_in, h_in, w_in = i.shape
                o = out if isinstance(out, torch.Tensor) else out[0]
                n2, c_out, h_out, w_out = o.shape
            except Exception:
                return
            kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size))
            sy, sx = _as_2tuple(m.stride)
            records.append({
                'name': _sanitize_layer_name(name_by_module[m], prefix=name_prefix),
                'Type': 'TRCONV',
                'N': int(n),
                'K': int(m.out_channels),
                'C': int(m.in_channels),
                'Y': int(h_out),
                'X': int(w_out),
                'R': int(kh),
                'S': int(kw),
                'stride_y': int(sy),
                'stride_x': int(sx),
            })
        return _hook

    def make_fc_hook(mod: nn.Linear):
        def _hook(m: nn.Linear, inp, out):
            try:
                i = inp[0] if isinstance(inp, (list, tuple)) else inp
                n = int(i.shape[0])
                _ = out.shape  # validate
            except Exception:
                return
            records.append({
                'name': _sanitize_layer_name(name_by_module[m], prefix=name_prefix),
                'Type': 'CONV',          # represent FC as 1x1 conv
                'N': n,
                'K': int(m.out_features),
                'C': int(m.in_features),
                'Y': 1,
                'X': 1,
                'R': 1,
                'S': 1,
                'stride_y': 1,
                'stride_x': 1,
            })
        return _hook

    def make_residual_hook(mod: nn.Module):
        def _hook(m: nn.Module, inp, out):
            # Represent the elementwise add as a depthwise 1x1 conv with K=1, C=channels, stride=1
            try:
                o = out if isinstance(out, torch.Tensor) else out[0]
                n, c, h, w = o.shape
            except Exception:
                return
            records.append({
                'name': _sanitize_layer_name(name_by_module[m] + "_Residual", prefix=name_prefix),
                'Type': 'DSCONV',
                'N': int(n),
                'K': 1,
                'C': int(c),
                'Y': int(h),
                'X': int(w),
                'R': 1,
                'S': 1,
                'stride_y': 1,
                'stride_x': 1,
            })
        return _hook

    def make_unpool_hook(mod: nn.Module):
        def _hook(m: nn.Module, inp, out):
            try:
                o = out if isinstance(out, torch.Tensor) else out[0]
                n, c, h, w = o.shape
            except Exception:
                return
            stride = int(getattr(m, 'stride', 2))
            channels = int(getattr(m, 'num_channels', c))
            records.append({
                'name': _sanitize_layer_name(name_by_module[m] + '_Unpool', prefix=name_prefix),
                'Type': 'TRCONV',
                'N': int(n),
                'K': channels,
                'C': channels,
                'Y': int(h),
                'X': int(w),
                'R': stride,
                'S': stride,
                'stride_y': stride,
                'stride_x': stride,
            })
        return _hook

    # Register on all Conv2d, ConvTranspose2d and Linear modules
    for mod in model.modules():
        cls_name = mod.__class__.__name__
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(make_hook(mod)))
        elif isinstance(mod, nn.ConvTranspose2d):
            handles.append(mod.register_forward_hook(make_tconv_hook(mod)))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_fc_hook(mod)))
        elif include_residual and cls_name in _RESIDUAL_BLOCK_NAMES:
            handles.append(mod.register_forward_hook(make_residual_hook(mod)))

    # Run one pass in eval mode
    was_training = model.training
    model.eval()
    try:
        if isinstance(example_input, tuple):
            example_input = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in example_input)
            _ = model(*example_input)
        else:
            _ = model(example_input.to(device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    # Emit MAESTRO text
    lines = [f"Network {network_name} {{\n"]
    if fps is not None:
        lines.append(f"\nFPS: {int(fps)}\n\n")

    for rec in records:
        lines.append(f"Layer {rec['name']} {{ \n")
        lines.append(f"\tType: {rec['Type']}\n")
        lines.append(f"\tStride {{ X: {rec['stride_x']}, Y: {rec['stride_y']} }}\n")
        lines.append(
            f"\tDimensions: {{ N: {rec['N']}, K: {rec['K']}, C: {rec['C']}, Y: {rec['Y']}, X: {rec['X']}, R: {rec['R']}, S: {rec['S']} }}\n"
        )
        lines.append("}\n\n")

    lines.append("}\n")
    text = "".join(lines)

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(text)
    return text