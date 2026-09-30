import torch
import torch.nn as nn
import functools

class LinearToConv(nn.Module):
    """
    A module that wraps a 1x1 Conv2d layer to behave like a Linear layer
    by handling input/output shape transformations internally.
    This allows it to be a drop-in replacement for a Linear layer.
    """
    def __init__(self, linear_layer: nn.Linear):
        """
        Initializes the 1x1 Conv2d layer and copies the weights from the
        provided Linear layer.

        Args:
            linear_layer (nn.Linear): The linear layer to be converted.
        """
        super().__init__()
        if not isinstance(linear_layer, nn.Linear):
            raise TypeError(f"Expected input to be of type nn.Linear, but got {type(linear_layer)}")

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        has_bias = linear_layer.bias is not None

        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=has_bias)

        self.conv.weight.data.copy_(linear_layer.weight.data.reshape(out_features, in_features, 1, 1))
        
        if has_bias:
            self.conv.bias.data.copy_(linear_layer.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass. It handles the shape transformations
        required to make a Conv2d layer behave like a Linear layer.
        """
        # Swin Transformer inputs to Linear layers can be 3D (batch, tokens, features)
        # or 2D. We only need to reshape if it's 2D.
        if x.dim() == 2:
            # Reshape to (batch_size, features, 1, 1)
            x = x.unsqueeze(-1).unsqueeze(-1)
        
        # If input is 3D, e.g. (batch, 49, 768), reshape to 4D for convolution
        elif x.dim() == 3:
             # Assuming H=W, find the spatial dim. Ex: 49 -> 7x7
            H = W = int(x.shape[1]**0.5)
            if H * W != x.shape[1]:
                # This case might happen if the token dim isn't a perfect square,
                # which is not typical for dense layers in vision models but is a safeguard.
                # We add dummy spatial dims.
                 x = x.unsqueeze(-1).unsqueeze(-1)
            else:
                 # Reshape to (batch, features, H, W)
                 x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], H, W)


        x = self.conv(x)
        return torch.flatten(x, 1)

    def __repr__(self):
        """
        Overrides the default string representation to display only the
        internal Conv2d layer for a cleaner model summary.
        """
        return self.conv.__repr__()


def replace_linear_with_conv(model: nn.Module, layer_path: str):
    """
    Finds a nn.Linear layer by its path in a model and replaces it
    in-place with a LinearToConv module.

    Args:
        model (nn.Module): The model to modify.
        layer_path (str): The path to the Linear layer (e.g., 'classifier.0').
    """
    try:
        parts = layer_path.split(".")
        parent = functools.reduce(getattr, [model] + parts[:-1])
        name = parts[-1]
        linear_layer = getattr(parent, name)
    except AttributeError:
        print(f"Warning: Layer with path '{layer_path}' not found. Model is unchanged.")
        return

    if not isinstance(linear_layer, nn.Linear):
         raise TypeError(f"Layer '{layer_path}' is of type {type(linear_layer).__name__}, not nn.Linear.")
    
    replacement = LinearToConv(linear_layer)
    setattr(parent, name, replacement)
    print(f"✅ Replaced {layer_path} with a Conv2d-equivalent layer.")