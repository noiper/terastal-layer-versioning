"""Resolve module names in legacy Sp2Dense checkpoints after relocation."""

import pickle
import copy
from collections.abc import Mapping

import torch


class LegacyUnpickler(pickle.Unpickler):
    """Remap known project module paths without replacing global sys.modules."""

    def find_class(self, module, name):
        short_name = module.removeprefix('experiments.sp2dense.')
        if short_name in {"models", "metrics", "criteria"}:
            from .upstream import load_upstream, NAMESPACE
            load_upstream()
            module = f"{NAMESPACE}.{short_name}"
        elif module == "layer_versioning_utils" or module.startswith("layer_versioning_utils."):
            module = "utils" + module[len("layer_versioning_utils"):]
        return super().find_class(module, name)


class _LegacyPickle:
    __name__ = "terastal_sp2dense_legacy_pickle"
    Unpickler = LegacyUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def load_checkpoint(path, **kwargs):
    """Load the same trusted research checkpoints with their old class paths."""
    kwargs.setdefault("weights_only", False)
    return torch.load(path, pickle_module=_LegacyPickle, **kwargs)


def checkpoint_model(checkpoint, output_size, layer_paths=()):
    """Restore either historical model objects or state dictionaries on CPU.

    State dictionaries require checkpoint architecture metadata and, for a
    variant, the replaced layer paths. Loading saved weights never downloads
    ImageNet weights. Whole-model objects retain their recorded architecture.
    """
    from .upstream import load_upstream
    from utils.layer_versioning.train_helper import setup_custom_layer_model

    stored = checkpoint['model']
    if isinstance(stored, torch.nn.Module):
        return copy.deepcopy(stored).cpu()
    if not isinstance(stored, Mapping):
        raise TypeError("checkpoint['model'] must be a model or a state dictionary")
    args = checkpoint['args']
    if args.arch not in ('resnet18', 'resnet50'):
        raise ValueError(f'Unsupported checkpoint architecture: {args.arch}')
    model = load_upstream().models.ResNet(layers=int(args.arch.removeprefix('resnet')),
                   decoder=args.decoder, output_size=output_size,
                   in_channels=len(args.modality), pretrained=False)
    if layer_paths:
        model = setup_custom_layer_model(model, layers_to_replace=list(layer_paths))
    model.load_state_dict(stored, strict=True)
    return model
