"""Load a user-supplied Sparse-to-Dense checkout without vendoring its source."""
import collections.abc
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_URL = 'https://github.com/fangchangma/sparse-to-dense.pytorch.git'
TESTED_REVISION = '10efc6d60bddedd6f28f0532c108bb1d7ccdfc49'
NAMESPACE = '_terastal_sp2dense_upstream'
_backend = None


class DeviceUnpool(nn.Module):
    """Insert zeros between pixels; keep historical checkpoint parameter keys."""
    def __init__(self, num_channels, stride=2):
        super().__init__()
        self.num_channels = num_channels
        self.stride = stride
        kernel = torch.zeros(num_channels, 1, stride, stride)
        kernel[:, :, 0, 0] = 1
        self.register_buffer('weights', kernel, persistent=False)

    def forward(self, value):
        # Historical whole-model files have a plain weights attribute.
        kernel = self.weights.to(value)
        return F.conv_transpose2d(value, kernel, stride=self.stride, groups=self.num_channels)


def _imresize(array, size, interp='bilinear', mode=None):
    """Pillow equivalent of the upstream scipy.misc.imresize call sites."""
    image = Image.fromarray(array, mode=mode)
    if isinstance(size, float):
        width, height = (int(dimension * size) for dimension in image.size)
    elif isinstance(size, int):
        width, height = (int(dimension * size / 100) for dimension in image.size)
    else:
        height, width = size
    resampling = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR,
                  'bicubic': Image.Resampling.BICUBIC, 'lanczos': Image.Resampling.LANCZOS}
    return np.asarray(image.resize((width, height), resample=resampling[interp]))


class _NumpyCompat:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asfarray(value, dtype=float):
        return np.asarray(value, dtype=dtype)


def load_upstream(path=None):
    """Load only model/data/metric modules, never upstream training entry points."""
    global _backend
    if path is None and _backend is not None:
        return _backend
    location = path or os.environ.get('TERASTAL_SP2DENSE_SOURCE')
    if not location:
        if _backend is not None:
            return _backend
        raise RuntimeError('Set TERASTAL_SP2DENSE_SOURCE or pass --upstream to your external checkout; see experiments/sp2dense/README.md')
    root = Path(location).expanduser().resolve()
    if _backend is not None:
        if root != _backend.root:
            raise RuntimeError('Use a new Python process to switch Sp2Dense checkouts')
        return _backend
    files = ['models', 'metrics', 'criteria', 'dataloaders.transforms',
             'dataloaders.dense_to_sparse', 'dataloaders.dataloader',
             'dataloaders.nyu_dataloader', 'dataloaders.kitti_dataloader']
    missing = [name for name in files if not (root / (name.replace('.', '/') + '.py')).is_file()]
    if missing:
        raise FileNotFoundError(f'Invalid Sp2Dense checkout {root}: missing {missing}')
    loaded = {}
    for package, directory in [(NAMESPACE, root), (NAMESPACE + '.dataloaders', root / 'dataloaders')]:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    setattr(sys.modules[NAMESPACE], 'dataloaders', sys.modules[NAMESPACE + '.dataloaders'])
    try:
        for name in files:
            fullname = NAMESPACE + '.' + name
            filename = root / (name.replace('.', '/') + '.py')
            spec = importlib.util.spec_from_file_location(fullname, filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            parent, child = fullname.rsplit('.', 1)
            setattr(sys.modules[parent], child, module)
            # Scope old absolute imports to this checkout, without editing it or
            # replacing unrelated global modules named models/dataloaders.
            source = filename.read_text().replace('import dataloaders.', f'import {NAMESPACE}.dataloaders.')
            source = source.replace('from dataloaders.', f'from {NAMESPACE}.dataloaders.')
            exec(compile(source, str(filename), 'exec'), module.__dict__)
            if name == 'models':
                module.Unpool = DeviceUnpool
            if name == 'dataloaders.transforms':
                module.collections = types.SimpleNamespace(Iterable=collections.abc.Iterable)
                module.misc = types.SimpleNamespace(imresize=_imresize)
            if hasattr(module, 'np'):
                module.np = _NumpyCompat()
            loaded[name] = module
    except Exception:
        for name in list(sys.modules):
            if name == NAMESPACE or name.startswith(NAMESPACE + '.'):
                del sys.modules[name]
        raise
    _backend = types.SimpleNamespace(root=root, modules=loaded,
                                     models=loaded['models'], metrics=loaded['metrics'],
                                     criteria=loaded['criteria'])
    return _backend
