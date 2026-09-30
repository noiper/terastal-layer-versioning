"""Terastal experiment loops using an external Sparse-to-Dense implementation."""
import argparse
import copy
import csv
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.checkpoints import require_checkpoint_files
from .checkpoint_compat import checkpoint_model, load_checkpoint
from .upstream import load_upstream

LAYERS = ('layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2')
METRICS = ('mse', 'rmse', 'absrel', 'lg10', 'mae', 'delta1', 'delta2', 'delta3')


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--upstream', help='External checkout; otherwise use TERASTAL_SP2DENSE_SOURCE')
    parser.add_argument('--checkpoint', '--evaluate', required=True, help='Trusted baseline checkpoint')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--workers', type=int, default=0)
    return parser


def prepare(options):
    require_checkpoint_files([options.checkpoint])
    load_upstream(options.upstream)
    checkpoint = load_checkpoint(options.checkpoint, map_location='cpu')
    device = options.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if options.workers < 0:
        raise ValueError('workers must be non-negative')
    return checkpoint, torch.device(device)


def data_loader(args, data_root, training=False, batch_size=1, workers=0):
    backend = load_upstream()
    classes = {'nyudepthv2': ('nyu_dataloader', 'NYUDataset'), 'kitti': ('kitti_dataloader', 'KITTIDataset')}
    if args.data not in classes:
        raise ValueError(f'Unknown checkpoint dataset: {args.data}')
    module, name = classes[args.data]
    dataset_class = getattr(backend.modules['dataloaders.' + module], name)
    sparse_module = backend.modules['dataloaders.dense_to_sparse']
    sparsifiers = {cls.name: cls for cls in (sparse_module.UniformSampling, sparse_module.SimulatedStereo)}
    sparsifier = sparsifiers[args.sparsifier](num_samples=args.num_samples,
                                            max_depth=args.max_depth if args.max_depth >= 0 else np.inf)
    dataset = dataset_class(str(Path(data_root) / args.data / ('train' if training else 'val')),
                            type='train' if training else 'val', modality=args.modality, sparsifier=sparsifier)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size if training else 1,
                                       shuffle=training, num_workers=workers)


def evaluate(model, loader, device):
    backend = load_upstream()
    meter = backend.metrics.AverageMeter()
    model.to(device).eval()
    with torch.no_grad():
        for inputs, target in loader:
            started = time.perf_counter()
            prediction = model(inputs.to(device))
            result = backend.metrics.Result()
            result.evaluate(prediction, target.to(device))
            if not all(math.isfinite(getattr(result, name)) for name in METRICS):
                raise ValueError('Depth evaluation produced non-finite metrics; check predictions and valid target pixels')
            meter.update(result, time.perf_counter() - started, 0, inputs.shape[0])
    if meter.count == 0:
        raise ValueError('Validation dataset is empty')
    return {name: getattr(meter.average(), name) for name in METRICS}


def assemble_combination(baseline, variants, layers):
    model = copy.deepcopy(baseline)
    for layer in layers:
        parent, name = layer.rsplit('.', 1)
        source = variants[layer].get_submodule(parent)
        target = model.get_submodule(parent)
        for component in (name, name.replace('conv', 'bn')):
            setattr(target, component, copy.deepcopy(getattr(source, component)))
    return model


def write_results(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write only completed runs. A failed run never replaces a previous table.
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
