"""Evaluate the 16 Sp2Dense layer combinations from trusted checkpoints."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'experiments.sp2dense'

from itertools import combinations
from pathlib import Path

from experiments.checkpoints import require_checkpoint_files
from .checkpoint_compat import load_checkpoint, checkpoint_model
from .runtime import (argument_parser, prepare, data_loader, evaluate, assemble_combination,
                      write_results, LAYERS, METRICS)


def main(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument('--variants-dir', default='results')
    parser.add_argument('--variant-filename', default='checkpoint-0.pth.tar')
    parser.add_argument('--output', default='outputs/layer_combination_results.csv')
    options = parser.parse_args(argv)
    checkpoint, device = prepare(options)
    args = checkpoint['args']
    paths = {layer: Path(options.variants_dir) /
             f"sparse_to_dense_{args.arch}_{args.decoder}_{layer.replace('.', '_')}" /
             options.variant_filename for layer in LAYERS}
    require_checkpoint_files(paths.values())
    loader = data_loader(args, options.data_root, workers=options.workers)
    baseline = checkpoint_model(checkpoint, loader.dataset.output_size)
    variants = {layer: checkpoint_model(load_checkpoint(path, map_location='cpu'),
                                        loader.dataset.output_size, [layer])
                for layer, path in paths.items()}
    rows = []
    for count in range(len(LAYERS) + 1):
        for layers in combinations(LAYERS, count):
            model = assemble_combination(baseline, variants, layers)
            metrics = evaluate(model, loader, device)
            rows.append({'combination_id': len(rows), 'num_layers': count,
                         'layers_list': '|'.join(layers) or 'baseline', **metrics})
    write_results(options.output, rows, ['combination_id', 'num_layers', 'layers_list', *METRICS])
    print(f'Saved {len(rows)} combinations to {options.output}')


if __name__ == '__main__':
    main()
