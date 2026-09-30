"""Evaluate a Sp2Dense baseline checkpoint using an external checkout."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'experiments.sp2dense'

from .runtime import argument_parser, prepare, data_loader, checkpoint_model, evaluate, write_results, METRICS


def main(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument('--output', default='outputs/sp2dense_baseline.csv')
    options = parser.parse_args(argv)
    checkpoint, device = prepare(options)
    loader = data_loader(checkpoint['args'], options.data_root, workers=options.workers)
    model = checkpoint_model(checkpoint, loader.dataset.output_size)
    metrics = evaluate(model, loader, device)
    write_results(options.output, [metrics], METRICS)
    print(metrics)


if __name__ == '__main__':
    main()
