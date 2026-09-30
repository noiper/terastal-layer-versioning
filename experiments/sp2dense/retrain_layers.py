"""Retrain Terastal's four Sp2Dense variants with a user-supplied upstream checkout."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'experiments.sp2dense'

import copy
from pathlib import Path

import torch

from utils.layer_versioning.train_helper import (freeze_all_layers, unfreeze_specific_layers,
                                                setup_custom_layer_model, partial_train)
from utils.layer_versioning import model_analysis
from .upstream import load_upstream
from .runtime import (argument_parser, prepare, data_loader, checkpoint_model, evaluate,
                      write_results, LAYERS, METRICS)


def train_epoch(model, loader, optimizer, criterion, trainable, device):
    partial_train(model, trainable)
    for inputs, target in loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs.to(device)), target.to(device))
        if not torch.isfinite(loss):
            raise ValueError('Training produced a non-finite loss')
        loss.backward()
        optimizer.step()


def main(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--layers', nargs='+', choices=LAYERS, default=LAYERS)
    options = parser.parse_args(argv)
    if options.epochs < 1 or options.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    checkpoint, device = prepare(options)
    args = copy.deepcopy(checkpoint['args'])
    args.epochs, args.batch_size = options.epochs, options.batch_size
    train = data_loader(args, options.data_root, training=True, batch_size=options.batch_size, workers=options.workers)
    validation = data_loader(args, options.data_root, workers=options.workers)
    loss_class = {'l1': load_upstream().criteria.MaskedL1Loss, 'l2': load_upstream().criteria.MaskedMSELoss}[args.criterion]
    for layer in options.layers:
        model = checkpoint_model(checkpoint, validation.dataset.output_size)
        setup_custom_layer_model(model, [layer])
        freeze_all_layers(model)
        parent, name = layer.rsplit('.', 1)
        trainable = [layer, parent + '.' + name.replace('conv', 'bn')]
        unfreeze_specific_layers(model, trainable)
        model.to(device)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        output = Path(options.output_dir) / f"sparse_to_dense_{args.arch}_{args.decoder}_{layer.replace('.', '_')}"
        output.mkdir(parents=True, exist_ok=True)
        example = next(iter(validation))[0].to(device)
        model_analysis.export_maestro_conv_layers(model, example, network_name=args.arch,
                                                  filename=str(output / 'variant.m'))
        model_analysis.export_torch_layers(model, str(output / 'variant_layers.txt'))
        best_rmse, rows = float('inf'), []
        for epoch in range(options.epochs):
            for group in optimizer.param_groups:
                group['lr'] = args.lr * (0.1 ** (epoch // 5))
            train_epoch(model, train, optimizer, loss_class().to(device), trainable, device)
            metrics = evaluate(model, validation, device)
            rows.append({'epoch': epoch, **metrics})
            saved = {'args': args, 'model': model.state_dict(), 'epoch': epoch,
                     'metrics': metrics, 'optimizer': optimizer.state_dict(), 'variant_layer': layer}
            torch.save(saved, output / f'checkpoint-{epoch}.pth.tar')
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                torch.save(saved, output / 'model_best.pth.tar')
            write_results(output / 'test.csv', rows, ['epoch', *METRICS])


if __name__ == '__main__':
    main()
