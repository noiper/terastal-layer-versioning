"""Regression checks for failures that used to look like successful runs."""
import contextlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.checkpoints import copy_variant_state, require_checkpoint_files
from experiments.vgg11.layers import selected_layers


class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_checkpoint_patch_is_complete_and_prefix_exact(self):
        target = {'layer.1.weight': torch.zeros(2), 'layer.10.weight': torch.zeros(2)}
        source = {'layer.1.weight': torch.ones(2), 'layer.10.weight': torch.ones(2)}
        copy_variant_state(target, source, ['layer.1'])
        torch.testing.assert_close(target['layer.1.weight'], torch.ones(2))
        torch.testing.assert_close(target['layer.10.weight'], torch.zeros(2))
        with self.assertRaises(ValueError):
            copy_variant_state(target, {}, ['layer.1'])
        with self.assertRaises(ValueError):
            copy_variant_state(target, {'layer.1.weight': torch.ones(3)}, ['layer.1'])

    def test_missing_checkpoints_raise_with_all_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / 'first.pth', Path(directory) / 'second.pth']
            with self.assertRaises(FileNotFoundError) as error:
                require_checkpoint_files(paths)
            for path in paths:
                self.assertIn(str(path), str(error.exception))

    def test_vgg_subset_retains_original_checkpoint_indices(self):
        layers = selected_layers([4, 1, 3, 2])
        self.assertEqual([spec['idx'] for spec in layers.values()], [1, 2, 3, 4])
        self.assertEqual(layers['classifier.0'], {'idx': 3, 'r': 2})
        for selection in ([], [3, 3], [5]):
            with self.assertRaises(ValueError):
                selected_layers(selection)

    def test_zero_accuracy_still_saves_first_checkpoint(self):
        from experiments.vgg11 import vgg_retrain
        from experiments.inceptionv3 import inception_retrain

        class TinyClassifier(nn.Module):
            def __init__(self, auxiliary):
                super().__init__()
                self.linear = nn.Linear(2, 6)
                self.auxiliary = auxiliary

            def forward(self, x):
                logits = self.linear(x)
                return types.SimpleNamespace(logits=logits, aux_logits=logits) if self.auxiliary else logits

        loader = [(torch.randn(2, 2), torch.zeros(2, dtype=torch.long))]
        for module in (vgg_retrain, inception_retrain):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(module, 'get_classification_metrics', return_value={'top1_accuracy': 0.0}):
                target = Path(directory) / 'first.pth'
                module.train_model(TinyClassifier(module is inception_retrain), torch.device('cpu'),
                                   loader, loader, 1, target)
                self.assertTrue(target.is_file())

    def test_classification_failure_raises_without_zero_result(self):
        from experiments.resnet50 import test_comb as resnet
        from experiments.inceptionv3 import test_comb_inception as inception
        for module in (resnet, inception):
            args = types.SimpleNamespace(val_dir='unused', gt_txt='unused', meta_mat='unused',
                                         checkpoints_path='unused', saved_models_path='unused',
                                         local_imagenet_path='unused', batch_size=1, workers=0)
            constructor = 'resnet50' if module is resnet else 'inception_v3'
            with self.subTest(model=constructor), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(module, 'require_checkpoint_files'), \
                 patch.object(module, 'build_test_loader', return_value=[]), \
                 patch.object(module.models, constructor, side_effect=ValueError('broken checkpoint')), \
                 patch('builtins.open', side_effect=AssertionError('A failed run must not write results')):
                with self.assertRaisesRegex(RuntimeError, 'no result recorded'):
                    module.main(args)

    def test_resnet_trainer_saves_zero_accuracy_and_tracks_top5(self):
        from utils.layer_versioning.layer_trainer import LayerVersionTrainer
        model = nn.Linear(2, 6)
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            name = str(Path(directory) / 'model')
            trainer = LayerVersionTrainer(model, name, None, [], [], [],
                                          torch.optim.SGD(model.parameters(), lr=0.01),
                                          nn.CrossEntropyLoss(), 2, torch.device('cpu'))
            with patch.object(trainer, '_train_epoch', return_value=(1.0, 0.0, 20.0)), \
                 patch.object(trainer, 'validate', side_effect=[(1.0, 0.0, 40.0), (0.9, 0.0, 60.0)]):
                result = trainer.train()
            saved = torch.load(name + '_best.pth', map_location='cpu', weights_only=True)
            self.assertEqual(saved['epoch'], 0)
            self.assertEqual(saved['val_top1_acc'], 0.0)
            self.assertEqual(result['best_val_top5_acc'], 60.0)
            self.assertEqual(trainer.epoch, 2)

    def test_small_validation_set_requires_opt_in_and_matching_labels(self):
        from experiments.resnet50 import resnet50_retrain
        from experiments.vgg11 import vgg_retrain
        from experiments.inceptionv3 import inception_retrain
        from experiments.swin_tiny import swin_retrain
        from PIL import Image
        modules = (resnet50_retrain, vgg_retrain, inception_retrain, swin_retrain)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (8, 8)).save(root / 'ILSVRC2012_val_00000001.JPEG')
            labels = root / 'labels.txt'
            for module in modules:
                with self.subTest(module=module.__name__):
                    labels.write_text('1\n')
                    kwargs = dict(root=str(root), gt_file=str(labels))
                    if module is not resnet50_retrain:
                        kwargs['meta_file'] = 'unused-without-transform'
                    with self.assertRaisesRegex(ValueError, '50,000'):
                        module.ImageNetValFlat(**kwargs)
                    dataset = module.ImageNetValFlat(**kwargs, allow_subset=True)
                    self.assertEqual(len(dataset), 1)
                    self.assertEqual(dataset[0][1], 0)
                    labels.write_text('1\n2\n')
                    with self.assertRaisesRegex(ValueError, 'mismatch'):
                        module.ImageNetValFlat(**kwargs, allow_subset=True)
                    labels.write_text('')
                    with self.assertRaises(ValueError):
                        module.ImageNetValFlat(**kwargs, allow_subset=True)

    def test_invalid_layer_replacement_raises(self):
        from utils.layer_versioning.train_helper import setup_custom_layer_model
        with self.assertRaises(RuntimeError):
            setup_custom_layer_model(nn.Sequential(nn.ReLU()), ['0'])

    def test_ssd_predictor_failure_is_not_zero_accuracy(self):
        from experiments.mobilenetv2_ssd import test_comb as ssd
        with patch.object(ssd, 'create_mobilenetv2_ssd_lite_predictor', side_effect=ValueError('bad model')):
            with self.assertRaisesRegex(RuntimeError, 'predictor'):
                ssd.evaluate_model_map(nn.Identity(), [], torch.device('cpu'))

    def test_frozen_export_uses_model_device(self):
        from utils.layer_versioning.model_analysis import export_maestro_conv_layers
        model = nn.Sequential(nn.Linear(4, 4, device='meta'))
        model.requires_grad_(False)
        text = export_maestro_conv_layers(model, torch.zeros(1, 4), network_name='frozen')
        self.assertIn('Network frozen', text)
        self.assertIn('Layer 0', text)


if __name__ == '__main__':
    unittest.main()
