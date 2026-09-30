"""Local adapter tests; external tests run when the checkout path is supplied."""
import copy
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.sp2dense.upstream import DeviceUnpool, load_upstream, _imresize
from experiments.sp2dense.runtime import assemble_combination


class AdapterChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_unpool_preserves_values_gradients_and_legacy_state(self):
        x = torch.randn(1, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        for legacy in (False, True):
            module = DeviceUnpool(3)
            if legacy:
                kernel = module._buffers.pop('weights')
                module.weights = kernel
            y = module(x)
            expected = torch.zeros(1, 3, 8, 10, dtype=x.dtype)
            expected[:, :, ::2, ::2] = x
            torch.testing.assert_close(y, expected)
            torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], torch.ones_like(x))
            self.assertEqual(list(module.state_dict()), [])

    def test_combination_copies_only_selected_conv_and_bn(self):
        model = nn.Module()
        model.block = nn.Module()
        model.block.conv1 = nn.Conv2d(4, 4, 1)
        model.block.bn1 = nn.BatchNorm2d(4)
        model.untouched = nn.Linear(2, 2)
        variant = copy.deepcopy(model)
        with torch.no_grad():
            variant.block.conv1.weight.fill_(2)
            variant.block.bn1.running_mean.fill_(3)
            variant.untouched.weight.fill_(4)
        assembled = assemble_combination(model, {'block.conv1': variant}, ['block.conv1'])
        torch.testing.assert_close(assembled.block.conv1.weight, variant.block.conv1.weight)
        torch.testing.assert_close(assembled.block.bn1.running_mean, variant.block.bn1.running_mean)
        torch.testing.assert_close(assembled.untouched.weight, model.untouched.weight)
        self.assertNotEqual(assembled.block.conv1.weight.data_ptr(), variant.block.conv1.weight.data_ptr())

    def test_float_depth_resize_retains_metric_values(self):
        depth = np.full((12, 16), 3.25, dtype=np.float32)
        resized = _imresize(depth, 0.5, 'nearest', 'F')
        np.testing.assert_array_equal(resized, np.full((6, 8), 3.25, dtype=np.float32))


@unittest.skipUnless(os.environ.get('TERASTAL_SP2DENSE_SOURCE'), 'External Sp2Dense checkout not configured')
class ExternalSp2DenseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backend = load_upstream()

    def test_legacy_checkpoint_classes_and_torch_archive(self):
        import sys
        from experiments.sp2dense.checkpoint_compat import LegacyUnpickler, load_checkpoint
        result_class = self.backend.metrics.Result
        unpickler = LegacyUnpickler(io.BytesIO())
        self.assertIs(unpickler.find_class('models', 'ResNet'), self.backend.models.ResNet)
        self.assertIs(unpickler.find_class('experiments.sp2dense.metrics', 'Result'), result_class)
        legacy = types.ModuleType('metrics')
        legacy.Result = result_class
        previous = result_class.__module__
        try:
            with patch.dict(sys.modules, {'metrics': legacy}), tempfile.TemporaryDirectory() as directory:
                result_class.__module__ = 'metrics'
                path = Path(directory) / 'old.pth'
                torch.save({'best_result': result_class()}, path)
                result_class.__module__ = previous
                self.assertIsInstance(load_checkpoint(path, map_location='cpu')['best_result'], result_class)
        finally:
            result_class.__module__ = previous

    def test_hdf5_training_and_validation_preprocessing(self):
        import h5py
        from experiments.sp2dense.runtime import data_loader
        args = types.SimpleNamespace(data='nyudepthv2', sparsifier='uar', num_samples=100,
                                     max_depth=-1, modality='rgbd')
        with tempfile.TemporaryDirectory() as directory:
            for split in ('train', 'val'):
                path = Path(directory) / args.data / split / 'room' / 'sample.h5'
                path.parent.mkdir(parents=True)
                with h5py.File(path, 'w') as data:
                    data['rgb'] = np.full((3, 480, 640), 127, dtype=np.uint8)
                    data['depth'] = np.full((480, 640), 2, dtype=np.float32)
                inputs, target = next(iter(data_loader(args, directory, training=split == 'train')))
                self.assertEqual(inputs.shape, (1, 4, 228, 304))
                self.assertEqual(target.shape, (1, 1, 228, 304))
                self.assertTrue(torch.isfinite(inputs).all())

    def test_retraining_step_updates_only_selected_parameters(self):
        from experiments.sp2dense.retrain_layers import train_epoch
        from utils.layer_versioning.train_helper import freeze_all_layers, unfreeze_specific_layers
        model = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 1, 1), nn.Softplus())
        freeze_all_layers(model)
        unfreeze_specific_layers(model, ['0'])
        frozen = model[1].weight.detach().clone()
        selected = model[0].weight.detach().clone()
        optimizer = torch.optim.SGD(model[0].parameters(), lr=0.01)
        loader = [(torch.randn(2, 4, 8, 8), torch.full((2, 1, 8, 8), 2.0))]
        train_epoch(model, loader, optimizer, self.backend.criteria.MaskedL1Loss(), ['0'], torch.device('cpu'))
        self.assertFalse(torch.equal(model[0].weight, selected))
        torch.testing.assert_close(model[1].weight, frozen, rtol=0, atol=0)

    def test_full_model_cpu_forward_and_state_checkpoint_without_download(self):
        from experiments.sp2dense.checkpoint_compat import checkpoint_model
        from utils.layer_versioning.train_helper import setup_custom_layer_model
        args = types.SimpleNamespace(arch='resnet18', decoder='upproj', modality='rgbd')
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('No downloads')):
            model = self.backend.models.ResNet(18, 'upproj', (32, 48), in_channels=4, pretrained=False).eval()
            x = torch.randn(1, 4, 64, 64)
            with torch.no_grad():
                original = model(x)
            restored = checkpoint_model({'args': args, 'model': model.state_dict()}, (32, 48)).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(x), original)
            setup_custom_layer_model(model, ['layer4.0.conv1'])
            restored = checkpoint_model({'args': args, 'model': model.state_dict()}, (32, 48), ['layer4.0.conv1']).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(x), model(x))
            self.assertEqual(original.shape, (1, 1, 32, 48))


if __name__ == '__main__':
    unittest.main()
