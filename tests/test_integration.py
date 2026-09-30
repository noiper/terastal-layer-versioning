"""Small CPU checks; no datasets, checkpoints, downloads, or full training."""

import contextlib
import hashlib
import importlib
import io
import json
import pickle
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from utils.layer_versioning.models import D2SConvS2D, LinearToConv
from utils.layer_versioning.train_helper import freeze_all_layers, unfreeze_specific_layers

ROOT = Path(__file__).resolve().parents[1]


class LayerChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_shared_variant_shapes_and_gradients(self):
        for ratio, channels, stride in [(2, 8, 1), (2, 8, 2), (3, 18, 1)]:
            with self.subTest(ratio=ratio, stride=stride):
                layer = D2SConvS2D(channels, channels * 2, 3, stride=stride, padding=1, r=ratio)
                x = torch.randn(2, channels, 7, 9, requires_grad=True)
                y = layer(x)
                self.assertEqual(y.shape, (2, channels * 2, (7 - 1) // stride + 1, (9 - 1) // stride + 1))
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(layer.conv.weight.grad).all())

    def test_linear_conversion_preserves_2d_values_and_gradients(self):
        linear = nn.Linear(8, 12)
        converted = LinearToConv(linear)
        x = torch.randn(3, 8, requires_grad=True)
        y = converted(x)
        torch.testing.assert_close(y.reshape(3, 12), linear(x))
        actual = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
        expected = torch.autograd.grad(linear(x).sum(), x)[0]
        torch.testing.assert_close(actual, expected)

    def test_model_specific_wrappers(self):
        from experiments.vgg11.vgg_retrain import VGGD2SWrapper
        from experiments.swin_tiny.swin_retrain import SwinD2SWrapper
        for wrapper, shape in [(VGGD2SWrapper, (2, 8)), (SwinD2SWrapper, (2, 5, 8))]:
            layer = wrapper(D2SConvS2D(8, 12, r=2))
            x = torch.randn(*shape, requires_grad=True)
            y = layer(x)
            self.assertEqual(y.shape, (*shape[:-1], 12))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_ssd_training_and_evaluation_share_checkpoint_keys(self):
        from experiments.mobilenetv2_ssd.mobilenet_retrain import D2SConvS2D as TrainLayer
        from experiments.mobilenetv2_ssd.eval_ssd import D2SConvS2D as EvalLayer
        for ratio, channels in [(2, 8), (3, 18)]:
            training = TrainLayer(channels, channels, 1, 1, 0, ratio, 1, False)
            evaluation = EvalLayer(channels, channels, 1, 1, 0, ratio, 1, False)
            evaluation.load_state_dict(training.state_dict(), strict=True)
            x = torch.randn(2, channels, 5, 5)
            torch.testing.assert_close(training(x), evaluation(x), rtol=0, atol=0)
            training(x).square().mean().backward()
            self.assertIsNotNone(training.conv.weight.grad)

    def test_ssd_profiling_group_handling_is_retained(self):
        from experiments.mobilenetv2_ssd.profiling_layer import D2SConvS2D as ProfileLayer
        layer = ProfileLayer(16, 16, kernel_size=3, padding=1, groups=16, r=2)
        self.assertEqual(layer.conv.groups, 4)
        self.assertEqual(layer(torch.randn(1, 16, 5, 5)).shape, (1, 16, 5, 5))

    def test_freezing_keeps_non_target_parameters_unchanged(self):
        model = nn.Sequential(nn.Conv2d(8, 8, 1), D2SConvS2D(8, 8, r=2))
        freeze_all_layers(model)
        unfreeze_specific_layers(model, ['1'])
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        model(torch.randn(2, 8, 3, 3)).square().mean().backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name.startswith('0.'):
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model[1].conv.weight, before['1.conv.weight']))

    def test_legacy_shared_checkpoint_module_names_resolve(self):
        from experiments.sp2dense.checkpoint_compat import LegacyUnpickler
        unpickler = LegacyUnpickler(io.BytesIO())
        self.assertIs(unpickler.find_class('layer_versioning_utils.layer_versioning.models.D2SConvS2D', 'D2SConvS2D'), D2SConvS2D)


class IntegrationChecks(unittest.TestCase):
    def test_every_python_source_compiles(self):
        for folder in ['experiments', 'utils', 'vision']:
            for path in (ROOT / folder).rglob('*.py'):
                with self.subTest(path=path.relative_to(ROOT)):
                    compile(path.read_text(), str(path), 'exec')

    def test_imports_do_not_load_checkpoints_or_datasets(self):
        code = '''
import importlib
from pathlib import Path
from unittest.mock import patch
with patch('torch.load', side_effect=AssertionError('Checkpoint loaded during import')), \\
     patch('torchvision.datasets.ImageFolder', side_effect=AssertionError('Dataset created during import')), \\
     patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('Weights downloaded during import')):
    for folder in ['experiments', 'utils', 'vision']:
        for path in sorted(Path(folder).rglob('*.py')):
            if path.name != '__init__.py':
                importlib.import_module('.'.join(path.with_suffix('').parts))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_archived_paper_tables_are_byte_identical(self):
        manifest = json.loads((ROOT / 'docs/source_manifest.json').read_text())
        entries = {p: m for p, m in manifest['sources'].items() if p.startswith('results/paper/')}
        self.assertEqual(len(entries), 6)
        for relative, source in entries.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), source['sha256'])

    def test_combination_counts_and_vgg_subset(self):
        import csv
        import re
        import statistics
        expected = {'vgg11': 32, 'inceptionv3': 8, 'resnet50': 128, 'swin_tiny': 16, 'mobilenetv2_ssd': 16}
        for model, count in expected.items():
            table = next((ROOT / 'results/paper' / model).glob('*.txt')).read_text()
            self.assertEqual(len(re.findall(r'^Model:', table, re.M)), count)
        vgg = (ROOT / 'results/paper/vgg11/vgg11_combination_results.txt').read_text()
        rows = re.findall(r'Model: (.+)\n\s*> Top-1 Accuracy: [\d.]+%\n\s*> Top-5 Accuracy: ([\d.]+)%', vgg)
        means = []
        for n in range(5):
            values = [float(v) / 100 for name, v in rows if 'idx0' not in name and len(re.findall(r'idx\d+', name)) == n]
            means.append(statistics.mean(values))
        for actual, expected_mean in zip(means, [0.8863, 0.7893, 0.4719166667, 0.12205, 0.0205]):
            self.assertAlmostEqual(actual, expected_mean)
        with next((ROOT / 'results/paper/sp2dense').glob('*.csv')).open() as source:
            self.assertEqual(len(list(csv.DictReader(source))), 16)


if __name__ == '__main__':
    unittest.main()
