# VGG11

| Index | Layer |
|---|---|
| 0 | `features.8` |
| 1 | `features.16` |
| 2 | `features.18` |
| 3 | `classifier.0` |
| 4 | `classifier.3` |

The saved result table contains all 32 combinations of these five candidates,
including the baseline. Use `--layer-indices` to train or evaluate a subset.

ImageNet layout (pass its parent as `--local_imagenet_path`):

```text
imagenet/
  ILSVRC2012_img_train/<class>/*.JPEG
  ILSVRC2012_img_val/ILSVRC2012_val_*.JPEG
  ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
  ILSVRC2012_devkit_t12/data/meta.mat
```

After installation, run commands from a separate output directory. Paths are
relative to that directory. Baseline weights download if not cached.

```bash
python -m experiments.vgg11.vgg_retrain --local_imagenet_path /data/imagenet --epochs 5 --layer-indices 1 2 3 4
python -m experiments.vgg11.vgg_comb_test --local_imagenet_path /data/imagenet --saved_models_path saved_models --layer-indices 1 2 3 4
python -m experiments.vgg11.vgg_versioning --help
```

Training defaults to `classifier.0` (`idx3`); the commands above select indices
1–4. Use `--layer-indices 0 1 2 3 4` for all candidates. Training uses `r=2`
and also updates the classifier. Checkpoints are named
`vgg11_idx<index>_trained_imagenet.pth`; `--saved_models_path` sets their directory.

Evaluation defaults to all five candidates (32 combinations); selecting four
produces 16 combinations and requires only those four checkpoints.

For profiling, pass `--classifier-r 2` to match training. The profiling default
for `classifier.0` is `r=4`.
