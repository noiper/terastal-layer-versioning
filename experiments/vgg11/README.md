# VGG11

Source: `vgg_retrain@1d2294c`.

| Index | Layer | Inferred paper subset |
|---|---|---|
| 0 | `features.8` | No |
| 1 | `features.16` | Yes |
| 2 | `features.18` | Yes |
| 3 | `classifier.0` | Yes |
| 4 | `classifier.3` | Yes |

The original 32-row results are preserved. For Figure 4, filter combinations
containing index 0 without changing that source table. The four-layer selection
is inferred from the paper's curves, not recovered plotting code.

ImageNet layout (pass its parent as `--local_imagenet_path`):

```text
imagenet/
  ILSVRC2012_img_train/<class>/*.JPEG
  ILSVRC2012_img_val/ILSVRC2012_val_*.JPEG
  ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
  ILSVRC2012_devkit_t12/data/meta.mat
```

Use an isolated run directory. Paths and generated filenames are relative to the
current working directory; install the repository first so module invocation
works from that directory. Training explicitly downloads upstream baseline
weights if they are not already cached. Custom variant weights are not bundled.


```bash
python -m experiments.vgg11.vgg_retrain --local_imagenet_path /data/imagenet --epochs 5
python -m experiments.vgg11.vgg_retrain --local_imagenet_path /data/imagenet --epochs 5 --layer-indices 1 2 3 4
python -m experiments.vgg11.vgg_comb_test --local_imagenet_path /data/imagenet --saved_models_path saved_models
python -m experiments.vgg11.vgg_comb_test --local_imagenet_path /data/imagenet --saved_models_path saved_models --layer-indices 1 2 3 4
python -m experiments.vgg11.vgg_versioning --help
```

The trainer defaults to `classifier.0` (`idx3`) to preserve its original setting.
Use `--layer-indices 1 2 3 4` for the inferred paper subset, or `0 1 2 3 4` for
all candidates. Saved indices stay fixed when selecting a subset. Training uses
`r=2` and also unfreezes the classifier; `--saved_models_path` selects the output
directory. Files are named `vgg11_idx<index>_trained_imagenet.pth`.

The evaluator defaults to all five indices (32 combinations). The four-layer
command above evaluates 16 combinations and requires only those four checkpoints.
Missing or incompatible weights stop evaluation without recording a zero score.

The profiling script retains its historical `r=4` default for `classifier.0`.
Pass `--classifier-r 2` to match the training and evaluation architecture. Saved
historical exports are unchanged; their ratios must be checked before reuse.
