# Swin-Tiny

Install the `[swin]` or `[all]` dependency extra.
The upstream baseline is `microsoft/swin-tiny-patch4-window7-224`.
Use `--model-name-or-path /path/to/local/model` to load a local Hugging Face
checkpoint without downloading it.

Four `r=2` variants: `swin.encoder.layers.3.blocks.0.intermediate.dense`,
`swin.encoder.layers.3.blocks.0.output.dense`,
`swin.encoder.layers.3.blocks.1.intermediate.dense`, and
`swin.encoder.layers.3.blocks.1.output.dense`.

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
python -m experiments.swin_tiny.swin_retrain --local_imagenet_path /data/imagenet --epochs 5
python -m experiments.swin_tiny.test_comb --local_imagenet_path /data/imagenet --saved_models_path saved_models
python -m experiments.swin_tiny.swin_analysis --help
```

Checkpoints: `swin-tiny_idx0_trained_imagenet.pth` through
`swin-tiny_idx3_trained_imagenet.pth`. The evaluator writes
`combination_results.txt` (16 combinations). Training also updates the classifier;
evaluation uses classifier weights from index 0, including for the baseline.
