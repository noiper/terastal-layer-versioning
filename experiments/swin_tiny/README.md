# Swin-Tiny

Source: `swin_retrain@a5058c0`. Install the `[swin]` or `[all]` dependency extra.
The upstream baseline is `microsoft/swin-tiny-patch4-window7-224`.

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

Use an isolated run directory. Paths and generated filenames are relative to the
current working directory; install the repository first so module invocation
works from that directory. Training explicitly downloads upstream baseline
weights if they are not already cached. Custom variant weights are not bundled.


```bash
python -m experiments.swin_tiny.swin_retrain --local_imagenet_path /data/imagenet --epochs 5
python -m experiments.swin_tiny.test_comb --local_imagenet_path /data/imagenet --saved_models_path saved_models
python -m experiments.swin_tiny.swin_analysis --help
```

Checkpoints: `swin-tiny_idx0_trained_imagenet.pth` through
`swin-tiny_idx3_trained_imagenet.pth`. The evaluator writes
`combination_results.txt` (16 combinations). It loads classifier weights from
index 0 even for the baseline; this historical behavior is preserved.
