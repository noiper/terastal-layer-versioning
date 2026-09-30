# ResNet50

Source: ordinary trainer from `master@da568c3`, evaluator/results from
`inception_retrain@568f00f`, exporter from `vgg_retrain@1d2294c`.
The author confirmed ordinary training rather than KD.

Seven layers: `layer2.0.conv1`, `layer3.0.conv1`, `layer3.0.conv2`,
`layer4.0.conv1`, `layer4.0.conv2`, `layer4.1.conv2`, `layer4.2.conv2`; all use `r=2`.
The trainer preserves its one-epoch, batch-size-128 defaults. The exact epoch
count used for the paper remains unresolved.

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
python -m experiments.resnet50.resnet50_retrain --local_imagenet_path /data/imagenet --checkpoint_dir checkpoints
python -m experiments.resnet50.test_comb --val_dir /data/imagenet/ILSVRC2012_img_val --gt_txt /data/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt --meta_mat /data/imagenet/ILSVRC2012_devkit_t12/data/meta.mat --checkpoints_path checkpoints
python -m experiments.resnet50.resnet50_analysis --help
```

The evaluator expects `resnet50_d2sconv_s2d_<layer_path_with_underscores>.pth`
for each of the seven layers. It writes `resnet50_combination_results.txt`.
The relevant BatchNorm is trained/copied along with each variant, as in the source.
