# InceptionV3

Three layers: `Conv2d_4a_3x3.conv`, `Mixed_6a.branch3x3.conv`, and
`Mixed_7c.branch3x3dbl_1.conv`, all with `r=2`. Input images are 299 x 299.

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
python -m experiments.inceptionv3.inception_retrain --local_imagenet_path /data/imagenet --save_dir saved_models --epochs 5
python -m experiments.inceptionv3.test_comb_inception --local_imagenet_path /data/imagenet --saved_models_path saved_models
python -m experiments.inceptionv3.inception_versioning --help
```

The evaluator expects `inceptionv3_idx0_trained_imagenet.pth` through
`inceptionv3_idx2_trained_imagenet.pth` and writes
`inceptionv3_combination_results.txt` (eight combinations including baseline).
Training also updates `fc`. Combination evaluation uses classifier and auxiliary
weights from the last selected checkpoint.
