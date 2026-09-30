# MobileNetV2-SSD-Lite

Install the `[ssd]` or `[all]` extra.
This implementation corresponds to the paper's MobileNetV2-SSD label.

| Index | Layer | Ratio |
|---|---|---:|
| 0 | `classification_headers.0.3` | 3 |
| 1 | `base_net.17.conv.6` | 2 |
| 2 | `base_net.18.0` | 2 |
| 3 | `extras.0.conv.0` | 2 |

Use Pascal VOC2007/VOC2012 training directories and VOC2007 test data. Each VOC
root contains `Annotations`, `JPEGImages`, and `ImageSets/Main`. Supply the
21-class baseline `mb2-ssd-lite.pth`; custom weights are not bundled.

Example from a separate run directory after installing this repository:

```bash
mkdir -p models/0 models/1 models/2 models/3
python -m experiments.mobilenetv2_ssd.mobilenet_retrain --resume /weights/mb2-ssd-lite.pth --datasets /data/VOC2007 /data/VOC2012 --validation_dataset /data/test/VOC2007 --layer_to_replace classification_headers.0.3 --R 3 --batch_size 32 --num_epochs 10 --lr 0.001 --checkpoint_folder models/0
python -m experiments.mobilenetv2_ssd.test_comb --dataset /data/test/VOC2007 --label_file /path/to/checkout/experiments/mobilenetv2_ssd/voc-model-labels.txt --baseline_model /weights/mb2-ssd-lite.pth --models_base_dir models
```

Repeat training for indices 1–3 using the layer, ratio, and checkpoint directory
from the table. Evaluation loads checkpoints from `models/0` through `models/3`
and writes `mb2ssd_combination_map_results.txt`. `train_ssd.py` supports baseline
training.
