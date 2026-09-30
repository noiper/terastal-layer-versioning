# Terastal: layer-variant experiments

Layer modification, retraining, combination evaluation, and MAESTRO export for
*Terastal: Layer-Variant-based Scheduling for Real-Time Multi-DNN Workloads on
Heterogeneous Accelerators*. Scheduling code is maintained separately in
`rt-dnn-scheduler`.

## Installation

Use Python 3.10–3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Models

Each model's README provides data layouts and training/evaluation commands:

- [Sp2Dense](experiments/sp2dense/README.md) — requires an external checkout
- [MobileNetV2-SSD-Lite](experiments/mobilenetv2_ssd/README.md)
- [ResNet50](experiments/resnet50/README.md)
- [VGG11](experiments/vgg11/README.md)
- [InceptionV3](experiments/inceptionv3/README.md)
- [Swin-Tiny](experiments/swin_tiny/README.md)

Datasets and checkpoints are not included. Saved results are in
[results/paper](results/paper/README.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

Set `TERASTAL_SP2DENSE_SOURCE` to include the external Sp2Dense tests.

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream sources and terms.
