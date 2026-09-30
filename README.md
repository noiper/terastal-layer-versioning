# Terastal: layer-variant experiments

Offline layer modification, retraining, combination evaluation, and MAESTRO
export code associated with *Terastal: Layer-Variant-based Scheduling for
Real-Time Multi-DNN Workloads on Heterogeneous Accelerators*.

The six paper experiments are available together: five include their required
model code, and Sp2Dense uses an optional external checkout.
The scheduling simulator and Figures 5-6 belong to the separate
`rt-dnn-scheduler` project and are not included here.

## Experiments

| Experiment | Directory | Metric | Paper layer variants |
|---|---|---|---:|
| Sp2Dense | [sp2dense](experiments/sp2dense/README.md) | delta1 | 4 |
| MobileNetV2-SSD-Lite | [mobilenetv2_ssd](experiments/mobilenetv2_ssd/README.md) | VOC mAP | 4 |
| ResNet50 | [resnet50](experiments/resnet50/README.md) | ImageNet top-5 | 7 |
| VGG11 | [vgg11](experiments/vgg11/README.md) | ImageNet top-5 | 4 |
| InceptionV3 | [inceptionv3](experiments/inceptionv3/README.md) | ImageNet top-5 | 3 |
| Swin-Tiny | [swin_tiny](experiments/swin_tiny/README.md) | ImageNet top-5 | 4 |

Shared code retains its `utils.layer_versioning` namespace. SSD's vendored
`vision` package and model-specific D2S implementations are retained where their
behavior differs from the shared implementation.

## Installation

Use Python 3.10–3.12 in an isolated environment. The smoke tests were run
with Python 3.12, PyTorch 2.3.1, and torchvision 0.18.1. Choose the PyTorch build
appropriate for your CUDA installation before installing this package.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For fewer dependencies, install `pip install -e .` for the torchvision
classification experiments, or add `[swin]`, `[ssd]`, or `[depth]` for the
corresponding experiment. `requirements.txt` installs all Python dependencies.
Sp2Dense additionally requires the [external checkout setup](experiments/sp2dense/README.md);
its upstream source is not bundled or downloaded during installation.

After installation, use module invocation from your chosen run directory, for
example:

```bash
python -m experiments.resnet50.resnet50_retrain --help
python -m experiments.vgg11.vgg_comb_test --help
python -m experiments.mobilenetv2_ssd.test_comb --help
```

Direct script execution is also supported. Sp2Dense entry points accept explicit
`--checkpoint`, `--data-root`, and `--upstream` paths. Relative dataset, checkpoint, and output paths still refer to the
current working directory. Use separate run directories to avoid result-file
collisions. Model READMEs give commands and describe their source configurations. Missing
checkpoints, incomplete layer weights, and evaluation failures stop the run; they
are not recorded as zero accuracy. Saved historical tables are unchanged.

## Data and checkpoints

- ImageNet experiments expect the training image tree, flat validation images,
  and ILSVRC2012 devkit metadata. The exact layout is documented per experiment.
- SSD expects Pascal VOC directories, its 21-class baseline, and separately
  trained variant checkpoints.
- Sp2Dense supports NYU Depth V2 and KITTI; the supplied checkpoint specifies
  the dataset and model configuration. See its README for the expected layout.

Datasets and trained weights are not bundled. There is currently no verified
public download location for the project's custom variant checkpoints. Model
READMEs list the expected filenames. Baseline constructors may download upstream
weights when training/export is explicitly run; the import/unit checks do not.

## Saved paper results

The six saved result tables are available under
[results/paper](results/paper/README.md). VGG11 includes all 32 combinations of
the five implemented candidates, including the baseline.

Exports under each experiment are historical MAESTRO text descriptions and layer
maps. Their original ratios and layer selection are retained; verify them against
the intended profiling configuration before reuse.

New `.m` exports, `.txt` layer maps and evaluation summaries, training logs,
checkpoint binaries, and other run outputs are ignored by default. The reviewed
snapshots in `experiments/*/exports/`, tables in `results/paper/`, requirements,
and SSD label file are explicit exceptions. External checkouts, PDFs, and local
working notes are excluded from version control.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests check imports, layer conversion, parameter freezing, checkpoint loading,
and saved result integrity on CPU. Set `TERASTAL_SP2DENSE_SOURCE` to enable the
optional external Sp2Dense tests. These checks do not measure dataset accuracy.

## Third-party code

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream sources and notices.
Sp2Dense requires a separate [upstream checkout](experiments/sp2dense/README.md).
