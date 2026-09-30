# Terastal: layer-variant experiments

Offline layer modification, retraining, combination evaluation, and MAESTRO
export code associated with *Terastal: Layer-Variant-based Scheduling for
Real-Time Multi-DNN Workloads on Heterogeneous Accelerators*.

The six paper experiments are available together: five include their required
model code, and Sp2Dense uses an optional external checkout. The saved results
and model-specific checkpoint assembly are retained. This is research code, not
a verified end-to-end reproduction; see [reproduction limits](docs/KNOWN_ISSUES.md).
The scheduling simulator and Figures 5-6 belong to the separate
`rt-dnn-scheduler` project and are not included here.

## Experiments

| Experiment | Directory | Figure 4 metric | Selected paper variants |
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
corresponding experiment. `requirements.txt` installs all Python dependencies.f
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
- The retained Sp2Dense checkpoints inspected during the audit name NYU Depth V2.
  The loaders also support KITTI. The precise checkpoint-to-paper linkage remains
  unresolved; do not relabel the existing results as KITTI.

Datasets and trained weights are not bundled. There is currently no verified
public download location for the project's custom variant checkpoints. Model
READMEs list the expected filenames. Baseline constructors may download upstream
weights when training/export is explicitly run; the import/unit checks do not.

## Saved paper results

The six source result tables under [results/paper](results/paper/README.md) are
preserved byte-for-byte, including poor-accuracy combinations. VGG retains all
32 original rows; its likely four-layer paper subset excludes combinations with
`idx0`. Do not filter the stored tables using the scheduler's accuracy threshold.

Exports under each experiment are historical MAESTRO text descriptions and layer
maps. Their original ratios and layer selection are retained; verify them against
the intended profiling configuration before reuse.

New `.m` exports, `.txt` layer maps and evaluation summaries, training logs,
checkpoint binaries, and other run outputs are ignored by default. The reviewed
snapshots in `experiments/*/exports/`, tables in `results/paper/`, requirements,
and SSD label file are explicit exceptions. External checkouts, PDFs, and local
working notes are excluded from version control.

## Validation

```bash
python -m unittest discover -s tests -v
```

These CPU checks cover imports, compilation, tensor shapes/gradients, freezing,
wrappers, checkpoint validation, layer selection, and integrity of saved result tables.
With `TERASTAL_SP2DENSE_SOURCE` set, additional tests exercise synthetic HDF5
preprocessing, a complete Sp2Dense CPU forward pass, and legacy checkpoints.
They do not retrain models, download weights, or reproduce scheduling results.
The integration checks passed locally with the environment described above.
Full dataset evaluation and paper-checkpoint provenance remain unverified.

Separate one-epoch CPU training checks passed for all 27 candidate layer variants
(ResNet50: 7, VGG11: 5, InceptionV3: 3, Swin-Tiny: 4, SSD: 4, Sp2Dense: 4), plus
the SSD baseline trainer. These checks used synthetic images/annotations at the
normal input resolutions, batch size 2, and zero data-loader workers. ImageNet
fixtures contained 10 training and 2 validation images; VOC and NYU fixtures
contained 2 training and 2 validation samples each. ResNet50 used its usual 90/10
training split in addition to the validation fixtures.

ResNet50, VGG11, InceptionV3, and SSD loaded cached pretrained baselines. Swin and
Sp2Dense used generated checkpoints with the full model architectures; the depth
fixture used a positive output head so logarithmic depth metrics were defined.
The checks verified finite losses/gradients, optimizer updates, unchanged frozen
parameters, and saved checkpoint reloads. They do not establish real-data
accuracy, convergence, GPU execution, or paper reproduction.

For a small ImageNet-format training check, pass `--epochs 1 --batch_size 2
--workers 0 --allow-val-subset` to the classification trainer. The validation
directory must contain matching image and ground-truth counts; the flag opts out
of the default 50,000-image requirement. Preserve ImageNet class indices and
devkit metadata when preparing a real subset. Swin also accepts
`--model-name-or-path /path/to/local/model` for an offline Hugging Face checkpoint.

## Provenance and release status

[source_manifest.json](docs/source_manifest.json) records the source repository,
branch, commit, original path, and SHA-256 for every imported file.
The manifest also records the tested external Sp2Dense revision. Newly written
integration adapters are separate from imported source.

## License and third-party code

The original Terastal code and integration adapters are available under
[Apache-2.0](LICENSE). Retained SSD code keeps its [MIT license](vision/LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and scope.

The external Sp2Dense repository has no license file at the tested revision.
It is not included in this source distribution, and Apache-2.0 does not grant
rights to it. Obtain the applicable upstream permission before using or
redistributing that dependency. Datasets and model weights retain their own terms.
