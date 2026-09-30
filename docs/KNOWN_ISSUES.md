# Reproduction limits

The saved paper result tables are historical measurements. Runtime and packaging
repairs do not establish the exact provenance of the original training runs, and
no saved result values were changed.

## Historical configurations

- **Sp2Dense dataset:** two archived checkpoint metadata records name NYU Depth V2,
  ResNet18, RGBD, `upproj`, and simulated stereo sampling. Their metrics do not
  exactly match the saved combination CSV. The paper names KITTI, so the precise
  dataset/checkpoint linkage remains unresolved. Both data loaders are supported;
  the adapter uses the dataset named in the supplied checkpoint.
- **ResNet50:** ordinary training, rather than KD, was confirmed by the author.
  The retained seven-layer trainer defaults to one epoch and batch size 128.
  The original paper run's epoch count remains unverified.
- **VGG11:** the inferred paper subset is `idx1`–`idx4`; the full source table
  contains 32 combinations of five candidates. Training/evaluation use `r=2`.
  Profiling historically used `r=4` for `classifier.0`; use `--classifier-r 2`
  to profile the current training architecture. Historical exports retain their
  original settings.
- **Freezing and checkpoint assembly:** VGG and Swin also train the classifier;
  Inception also trains `fc`; ResNet50 and Sp2Dense also train the associated
  BatchNorm. Swin's evaluator loads its classifier from `idx0`, including for its
  baseline. Inception's combination classifier/auxiliary state comes from the
  last checkpoint in each combination. These behaviors are retained and should
  be distinguished from experiments that freeze every unmodified layer.
- **Historical failures:** older evaluators sometimes substituted zero accuracy
  after exceptions or continued with missing weights. Whether this affected any
  saved paper table is unknown. Current evaluators raise errors instead.

## Sp2Dense external adapter

The public source uses a user-supplied upstream checkout; the upstream model,
loaders, losses, and metrics are not copied into this release. See its
[setup instructions](../experiments/sp2dense/README.md) and the
[dependency notice](../THIRD_PARTY_NOTICES.md).

The adapter supports state-dictionary and historical whole-model checkpoints,
CPU unpooling, explicit input/output paths, and all 16 layer combinations. It
provides local compatibility shims for removed NumPy/SciPy/collections APIs without
editing the external checkout. Checkpoints produced by the new trainer contain
state dictionaries and identify their modified layer. These implementation and
logging changes are not a claim to reproduce the original training run exactly.

Synthetic HDF5 preprocessing, full ResNet18 forward execution, one training epoch
for each of the four variants, and checkpoint round trips are tested on CPU.
The training check used a generated baseline with a positive output head;
it does not validate a historical paper checkpoint. Use zero data-loader workers for portability;
multiple workers require the Linux `fork` start method with the dynamic upstream
modules. No full dataset training/evaluation or GPU validation was performed.

## Paper scope and unavailable artifacts

Custom baseline/variant weights and datasets are not distributed, and no verified
public download location is available for the project's custom checkpoints.
Readers need their own baseline and variant files to run accuracy experiments.
The independent scheduler, workload/hardware configurations, ablations, and
plotting pipeline for Figures 5–6 are outside this repository. This is the
offline layer-variant component, not a complete scheduling reproduction.
