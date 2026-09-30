# Sp2Dense

Requires an external checkout of
[Sparse-to-Dense](https://github.com/fangchangma/sparse-to-dense.pytorch).
See [third-party notices](../../THIRD_PARTY_NOTICES.md) for upstream license information.

## External checkout

The adapter was tested against commit `10efc6d60bddedd6f28f0532c108bb1d7ccdfc49`.
For a checkout you are authorized to use:

```bash
git clone https://github.com/fangchangma/sparse-to-dense.pytorch.git external/sp2dense
git -C external/sp2dense checkout 10efc6d60bddedd6f28f0532c108bb1d7ccdfc49
export TERASTAL_SP2DENSE_SOURCE="$(pwd)/external/sp2dense"
python -m pip install -e '.[depth]'
```

Alternatively, pass `--upstream /absolute/path/to/checkout` to each command.

## Data and checkpoint inputs

Prepare the upstream HDF5 layout under `data/<dataset>/{train,val}/<scene>/*.h5`.
Use `--data-root` for a different parent directory. Each HDF5 file contains
`rgb` with shape `(3,H,W)` and `depth` with shape `(H,W)`.

The checkpoint's `args` metadata selects NYU or KITTI and the model configuration.
Archived checkpoint metadata names NYU Depth V2; its relationship to the paper's
KITTI results remains unconfirmed.

Supply a trusted baseline checkpoint with `args` and `model` entries. The model
may be a state dictionary or model object. Pickle checkpoints can execute code
when loaded; use trusted files only.

## Commands

Install the package first; use a separate run directory for outputs:

```bash
python -m experiments.sp2dense.main --checkpoint /weights/baseline.pth.tar --data-root /data --device cpu
python -m experiments.sp2dense.retrain_layers --checkpoint /weights/baseline.pth.tar --data-root /data --output-dir results --epochs 1 --batch-size 32
python -m experiments.sp2dense.test_layer_comb --checkpoint /weights/baseline.pth.tar --data-root /data --variants-dir results --variant-filename checkpoint-0.pth.tar --output outputs/layer_combination_results.csv
```

Four `r=2` variants modify `layer4.0.conv1`, `layer4.0.conv2`, `layer4.1.conv1`,
and `layer4.1.conv2`, each trained with its corresponding BatchNorm. Use `--layers`
to select retraining targets. Architecture and training hyperparameters come from
the checkpoint. `main` evaluates the supplied baseline.

Variant directories are named
`results/sparse_to_dense_<arch>_<decoder>_<layer_with_underscores>/`.
The trainer saves `checkpoint-<epoch>.pth.tar`, `model_best.pth.tar`, `test.csv`,
and MAESTRO/layer-map exports. Use `--variant-filename model_best.pth.tar` to evaluate
best checkpoints. Combination evaluation requires all four variants and writes
16 rows, including the baseline.

Use `--workers 0` (the default) for portable loading. Additional workers require
Linux `fork`. The evaluation metric is `delta1`.
