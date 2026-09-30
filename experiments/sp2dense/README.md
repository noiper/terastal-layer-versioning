# Sp2Dense external integration

This package contains Terastal's checkpoint, retraining, and combination-evaluation
adapter. Obtain the model, data loaders, losses, and depth metrics separately from
[Fangchang Ma and Sertac Karaman's PyTorch implementation](https://github.com/fangchangma/sparse-to-dense.pytorch).
No upstream source is bundled here. No license file was found in the tested
upstream revision. Determine the applicable upstream permission for your use.

## External checkout

The adapter was tested against commit `10efc6d60bddedd6f28f0532c108bb1d7ccdfc49`.
For a checkout you are authorized to use:

```bash
git clone https://github.com/fangchangma/sparse-to-dense.pytorch.git external/sp2dense
git -C external/sp2dense checkout 10efc6d60bddedd6f28f0532c108bb1d7ccdfc49
export TERASTAL_SP2DENSE_SOURCE="$(pwd)/external/sp2dense"
python -m pip install -e '.[depth]'
```

`external/` is ignored. You can instead pass `--upstream /absolute/path/to/checkout`
to each command. The adapter imports the external code only when a Sp2Dense run
starts; other experiments and command-line help work without it. It leaves the
checkout unchanged and handles its legacy CPU/NumPy/SciPy compatibility locally.

## Data and checkpoint inputs

Prepare the upstream HDF5 layout under `data/<dataset>/{train,val}/<scene>/*.h5`.
Use `--data-root` for a different parent directory. Each HDF5 file contains
`rgb` with shape `(3,H,W)` and `depth` with shape `(H,W)`.

The archived checkpoint metadata names NYU Depth V2, ResNet18, RGBD, `upproj`,
and simulated stereo sampling. Its exact relationship to the paper's Figure 4
CSV remains unresolved. The adapter uses the checkpoint's `args` metadata and
supports both NYU and KITTI; it does not relabel historical results as KITTI.

Supply a **trusted** baseline checkpoint with `args` and `model` entries. The
model may be a state dictionary or a historical model object. Legacy pickle
class paths are remapped; loading such checkpoints can execute serialized Python
code, so do not use untrusted checkpoint files. Custom weights are not bundled.

## Commands

Install the package first; use a separate run directory for outputs:

```bash
python -m experiments.sp2dense.main --checkpoint /weights/baseline.pth.tar --data-root /data --device cpu
python -m experiments.sp2dense.retrain_layers --checkpoint /weights/baseline.pth.tar --data-root /data --output-dir results --epochs 1 --batch-size 32
python -m experiments.sp2dense.test_layer_comb --checkpoint /weights/baseline.pth.tar --data-root /data --variants-dir results --variant-filename checkpoint-0.pth.tar --output outputs/layer_combination_results.csv
```

Four `r=2` variants modify `layer4.0.conv1`, `layer4.0.conv2`, `layer4.1.conv1`,
and `layer4.1.conv2`, each trained with its corresponding BatchNorm. Use `--layers`
to select individual retraining targets. The baseline architecture and training
hyperparameters come from its checkpoint; this adapter does not train an upstream
baseline from scratch. `main` evaluates the supplied baseline.

Variant directories are named
`results/sparse_to_dense_<arch>_<decoder>_<layer_with_underscores>/`.
The trainer saves `checkpoint-<epoch>.pth.tar`, `model_best.pth.tar`, `test.csv`,
and MAESTRO/layer-map exports. Use `--variant-filename model_best.pth.tar` to evaluate
best checkpoints. The combination evaluator requires all four files, applies only
the selected conv/BatchNorm pairs to a fresh baseline, and writes 16 rows after
all evaluations succeed. Failures are not written as zero accuracy.

Use `--workers 0` (the default) for portable loading. Additional workers require
Linux `fork` because the upstream modules are loaded dynamically. The Figure 4
metric is `delta1`. Historical paper results are preserved separately under
`results/paper/sp2dense`; new adapter outputs are not verified paper reproductions.
