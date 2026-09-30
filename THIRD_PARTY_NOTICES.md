# Third-party source provenance

## pytorch-ssd

The `vision/` package and SSD baseline/evaluation code derive from the
[`noiper/pytorch-ssd`](https://github.com/noiper/pytorch-ssd) checkout referenced
by this research repository at commit
`7a839cbc8c3fb39679856b4dc42a1ab19ec07581`. The retained license is
[vision/LICENSE](vision/LICENSE).

Most `vision/` files were copied from `mb2ssd_versioning@185b09b`. Dataset loaders
that were missing from that tree were recovered from the exact dependency commit
above, along with its license. The manifest identifies their individual origins.
Existing upstream notices in the source are preserved.

## Sparse-to-Dense (external dependency)

The optional Sp2Dense adapter calls Fangchang Ma and Sertac Karaman's
[Sparse-to-Dense PyTorch implementation](https://github.com/fangchangma/sparse-to-dense.pytorch).
The tested upstream revision is `10efc6d60bddedd6f28f0532c108bb1d7ccdfc49`.
Its model, loader, loss, and metric source is not vendored in this release.

No license file was found in that revision or its inspected history. The
separate Torch implementation's license is not asserted to cover the PyTorch
implementation. Users must determine applicable upstream permission; this
project's Apache-2.0 license does not grant rights to that external code.
The adapter loads a user-supplied checkout and does not modify it on disk.

Upstream paper: Fangchang Ma and Sertac Karaman, “Sparse-to-Dense: Depth Prediction
from Sparse Depth Samples and a Single Image,” ICRA 2018,
[arXiv:1709.07492](https://arxiv.org/abs/1709.07492).

## Libraries, data, and weights

PyTorch, torchvision, Transformers, NumPy, SciPy, Pillow, OpenCV, pandas, h5py,
matplotlib, tqdm, and THOP are installed separately. Their code, model weights,
and datasets retain their respective terms. No pretrained or custom checkpoint
binaries are included in this source distribution.

The root [Apache-2.0 license](LICENSE) applies to original Terastal code and
integration adapters, subject to the retained third-party notices above.
