# Historical combination results

Every `.txt`/`.csv` table in this directory is byte-identical to its source
snapshot, with hashes in `docs/source_manifest.json`. No new accuracy results
were generated during consolidation.

| Model | Rows, including baseline | Metric |
|---|---:|---|
| MobileNetV2-SSD | 16 | VOC mAP |
| VGG11 | 32 original; 16 in inferred paper subset | ImageNet top-5 |
| InceptionV3 | 8 | ImageNet top-5 |
| ResNet50 | 128 | ImageNet top-5 |
| Swin-Tiny | 16 | ImageNet top-5 |
| Sp2Dense | 16 | delta1 |

For the likely Figure 4 VGG subset, exclude rows whose combination contains
`idx0` (`features.8`); retain the baseline and every combination of `idx1`-`idx4`.
The original table stays intact. The aggregation check is in the test suite.

Group combinations by the number of applied variants and compute mean, minimum,
and maximum. Do not discard low-accuracy combinations: they belong in Figure 4.
The scheduler's valid-combination threshold is a separate downstream operation.
