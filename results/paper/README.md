# Historical combination results

Every `.txt`/`.csv` table in this directory is byte-identical to its source
snapshot, with hashes in `docs/source_manifest.json`. No new accuracy results
were generated during consolidation.

| Model | Rows, including baseline | Metric |
|---|---:|---|
| MobileNetV2-SSD | 16 | VOC mAP |
| VGG11 | 32 | ImageNet top-5 |
| InceptionV3 | 8 | ImageNet top-5 |
| ResNet50 | 128 | ImageNet top-5 |
| Swin-Tiny | 16 | ImageNet top-5 |
| Sp2Dense | 16 | delta1 |

The VGG11 table covers all five implemented layer candidates. A four-layer
subset has 16 combinations, including the baseline; select the intended indices
explicitly when comparing a subset.

Group combinations by the number of applied variants and compute mean, minimum,
and maximum. Include all combinations for the selected layer indices.
The scheduler's valid-combination threshold is a separate downstream operation.
