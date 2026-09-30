#!/usr/bin/env bash
set -euo pipefail
python -m experiments.mobilenetv2_ssd.mobilenet_retrain --resume models/mb2-ssd-lite.pth --batch_size 32 --num_epochs 10 --lr 0.001 --layer_to_replace "classification_headers.0.3" --R 3 --checkpoint_folder "models/0/"
python -m experiments.mobilenetv2_ssd.mobilenet_retrain --resume models/mb2-ssd-lite.pth --batch_size 32 --num_epochs 10 --lr 0.001 --layer_to_replace "base_net.17.conv.6" --checkpoint_folder "models/1/"
python -m experiments.mobilenetv2_ssd.mobilenet_retrain --resume models/mb2-ssd-lite.pth --batch_size 32 --num_epochs 10 --lr 0.001 --layer_to_replace "base_net.18.0" --checkpoint_folder "models/2/"
python -m experiments.mobilenetv2_ssd.mobilenet_retrain --resume models/mb2-ssd-lite.pth --batch_size 32 --num_epochs 10 --lr 0.001 --layer_to_replace "extras.0.conv.0" --checkpoint_folder "models/3/"
