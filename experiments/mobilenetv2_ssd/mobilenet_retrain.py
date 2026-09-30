# Support both `python -m experiments.mobilenetv2_ssd.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.mobilenetv2_ssd"

# python mobilenet_retrain.py --resume models/mb2-ssd-lite.pth --batch_size 32 --num_epochs 10 --lr 0.001 --layer_to_replace "base_net.18.0"
import argparse
import os
import logging
import sys
import torch
import functools

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch import nn

from vision.utils.misc import str2bool, Timer, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config as config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

# Assume D2SConvS2D is in a utils directory as per the ResNet50 example
# from utils.layer_versioning.models.D2SConvS2D import D2SConvS2D

# Mock D2SConvS2D for demonstration if not available
class D2SConvS2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, r, groups, bias):
        super().__init__()
        self.depth_to_space = nn.PixelShuffle(r)
        self.conv = nn.Conv2d(in_channels=in_channels // (r*r), out_channels=out_channels // (r*r),
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.space_to_depth = nn.PixelUnshuffle(r)

    def forward(self, x):
        x = self.depth_to_space(x)
        x = self.conv(x)
        x = self.space_to_depth(x)
        return x

# --- Helper Functions from ResNet50 Example ---

def replace_with_d2sconv_s2d(model, layer_path: str, r=2):
    """
    Replaces a Conv2d layer with a D2SConvS2D layer at the given path.
    """
    try:
        parts = layer_path.split(".")
        parent = functools.reduce(getattr, [model] + parts[:-1])
        name = parts[-1]
        orig = getattr(parent, name)

        assert isinstance(orig, nn.Conv2d), "Target must be nn.Conv2d"
        k = orig.kernel_size[0] if isinstance(orig.kernel_size, tuple) else orig.kernel_size
        p = orig.padding[0] if isinstance(orig.padding, tuple) else orig.padding

        assert orig.in_channels % (r*r) == 0, f"in_channels {orig.in_channels} not divisible by r^2 ({r*r})"
        assert orig.out_channels % (r*r) == 0, f"out_channels {orig.out_channels} not divisible by r^2 ({r*r})"

        new_mod = D2SConvS2D(orig.in_channels, orig.out_channels,
                              kernel_size=k, stride=orig.stride, padding=p, r=r,
                              groups=orig.groups, bias=(orig.bias is not None))

        setattr(parent, name, new_mod)
        logging.info(f"✅ Replaced {layer_path} with D2SConvS2D")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to replace {layer_path}: {e}")
        return model

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_specific_layers(model, layers_to_unfreeze):
    layers_to_unfreeze_set = set(layers_to_unfreeze)
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze_set:
            if name.startswith(layer_name):
                param.requires_grad = True
                logging.info(f"Unfroze parameter: {name}")

# --- Main Training and Evaluation Functions ---

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    # (Implementation is the same as in layer_train_ssd.py)
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    # (Implementation is the same as in layer_train_ssd.py)
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Depth-to-Space Layer Replacement and Training for MobileNetV2 SSD-Lite')

    parser.add_argument("--dataset_type", default="voc", type=str)
    parser.add_argument('--datasets', nargs='+', default=['~/data/VOC2007', '~/data/VOC2012'], help='Dataset directory path')
    parser.add_argument('--validation_dataset', default="~/data/test/VOC2007/", help='Dataset directory path')
    parser.add_argument('--net', default="mb2-ssd-lite", help="The network architecture")
    parser.add_argument('--resume', type=str, help='Checkpoint file to resume training from')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--scheduler', default="cosine", type=str)
    parser.add_argument('--t_max', default=100, type=float)
    parser.add_argument('--validation_epochs', default=5, type=int)
    parser.add_argument('--debug_steps', default=100, type=int)
    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--checkpoint_folder', default='models/')
    parser.add_argument('--layer_to_replace', type=str, default="regression_headers.0.3",
                        help="The specific Conv2D layer to replace with D2SConvS2D.")
    parser.add_argument('--R', type=int, default=2,
                    help="The block size for Space-to-Depth and Depth-to-Space operations.")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_folder, exist_ok=True)
    args.datasets = [os.path.expanduser(path) for path in args.datasets]
    args.validation_dataset = os.path.expanduser(args.validation_dataset)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    timer = Timer()

    # --- Data Loading ---
    config = config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = [VOCDataset(path, transform=train_transform, target_transform=target_transform) for path in args.datasets]
    num_classes = len(datasets[0].class_names)
    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)

    logging.info("Prepare validation dataset.")
    val_dataset = VOCDataset(args.validation_dataset, transform=test_transform, target_transform=target_transform, is_test=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)

    # --- Model Setup ---
    logging.info("Build network.")
    net = create_mobilenetv2_ssd_lite(num_classes)

    if not args.resume:
        logging.fatal("This script requires a pretrained SSD model. Use the --resume argument.")
        sys.exit(1)

    logging.info(f"Loading pretrained model from {args.resume}")
    net.load(args.resume)

    # --- Layer Replacement ---
    net = replace_with_d2sconv_s2d(net, args.layer_to_replace, args.R)

    # Freeze all layers, then unfreeze the new D2S layer and its parent module
    freeze_all_layers(net)
    unfreeze_specific_layers(net, [args.layer_to_replace])

    net.to(DEVICE)

    # --- Training Setup ---
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    if not trainable_params:
        logging.fatal("No layers to train. Check the layer path and freezing logic.")
        sys.exit(1)

    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.t_max)

    # --- Main Loop ---
    for epoch in range(args.num_epochs):
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_reg_loss, val_clf_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, "
                f"Validation Regression Loss: {val_reg_loss:.4f}, "
                f"Validation Classification Loss: {val_clf_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-d2s-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
