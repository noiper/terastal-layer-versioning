# Support both `python -m experiments.resnet50.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.resnet50"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
from tqdm import tqdm
import functools
import pickle as pkl
import torchvision
import torchvision.transforms as T
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import scipy.io as sio
from utils.layer_versioning.models.D2SConvS2D import D2SConvS2D
from utils.layer_versioning.layer_trainer import LayerVersionTrainer
from utils.layer_versioning.train_helper import *

# Configuration parameters
DATA_DIR = "datasets/imagenet/ILSVRC2012_img_train"  # Your ImageNet path
IMAGENET_VAL_DIR = "datasets/imagenet/ILSVRC2012_img_val"
GROUND_TRUTH_TXT = "datasets/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
META_MAT         = "datasets/imagenet/ILSVRC2012_devkit_t12/data/meta.mat"


BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
VAL_SPLIT = 0.1
NUM_WORKERS = 8

CHECKPOINT_DIR = "checkpoints"

# Device setup
device = None

def replace_with_d2sconv_s2d(model, layer_path: str, r=2):
    parts = layer_path.split(".")
    parent = functools.reduce(getattr, [model] + parts[:-1])
    print(f"Parent is {parent}")
    name = parts[-1]
    orig = getattr(parent, name)
    
    assert isinstance(orig, nn.Conv2d), "Target must be nn.Conv2d"
    k = orig.kernel_size[0] if isinstance(orig.kernel_size, tuple) else orig.kernel_size
    p = orig.padding[0] if isinstance(orig.padding, tuple) else orig.padding
    
    # Check divisibility
    assert orig.in_channels % (r*r) == 0, f"in_channels {orig.in_channels} not divisible by r^2 ({r*r})"
    assert orig.out_channels % (r*r) == 0, f"out_channels {orig.out_channels} not divisible by r^2 ({r*r})"
    
    print(f"Orig is {orig}; k is {k}; p is {p}.")

    new_mod = D2SConvS2D(orig.in_channels, orig.out_channels,
                         kernel_size=k, stride=orig.stride, padding=p, r=r,
                         groups=orig.groups, bias=(orig.bias is not None))
    
    setattr(parent, name, new_mod)
    print(f"✅ Replaced {layer_path} with D2SConvS2D")
    return model


def setup_custom_resnet_model(layers_to_replace=None):
    model = models.resnet50(weights="IMAGENET1K_V1")

    if layers_to_replace is not None:
        # Replace specified layers with D2SConvS2D
        for layer_path in layers_to_replace:
            try:
                print(f"layer path is {layer_path}")
                model = replace_with_d2sconv_s2d(model, layer_path, r=2)
            except Exception as e:
                print(f"❌ Failed to replace {layer_path}: {e}")
                continue
    
    return model

def print_trainable_params(model, training_mode="D2SConvS2D only"):
    """Print information about trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print(f"📊 MODEL PARAMETERS SUMMARY")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print(f"Training mode: {training_mode}")
    
    # Show which layers are trainable
    print("\n🔓 Trainable layers:")
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):
            if isinstance(module, D2SConvS2D):
                print(f"  - {name} (D2SConvS2D)")
            elif isinstance(module, nn.Conv2d):
                print(f"  - {name} (Conv2d)")
            elif isinstance(module, nn.Linear):
                print(f"  - {name} (Linear)")
    print("="*50)


# --- Custom Dataset for flat ImageNet‑val folder ---
class ImageNetValFlat(Dataset):
    """
    ImageNet validation set stored in a single directory with 50 k JPEGs and
    an accompanying ILSVRC2012_validation_ground_truth.txt file.
    Each line in the txt is a 1‑based class index (1‥1000) in the alphabetical
    order of the filenames.
    """
    def __init__(self, root, gt_file, transform=None, allow_subset=False):
        self.root = root
        self.transform = transform
        # sort filenames by numeric ID to match ground-truth ordering
        import re
        pattern = re.compile(r"ILSVRC2012_val_(\d+)\.JPEG", re.IGNORECASE)
        self.filenames = sorted(
            (f for f in os.listdir(root) if pattern.match(f)),
            key=lambda x: int(pattern.match(x).group(1))
        )
        with open(gt_file) as f:
            # convert to 0‑based labels expected by PyTorch
            self.labels = [int(line.strip()) - 1 for line in f]
        # map ILSVRC2012_ID -> torchvision class index
        if transform is not None:  # locate meta.mat once
            meta_path = META_MAT
            self._id_map = build_id_to_classidx(meta_path)
            self.labels = [self._id_map[label] for label in self.labels]
        if not self.filenames or len(self.filenames) != len(self.labels):
            raise ValueError("Image/label count mismatch or empty validation dataset.")
        if not allow_subset and len(self.filenames) != 50_000:
            raise ValueError("Expected 50,000 validation images; use --allow-val-subset for a small test dataset.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.filenames[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# --- Helper: map ILSVRC2012_ID (1‥1000) to torchvision class index (0‥999)
def build_id_to_classidx(devkit_meta_path: str):
    """
    Returns a list idx_map where idx_map[ILSVRC2012_ID - 1] = torchvision_class_idx
    so we can convert ground‑truth labels to the indices expected by pretrained models.
    """
    mat = sio.loadmat(devkit_meta_path, squeeze_me=True, struct_as_record=False)
    synsets = mat["synsets"]
    # Collect the 1000 leaf synsets (num_children == 0) with their WNIDs
    leaf_tuples = [
        (s.ILSVRC2012_ID, s.WNID)
        for s in synsets
        if int(s.num_children) == 0
    ]  # list of (1‑based id, 'n01440764')
    # torchvision class indices are alphabetical by WNID
    sorted_wnids = sorted(wnid for _, wnid in leaf_tuples)
    wnid_to_classidx = {wnid: i for i, wnid in enumerate(sorted_wnids)}
    id_to_classidx = [None] * 1000
    for ilsvrc_id, wnid in leaf_tuples:
        id_to_classidx[ilsvrc_id - 1] = wnid_to_classidx[wnid]
    return id_to_classidx

def build_test_loader(allow_subset=False):
    imagenet_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageNetValFlat(
        root=IMAGENET_VAL_DIR,
        gt_file=GROUND_TRUTH_TXT,
        transform=imagenet_transforms,
        allow_subset=allow_subset,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

# --- 1. Metric Calculation Functions ---

def get_classification_metrics(model, data_loader, device):
    """
    Calculates top-1 and top-5 accuracy for a given classification model.
    This is the standard evaluation method for models trained on ImageNet.
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Classification Eval"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get top-5 predictions
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            correct = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))

            # Top-1 Correct
            correct_top1 += correct[0].reshape(-1).float().sum(0, keepdim=True).item()
            # Top-5 Correct
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total_samples += labels.size(0)

    top1_accuracy = (correct_top1 / total_samples) * 100
    top5_accuracy = (correct_top5 / total_samples) * 100

    return {"top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy}

def evaluate(model, name, loader, device):
    metrics = get_classification_metrics(model, loader, device)
    print(f"  > {name} Metrics (on ImageNet Val):")
    for m, v in metrics.items():
        print(f"    - {m}: {v:.2f}%")
    print("-" * 40)
    return metrics

# Setup data transforms
def main(args):
    global DATA_DIR, IMAGENET_VAL_DIR, GROUND_TRUTH_TXT, META_MAT
    global NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, CHECKPOINT_DIR, device
    DATA_DIR = os.path.join(args.local_imagenet_path, "ILSVRC2012_img_train")
    IMAGENET_VAL_DIR = os.path.join(args.local_imagenet_path, "ILSVRC2012_img_val")
    GROUND_TRUTH_TXT = os.path.join(args.local_imagenet_path, "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
    META_MAT = os.path.join(args.local_imagenet_path, "ILSVRC2012_devkit_t12/data/meta.mat")
    NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS = args.epochs, args.batch_size, args.workers
    CHECKPOINT_DIR = args.checkpoint_dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset
    full_dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_transform
    )

    # Get number of classes from dataset
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Dataset size: {len(full_dataset)}")


    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create separate validation dataset with validation transforms
    val_dataset_transformed = datasets.ImageFolder(
        root=DATA_DIR,
        transform=val_transform
    )

    # Get the same validation indices
    val_indices = val_dataset.indices
    val_dataset_subset = torch.utils.data.Subset(val_dataset_transformed, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    # Test the model with replacing one layer one by one
    RETRAIN_LAYERS = [
        "layer2.0.conv1",  # Layer id 13 CONV3_1_1
        "layer3.0.conv1",  # Layer id 29 CONV4_1_1
        "layer3.0.conv2",  # Layer id 30 CONV4_1_2
        "layer4.0.conv1",  # Layer id 53 CONV5_1_1
        "layer4.0.conv2",  # Layer id 54 CONV5_1_2
        "layer4.1.conv2",  # Layer id 58 CONV5_2_2
        "layer4.2.conv2",  # Layer id 63 CONV5_3_2
    ]

    test_models = []
    test_loader = build_test_loader(args.allow_val_subset)

    state_dict = models.resnet50(weights="IMAGENET1K_V1").state_dict()
    print("Model's state_dict:")
    for param_tensor in state_dict:
        print(f"{param_tensor}\t {state_dict[param_tensor].size()}")

    # Setup model with custom layers
    for layer_path in RETRAIN_LAYERS:
        print(f"🔧 Setting up custom ResNet model with layer: {layer_path}")
        layers_to_replace = [layer_path]
        model = setup_custom_resnet_model(layers_to_replace=layers_to_replace)
        model_name = f"resnet50_d2sconv_s2d_{layer_path.replace('.', '_')}"
        state_dict = model.state_dict()
        print("Converted model's state_dict:")
        for param_tensor in state_dict:
            print(f"{param_tensor}\t {state_dict[param_tensor].size()}")
        model = model.to(device)
        freeze_all_layers(model)

        parts = layer_path.split('.')
        base_path = '.'.join(parts[:-1])
        conv_name = parts[-1]
        bn_name = conv_name.replace('conv', 'bn')  # conv1->bn1, conv2->bn2
        bn_layer_path = f"{base_path}.{bn_name}"
        training_mode = "D2SConvS2D layers only"

        unfreeze_specific_layers(model, layers_to_replace + [bn_layer_path])
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print("model structure:")
        print(model)

        # Setup optimizer only for trainable parameters
        optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

        train_layers = layers_to_replace + [bn_layer_path]
        trainer = LayerVersionTrainer(
            model=model,
            model_name=model_name,
            teacher_model=models.resnet50(pretrained=True),
            train_layers=train_layers,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            num_epochs=NUM_EPOCHS,
            device=device
        )
        trainer.train()
        model.eval().to(device)
    
        test_metrics = evaluate(model.to(device), model_name, test_loader, device)

        save_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pth")
        # We save the state_dict, not the whole model object
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved model state_dict to {save_path}")

        model.to('cpu')  # Move model back to CPU after evaluation
        test_models.append({
            'model_name': model_name,
            'model_path': save_path,  # Store the path for reference
            'train_metrics': None,
            'test_metrics': test_metrics
        })

    # save the test_models
    metrics_save_path = 'test_metrics.pkl'
    print(f"\nSaving metrics and model paths to {metrics_save_path}...")
    with open(metrics_save_path, 'wb') as f:
        pkl.dump(test_models, f)
    print("✅ Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ordinary ResNet50 seven-layer retraining")
    parser.add_argument("--local_imagenet_path", default="datasets/imagenet")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument('--allow-val-subset', action='store_true', help='Allow a smaller validation set with matching labels for smoke tests.')
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    main(parser.parse_args())
