# Support both `python -m experiments.vgg11.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.vgg11"

from .layers import selected_layers

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import functools
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.datasets as datasets
import scipy.io as sio

import importlib

MODEL_ANALYSIS_AVAILABLE = True

# Import the focused conversion utility from our separate module
try:
    from utils.layer_versioning.models import D2SConvS2D, LinearToConv, replace_linear_with_conv
    model_analysis = importlib.import_module("utils.layer_versioning.model_analysis")
except ImportError:
    print("="*50)
    print("WARNING: Could not import local modules (D2SConvS2D, model_analysis).")
    print("The script may fail if these are required.")
    print("="*50)
    MODEL_ANALYSIS_AVAILABLE = False


class VGGD2SWrapper(nn.Module):
    """
    A wrapper for D2SConvS2D to handle the 3D tensor shapes found in
    VGG models. This wrapper treats each token independently.
    """
    def __init__(self, d2s_module):
        super().__init__()
        self.d2s_module = d2s_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.shape
        x_reshaped = x.reshape(N, C).unsqueeze(-1).unsqueeze(-1)
        output_reshaped = self.d2s_module(x_reshaped)
        _B_N, C_out, _H, _W = output_reshaped.shape
        output = output_reshaped.view(N, C_out)
        return output

def replace_with_d2sconv_s2d(model: nn.Module, layer_path: str, device, r=2):
    """
    Replaces a target layer (Conv2d or Linear) with a D2SConvS2D module.
    Ensures the new module is on the correct device.
    """
    parts = layer_path.split(".")
    parent = functools.reduce(getattr, [model] + parts[:-1])
    name = parts[-1]
    orig = getattr(parent, name)
    was_linear = isinstance(orig, nn.Linear)
    
    if was_linear:
        print(f"Layer {layer_path} is Linear. Converting to Conv2d first.")
        replace_linear_with_conv(model, layer_path)
        orig = getattr(parent, name)

    conv_layer = orig.conv if isinstance(orig, LinearToConv) else orig
    
    if not isinstance(conv_layer, nn.Conv2d):
        raise TypeError(f"Target layer {layer_path} could not be resolved to a Conv2d layer.")

    k = conv_layer.kernel_size[0] if isinstance(conv_layer.kernel_size, tuple) else conv_layer.kernel_size
    p = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
    
    assert conv_layer.in_channels % (r*r) == 0, f"in_channels {conv_layer.in_channels} not divisible by r^2 ({r*r})"
    assert conv_layer.out_channels % (r*r) == 0, f"out_channels {conv_layer.out_channels} not divisible by r^2 ({r*r})"
    
    new_mod = D2SConvS2D(conv_layer.in_channels, conv_layer.out_channels,
                         kernel_size=k, stride=conv_layer.stride, padding=p, r=r,
                         groups=conv_layer.groups, bias=(conv_layer.bias is not None))
    
    if was_linear:
        final_module = VGGD2SWrapper(new_mod)
    else:
        final_module = new_mod
        
    # FIX: Move the newly created module to the same device as the model
    setattr(parent, name, final_module.to(device))
        
    print(f"✅ Replaced {layer_path} with D2SConvS2D")
    return model

def freeze_model_except(model, trainable_layer_names):
    """
    Freezes all model parameters except for the layers specified in the list.
    """
    if not isinstance(trainable_layer_names, list):
        trainable_layer_names = [trainable_layer_names]
        
    for name, param in model.named_parameters():
        is_trainable = any(trainable_name in name for trainable_name in trainable_layer_names)
        if is_trainable:
            param.requires_grad = True
            print(f"   - Unfrozen: {name}")
        else:
            param.requires_grad = False
    return model

class ImageNetValFlat(Dataset):
    """ Custom Dataset for the flat ImageNet validation folder structure. """
    def __init__(self, root, gt_file, meta_file, transform=None, allow_subset=False):
        self.root = root
        self.transform = transform
        import re
        pattern = re.compile(r"ILSVRC2012_val_(\d+)\.JPEG", re.IGNORECASE)
        self.filenames = sorted(
            (f for f in os.listdir(root) if pattern.match(f)),
            key=lambda x: int(pattern.match(x).group(1))
        )
        with open(gt_file) as f:
            self.labels = [int(line.strip()) - 1 for line in f]
        
        if transform is not None:
            self._id_map = build_id_to_classidx(meta_file)
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

def build_id_to_classidx(devkit_meta_path: str):
    """ Helper to map ILSVRC2012_ID to torchvision class index. """
    mat = sio.loadmat(devkit_meta_path, squeeze_me=True, struct_as_record=False)
    synsets = mat["synsets"]
    leaf_tuples = [(s.ILSVRC2012_ID, s.WNID) for s in synsets if int(s.num_children) == 0]
    sorted_wnids = sorted(wnid for _, wnid in leaf_tuples)
    wnid_to_classidx = {wnid: i for i, wnid in enumerate(sorted_wnids)}
    id_to_classidx = [None] * 1000
    for ilsvrc_id, wnid in leaf_tuples:
        id_to_classidx[ilsvrc_id - 1] = wnid_to_classidx[wnid]
    return id_to_classidx

def get_imagenet_loaders(data_dir, batch_size, num_workers, allow_val_subset=False):
    """ Creates train and validation DataLoaders for ImageNet. """
    train_dir = os.path.join(data_dir, 'ILSVRC2012_img_train')
    val_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    gt_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    meta_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/meta.mat')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    print("Setting up ImageNet training data loader...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    print("Setting up ImageNet validation data loader...")
    val_dataset = ImageNetValFlat(root=val_dir, gt_file=gt_file, meta_file=meta_file, transform=val_transform, allow_subset=allow_val_subset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

# --- Evaluation Logic (Moved before train_model) ---

def get_classification_metrics(model, data_loader, device):
    """ Calculates top-1 and top-5 accuracy. """
    model.eval()
    correct_top1, correct_top5, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Classification Eval"):
            images, labels = images.to(device), labels.to(device)
            
            # --- FIX 1 ---
            # torchvision models return logits directly
            # The input is just the image tensor
            outputs = model(images)
            # --- END FIX 1 ---

            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            correct = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))

            correct_top1 += correct[0].reshape(-1).float().sum(0, keepdim=True).item()
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total_samples += labels.size(0)

    top1_accuracy = (correct_top1 / total_samples) * 100
    top5_accuracy = (correct_top5 / total_samples) * 100
    return {"top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy}

def evaluate(model, name, loader, device):
    """ Wrapper to print evaluation metrics. """
    print(f"\n--- Evaluating {name} ---")
    metrics = get_classification_metrics(model, loader, device)
    print(f"   > {name} Metrics (on ImageNet Val):")
    for m, v in metrics.items():
        print(f"     - {m}: {v:.2f}%")
    print("-" * 40)
    return metrics

# --- MODIFIED train_model Function ---

def train_model(model, device, train_loader, val_loader, num_epochs, save_path):
    """
    Fine-tunes the model, evaluates after each epoch, and saves the best model.
    """
    print(f"\n--- Starting Model Training ---")
    print(f" ▷ Will save best model to: {save_path}")

    # --- FIX 2: Define Loss Function ---
    criterion = nn.CrossEntropyLoss()
    # --- END FIX 2 ---
    
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    
    best_accuracy = float("-inf")  # Track the best top-1 accuracy

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # --- Training Phase ---
        model.train() # Set model to training mode
        total_loss = 0
        
        for i, (pixel_values, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # --- FIX 3: Correct Model Call and Loss Calculation ---
            # 1. Call model with input tensor directly
            outputs = model(pixel_values) 
            # 2. Calculate loss manually using the criterion
            loss = criterion(outputs, labels) 
            # --- END FIX 3 ---
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # --- Evaluation Phase (NEW) ---
        # get_classification_metrics sets model.eval() internally
        metrics = get_classification_metrics(model, val_loader, device)
        top1_accuracy = metrics["top1_accuracy"]
        
        print(f"  Epoch {epoch + 1} Avg Loss: {avg_loss:.4f} | Val Top-1 Acc: {top1_accuracy:.2f}%")

        # --- Save Best Model (NEW) ---
        if top1_accuracy > best_accuracy:
            best_accuracy = top1_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"    ✨ New best model saved with {top1_accuracy:.2f}% accuracy.")

    print(f"--- Training Finished ---")
    print(f"✅ Best model saved to {save_path} (Top-1 Acc: {best_accuracy:.2f}%)")
    return model

# --- MODIFIED main Function ---

def main(args):
    if not MODEL_ANALYSIS_AVAILABLE:
        print("\nAborting due to missing local modules.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    layers = selected_layers(args.layer_indices)
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    train_loader, val_loader = get_imagenet_loaders(args.local_imagenet_path, args.batch_size, args.workers, args.allow_val_subset)
    
    # Create a directory to save models
    os.makedirs(args.saved_models_path, exist_ok=True)
    
    example_input = torch.randn(1, 3, 224, 224)
    network_name = "vgg11"
    weights_vgg11 = models.VGG11_Weights.DEFAULT
    
    for layer_path, spec in layers.items():
        idx = spec["idx"]
        print("-" * 50)
        print(f"Processing layer {idx}/{len(layers)}: {layer_path}")
        
        print("Loading a fresh VGG11 model...")
        model = models.vgg11(weights=weights_vgg11)
        model.to(device)
        model = replace_with_d2sconv_s2d(model, layer_path, device, r=2)
        
        trainable_layers = [layer_path, "classifier"]
        print(f"Freezing model weights except for: {trainable_layers}...")
        model = freeze_model_except(model, trainable_layers)
        
        # --- MODIFIED SECTION ---
        
        # 1. Define the save path for this layer's best model
        file_name = f"{network_name}_idx{idx}_trained_imagenet"
        save_path = os.path.join(args.saved_models_path, f"{file_name}.pth")
        
        # 2. Call the new train_model function
        #    It now handles training, per-epoch evaluation, and saving.
        model = train_model(model, device, train_loader, val_loader, 
                            num_epochs=args.epochs, save_path=save_path)
        
        # 3. The old evaluate() and torch.save() calls are removed from here.
        
        # --- END MODIFIED SECTION ---
        
        
        print(f"\nExporting analysis files for trained model: {file_name}...")
        
        # --- ENHANCEMENT: Load the BEST model for analysis ---
        print(f"Loading best weights from {save_path} for analysis...")
        model.load_state_dict(torch.load(save_path))
        # --- END ENHANCEMENT ---
        
        model.to('cpu')
        
        if MODEL_ANALYSIS_AVAILABLE:
            try:
                model_analysis.export_maestro_conv_layers(
                    model, example_input, network_name=network_name,
                    fps=30, filename=f"{file_name}.m"
                )
                model_analysis.export_torch_layers(model, f"{file_name}_layers.txt")
                print("✅ Analysis export successful.")
            except Exception as e:
                print(f"❌ Analysis failed for layer {layer_path}: {e}")
                continue

    print("-" * 50)
    print("\nAll independent layer conversions and analyses are complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGG Layer Conversion and Fine-tuning on ImageNet")
    parser.add_argument('--local_imagenet_path', type=str, required=True, help='Path to the local ImageNet dataset directory.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation.')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes for data loading.')
    parser.add_argument('--allow-val-subset', action='store_true', help='Allow a smaller validation set with matching labels for smoke tests.')
    parser.add_argument("--layer-indices", nargs="+", type=int, choices=range(5), default=[3], help="Stable checkpoint indices to train; use 0 1 2 3 4 for all candidates")
    parser.add_argument("--saved_models_path", default="saved_models")
    args = parser.parse_args()
    main(args)
