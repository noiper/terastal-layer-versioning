# Support both `python -m experiments.swin_tiny.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.swin_tiny"

import torch
import torch.nn as nn
import torch.optim as optim
import functools
import argparse
import os
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.datasets as datasets
import scipy.io as sio

import importlib

MODEL_ANALYSIS_AVAILABLE = True

# Import the focused conversion utility from our separate module
from utils.layer_versioning.models import D2SConvS2D, LinearToConv, replace_linear_with_conv
model_analysis = importlib.import_module("utils.layer_versioning.model_analysis")

class SwinD2SWrapper(nn.Module):
    """
    A wrapper for D2SConvS2D to handle the 3D tensor shapes found in
    Swin Transformer models. This wrapper treats each token independently.
    """
    def __init__(self, d2s_module):
        super().__init__()
        self.d2s_module = d2s_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return self.d2s_module(x)
        B, N, C = x.shape
        x_reshaped = x.reshape(B * N, C).unsqueeze(-1).unsqueeze(-1)
        output_reshaped = self.d2s_module(x_reshaped)
        _B_N, C_out, _H, _W = output_reshaped.shape
        output = output_reshaped.view(B, N, C_out)
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
        final_module = SwinD2SWrapper(new_mod)
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
            print(f"  - Unfrozen: {name}")
        else:
            param.requires_grad = False
    return model

# --- Data Loading Logic from ResNet50 Example ---

class ImageNetValFlat(Dataset):
    """ Custom Dataset for the flat ImageNet validation folder structure. """
    def __init__(self, root, gt_file, meta_file, transform=None):
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
        
        assert len(self.filenames) == len(self.labels) == 50_000, "Image/label count mismatch."

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

def get_imagenet_loaders(data_dir, batch_size, num_workers):
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
    val_dataset = ImageNetValFlat(root=val_dir, gt_file=gt_file, meta_file=meta_file, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

# --- Evaluation Logic from ResNet50 Example ---

def get_classification_metrics(model, data_loader, device):
    """ Calculates top-1 and top-5 accuracy. """
    model.eval()
    correct_top1, correct_top5, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Classification Eval"):
            images, labels = images.to(device), labels.to(device)
            # Adapt for Hugging Face model output
            outputs = model(pixel_values=images).logits

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
    print(f"  > {name} Metrics (on ImageNet Val):")
    for m, v in metrics.items():
        print(f"    - {m}: {v:.2f}%")
    print("-" * 40)
    return metrics

def train_model(model, device, train_loader, num_epochs):
    """Fine-tunes the model on the specified dataset."""
    print(f"\n--- Starting Model Training ---")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        for i, (pixel_values, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()
            # Use labels for loss calculation with Hugging Face models
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(train_loader):.4f}")
    
    print("--- Training Finished ---")
    return model

def main(args):
    if not MODEL_ANALYSIS_AVAILABLE:
        print("\nAborting due to missing local modules.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_imagenet_loaders(args.local_imagenet_path, args.batch_size, args.workers)
    
    # Create a directory to save models
    os.makedirs("saved_models", exist_ok=True)
    
    example_input = torch.randn(1, 3, 224, 224)
    network_name = "swin-tiny"
    RETRAIN_LAYERS = [
        "swin.encoder.layers.3.blocks.0.intermediate.dense",
        "swin.encoder.layers.3.blocks.0.output.dense",
        "swin.encoder.layers.3.blocks.1.intermediate.dense",
        "swin.encoder.layers.3.blocks.1.output.dense"
    ]
    
    for idx, layer_path in enumerate(RETRAIN_LAYERS):
        print("-" * 50)
        print(f"Processing layer {idx+1}/{len(RETRAIN_LAYERS)}: {layer_path}")
        
        print("Loading a fresh Swin Transformer model...")
        model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=1000, ignore_mismatched_sizes=True)
        model.to(device)

        # evaluate(model, "Original Model (Baseline)", val_loader, device)
        
        model = replace_with_d2sconv_s2d(model, layer_path, device, r=2)
        
        trainable_layers = [layer_path, "classifier"]
        print(f"Freezing model weights except for: {trainable_layers}...")
        model = freeze_model_except(model, trainable_layers)
        
        model = train_model(model, device, train_loader, num_epochs=args.epochs)
        
        evaluate(model, f"Fine-tuned model ({layer_path})", val_loader, device)
        
        file_name = f"{network_name}_idx{idx}_trained_imagenet"
        
        # Save the fine-tuned model's state dictionary
        save_path = os.path.join("saved_models", f"{file_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved fine-tuned model to {save_path}")
        
        print(f"\nExporting analysis files for trained model: {file_name}...")
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
    parser = argparse.ArgumentParser(description="Swin Transformer Layer Conversion and Fine-tuning on ImageNet")
    parser.add_argument('--local_imagenet_path', type=str, required=True, help='Path to the local ImageNet dataset directory.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation.')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes for data loading.')
    args = parser.parse_args()
    main(args)