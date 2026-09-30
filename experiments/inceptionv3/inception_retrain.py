# Support both `python -m experiments.inceptionv3.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.inceptionv3"

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
    print("Successfully imported D2SConvS2D and model_analysis.")
except ImportError:
    print("="*50)
    print("WARNING: Could not import local modules (D2SConvS2D, model_analysis).")
    print("The script may fail if these are required.")
    print("="*50)
    MODEL_ANALYSIS_AVAILABLE = False

def replace_with_d2sconv_s2d(model, layer_path: str, r=2):
    """
    Replaces a target nn.Conv2d layer with a D2SConvS2D module.
    """
    parts = layer_path.split(".")
    parent = functools.reduce(getattr, [model] + parts[:-1])
    name = parts[-1]
    orig = getattr(parent, name)
    
    assert isinstance(orig, nn.Conv2d), "Target must be nn.Conv2d"
    k = orig.kernel_size[0] if isinstance(orig.kernel_size, tuple) else orig.kernel_size
    p = orig.padding[0] if isinstance(orig.padding, tuple) else orig.padding
    
    # Check divisibility
    assert orig.in_channels % (r*r) == 0, f"in_channels {orig.in_channels} not divisible by r^2 ({r*r})"
    assert orig.out_channels % (r*r) == 0, f"out_channels {orig.out_channels} not divisible by r^2 ({r*r})"
    
    new_mod = D2SConvS2D(orig.in_channels, orig.out_channels,
                         kernel_size=k, stride=orig.stride, padding=p, r=r,
                         groups=orig.groups, bias=(orig.bias is not None))
    
    setattr(parent, name, new_mod)
    print(f"✅ Replaced {layer_path} with D2SConvS2D")
    return model

def freeze_model_except(model, trainable_layer_names):
    """
    Freezes all model parameters except for the layers specified in the list.
    """
    if not isinstance(trainable_layer_names, list):
        trainable_layer_names = [trainable_layer_names]
        
    for name, param in model.named_parameters():
        # --- FIX: Check if the name STARTS WITH the key ---
        # This prevents "fc" from matching "AuxLogits.fc"
        is_trainable = any(name.startswith(trainable_name) for trainable_name in trainable_layer_names)
        # --- END FIX ---
        
        if is_trainable:
            param.requires_grad = True
            print(f"   - Unfrozen: {name}")
        else:
            param.requires_grad = False
    return model

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

# --- MODIFIED FOR INCEPTIONV3 ---
def get_imagenet_loaders(data_dir, batch_size, num_workers):
    """ Creates train and validation DataLoaders for ImageNet (InceptionV3 size). """
    train_dir = os.path.join(data_dir, 'ILSVRC2012_img_train')
    val_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    gt_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    meta_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/meta.mat')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # InceptionV3 requires 299x299 input
    train_transform = T.Compose([
        T.RandomResizedCrop(299),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    
    val_transform = T.Compose([
        T.Resize(320), # Resize larger than crop
        T.CenterCrop(299),
        T.ToTensor(),
        normalize,
    ])

    print("Setting up ImageNet training data loader (299x299)...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    print("Setting up ImageNet validation data loader (299x299)...")
    val_dataset = ImageNetValFlat(root=val_dir, gt_file=gt_file, meta_file=meta_file, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

# --- Evaluation Logic (Unchanged, but crucial) ---
# This function works as-is because model.eval() causes InceptionV3
# to return only the main logits tensor, not the aux outputs.
def get_classification_metrics(model, data_loader, device):
    """ Calculates top-1 and top-5 accuracy. """
    model.eval()
    correct_top1, correct_top5, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Classification Eval"):
            images, labels = images.to(device), labels.to(device)
            
            # This works because model.eval() mode returns raw logits
            outputs = model(images)

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

# --- MODIFIED FOR INCEPTIONV3 ---
def train_model(model, device, train_loader, val_loader, num_epochs, save_path):
    """
    Fine-tunes the model, evaluates after each epoch, and saves the best model.
    Handles InceptionV3's auxiliary logits.
    """
    print(f"\n--- Starting Model Training ---")
    print(f" ▷ Will save best model to: {save_path}")

    criterion = nn.CrossEntropyLoss()
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    
    best_accuracy = float("-inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # --- Training Phase ---
        model.train() # Set model to training mode (enables aux_logits)
        total_loss = 0
        
        for i, (pixel_values, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # --- FIX FOR INCEPTIONV3 ---
            # In 'train' mode, model returns InceptionOutputs(logits, aux_logits)
            outputs = model(pixel_values) 
            
            # Calculate combined loss
            loss_main = criterion(outputs.logits, labels)
            loss_aux = criterion(outputs.aux_logits, labels)
            loss = loss_main + 0.4 * loss_aux # Standard InceptionV3 loss
            # --- END FIX ---
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # --- Evaluation Phase ---
        # get_classification_metrics sets model.eval() internally
        metrics = get_classification_metrics(model, val_loader, device)
        top1_accuracy = metrics["top1_accuracy"]
        
        print(f"   Epoch {epoch + 1} Avg Loss: {avg_loss:.4f} | Val Top-1 Acc: {top1_accuracy:.2f}%")

        # --- Save Best Model ---
        if top1_accuracy > best_accuracy:
            best_accuracy = top1_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"     ✨ New best model saved with {top1_accuracy:.2f}% accuracy.")

    print(f"--- Training Finished ---")
    print(f"✅ Best model saved to {save_path} (Top-1 Acc: {best_accuracy:.2f}%)")
    return model

# --- MODIFIED FOR INCEPTIONV3 ---
def main(args):
    if not MODEL_ANALYSIS_AVAILABLE:
        print("\nAborting due to missing local modules.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_imagenet_loaders(args.local_imagenet_path, args.batch_size, args.workers)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # InceptionV3 input size
    example_input = torch.randn(1, 3, 299, 299)
    network_name = "inceptionv3"
    
    RETRAIN_LAYERS = ["Conv2d_4a_3x3.conv", # 4
                      "Mixed_6a.branch3x3.conv", # 26
                      "Mixed_7c.branch3x3dbl_1.conv" # 89
                     ]

    # Load InceptionV3 weights
    weights_inception = models.Inception_V3_Weights.DEFAULT
    
    if not RETRAIN_LAYERS:
        print("\nWARNING: RETRAIN_LAYERS list is empty. No training will be performed.")
        return
        
    for idx, layer_path in enumerate(RETRAIN_LAYERS):
        print("-" * 50)
        print(f"Processing layer {idx+1}/{len(RETRAIN_LAYERS)}: {layer_path}")
        
        print("Loading a fresh InceptionV3 model...")
        # CRITICAL: Must set aux_logits=True for the training function
        model = models.inception_v3(weights=weights_inception, aux_logits=True)
        model.to(device)
        
        # This function should work for Inception layers (e.g., "fc" or "Conv2d_1a_3x3.conv")
        try:
            model = replace_with_d2sconv_s2d(model, layer_path, r=2)
        except Exception as e:
            print(f"❌ Failed to replace layer {layer_path}: {e}")
            print("Please check layer path. Skipping this layer.")
            continue
            
        model.to(device)
        
        # CRITICAL: The classifier is named 'fc', not 'classifier'
        trainable_layers = [layer_path, "fc"]
        
        print(f"Freezing model weights except for: {trainable_layers}...")
        model = freeze_model_except(model, trainable_layers)
        
        file_name = f"{network_name}_idx{idx}_trained_imagenet"
        save_path = os.path.join(args.save_dir, f"{file_name}.pth")
        
        model = train_model(model, device, train_loader, val_loader, 
                            num_epochs=args.epochs, save_path=save_path)
        
        print(f"\nExporting analysis files for trained model: {file_name}...")
        
        print(f"Loading best weights from {save_path} for analysis...")
        # We load into a model with aux_logits=True (as it was saved)
        analysis_model = models.inception_v3(weights=None, aux_logits=True)
        analysis_model = replace_with_d2sconv_s2d(analysis_model, layer_path, r=2)
        analysis_model.load_state_dict(torch.load(save_path))
        analysis_model.to('cpu')
        
        if MODEL_ANALYSIS_AVAILABLE:
            try:
                model_analysis.export_maestro_conv_layers(
                    analysis_model, example_input, network_name=network_name,
                    fps=30, filename=f"{file_name}.m"
                )
                model_analysis.export_torch_layers(analysis_model, f"{file_name}_layers.txt")
                print("✅ Analysis export successful.")
            except Exception as e:
                print(f"❌ Analysis failed for layer {layer_path}: {e}")
                continue

    print("-" * 50)
    print("\nAll independent layer conversions and analyses are complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="InceptionV3 Layer Conversion and Fine-tuning on ImageNet")
    parser.add_argument('--local_imagenet_path', type=str, required=True, help='Path to the local ImageNet dataset directory (parent of ILSVRC2012_img_train, etc.)')
    parser.add_argument('--save_dir', type=str, default="saved_models", help='Directory to save trained model checkpoints.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation. (e.g., 128 or 256)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes for data loading.')
    args = parser.parse_args()
    
    # Verify paths
    if not os.path.isdir(args.local_imagenet_path):
        print(f"Error: ImageNet directory not found at {args.local_imagenet_path}")
    else:
        main(args)