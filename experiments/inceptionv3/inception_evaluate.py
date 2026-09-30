# Support both `python -m experiments.inceptionv3.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.inceptionv3"

import torch
import torch.nn as nn
import torchvision.models as models
import functools
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.datasets as datasets
import scipy.io as sio

# --- Custom Layer Imports ---
# This script assumes your 'utils' directory is accessible.
try:
    from utils.layer_versioning.models import D2SConvS2D
    print("Successfully imported D2SConvS2D.")
except ImportError:
    print("="*50)
    print("WARNING: Could not import local module 'D2SConvS2D'.")
    print("The script will fail if it tries to load the custom model.")
    print("Make sure 'utils/layer_versioning/models.py' is in your PYTHONPATH.")
    print("="*50)

# --- Layer Replacement Utility (Copied from your script) ---
def replace_with_d2sconv_s2d(model, layer_path: str, r=2):
    """ Replaces a Conv2d layer with a D2SConvS2D layer. """
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
    print(f"✅ Replaced {layer_path} with D2SConvS2D for architecture matching.")
    return model

# --- Data Loading Utilities (Copied from your script) ---
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

def get_imagenet_val_loader(data_dir, batch_size, num_workers):
    """ Creates the validation DataLoader for ImageNet (InceptionV3 size). """
    val_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    gt_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    meta_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/meta.mat')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # InceptionV3 validation transform (299x299)
    val_transform = T.Compose([
        T.Resize(320), # Resize larger than crop
        T.CenterCrop(299),
        T.ToTensor(),
        normalize,
    ])

    print("Setting up ImageNet validation data loader (299x299)...")
    val_dataset = ImageNetValFlat(root=val_dir, gt_file=gt_file, meta_file=meta_file, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return val_loader

# --- Evaluation Logic (Copied from your script) ---
def get_classification_metrics(model, data_loader, device):
    """ Calculates top-1 and top-5 accuracy. """
    model.eval()
    correct_top1, correct_top5, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Classification Eval"):
            images, labels = images.to(device), labels.to(device)
            
            # In eval mode, InceptionV3 returns raw logits directly
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
    print(f" ▷ {name} Metrics (on ImageNet Val):")
    for m, v in metrics.items():
        print(f"    - {m}: {v:.2f}%")
    print("-" * 40)
    return metrics

# --- Main Evaluation Function ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    val_loader = get_imagenet_val_loader(args.local_imagenet_path, args.batch_size, args.workers)
    
    # --- 1. Evaluate the Original Pre-trained Model ---
    print("\nLoading original pre-trained InceptionV3 model...")
    # The fix
    original_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    original_model.to(device)
    evaluate(original_model, "Original Pre-trained InceptionV3", val_loader, device)

    # --- 2. Evaluate Your Custom-Trained Model ---
    
    # Check if the model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Skipping evaluation of custom model.")
        return

    print(f"\nLoading custom model from {args.model_path}...")
    
    # This is the layer you replaced in your training script
    layer_to_replace = "Mixed_6a.branch3x3.conv"
    
    # CRITICAL:
    # 1. Create a base InceptionV3 model (with no weights).
    #    We set aux_logits=False because we are only evaluating.
    custom_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    
    # 2. Apply the *exact same* layer replacement to this new model
    #    so its architecture matches the saved state_dict.
    try:
        custom_model = replace_with_d2sconv_s2d(custom_model, layer_to_replace, r=2)
    except Exception as e:
        print(f"Error during layer replacement for custom model: {e}")
        print("This likely means the D2SConvS2D import failed or the layer path is wrong.")
        return
        
    custom_model.to(device)
    
    # 3. Now, load the saved weights into the correctly-structured model
    try:
        custom_model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Successfully loaded custom model weights.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("This often happens if the model architecture does not match the saved weights.")
        return

    # 4. Evaluate the custom model
    evaluate(custom_model, "Custom-Trained InceptionV3", val_loader, device)
    
    print("\nEvaluation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="InceptionV3 Model Evaluation")
    parser.add_argument('--local_imagenet_path', type=str, required=True, help='Path to the local ImageNet dataset directory.')
    parser.add_argument('--model_path', type=str, 
                        default='saved_models/inceptionv3_idx0_trained_imagenet.pth',
                        help='Path to the saved custom model .pth file.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes for data loading.')
    
    args = parser.parse_args()
    main(args)
