# Support both `python -m experiments.vgg11.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.vgg11"

from experiments.checkpoints import require_checkpoint_files, copy_variant_state

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
from itertools import combinations
import importlib

# NOTE: This script assumes your custom modules are available in your Python path
try:
    from utils.layer_versioning.models import D2SConvS2D, LinearToConv, replace_linear_with_conv
    print("Successfully imported D2SConvS2D, LinearToConv, and replace_linear_with_conv modules.")
except ImportError as e:
    print(f"FATAL ERROR: Could not import required local modules: {e}")
    print("Please ensure your custom modules are in the PYTHONPATH.")
    raise ImportError("Required experiment modules could not be imported") from e

# --- Copied Helper Classes & Functions for VGG11 ---

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
    (Copied from your VGG training script)
    """
    parts = layer_path.split(".")
    parent = functools.reduce(getattr, [model] + parts[:-1])
    name = parts[-1]
    orig = getattr(parent, name)
    was_linear = isinstance(orig, nn.Linear)
    
    if was_linear:
        # print(f"Layer {layer_path} is Linear. Converting to Conv2d first.") # Verbose
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
        
    # Move the newly created module to the same device as the model
    setattr(parent, name, final_module.to(device))
        
    # print(f"✅ Replaced {layer_path} with D2SConvS2D (r={r})") # Verbose
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

def get_imagenet_val_loader(data_dir, batch_size, num_workers):
    """ Creates a validation DataLoader for ImageNet. """
    val_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    gt_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    meta_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12/data/meta.mat')
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    val_dataset = ImageNetValFlat(root=val_dir, gt_file=gt_file, meta_file=meta_file, transform=val_transform)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def get_classification_metrics(model, data_loader, device):
    """ 
    Calculates top-1 and top-5 accuracy.
    (Adapted for torchvision models)
    """
    model.eval()
    correct_top1, correct_top5, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Classification Eval"):
            images, labels = images.to(device), labels.to(device)
            
            # torchvision models return logits directly from image tensor
            outputs = model(images) 
            
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            correct = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
            correct_top1 += correct[0].reshape(-1).float().sum(0, keepdim=True).item()
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total_samples += labels.size(0)
    top1 = (correct_top1 / total_samples) * 100
    top5 = (correct_top5 / total_samples) * 100
    return {"top1_accuracy": top1, "top5_accuracy": top5}

# --- Main Inference Script ---

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Resolve and verify dataset path
    abs_path = os.path.abspath(os.path.expanduser(args.local_imagenet_path))
    print(f"Resolved local ImageNet path to absolute path: {abs_path}")
    args.local_imagenet_path = abs_path
    
    val_dir = os.path.join(args.local_imagenet_path, 'ILSVRC2012_img_val')
    gt_file = os.path.join(args.local_imagenet_path, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    meta_file = os.path.join(args.local_imagenet_path, 'ILSVRC2012_devkit_t12/data/meta.mat')

    if not (os.path.isdir(val_dir) and os.path.isfile(gt_file) and os.path.isfile(meta_file)):
        raise FileNotFoundError(f"ImageNet validation files not found in {abs_path}")
    print("Successfully verified local ImageNet validation directory structure.")

    # 2. Define layers to test, their file indices, and 'r' values
    # (Based on your VGG training script)
    layers_to_test = selected_layers(args.layer_indices)
    require_checkpoint_files(os.path.join(args.saved_models_path, f"vgg11_idx{spec['idx']}_trained_imagenet.pth") for spec in layers_to_test.values())
    layer_paths = list(layers_to_test.keys())
    
    # 3. Load the validation data loader
    print("\nSetting up ImageNet validation data loader...")
    val_loader = get_imagenet_val_loader(args.local_imagenet_path, args.batch_size, args.workers)
    
    # 4. Load baseline weights (used for loading fresh model)
    weights_vgg11 = models.VGG11_Weights.DEFAULT
        
    all_results = []
    num_combinations = 2**len(layer_paths)
    print(f"\nStarting combination testing for {len(layer_paths)} layers ({num_combinations} total combinations)...")

    # 5. Iterate through all combinations
    for r in range(len(layer_paths) + 1):
        for combo in combinations(layer_paths, r):
            if not combo:
                combo_name = "Original VGG11 (Baseline)"
            else:
                combo_name = "Converted: " + ", ".join([f"idx{layers_to_test[p]['idx']}" for p in combo])
            
            print("\n" + "="*50)
            print(f"Testing combination: {combo_name}")
            print(f"Layers: {combo or 'None'}")
            
            # 1. Load a fresh base model with pre-trained weights
            model = models.vgg11(weights=weights_vgg11)
            
            # 2. Architecturally convert layers in this combination
            for layer_path in combo:
                r_val = layers_to_test[layer_path]['r']
                model = replace_with_d2sconv_s2d(model, layer_path, device, r=r_val)
            
            # 3. Load weights for this combination
            # Start with the fresh model's state dict (has all baseline weights)
            final_state_dict = model.state_dict()
            
            # Load the specific fine-tuned weights for each converted layer
            for layer_path in combo:
                idx = layers_to_test[layer_path]['idx']
                model_file = f"vgg11_idx{idx}_trained_imagenet.pth"
                model_path = os.path.join(args.saved_models_path, model_file)
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(model_path)
                
                try:
                    trained_state_dict = torch.load(model_path, map_location='cpu')
                    # Copy only the keys relevant to this replaced layer
                    copy_variant_state(final_state_dict, trained_state_dict, [layer_path])
                except Exception as e:
                    raise RuntimeError(f"Cannot apply checkpoint {model_path}") from e

            try:
                model.load_state_dict(final_state_dict)
            except Exception as e:
                print(f"FATAL: Error loading final state dict for combo {combo_name}: {e}")
                raise RuntimeError(f"Architecture mismatch for {combo_name}") from e
                
            model.to(device)
            
            # 4. Evaluate
            metrics = get_classification_metrics(model, val_loader, device)
            all_results.append({'name': combo_name, 'layers': combo, 'metrics': metrics})

            # --- Print immediate results ---
            print(f"  > Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
            print(f"  > Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")

    # 6. Print final summary and save to file
    print("\n" + "="*70)
    print("--- FINAL VGG11 COMBINATION ANALYSIS SUMMARY ---")
    print("="*70)
    
    results_filename = "vgg11_combination_results.txt"
    try:
        with open(results_filename, "w") as f:
            f.write("="*70 + "\n")
            f.write("--- FINAL VGG11 COMBINATION ANALYSIS SUMMARY ---\n")
            f.write("="*70 + "\n")
            
            for result in all_results:
                summary_line_1 = f"\nModel: {result['name']}"
                summary_line_2 = f"  > Top-1 Accuracy: {result['metrics']['top1_accuracy']:.2f}%"
                summary_line_3 = f"  > Top-5 Accuracy: {result['metrics']['top5_accuracy']:.2f}%"
                
                # Print to console
                print(summary_line_1)
                print(summary_line_2)
                print(summary_line_3)
                
                # Write to file
                f.write(summary_line_1 + "\n")
                f.write(summary_line_2 + "\n")
                f.write(summary_line_3 + "\n")

            f.write("\n" + "="*70 + "\n")
            f.write("Combination analysis complete.\n")

        print("\n" + "="*70)
        print("Combination analysis complete.")
        print(f"✅ Results saved to {results_filename}")
    
    except IOError as e:
        raise OSError("Could not save evaluation results") from e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGG11 Combination Inference")
    parser.add_argument('--local_imagenet_path', type=str, required=True, help='Path to the local ImageNet dataset directory.')
    parser.add_argument('--saved_models_path', type=str, default="saved_models", help='Path to the directory containing the saved .pth model files.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes for data loading.')
    parser.add_argument("--layer-indices", nargs="+", type=int, choices=range(5), default=list(range(5)), help="Stable checkpoint indices to evaluate; defaults to all five candidates")
    args = parser.parse_args()
    main(args)
