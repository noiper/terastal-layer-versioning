# Support both `python -m experiments.inceptionv3.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.inceptionv3"

from experiments.checkpoints import require_checkpoint_files, copy_variant_state

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import torchvision.transforms as T
import os
import functools
import importlib
from itertools import combinations
import argparse
from tqdm import tqdm
from PIL import Image
import scipy.io as sio

# --- Imports from your custom utils ---
try:
    from utils.layer_versioning.models.D2SConvS2D import D2SConvS2D
    print("Successfully imported D2SConvS2D.")
except ImportError as e:
    print(f"FATAL ERROR: Could not import required local modules: {e}")
    print("Please ensure your custom modules are in the PYTHONPATH.")
    raise ImportError("Required experiment modules could not be imported") from e

# --- Helper: `replace_with_d2sconv_s2d` (from training script) ---
def replace_with_d2sconv_s2d(model, layer_path: str, r=2):
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
        return model
    except Exception as e:
        print(f"ERROR: Failed to replace layer {layer_path}: {e}")
        raise

# --- Helper: `ImageNetValFlat` (from training script) ---
class ImageNetValFlat(Dataset):
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

# --- Helper: `build_id_to_classidx` (from training script) ---
def build_id_to_classidx(devkit_meta_path: str):
    mat = sio.loadmat(devkit_meta_path, squeeze_me=True, struct_as_record=False)
    synsets = mat["synsets"]
    leaf_tuples = [(s.ILSVRC2012_ID, s.WNID) for s in synsets if int(s.num_children) == 0]
    sorted_wnids = sorted(wnid for _, wnid in leaf_tuples)
    wnid_to_classidx = {wnid: i for i, wnid in enumerate(sorted_wnids)}
    id_to_classidx = [None] * 1000
    for ilsvrc_id, wnid in leaf_tuples:
        id_to_classidx[ilsvrc_id - 1] = wnid_to_classidx[wnid]
    return id_to_classidx

# --- Helper: `build_test_loader` for InceptionV3 ---
def build_test_loader(val_dir, gt_file, meta_file, batch_size, num_workers):
    # InceptionV3 requires 299x299 input
    val_transform = T.Compose([
        T.Resize(320), # Resize larger than crop
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageNetValFlat(
        root=val_dir,
        gt_file=gt_file,
        meta_file=meta_file,
        transform=val_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

# --- Helper: `get_classification_metrics` (from training script) ---
def get_classification_metrics(model, data_loader, device):
    model.eval()
    correct_top1, correct_top5, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Classification Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if isinstance(outputs, models.inception.InceptionOutputs):
                outputs = outputs.logits
            
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            correct = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
            correct_top1 += correct[0].reshape(-1).float().sum(0, keepdim=True).item()
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total_samples += labels.size(0)
    top1 = (correct_top1 / total_samples) * 100
    top5 = (correct_top5 / total_samples) * 100
    return {"top1_accuracy": top1, "top5_accuracy": top5}

# --- Main Combination Test Script ---
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- ImageNet Paths (Constructed from single arg) ---
    IMAGENET_VAL_DIR = os.path.join(args.local_imagenet_path, 'ILSVRC2012_img_val')
    GROUND_TRUTH_TXT = os.path.join(args.local_imagenet_path, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    META_MAT = os.path.join(args.local_imagenet_path, 'ILSVRC2012_devkit_t12/data/meta.mat')

    # --- Layers to Test (from your training script) ---
    layers_to_test = {
        "Conv2d_4a_3x3.conv": "inceptionv3_idx0_trained_imagenet.pth",
        "Mixed_6a.branch3x3.conv": "inceptionv3_idx1_trained_imagenet.pth",
        "Mixed_7c.branch3x3dbl_1.conv": "inceptionv3_idx2_trained_imagenet.pth",
    }
    require_checkpoint_files(os.path.join(args.saved_models_path, name) for name in layers_to_test.values())
    layer_paths = list(layers_to_test.keys())
    
    print(f"Found {len(layer_paths)} replaceable layers.")
    print(f"Starting combination testing for {2**len(layer_paths)} total combinations...")

    # 1. Load the validation data loader
    print("\nSetting up ImageNet validation data loader (299x299)...")
    val_loader = build_test_loader(
        IMAGENET_VAL_DIR, GROUND_TRUTH_TXT, META_MAT, 
        args.batch_size, args.workers
    )
    
    all_results = []
    
    # 2. Iterate through all combinations
    for r in tqdm(range(len(layer_paths) + 1), desc="Total Combinations"):
        for combo in combinations(layer_paths, r):
            model = None
            combo_name = ""
            
            try:
                # --- A: Handle Model Loading ---
                
                # Case 0: Baseline Model
                if r == 0:
                    combo_name = "Original InceptionV3 (Baseline)"
                    # Load with aux_logits=True (to match arch) and weights
                    # This correctly sets transform_input=True
                    model = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
                
                # Case 1: Single Layer Replacement (strict load)
                # This should now match your training results
                elif r == 1:
                    layer_path = combo[0]
                    combo_name = f"Converted: {layer_path}"
                    
                    # --- *** CRITICAL FIX *** ---
                    # 1. Create the custom architecture. Must match saved file.
                    #    - weights=None (we will load them)
                    #    - aux_logits=True (to match saved arch)
                    #    - transform_input=True (to match training conditions)
                    model = models.inception_v3(weights=None, aux_logits=True, transform_input=True)
                    # --- *** END FIX *** ---

                    model = replace_with_d2sconv_s2d(model, layer_path, r=2)
                    
                    # 2. Load the full state dict from your checkpoint
                    model_file = layers_to_test[layer_path]
                    model_path = os.path.join(args.saved_models_path, model_file)
                    
                    # 3. Load with strict=True. This verifies the .pth file is a
                    #    complete model, which it should be.
                    print(f"Loading single-layer model from: {model_path}")
                    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
                
                # Case 2: Hybrid Model (2+ layers)
                else:
                    combo_name = "Converted: " + ", ".join(combo)
                    
                    # 1. Start with a pre-trained baseline model (This has correct fc, transform_input)
                    model = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
                    
                    # 2. Modify the architecture for all layers in the combo
                    for layer_path in combo:
                        model = replace_with_d2sconv_s2d(model, layer_path, r=2)
                    
                    # 3. Get the state_dict of the new hybrid architecture
                    final_state_dict = model.state_dict()

                    # 4. "Patch" the state_dict with your fine-tuned weights
                    #    This restores your original "Frankenstein" logic:
                    #    Model = (Layer A + Layer B) + (fc_B + AuxLogits_B)
                    for layer_path in combo:
                        model_file = layers_to_test[layer_path]
                        model_path = os.path.join(args.saved_models_path, model_file)
                        trained_state_dict = torch.load(model_path, map_location='cpu')

                        # --- *** CORRECTED FIX: Restore original logic *** ---
                        # Copy all keys for the custom layer AND the final
                        # fc/AuxLogits layers. This will overwrite with the
                        # weights from the *last* model in the 'combo' list.
                        print(f"Patching weights for: {layer_path}, fc, and AuxLogits")
                        copy_variant_state(final_state_dict, trained_state_dict, [layer_path, "fc", "AuxLogits"])

                    model.load_state_dict(final_state_dict, strict=True)
                
                # --- B: Evaluate the Loaded Model ---
                print(f"\nEvaluating: {combo_name}")
                model.to(device)
                model.eval() # Explicitly set to eval mode
                metrics = get_classification_metrics(model, val_loader, device)
                
                print(f"  > Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
                print(f"  > Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
                all_results.append({'name': combo_name, 'metrics': metrics})

            except Exception as e:
                print(f"❌ FAILED combination: {combo_name}")
                print(f"  > Error: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for easier debugging
                raise RuntimeError(f'Evaluation failed for {combo_name}; no result recorded') from e

            # Clean up memory
            if model is not None:
                model.to('cpu')
                del model
            torch.cuda.empty_cache()

    # 3. Print final summary and save to file
    print("\n" + "="*70)
    print("--- FINAL InceptionV3 COMBINATION ANALYSIS SUMMARY ---")
    print("="*70)
    
    all_results.sort(key=lambda x: x['metrics']['top1_accuracy'], reverse=True)
    
    results_filename = "inceptionv3_combination_results.txt"
    try:
        with open(results_filename, "w") as f:
            f.write("="*70 + "\n")
            f.write("--- FINAL InceptionV3 COMBINATION ANALYSIS SUMMARY ---\n")
            f.write("(Sorted by Top-1 Accuracy)\n")
            f.write("="*70 + "\n")
            
            for result in all_results:
                summary_line_1 = f"\nModel: {result['name']}"
                summary_line_2 = f"  > Top-1 Accuracy: {result['metrics']['top1_accuracy']:.2f}%"
                summary_line_3 = f"  > Top-5 Accuracy: {result['metrics']['top5_accuracy']:.2f}%"
                
                print(summary_line_1)
                print(summary_line_2)
                print(summary_line_3)
                
                f.write(summary_line_1 + "\n")
                f.write(summary_line_2 + "\n")
                f.write(summary_line_3 + "\n")

            f.write("\n" + "="*70 + "\n")
            f.write("Combination analysis complete.\n")

        print("\n" + "="*70)
        print(f"✅ Results saved to {results_filename}")
    
    except IOError as e:
        raise OSError("Could not save evaluation results") from e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="InceptionV3 Combination Inference")
    
    parser.add_argument('--local_imagenet_path', type=str, required=True, 
                        help='Path to the local ImageNet dataset directory (parent of ILSVRC2012_img_val, etc.)')
    
    parser.add_argument('--saved_models_path', type=str, default="saved_models", 
                        help='Path to the directory containing the saved .pth model files.')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=8, 
                        help='Number of worker processes for data loading.')
    args = parser.parse_args()
    
    # Verify paths based on the single --local_imagenet_path
    val_dir = os.path.join(args.local_imagenet_path, 'ILSVRC2012_img_val')
    gt_txt = os.path.join(args.local_imagenet_path, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    meta_mat = os.path.join(args.local_imagenet_path, 'ILSVRC2012_devkit_t12/data/meta.mat')
    model_dir = args.saved_models_path

    # --- CLEANUP: More robust path checking ---
    paths_ok = True
    if not os.path.isdir(args.local_imagenet_path):
        print(f"Error: ImageNet directory not found at {args.local_imagenet_path}")
        paths_ok = False
    if not os.path.isdir(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        paths_ok = False
    if not os.path.isfile(gt_txt):
        print(f"Error: Ground truth file not found at {gt_txt}")
        paths_ok = False
    if not os.path.isfile(meta_mat):
        print(f"Error: meta.mat file not found at {meta_mat}")
        paths_ok = False
    if not os.path.isdir(model_dir):
        print(f"Error: Saved models directory not found at {model_dir}")
        paths_ok = False

    if paths_ok:
        print("All required paths found.")
        main(args)
    else:
        print("\nPlease fix the paths and try again.")