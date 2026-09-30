# Support both `python -m experiments.mobilenetv2_ssd.<script>` and direct execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.mobilenetv2_ssd"

from experiments.checkpoints import require_checkpoint_files, copy_variant_state

import torch
import torch.nn as nn
import argparse
import os
import logging
import sys
import functools
from itertools import combinations
import re
import tempfile
import pathlib
import numpy as np

# --- Imports from your project files ---
# (Ensure these files are in the same directory or in your PYTHONPATH)

try:
    # From mobilenet_retrain.py
    from .mobilenet_retrain import D2SConvS2D, replace_with_d2sconv_s2d
    
    # From eval_ssd.py
    from .eval_ssd import group_annotation_by_class, compute_average_precision_per_class
    
    # From vision library
    from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
    from vision.datasets.voc_dataset import VOCDataset
    from vision.utils import box_utils, measurements
    from vision.utils.misc import Timer
except ImportError as e:
    print(f"FATAL ERROR: Could not import required modules.")
    print(f"Please make sure 'mobilenet_retrain.py', 'eval_ssd.py', and the 'vision' library are accessible.")
    print(f"Import error: {e}")
    sys.exit(1)


# --- mAP Evaluation Function ---
# (This function is unchanged from the previous version)

def evaluate_model_map(net, dataset, device, use_2007_metric=True, iou_threshold=0.5):
    """
    Runs the full mAP evaluation on a given model and dataset.
    """
    net.eval()
    
    # Create a predictor
    try:
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method="hard", device=device)
    except Exception as e:
        raise RuntimeError('Failed to create SSD predictor') from e

    class_names = dataset.class_names
    
    # Group ground truths
    true_case_stat, all_gt_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if not any(true_case_stat.values()):
        raise ValueError('VOC evaluation has no non-difficult ground-truth objects')
    
    results = []
    
    # Run predictions on all images
    logging.info("  > Running predictions on test set...")
    for i in range(len(dataset)):
        image = dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # Convert to 1-based index for eval
        ], dim=1))
    
    if not results:
        raise ValueError('VOC evaluation dataset is empty')
        
    results = torch.cat(results)

    # Use a temporary directory to store detection files
    with tempfile.TemporaryDirectory() as eval_dir:
        eval_path = pathlib.Path(eval_dir)

        # Write prediction files for each class
        for class_index, class_name in enumerate(class_names):
            if class_index == 0: continue  # Skip background
            prediction_path = eval_path / f"det_test_{class_name}.txt"
            
            with open(prediction_path, "w") as f:
                sub = results[results[:, 1] == class_index, :]
                for i in range(sub.size(0)):
                    prob_box = sub[i, 2:].numpy()
                    image_id = dataset.ids[int(sub[i, 0])]
                    print(
                        image_id + " " + " ".join([str(v) for v in prob_box]),
                        file=f
                    )
        
        # Compute AP for each class
        logging.info("  > Calculating AP for each class...")
        
        class_aps = {}
        
        for class_index, class_name in enumerate(class_names):
            if class_index == 0:
                continue
            
            if class_index not in true_case_stat:
                class_aps[class_name] = 0.0 # No ground truth for this class
                continue

            prediction_path = eval_path / f"det_test_{class_name}.txt"
            
            ap = compute_average_precision_per_class(
                true_case_stat[class_index],
                all_gt_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                iou_threshold,
                use_2007_metric
            )
            class_aps[class_name] = ap
    
    aps_values = list(class_aps.values())
    
    if not aps_values:
        raise ValueError('VOC evaluation has no foreground classes')

    mAP = sum(aps_values) / len(aps_values)
    return mAP


# --- Main Combination Script ---

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Define Layer Configuration
    LAYERS_TO_TEST = {
        "classification_headers.0.3": (0, 3),
        "base_net.17.conv.6": (1, 2),
        "base_net.18.0": (2, 2),
        "extras.0.conv.0": (3, 2),
    }
    layer_paths = list(LAYERS_TO_TEST.keys())
    fixed_model_name = "mb2-ssd-lite-d2s.pth"
    require_checkpoint_files([args.baseline_model, *[os.path.join(args.models_base_dir, str(i), fixed_model_name) for i in range(4)]])

    # 2. Load Dataset and Labels
    logging.info(f"Loading VOC test dataset from: {args.dataset}")
    try:
        dataset = VOCDataset(args.dataset, is_test=True)
        class_names = [name.strip() for name in open(args.label_file).readlines()]
        num_classes = len(class_names)
        logging.info(f"Found {num_classes} classes.")
    except Exception as e:
        raise RuntimeError("Failed to load VOC dataset or labels") from e

    # 3. Load Baseline Model State Dict
    logging.info(f"\nLoading baseline model weights from: {args.baseline_model}")
    if not os.path.exists(args.baseline_model):
        raise FileNotFoundError(args.baseline_model)
    # Load baseline state dict to CPU
    base_state_dict = torch.load(args.baseline_model, map_location='cpu')

    # 4. Find paths and pre-load patches for all trained models
    logging.info("\nFinding and pre-loading trained models for each layer...")
    all_trained_model_paths = {}
    all_trained_patches = {}

    for layer_path, (idx, r_val) in LAYERS_TO_TEST.items():
        model_dir = os.path.join(args.models_base_dir, str(idx))
        model_file = os.path.join(model_dir, fixed_model_name)
        
        if os.path.exists(model_file):
            logging.info(f"  > Found model for idx {idx}: {model_file}")
            all_trained_model_paths[idx] = model_file
            
            # Pre-load the full state dict and extract the patch
            trained_state_dict = torch.load(model_file, map_location='cpu')
            patch = {}
            for key, value in trained_state_dict.items():
                if key.startswith(layer_path):
                    patch[key] = value
            all_trained_patches[idx] = patch
            if not patch:
                raise ValueError(f"No weights for {layer_path} in {model_file}")
        else:
            raise FileNotFoundError(model_file)
            
    all_results = []

    # 5. Iterate through all 2^4 = 16 combinations
    for i in range(len(layer_paths) + 1):
        for combo in combinations(layer_paths, i):
            if not combo:
                combo_name = "Original Model (Baseline)"
            else:
                combo_name = "Converted: " + ", ".join([f"idx{LAYERS_TO_TEST[p][0]} (R={LAYERS_TO_TEST[p][1]})" for p in combo])
            
            print("\n" + "="*70)
            logging.info(f"Testing combination: {combo_name}")
            logging.info(f"Layers: {combo or 'None'}")
            
            # --- NEW LOGIC: Handle baseline, single, and multi-layer cases differently ---
            
            # 1. Create fresh model architecture
            net = create_mobilenetv2_ssd_lite(num_classes, width_mult=1.0, is_test=True)

            try:
                if not combo:
                    # --- CASE 1: Baseline Model ---
                    # Replicates: eval_ssd.py --trained_model models/mb2-ssd-lite.pth
                    logging.info("  > Loading baseline weights (strict=True)")
                    net.load_state_dict(base_state_dict, strict=True)
                
                elif len(combo) == 1:
                    # --- CASE 2: Single-Layer Model ---
                    # Replicates: eval_ssd.py --trained_model models/0/mb2-ssd-lite-d2s.pth
                    layer_path = combo[0]
                    idx, r_val = LAYERS_TO_TEST[layer_path]
                    
                    # 1. Modify architecture
                    logging.info(f"  > Replacing {layer_path} with R={r_val}")
                    net = replace_with_d2sconv_s2d(net, layer_path, r=r_val)
                    
                    # 2. Load the *entire* trained model file
                    model_file = all_trained_model_paths[idx]
                    logging.info(f"  > Loading full model weights from: {model_file} (strict=True)")
                    trained_state_dict = torch.load(model_file, map_location='cpu')
                    net.load_state_dict(trained_state_dict, strict=True)

                else:
                    # --- CASE 3: Multi-Layer Combination ---
                    # Reconstructs from baseline + patches
                    
                    # 1. Modify architecture for all layers in combo
                    for layer_path in combo:
                        idx, r_val = LAYERS_TO_TEST[layer_path]
                        logging.info(f"  > Replacing {layer_path} with R={r_val}")
                        net = replace_with_d2sconv_s2d(net, layer_path, r=r_val)
                    
                    # 2. Load baseline weights (non-strict)
                    logging.info("  > Loading baseline weights (strict=False)")
                    net.load_state_dict(base_state_dict, strict=False)
                    
                    # 3. Load all required patches
                    for layer_path in combo:
                        idx, r_val = LAYERS_TO_TEST[layer_path]
                        logging.info(f"  > Patching weights for layer idx {idx} ({layer_path})")
                        patch = all_trained_patches[idx]
                        if patch:
                            state = net.state_dict()
                            copy_variant_state(state, patch, [layer_path])
                            net.load_state_dict(state, strict=True)
                        else:
                            raise ValueError(f"No patch for layer index {idx}")

            except RuntimeError as e:
                raise RuntimeError(f"Checkpoint loading failed for {combo_name}") from e

            # 4. Evaluate mAP
            net.to(device)
            logging.info("Starting mAP evaluation for this combination...")
            try:
                mAP = evaluate_model_map(net, dataset, device, use_2007_metric=True)
                metrics = {"mAP": mAP}
                logging.info(f"--- mAP for {combo_name}: {mAP:.4f} ---")
            except Exception as e:
                logging.error(f"Evaluation failed for {combo_name}: {e}", exc_info=True)
                raise RuntimeError(f"Evaluation failed for {combo_name}; no result recorded") from e

            all_results.append({'name': combo_name, 'metrics': metrics})

    # 6. Print final summary
    print("\n" + "="*70)
    print("--- FINAL COMBINATION mAP SUMMARY ---")
    print("="*70)
    
    results_filename = "mb2ssd_combination_map_results.txt"
    with open(results_filename, "w") as f:
        f.write("--- FINAL SSD COMBINATION mAP SUMMARY ---\n")
        
        for result in all_results:
            summary_line_1 = f"\nModel: {result['name']}"
            summary_line_2 = f"  > Average Precision Across All Classes (mAP): {result['metrics']['mAP']:.4f}"
            
            print(summary_line_1)
            print(summary_line_2)
            
            f.write(summary_line_1 + "\n")
            f.write(summary_line_2 + "\n")

        f.write("\n" + "="*70 + "\n")

    print("\n" + "="*70)
    print("Combination analysis complete.")
    print(f"✅ Results saved to {results_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileNetV2-SSD Combination mAP Evaluation")
    
    # Paths
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to the VOC test dataset (e.g., /data/test/VOC2007/)')
    parser.add_argument('--label_file', type=str, required=True, 
                        help='Path to the label file (e.g., models/voc-model-labels.txt)')
    parser.add_argument('--baseline_model', type=str, required=True, 
                        help='Path to the baseline pretrained model (e.g., models/mb2-ssd-lite.pth)')
    parser.add_argument('--models_base_dir', type=str, default="models", 
                        help='Path to the base directory containing checkpoint folders (e.g., "models", which contains "0", "1", etc.)')
    
    args = parser.parse_args()
    
    # Simple check for dependencies
    if not os.path.exists('mobilenet_retrain.py') or not os.path.exists('eval_ssd.py'):
        logging.warning("Warning: 'mobilenet_retrain.py' or 'eval_ssd.py' not found in current directory.")
        logging.warning("Script will continue, but may fail if they are not in your PYTHONPATH.")
        
    main(args)
