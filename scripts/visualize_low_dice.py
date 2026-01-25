#!/usr/bin/env python3
"""
Visualize TransUNet predictions for samples with Dice below a threshold.
Helps identify failure cases and understand where the model struggles.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_kfold_dataloaders


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice score between prediction and target."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.item()


def compute_iou(pred, target, threshold=0.5):
    """Compute IoU score between prediction and target."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def visualize_sample(image, gt_mask, pred_mask, dice, iou, name, save_path):
    """Visualize a single sample with image, GT, prediction, and overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    img_np = image.cpu().numpy()
    if img_np.ndim == 3:
        img_np = img_np[0]  # Take first channel if multi-channel
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Ground truth mask
    gt_np = gt_mask.cpu().numpy()
    axes[1].imshow(gt_np, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction mask
    pred_np = pred_mask.cpu().numpy()
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title(f'TransUNet Pred\nDice: {dice:.4f}, IoU: {iou:.4f}')
    axes[2].axis('off')

    # Overlay: GT in green, Pred in red, Overlap in yellow
    overlay = np.zeros((*img_np.shape, 3))
    # Normalize image to 0-1 for background
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    overlay[:, :, 0] = img_norm * 0.5  # Red channel
    overlay[:, :, 1] = img_norm * 0.5  # Green channel
    overlay[:, :, 2] = img_norm * 0.5  # Blue channel

    # GT in green
    gt_binary = (gt_np > 0.5).astype(float)
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], gt_binary * 0.7)

    # Pred in red
    pred_binary = (pred_np > 0.5).astype(float)
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_binary * 0.7)

    # Overlap becomes yellow (red + green)
    overlap = gt_binary * pred_binary
    overlay[:, :, 0] = np.where(overlap > 0, 1.0, overlay[:, :, 0])
    overlay[:, :, 1] = np.where(overlap > 0, 1.0, overlay[:, :, 1])

    axes[3].imshow(np.clip(overlay, 0, 1))
    axes[3].set_title('Overlay\n(Green=GT, Red=Pred, Yellow=Overlap)')
    axes[3].axis('off')

    plt.suptitle(f'{name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize low-Dice TransUNet predictions')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of processed dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., BUSI)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number for cross-validation')
    parser.add_argument('--transunet_checkpoint', type=str, required=True,
                        help='Path to TransUNet checkpoint')
    parser.add_argument('--dice_threshold', type=float, default=0.8,
                        help='Visualize samples with Dice below this threshold')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Which split to visualize')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Maximum number of samples to visualize')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for TransUNet')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_fold{args.fold}_{args.split}_dice_lt_{args.dice_threshold}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    # Load TransUNet
    from models import VisionTransformer, CONFIGS
    config = CONFIGS['R50-ViT-B_16']
    config.n_classes = 2
    config.n_skip = 3
    transunet = VisionTransformer(config, img_size=args.img_size, num_classes=2)

    ckpt = torch.load(args.transunet_checkpoint, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    transunet.load_state_dict(ckpt, strict=False)
    transunet = transunet.to(device).eval()
    print(f"Loaded TransUNet from {args.transunet_checkpoint}")

    # Get dataloader
    train_loader, val_loader = get_kfold_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset,
        fold_idx=args.fold,
        batch_size=1,
        img_size=args.img_size,
    )

    dataloader = val_loader if args.split == 'val' else train_loader
    print(f"Evaluating on {args.dataset} {args.split} set ({len(dataloader)} samples)")

    # Collect all results
    all_results = []

    print("\nComputing predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image = batch['image'].to(device)  # (B, 1, H, W)
            label = batch['label'].to(device)  # (B, H, W)
            name = batch.get('name', ['unknown'])[0]

            # Get TransUNet prediction
            transunet_output = transunet(image)
            pred_mask = torch.softmax(transunet_output, dim=1)[:, 1]  # (B, H, W)

            # Compute metrics
            dice = compute_dice(pred_mask[0], label[0])
            iou = compute_iou(pred_mask[0], label[0])

            all_results.append({
                'name': name,
                'image': image[0].cpu(),
                'gt_mask': label[0].cpu(),
                'pred_mask': pred_mask[0].cpu(),
                'dice': dice,
                'iou': iou,
            })

    # Sort by Dice (ascending - worst first)
    all_results.sort(key=lambda x: x['dice'])

    # Filter by threshold
    low_dice_results = [r for r in all_results if r['dice'] < args.dice_threshold]

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(all_results)}")
    print(f"Samples with Dice < {args.dice_threshold}: {len(low_dice_results)}")
    print(f"Average Dice (all): {np.mean([r['dice'] for r in all_results]):.4f}")
    if low_dice_results:
        print(f"Average Dice (low): {np.mean([r['dice'] for r in low_dice_results]):.4f}")
    print(f"{'=' * 60}")

    if not low_dice_results:
        print(f"\n✓ No samples with Dice < {args.dice_threshold}")
        return

    # Limit number of visualizations
    samples_to_viz = low_dice_results[:args.max_samples]
    print(f"\nVisualizing {len(samples_to_viz)} samples...")

    # Create visualizations
    for i, result in enumerate(tqdm(samples_to_viz)):
        save_path = os.path.join(output_dir, f'{i:03d}_dice_{result["dice"]:.4f}_{result["name"]}.png')
        visualize_sample(
            image=result['image'],
            gt_mask=result['gt_mask'],
            pred_mask=result['pred_mask'],
            dice=result['dice'],
            iou=result['iou'],
            name=result['name'],
            save_path=save_path,
        )

    # Create summary image with worst cases
    n_summary = min(10, len(samples_to_viz))
    if n_summary > 0:
        fig, axes = plt.subplots(n_summary, 4, figsize=(16, 4 * n_summary))
        if n_summary == 1:
            axes = axes.reshape(1, -1)

        for i, result in enumerate(samples_to_viz[:n_summary]):
            img_np = result['image'].numpy()
            if img_np.ndim == 3:
                img_np = img_np[0]
            gt_np = result['gt_mask'].numpy()
            pred_np = result['pred_mask'].numpy()

            axes[i, 0].imshow(img_np, cmap='gray')
            axes[i, 0].set_title(f'{result["name"]}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gt_np, cmap='gray')
            axes[i, 1].set_title('GT')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].set_title(f'Pred (Dice: {result["dice"]:.3f})')
            axes[i, 2].axis('off')

            # Overlay
            overlay = np.zeros((*img_np.shape, 3))
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
            overlay[:, :, 0] = img_norm * 0.5
            overlay[:, :, 1] = img_norm * 0.5
            overlay[:, :, 2] = img_norm * 0.5
            gt_binary = (gt_np > 0.5).astype(float)
            pred_binary = (pred_np > 0.5).astype(float)
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], gt_binary * 0.7)
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_binary * 0.7)
            overlap = gt_binary * pred_binary
            overlay[:, :, 0] = np.where(overlap > 0, 1.0, overlay[:, :, 0])
            overlay[:, :, 1] = np.where(overlap > 0, 1.0, overlay[:, :, 1])
            axes[i, 3].imshow(np.clip(overlay, 0, 1))
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')

        plt.suptitle(f'Worst {n_summary} Predictions (Dice < {args.dice_threshold})', fontsize=14)
        plt.tight_layout()
        summary_path = os.path.join(output_dir, '_summary_worst_cases.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved summary to: {summary_path}")

    # Save stats to text file
    stats_path = os.path.join(output_dir, '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Dice Threshold: {args.dice_threshold}\n")
        f.write(f"TransUNet Checkpoint: {args.transunet_checkpoint}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Total samples: {len(all_results)}\n")
        f.write(f"Samples with Dice < {args.dice_threshold}: {len(low_dice_results)}\n")
        f.write(f"Average Dice (all): {np.mean([r['dice'] for r in all_results]):.4f}\n")
        if low_dice_results:
            f.write(f"Average Dice (low): {np.mean([r['dice'] for r in low_dice_results]):.4f}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Low Dice Samples (sorted by Dice ascending):\n")
        f.write(f"{'=' * 60}\n")
        for r in low_dice_results:
            f.write(f"  {r['name']}: Dice={r['dice']:.4f}, IoU={r['iou']:.4f}\n")

    print(f"Saved stats to: {stats_path}")
    print(f"\nDone! Check {output_dir} for visualizations.")


if __name__ == '__main__':
    main()
