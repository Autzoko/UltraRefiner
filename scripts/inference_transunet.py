"""
Inference TransUNet on validation sets and save predictions.

This script runs inference using each fold's best checkpoint on its validation set,
saves predictions as numpy arrays, and optionally generates visualizations.

Usage:
    python scripts/inference_transunet.py \
        --data_root ./dataset/processed \
        --checkpoint_root ./checkpoints/transunet \
        --dataset BUSI \
        --n_splits 5 \
        --visualize
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transunet import VisionTransformer, CONFIGS
from data import get_kfold_dataloaders, SUPPORTED_DATASETS
from utils import dice_score, iou_score, precision_score, recall_score


def get_args():
    parser = argparse.ArgumentParser(description='Inference TransUNet on validation sets')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing preprocessed datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/transunet',
                        help='Root directory containing checkpoints')

    # K-fold arguments
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Specific folds to run (default: all)')

    # Model arguments
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='ViT model variant')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./predictions/transunet',
                        help='Output directory for predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--num_vis', type=int, default=20,
                        help='Number of samples to visualize per fold')

    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def load_model(checkpoint_path, config, img_size, num_classes, device):
    """Load model from checkpoint."""
    model = VisionTransformer(
        config=config,
        img_size=img_size,
        num_classes=num_classes
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, checkpoint.get('best_dice', 0.0)


def visualize_prediction(image, label, pred, save_path, metrics=None):
    """
    Visualize image, ground truth, and prediction side by side.

    Args:
        image: Input image (H, W, 3) or (3, H, W)
        label: Ground truth mask (H, W)
        pred: Predicted mask (H, W)
        save_path: Path to save the visualization
        metrics: Optional dict of metrics to display
    """
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    # Normalize image for display
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Overlay
    overlay = image.copy()
    # Green for true positive, Red for false positive, Blue for false negative
    pred_binary = pred > 0.5
    label_binary = label > 0.5

    tp = pred_binary & label_binary
    fp = pred_binary & ~label_binary
    fn = ~pred_binary & label_binary

    overlay[tp] = [0, 1, 0]  # Green
    overlay[fp] = [1, 0, 0]  # Red
    overlay[fn] = [0, 0, 1]  # Blue

    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (G:TP, R:FP, B:FN)')
    axes[3].axis('off')

    # Add metrics text
    if metrics:
        metrics_text = f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}"
        fig.suptitle(metrics_text, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(model, dataloader, device):
    """
    Run inference and return predictions with metrics.

    Returns:
        results: List of dicts with image, label, pred, and metrics
    """
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            outputs = model(image)
            pred = torch.softmax(outputs, dim=1)[:, 1]  # Get foreground probability

            # Move to CPU
            for i in range(image.shape[0]):
                img_np = image[i].cpu().numpy()
                label_np = label[i].cpu().numpy()
                pred_np = pred[i].cpu().numpy()

                # Compute metrics
                metrics = {
                    'dice': dice_score(pred_np, label_np),
                    'iou': iou_score(pred_np, label_np),
                    'precision': precision_score(pred_np, label_np),
                    'recall': recall_score(pred_np, label_np),
                }

                results.append({
                    'image': img_np,
                    'label': label_np,
                    'pred': pred_np,
                    'metrics': metrics,
                    'case_name': batch.get('case_name', [f'sample_{len(results)}'])[i]
                                 if 'case_name' in batch else f'sample_{len(results)}'
                })

    return results


def main():
    args = get_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Setup folds
    folds = args.folds if args.folds else list(range(args.n_splits))

    # Build model config
    config = CONFIGS[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    if args.vit_name.startswith('R50'):
        config.patches.grid = (args.img_size // 16, args.img_size // 16)

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Process each fold
    all_fold_metrics = []

    for fold in folds:
        print(f'\n{"="*60}')
        print(f'Processing Fold {fold + 1}/{args.n_splits}')
        print(f'{"="*60}')

        # Load checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_root,
            f'{args.dataset}_fold{fold}',
            'best.pth'
        )

        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}')
            continue

        model, best_dice = load_model(checkpoint_path, config, args.img_size, args.num_classes, device)
        print(f'Loaded checkpoint with best dice: {best_dice:.4f}')

        # Get validation dataloader for this fold
        _, val_loader = get_kfold_dataloaders(
            data_root=args.data_root,
            dataset_name=args.dataset,
            fold_idx=fold,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            for_sam=False,
            seed=args.seed
        )

        print(f'Validation samples: {len(val_loader.dataset)}')

        # Run inference
        results = run_inference(model, val_loader, device)

        # Compute fold-level metrics
        fold_metrics = {
            'dice': np.mean([r['metrics']['dice'] for r in results]),
            'iou': np.mean([r['metrics']['iou'] for r in results]),
            'precision': np.mean([r['metrics']['precision'] for r in results]),
            'recall': np.mean([r['metrics']['recall'] for r in results]),
        }
        all_fold_metrics.append(fold_metrics)

        print(f'\nFold {fold} Metrics:')
        print(f'  Dice:      {fold_metrics["dice"]:.4f}')
        print(f'  IoU:       {fold_metrics["iou"]:.4f}')
        print(f'  Precision: {fold_metrics["precision"]:.4f}')
        print(f'  Recall:    {fold_metrics["recall"]:.4f}')

        # Save predictions
        fold_output_dir = os.path.join(output_dir, f'fold{fold}')
        os.makedirs(fold_output_dir, exist_ok=True)

        # Save as numpy arrays
        predictions_dir = os.path.join(fold_output_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)

        for i, result in enumerate(results):
            np.save(os.path.join(predictions_dir, f'{i:04d}_pred.npy'), result['pred'])
            np.save(os.path.join(predictions_dir, f'{i:04d}_label.npy'), result['label'])

        # Save metrics
        metrics_path = os.path.join(fold_output_dir, 'metrics.npy')
        np.save(metrics_path, {
            'fold_metrics': fold_metrics,
            'sample_metrics': [r['metrics'] for r in results]
        })

        # Generate visualizations
        if args.visualize:
            vis_dir = os.path.join(fold_output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # Sort by dice score to get best and worst cases
            sorted_results = sorted(results, key=lambda x: x['metrics']['dice'])

            # Visualize worst cases
            num_worst = min(args.num_vis // 2, len(sorted_results))
            for i, result in enumerate(sorted_results[:num_worst]):
                save_path = os.path.join(vis_dir, f'worst_{i:02d}_dice{result["metrics"]["dice"]:.3f}.png')
                visualize_prediction(
                    result['image'], result['label'], result['pred'],
                    save_path, result['metrics']
                )

            # Visualize best cases
            num_best = min(args.num_vis // 2, len(sorted_results))
            for i, result in enumerate(sorted_results[-num_best:]):
                save_path = os.path.join(vis_dir, f'best_{i:02d}_dice{result["metrics"]["dice"]:.3f}.png')
                visualize_prediction(
                    result['image'], result['label'], result['pred'],
                    save_path, result['metrics']
                )

            # Visualize random samples
            num_random = min(args.num_vis, len(results))
            random_indices = np.random.choice(len(results), num_random, replace=False)
            for i, idx in enumerate(random_indices):
                result = results[idx]
                save_path = os.path.join(vis_dir, f'random_{i:02d}_dice{result["metrics"]["dice"]:.3f}.png')
                visualize_prediction(
                    result['image'], result['label'], result['pred'],
                    save_path, result['metrics']
                )

            print(f'Saved {num_worst + num_best + num_random} visualizations to {vis_dir}')

    # Print overall summary
    print(f'\n{"="*60}')
    print(f'OVERALL SUMMARY - {args.dataset}')
    print(f'{"="*60}')

    if all_fold_metrics:
        avg_metrics = {
            'dice': np.mean([m['dice'] for m in all_fold_metrics]),
            'iou': np.mean([m['iou'] for m in all_fold_metrics]),
            'precision': np.mean([m['precision'] for m in all_fold_metrics]),
            'recall': np.mean([m['recall'] for m in all_fold_metrics]),
        }
        std_metrics = {
            'dice': np.std([m['dice'] for m in all_fold_metrics]),
            'iou': np.std([m['iou'] for m in all_fold_metrics]),
            'precision': np.std([m['precision'] for m in all_fold_metrics]),
            'recall': np.std([m['recall'] for m in all_fold_metrics]),
        }

        print(f'Average across {len(all_fold_metrics)} folds:')
        print(f'  Dice:      {avg_metrics["dice"]:.4f} +/- {std_metrics["dice"]:.4f}')
        print(f'  IoU:       {avg_metrics["iou"]:.4f} +/- {std_metrics["iou"]:.4f}')
        print(f'  Precision: {avg_metrics["precision"]:.4f} +/- {std_metrics["precision"]:.4f}')
        print(f'  Recall:    {avg_metrics["recall"]:.4f} +/- {std_metrics["recall"]:.4f}')

        # Save overall summary
        summary_path = os.path.join(output_dir, 'summary.npy')
        np.save(summary_path, {
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'fold_metrics': all_fold_metrics
        })
        print(f'\nSaved summary to {summary_path}')


if __name__ == '__main__':
    main()
