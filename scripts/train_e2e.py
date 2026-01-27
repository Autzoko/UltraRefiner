"""
Phase 3: End-to-end training of TransUNet + SAMRefiner.

This script trains the complete UltraRefiner pipeline with gradients flowing
from the SAM refinement back to TransUNet. Both models are jointly optimized.

Uses K-fold cross-validation within the training set for validation.

IMPORTANT: Mask Prompt Style Selection
======================================
The mask_prompt_style parameter is CRITICAL for SAM performance:

**gaussian (DEFAULT, recommended for unfinetuned SAM)**
- Applies Gaussian blur to soft masks before passing to SAM
- Creates softer, more natural boundaries that match SAM's training distribution
- As shown in SAMRefiner paper: unfinetuned SAM + gaussian style + pos/neg points + box = good results
- Use this when: skipping Phase 2, or using original SAM/MedSAM checkpoint

**direct (recommended for Phase 2-finetuned SAM)**
- Passes soft masks directly without blur
- Preserves sharp boundaries from TransUNet output
- Use this when: Phase 2 was trained with --soft_masks and --mask_prompt_style direct
- Phase 2 and Phase 3 must use the same style for consistency

Usage:
    # E2E training WITHOUT Phase 2 (using unfinetuned SAM/MedSAM) - GAUSSIAN STYLE
    python scripts/train_e2e.py \\
        --data_root ./dataset/processed \\
        --transunet_checkpoint ./checkpoints/transunet/best.pth \\
        --sam_checkpoint ./checkpoints/medsam_vit_b.pth \\
        --fold 0 \\
        --mask_prompt_style gaussian  # DEFAULT, matches SAMRefiner paper

    # E2E training WITH Phase 2 finetuned SAM - DIRECT STYLE (must match Phase 2)
    python scripts/train_e2e.py \\
        --data_root ./dataset/processed \\
        --transunet_checkpoint ./checkpoints/transunet/best.pth \\
        --sam_checkpoint ./checkpoints/sam_finetuned/best_sam.pth \\
        --fold 0 \\
        --mask_prompt_style direct  # Must match Phase 2 setting
"""
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_ultra_refiner, build_gated_ultra_refiner, CONFIGS
from data import (
    get_combined_kfold_dataloaders,
    RandomGenerator,
    SUPPORTED_DATASETS
)
from utils import DiceLoss, BCEDiceLoss, MetricTracker, TrainingLogger


def get_args():
    parser = argparse.ArgumentParser(description='End-to-end training of UltraRefiner')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing preprocessed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names to use for training (default: all)')

    # K-fold cross-validation arguments
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for K-fold cross-validation (0 to n_splits-1)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for K-fold cross-validation')

    # Model arguments
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='TransUNet ViT model variant')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        help='SAM model variant')
    parser.add_argument('--transunet_checkpoint', type=str, default=None,
                        help='Path to pre-trained TransUNet checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, default=None,
                        help='Path to finetuned SAM checkpoint')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size for TransUNet')
    parser.add_argument('--refined_eval_size', type=int, default=224,
                        help='Resolution for evaluating SAM refined output. '
                             '224 = downsample SAM to match label (default), '
                             '1024 = upsample label to match SAM (preserves boundary details)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of segmentation classes')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--transunet_lr', type=float, default=1e-5,
                        help='Learning rate for TransUNet (use low value to prevent destabilization)')
    parser.add_argument('--sam_lr', type=float, default=1e-5,
                        help='Learning rate for SAM components')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Loss weights
    parser.add_argument('--coarse_loss_weight', type=float, default=0.5,
                        help='Weight for coarse mask loss (higher = more stable TransUNet)')
    parser.add_argument('--refined_loss_weight', type=float, default=0.5,
                        help='Weight for refined mask loss')

    # SAM freezing options
    parser.add_argument('--freeze_sam_image_encoder', action='store_true', default=True,
                        help='Freeze SAM image encoder')
    parser.add_argument('--freeze_sam_prompt_encoder', action='store_true', default=False,
                        help='Freeze SAM prompt encoder')
    parser.add_argument('--freeze_sam_mask_decoder', action='store_true', default=False,
                        help='Freeze SAM mask decoder')
    parser.add_argument('--freeze_sam_all', action='store_true', default=False,
                        help='Freeze entire SAM (only train TransUNet)')
    parser.add_argument('--unfreeze_sam_epoch', type=int, default=0,
                        help='Epoch to unfreeze SAM (0=never freeze, >0=two-stage training)')

    # Coarse mask processing (to match Phase 2 training distribution)
    parser.add_argument('--sharpen_coarse_mask', action='store_true', default=False,
                        help='Sharpen soft TransUNet outputs to be more binary-like '
                             '(only needed if Phase 2 was trained with binary masks)')
    parser.add_argument('--sharpen_temperature', type=float, default=10.0,
                        help='Temperature for sharpening (higher = more binary-like)')
    parser.add_argument('--mask_prompt_style', type=str, default='gaussian',
                        choices=['gaussian', 'direct', 'distance'],
                        help='Mask prompt style: gaussian (RECOMMENDED for unfinetuned SAM, matches SAMRefiner paper), '
                             'direct (use if Phase 2 was trained with soft masks to match distribution)')

    # ROI cropping (focuses SAM on lesion area)
    parser.add_argument('--use_roi_crop', action='store_true',
                        help='Enable ROI cropping: crop to lesion bounding box, process at full '
                             'SAM resolution, paste back. Must match Phase 2 setting.')
    parser.add_argument('--roi_expand_ratio', type=float, default=0.2,
                        help='Ratio to expand ROI bounding box (0.2 = 20%% on each side)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/ultra_refiner',
                        help='Output directory for checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')

    # TransUNet protection options (prevent performance degradation)
    parser.add_argument('--transunet_grad_scale', type=float, default=1.0,
                        help='Scale factor for gradients flowing to TransUNet from refined loss. '
                             'Values < 1.0 reduce SAM\'s influence on TransUNet. '
                             'Recommended: 0.1-0.5 to prevent TransUNet degradation.')
    parser.add_argument('--transunet_weight_reg', type=float, default=0.0,
                        help='L2 regularization weight to anchor TransUNet to Phase 1 weights. '
                             'Penalizes deviation from initial checkpoint. '
                             'Recommended: 0.001-0.01 to prevent drift.')
    parser.add_argument('--freeze_transunet_epochs', type=int, default=0,
                        help='Number of epochs to freeze TransUNet at the start. '
                             'Allows SAM to adapt first before joint training.')
    parser.add_argument('--transunet_unfreeze_decoder_layers', type=int, default=-1,
                        help='Number of decoder layers to unfreeze from the end (-1 = all layers trainable). '
                             'E.g., 2 = only last 2 decoder blocks + segmentation head are trainable. '
                             'Useful for fine-grained control over TransUNet adaptation.')

    # Gated residual refinement options (alternative to Phase 2)
    parser.add_argument('--use_gated_refinement', action='store_true',
                        help='Enable gated residual refinement: final = coarse + gate * (SAM - coarse). '
                             'Constrains SAM to act as controlled error corrector. '
                             'Recommended when skipping Phase 2 to prevent SAM from degrading good predictions.')
    parser.add_argument('--gate_type', type=str, default='uncertainty',
                        choices=['uncertainty', 'learned', 'hybrid'],
                        help='Gate type: uncertainty (based on coarse confidence, no extra params), '
                             'learned (CNN predicts correction regions), '
                             'hybrid (uncertainty * learned)')
    parser.add_argument('--gate_gamma', type=float, default=1.0,
                        help='Gamma for uncertainty gate curve. '
                             'Higher = more aggressive (only very uncertain regions). '
                             'Lower = softer (more regions get corrections).')
    parser.add_argument('--gate_min', type=float, default=0.0,
                        help='Minimum gate value. 0 = fully preserve confident regions.')
    parser.add_argument('--gate_max', type=float, default=0.8,
                        help='Maximum gate value. Caps correction strength. '
                             'Recommended: 0.3-0.5 for unfinetuned SAM, 0.8 for finetuned SAM.')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def freeze_transunet_except_decoder_layers(model, num_unfreeze_layers):
    """Freeze TransUNet except the last N decoder layers and segmentation head.

    Args:
        model: UltraRefiner model
        num_unfreeze_layers: Number of decoder blocks to unfreeze from the end.
                            -1 = all layers trainable (no freezing)
                             0 = only segmentation head trainable
                             1 = last decoder block + segmentation head
                             2 = last 2 decoder blocks + segmentation head
                             etc.
    Returns:
        List of trainable parameter names
    """
    if num_unfreeze_layers < 0:
        # All layers trainable
        for param in model.transunet.parameters():
            param.requires_grad = True
        return [name for name, _ in model.transunet.named_parameters()]

    # First freeze everything
    for param in model.transunet.parameters():
        param.requires_grad = False

    trainable_names = []

    # Always unfreeze segmentation head
    for name, param in model.transunet.named_parameters():
        if 'segmentation_head' in name:
            param.requires_grad = True
            trainable_names.append(name)

    # Unfreeze last N decoder blocks
    if num_unfreeze_layers > 0:
        # Get total number of decoder blocks
        num_decoder_blocks = len(model.transunet.decoder.blocks)

        # Calculate which blocks to unfreeze (from the end)
        start_unfreeze_idx = max(0, num_decoder_blocks - num_unfreeze_layers)

        for name, param in model.transunet.named_parameters():
            # Match decoder.blocks.{idx}
            if 'decoder.blocks' in name:
                # Extract block index
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'blocks' and i + 1 < len(parts):
                        try:
                            block_idx = int(parts[i + 1])
                            if block_idx >= start_unfreeze_idx:
                                param.requires_grad = True
                                trainable_names.append(name)
                        except ValueError:
                            pass
                        break

    return trainable_names


def get_transunet_trainable_params(model):
    """Get trainable parameters from TransUNet."""
    return [p for p in model.transunet.parameters() if p.requires_grad]


class BCEDiceLossProb(nn.Module):
    """
    Combined BCE and Dice loss for binary segmentation with probability inputs.
    Unlike BCEDiceLoss, this expects probabilities (after sigmoid), not logits.
    Used for gated refinement where output is already in probability space.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5, eps=1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: Probability predictions (after sigmoid, range [0, 1])
            target: Ground truth binary mask
        """
        # Clamp to avoid log(0)
        pred_clamped = torch.clamp(pred, self.eps, 1 - self.eps)

        # BCE loss for probabilities
        bce_loss = F.binary_cross_entropy(pred_clamped, target, reduction='mean')

        # Dice loss
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class EndToEndLoss(nn.Module):
    """
    Combined loss for end-to-end training.
    Supports both standard UltraRefiner (logits) and GatedUltraRefiner (probabilities).
    """
    def __init__(self, coarse_weight=0.3, refined_weight=0.7, n_classes=2, use_gated_refinement=False):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.refined_weight = refined_weight
        self.n_classes = n_classes
        self.use_gated_refinement = use_gated_refinement

        # Loss for coarse mask (TransUNet output)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=n_classes)

        # Loss for refined mask
        if use_gated_refinement:
            # Gated refinement outputs probabilities, not logits
            self.refined_loss_fn = BCEDiceLossProb()
        else:
            # Standard refinement outputs logits
            self.refined_loss_fn = BCEDiceLoss()

    def forward(self, outputs, target):
        """
        Args:
            outputs: Dictionary from UltraRefiner forward pass
            target: Ground truth mask (B, H, W)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        coarse_logits = outputs['coarse_logits']

        # Get refined mask based on mode
        if self.use_gated_refinement:
            # Gated mode: 'refined_mask' is probabilities
            refined_mask = outputs['refined_mask']
        else:
            # Standard mode: 'refined_mask_logits' is logits
            refined_mask = outputs['refined_mask_logits']

        # Resize target for refined mask if needed
        if refined_mask.shape[-2:] != target.shape[-2:]:
            target_refined = F.interpolate(
                target.unsqueeze(1),
                size=refined_mask.shape[-2:],
                mode='nearest'
            ).squeeze(1)
        else:
            target_refined = target

        # Coarse loss
        loss_ce = self.ce_loss(coarse_logits, target.long())
        loss_dice = self.dice_loss(coarse_logits, target, softmax=True)
        coarse_loss = 0.5 * loss_ce + 0.5 * loss_dice

        # Refined loss
        refined_loss = self.refined_loss_fn(refined_mask.unsqueeze(1), target_refined.unsqueeze(1).float())

        # Combined loss
        total_loss = self.coarse_weight * coarse_loss + self.refined_weight * refined_loss

        loss_dict = {
            'coarse_ce': loss_ce.item(),
            'coarse_dice': loss_dice.item(),
            'coarse_total': coarse_loss.item(),
            'refined': refined_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, args, weight_regularizer=None):
    """Train for one epoch."""
    model.train()
    metric_tracker_coarse = MetricTracker()
    metric_tracker_refined = MetricTracker()
    iter_num = epoch * len(dataloader)
    total_weight_reg_loss = 0.0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f'  Train', leave=False)

    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        label = batch['label'].to(device)

        # Forward pass (return_all=True for first batch to debug SAM masks)
        return_all = (batch_idx == 0 and epoch == 0)
        outputs = model(image, return_all=return_all)

        # DEBUG: Print mask statistics for first batch of first epoch
        if batch_idx == 0 and epoch == 0:
            with torch.no_grad():
                refined_out = outputs['refined_mask']
                # In gated mode, refined_mask is already probabilities; otherwise it's logits
                if args.use_gated_refinement:
                    refined_prob = refined_out
                    refined_label = "Refined prob"
                else:
                    refined_prob = torch.sigmoid(refined_out)
                    refined_label = "Refined logits (224)"
                coarse_mask = outputs['coarse_mask']
                print(f"\n  [DEBUG] Mask Statistics (gated={args.use_gated_refinement}):")
                print(f"  ├── Coarse mask:  min={coarse_mask.min():.4f}, max={coarse_mask.max():.4f}, mean={coarse_mask.mean():.4f}")
                print(f"  ├── Coarse shape: {coarse_mask.shape}")
                print(f"  ├── {refined_label}: min={refined_out.min():.4f}, max={refined_out.max():.4f}, mean={refined_out.mean():.4f}")
                print(f"  ├── Refined shape: {refined_out.shape}")
                print(f"  ├── Refined prob:   min={refined_prob.min():.4f}, max={refined_prob.max():.4f}, mean={refined_prob.mean():.4f}")
                print(f"  ├── Refined > 0.5:  {(refined_prob > 0.5).float().mean():.4f} of pixels")
                print(f"  └── Label > 0.5:    {(label > 0.5).float().mean():.4f} of pixels")

                # Check 1024x1024 refined mask (before resize) vs 224x224 (after resize)
                if 'refined_mask_logits' in outputs:
                    rm_1024 = outputs['refined_mask_logits']  # (B, H, W) at 1024x1024
                    rm_1024_prob = torch.sigmoid(rm_1024)
                    print(f"\n  [DEBUG] Refined mask BEFORE resize (1024x1024):")
                    print(f"  ├── Shape: {rm_1024.shape}")
                    print(f"  ├── Logits: min={rm_1024.min():.4f}, max={rm_1024.max():.4f}, mean={rm_1024.mean():.4f}")
                    print(f"  └── Area > 0.5: {(rm_1024_prob > 0.5).float().mean():.4f}")

                # Check SAM's 3 candidate masks and IoU predictions
                if 'sam_masks_all' in outputs:
                    masks_all = outputs['sam_masks_all']  # (B, 3, H, W)
                    iou_preds = outputs['iou_predictions']  # (B, 3)
                    print(f"\n  [DEBUG] SAM's 3 Candidate Masks (shape: {masks_all.shape}):")
                    for i in range(3):
                        mask_i = torch.sigmoid(masks_all[0, i])
                        area = (mask_i > 0.5).float().mean()
                        print(f"  ├── Mask {i}: area={area:.4f}, IoU_pred={iou_preds[0, i]:.4f}, logit_mean={masks_all[0, i].mean():.4f}")

                    # Compute soft selection weights
                    tau = 0.1  # Default selection temperature
                    selection_weights = torch.softmax(iou_preds / tau, dim=1)
                    print(f"  ├── Selection weights: {selection_weights[0].tolist()}")

                    # Manually compute soft-selected result at 1024x1024
                    selected = (masks_all * selection_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
                    selected_prob = torch.sigmoid(selected)
                    print(f"  ├── Soft-selected (manual): mean_logit={selected[0].mean():.4f}, area>0.5={(selected_prob[0] > 0.5).float().mean():.4f}")

                    # Check prompts if available
                    if 'prompts' in outputs:
                        prompts = outputs['prompts']
                        if 'boxes' in prompts:
                            boxes = prompts['boxes'][0]
                            print(f"\n  [DEBUG] Prompts sent to SAM:")
                            print(f"  ├── Box: [{boxes[0]:.1f}, {boxes[1]:.1f}, {boxes[2]:.1f}, {boxes[3]:.1f}]")
                            box_area = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1]) / (1024 * 1024)
                            print(f"  ├── Box area: {box_area:.4f} of image")
                        if 'point_coords' in prompts:
                            pts = prompts['point_coords'][0]
                            lbls = prompts['point_labels'][0]
                            pts_str = ", ".join([f"({p[0].item():.1f}, {p[1].item():.1f}, lbl={l.item():.0f})" for p, l in zip(pts, lbls)])
                            print(f"  ├── Points: [{pts_str}]")
                        if 'mask_inputs' in prompts:
                            mask_in = prompts['mask_inputs'][0, 0]  # (256, 256)
                            print(f"  └── Mask prompt: shape={mask_in.shape}, mean={mask_in.mean():.4f}, >0 area={(mask_in > 0).float().mean():.4f}")

        # Compute loss
        loss, loss_dict = criterion(outputs, label)

        # Add weight regularization loss (anchors TransUNet to Phase 1 weights)
        if weight_regularizer is not None:
            weight_reg_loss = weight_regularizer.compute_loss(model)
            loss = loss + weight_reg_loss
            loss_dict['weight_reg'] = weight_reg_loss.item()
            total_weight_reg_loss += weight_reg_loss.item()

        loss = loss / args.gradient_accumulation

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation == 0:
            # Gradient clipping to prevent TransUNet destabilization
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            coarse_pred = outputs['coarse_mask']
            # In gated mode, refined_mask is already probabilities; otherwise apply sigmoid
            if args.use_gated_refinement:
                refined_pred = outputs['refined_mask']
            else:
                refined_pred = torch.sigmoid(outputs['refined_mask'])

            # Coarse: evaluate at 224x224 (TransUNet native resolution)
            metric_tracker_coarse.update(coarse_pred, label)

            # Refined: evaluate at refined_eval_size (default 224, or 1024 to preserve SAM details)
            if args.refined_eval_size >= 1024 and refined_pred.shape[-2:] != label.shape[-2:]:
                # Upsample label to match SAM output (preserves boundary details)
                label_for_refined = F.interpolate(
                    label.unsqueeze(1),
                    size=refined_pred.shape[-2:],
                    mode='nearest'  # Use nearest for GT to preserve sharp boundaries
                ).squeeze(1)
            elif refined_pred.shape[-2:] != label.shape[-2:]:
                # Downsample SAM output to match label (default behavior)
                refined_pred = F.interpolate(
                    refined_pred.unsqueeze(1),
                    size=label.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                label_for_refined = label
            else:
                label_for_refined = label
            metric_tracker_refined.update(refined_pred, label_for_refined, loss_dict['total'])

        # Logging
        for key, value in loss_dict.items():
            writer.add_scalar(f'train/{key}', value, iter_num)

        iter_num += 1

        pbar.set_postfix({
            'loss': f'{loss_dict["total"]:.4f}',
            'dice_c': f'{metric_tracker_coarse.get_average()["dice"]:.4f}',
            'dice_r': f'{metric_tracker_refined.get_average()["dice"]:.4f}'
        })

    return {
        'coarse': metric_tracker_coarse.get_average(),
        'refined': metric_tracker_refined.get_average()
    }


def validate(model, dataloader, criterion, device, refined_eval_size=224):
    """Validate the model.

    Args:
        refined_eval_size: Resolution for SAM evaluation.
            224 = downsample SAM output to match label (default)
            1024 = upsample label to match SAM output (preserves boundary details)
    """
    model.eval()
    metric_tracker_coarse = MetricTracker(
        metrics=['dice', 'iou', 'jaccard', 'precision', 'recall', 'accuracy']
    )
    metric_tracker_refined = MetricTracker(
        metrics=['dice', 'iou', 'jaccard', 'precision', 'recall', 'accuracy']
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='  Valid', leave=False):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            outputs = model(image)
            _, loss_dict = criterion(outputs, label)

            coarse_pred = outputs['coarse_mask']
            # In gated mode, refined_mask is already probabilities; otherwise apply sigmoid
            if criterion.use_gated_refinement:
                refined_pred = outputs['refined_mask']
            else:
                refined_pred = torch.sigmoid(outputs['refined_mask'])

            # Coarse: evaluate at 224x224 (TransUNet native resolution)
            metric_tracker_coarse.update(coarse_pred, label)

            # Refined: evaluate at refined_eval_size
            if refined_eval_size >= 1024 and refined_pred.shape[-2:] != label.shape[-2:]:
                # Upsample label to match SAM output (preserves boundary details)
                label_for_refined = F.interpolate(
                    label.unsqueeze(1),
                    size=refined_pred.shape[-2:],
                    mode='nearest'  # Use nearest for GT to preserve sharp boundaries
                ).squeeze(1)
            elif refined_pred.shape[-2:] != label.shape[-2:]:
                # Downsample SAM output to match label (default behavior)
                refined_pred = F.interpolate(
                    refined_pred.unsqueeze(1),
                    size=label.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                label_for_refined = label
            else:
                label_for_refined = label
            metric_tracker_refined.update(refined_pred, label_for_refined, loss_dict['total'])

    return {
        'coarse': metric_tracker_coarse.get_average(),
        'refined': metric_tracker_refined.get_average()
    }


def evaluate_transunet_baseline(model, dataloader, device):
    """
    Evaluate TransUNet performance BEFORE E2E training starts.
    This provides a baseline to compare against during training.
    """
    model.eval()
    metric_tracker = MetricTracker(
        metrics=['dice', 'iou', 'precision', 'recall', 'accuracy']
    )

    print("\n" + "=" * 70)
    print("  TRANSUNET BASELINE EVALUATION (Before E2E Training)")
    print("=" * 70)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='  Evaluating TransUNet', leave=False):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            # Forward through TransUNet only
            transunet_output = model.forward_transunet_only(image)

            # Get soft mask (probability) for the foreground class
            if model.num_classes == 2:
                coarse_pred = torch.softmax(transunet_output, dim=1)[:, 1]
            else:
                coarse_pred = torch.sigmoid(transunet_output[:, 1:].sum(dim=1))

            metric_tracker.update(coarse_pred, label)

    metrics = metric_tracker.get_average()

    print(f"\n  TransUNet Baseline Performance:")
    print(f"  ├── Dice:      {metrics['dice']:.4f}")
    print(f"  ├── IoU:       {metrics['iou']:.4f}")
    print(f"  ├── Precision: {metrics['precision']:.4f}")
    print(f"  ├── Recall:    {metrics['recall']:.4f}")
    print(f"  └── Accuracy:  {metrics['accuracy']:.4f}")
    print("=" * 70 + "\n")

    return metrics


class TransUNetWeightRegularizer:
    """
    Regularization loss that penalizes TransUNet weights deviating from Phase 1 checkpoint.
    This prevents TransUNet from drifting too far during E2E training.
    """

    def __init__(self, model, weight=0.01):
        """
        Args:
            model: UltraRefiner model
            weight: Regularization weight (0.001-0.01 recommended)
        """
        self.weight = weight
        # Store initial TransUNet weights (from Phase 1 checkpoint)
        self.initial_weights = {}
        for name, param in model.transunet.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()

        print(f"  Weight regularizer initialized with {len(self.initial_weights)} parameters")

    def compute_loss(self, model):
        """Compute L2 regularization loss for TransUNet weight deviation."""
        if self.weight == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        reg_loss = 0.0
        for name, param in model.transunet.named_parameters():
            if name in self.initial_weights:
                reg_loss += torch.sum((param - self.initial_weights[name]) ** 2)

        return self.weight * reg_loss


class GradientScaler:
    """
    Scales gradients flowing to TransUNet from the refined loss.
    This reduces SAM's influence on TransUNet, preventing destabilization.
    """

    def __init__(self, model, scale=0.1):
        """
        Args:
            model: UltraRefiner model
            scale: Scale factor (0.1 = TransUNet receives 10% of gradients from refined loss)
        """
        self.scale = scale
        self.handles = []

        if scale < 1.0:
            # Register backward hooks on TransUNet parameters
            for param in model.transunet.parameters():
                if param.requires_grad:
                    handle = param.register_hook(lambda grad: grad * scale)
                    self.handles.append(handle)

            print(f"  Gradient scaler initialized: TransUNet receives {scale*100:.0f}% of gradients")

    def remove_hooks(self):
        """Remove all gradient scaling hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []


def main():
    args = get_args()

    # Set default datasets
    if args.datasets is None:
        args.datasets = SUPPORTED_DATASETS

    # Auto-generate experiment name
    if args.exp_name is None:
        # Structure: checkpoints/ultra_refiner/fold_{i}/
        args.exp_name = f'fold_{args.fold}'

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    snapshot_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(snapshot_path, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'Arguments: {args}')

    # Create dataloaders with K-fold cross-validation
    train_loader, val_loader = get_combined_kfold_dataloaders(
        data_root=args.data_root,
        dataset_names=args.datasets,
        fold_idx=args.fold,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        for_sam=False,
        seed=args.seed
    )

    logging.info(f'Training on datasets: {args.datasets}')
    logging.info(f'Fold {args.fold}/{args.n_splits}')
    logging.info(f'Train samples: {len(train_loader.dataset)}')
    logging.info(f'Val samples: {len(val_loader.dataset)}')

    # Initialize beautiful logger
    logger = TrainingLogger(
        experiment_name=args.exp_name,
        total_epochs=args.max_epochs
    )
    logger.print_header(
        dataset='Combined (End-to-End)',
        fold=args.fold,
        n_splits=args.n_splits,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset)
    )

    # Build UltraRefiner model
    logging.info(f'mask_prompt_style: {args.mask_prompt_style}')
    logging.info(f'use_roi_crop: {args.use_roi_crop}')
    logging.info(f'use_gated_refinement: {args.use_gated_refinement}')
    logging.info(f'refined_eval_size: {args.refined_eval_size} (SAM metrics evaluated at this resolution)')

    if args.use_gated_refinement:
        # Use gated residual refinement: final = coarse + gate * (SAM - coarse)
        logging.info(f'Gated refinement enabled: gate_type={args.gate_type}, '
                     f'gamma={args.gate_gamma}, min={args.gate_min}, max={args.gate_max}')
        model = build_gated_ultra_refiner(
            vit_name=args.vit_name,
            img_size=args.img_size,
            num_classes=args.num_classes,
            sam_model_type=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint,
            transunet_checkpoint=args.transunet_checkpoint,
            n_skip=args.n_skip,
            freeze_sam_image_encoder=args.freeze_sam_image_encoder,
            freeze_sam_prompt_encoder=args.freeze_sam_prompt_encoder,
            sharpen_coarse_mask=args.sharpen_coarse_mask,
            sharpen_temperature=args.sharpen_temperature,
            mask_prompt_style=args.mask_prompt_style,
            use_roi_crop=args.use_roi_crop,
            roi_expand_ratio=args.roi_expand_ratio,
            # Gated refinement parameters
            gate_type=args.gate_type,
            gate_gamma=args.gate_gamma,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
        ).to(device)
        logging.info('Built GatedUltraRefiner model')
    else:
        # Standard refinement: final = SAM(coarse)
        model = build_ultra_refiner(
            vit_name=args.vit_name,
            img_size=args.img_size,
            num_classes=args.num_classes,
            sam_model_type=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint,
            transunet_checkpoint=args.transunet_checkpoint,
            n_skip=args.n_skip,
            freeze_sam_image_encoder=args.freeze_sam_image_encoder,
            freeze_sam_prompt_encoder=args.freeze_sam_prompt_encoder,
            sharpen_coarse_mask=args.sharpen_coarse_mask,
            sharpen_temperature=args.sharpen_temperature,
            mask_prompt_style=args.mask_prompt_style,
            use_roi_crop=args.use_roi_crop,
            roi_expand_ratio=args.roi_expand_ratio,
        ).to(device)
        logging.info('Built UltraRefiner model')
    if args.use_roi_crop:
        logging.info(f'ROI cropping enabled: expand_ratio={args.roi_expand_ratio}')

    # Additional SAM freezing options
    if args.freeze_sam_all:
        # Freeze entire SAM (only train TransUNet)
        for param in model.sam.parameters():
            param.requires_grad = False
        logging.info('Frozen entire SAM - only training TransUNet')
    elif args.freeze_sam_mask_decoder:
        # Freeze mask decoder specifically
        for param in model.sam.mask_decoder.parameters():
            param.requires_grad = False
        logging.info('Frozen SAM mask decoder')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters: {total_params:,}')
    logging.info(f'Trainable parameters: {trainable_params:,}')

    # =========================================================================
    # BASELINE EVALUATION: Evaluate TransUNet before E2E training
    # =========================================================================
    baseline_metrics = evaluate_transunet_baseline(model, val_loader, device)
    logging.info(f'TransUNet Baseline - Dice: {baseline_metrics["dice"]:.4f}, IoU: {baseline_metrics["iou"]:.4f}')

    # =========================================================================
    # TRANSUNET PROTECTION: Initialize gradient scaler and weight regularizer
    # =========================================================================
    gradient_scaler = None
    weight_regularizer = None

    if args.transunet_grad_scale < 1.0:
        gradient_scaler = GradientScaler(model, scale=args.transunet_grad_scale)
        logging.info(f'Gradient scaling enabled: {args.transunet_grad_scale}')

    if args.transunet_weight_reg > 0:
        weight_regularizer = TransUNetWeightRegularizer(model, weight=args.transunet_weight_reg)
        logging.info(f'Weight regularization enabled: {args.transunet_weight_reg}')

    # Freeze TransUNet for initial epochs if specified
    transunet_frozen = False
    transunet_partial_freeze = False  # True if only some decoder layers are trainable

    if args.freeze_transunet_epochs > 0:
        # Fully freeze TransUNet for initial epochs
        for param in model.transunet.parameters():
            param.requires_grad = False
        transunet_frozen = True
        logging.info(f'TransUNet frozen for first {args.freeze_transunet_epochs} epochs')
    elif args.transunet_unfreeze_decoder_layers >= 0:
        # Partial freeze: only unfreeze last N decoder layers + segmentation head
        trainable_names = freeze_transunet_except_decoder_layers(
            model, args.transunet_unfreeze_decoder_layers
        )
        transunet_partial_freeze = True
        logging.info(f'TransUNet partial freeze: only {len(trainable_names)} params trainable '
                     f'(last {args.transunet_unfreeze_decoder_layers} decoder layers + segmentation head)')
        for name in trainable_names[:5]:  # Show first 5 trainable layer names
            logging.info(f'  Trainable: {name}')
        if len(trainable_names) > 5:
            logging.info(f'  ... and {len(trainable_names) - 5} more')

    # Loss function
    criterion = EndToEndLoss(
        coarse_weight=args.coarse_loss_weight,
        refined_weight=args.refined_loss_weight,
        n_classes=args.num_classes,
        use_gated_refinement=args.use_gated_refinement
    )

    # Optimizer with different learning rates
    # Only include params that are not frozen initially
    # Frozen params will be added when unfreezing
    if transunet_partial_freeze:
        # Use only trainable params when partially frozen
        transunet_params = get_transunet_trainable_params(model)
    else:
        transunet_params = list(model.get_transunet_params())
    sam_params = list(model.get_sam_params())

    # Check if SAM is initially frozen
    sam_frozen = args.freeze_sam_all or args.freeze_sam_mask_decoder

    # Build optimizer param groups based on what's trainable
    param_groups = []
    if not transunet_frozen and transunet_params:
        param_groups.append({'params': transunet_params, 'lr': args.transunet_lr})
        logging.info(f'Optimizer: Added {len(transunet_params)} TransUNet params')
    if not sam_frozen and sam_params:
        param_groups.append({'params': sam_params, 'lr': args.sam_lr})
        logging.info(f'Optimizer: Added {len(sam_params)} SAM params')

    if not param_groups:
        raise ValueError('No trainable parameters! Both TransUNet and SAM are frozen.')

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        # Restore baseline metrics if available
        if 'baseline_metrics' in checkpoint:
            baseline_metrics = checkpoint['baseline_metrics']
            logging.info(f'Restored baseline metrics from checkpoint')
        logging.info(f'Resumed from epoch {start_epoch}')

    # Tensorboard writer
    writer = SummaryWriter(os.path.join(snapshot_path, 'logs'))

    # Training loop
    best_epoch = 0
    sam_unfrozen = not (args.freeze_sam_all or args.freeze_sam_mask_decoder)

    for epoch in range(start_epoch, args.max_epochs):
        # Unfreeze TransUNet after specified epochs
        if transunet_frozen and epoch >= args.freeze_transunet_epochs:
            logging.info(f'Epoch {epoch}: Unfreezing TransUNet for joint training')

            # Check if we should apply partial freeze after unfreezing
            if args.transunet_unfreeze_decoder_layers >= 0:
                # Partial unfreeze: only last N decoder layers + segmentation head
                trainable_names = freeze_transunet_except_decoder_layers(
                    model, args.transunet_unfreeze_decoder_layers
                )
                logging.info(f'Partial unfreeze: {len(trainable_names)} params trainable '
                             f'(last {args.transunet_unfreeze_decoder_layers} decoder layers + segmentation head)')
                transunet_params = get_transunet_trainable_params(model)
            else:
                # Full unfreeze
                for param in model.transunet.parameters():
                    param.requires_grad = True
                transunet_params = list(model.get_transunet_params())

            transunet_frozen = False

            # Re-initialize gradient scaler with hooks
            if args.transunet_grad_scale < 1.0:
                gradient_scaler = GradientScaler(model, scale=args.transunet_grad_scale)

            # Update optimizer to include TransUNet params
            optimizer.add_param_group({'params': transunet_params, 'lr': args.transunet_lr})
            logging.info(f'Added {len(transunet_params)} TransUNet parameters to optimizer')

        # Two-stage training: unfreeze SAM at specified epoch
        if args.unfreeze_sam_epoch > 0 and epoch == args.unfreeze_sam_epoch and not sam_unfrozen:
            logging.info(f'Epoch {epoch}: Unfreezing SAM for joint training')
            for param in model.sam.mask_decoder.parameters():
                param.requires_grad = True
            if not args.freeze_sam_prompt_encoder:
                for param in model.sam.prompt_encoder.parameters():
                    param.requires_grad = True
            sam_unfrozen = True

            # Update optimizer to include SAM params
            sam_params = list(model.get_sam_params())
            optimizer.add_param_group({'params': sam_params, 'lr': args.sam_lr})
            logging.info(f'Added {len(sam_params)} SAM parameters to optimizer')

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, writer, args, weight_regularizer
        )
        logging.info(f'Epoch {epoch} Train Coarse: {train_metrics["coarse"]}')
        logging.info(f'Epoch {epoch} Train Refined: {train_metrics["refined"]}')

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, args.refined_eval_size)
        logging.info(f'Epoch {epoch} Val Coarse: {val_metrics["coarse"]}')
        logging.info(f'Epoch {epoch} Val Refined: {val_metrics["refined"]}')

        # Extract metrics for comparison (define early for use in tensorboard and printing)
        coarse_dice = val_metrics['coarse']['dice']
        refined_dice = val_metrics['refined']['dice']
        baseline_dice = baseline_metrics['dice']
        coarse_vs_baseline = coarse_dice - baseline_dice
        refined_vs_coarse = refined_dice - coarse_dice

        # Update scheduler
        scheduler.step()

        # Log to tensorboard
        for key, value in val_metrics['coarse'].items():
            writer.add_scalar(f'val/coarse_{key}', value, epoch)
        for key, value in val_metrics['refined'].items():
            writer.add_scalar(f'val/refined_{key}', value, epoch)

        # Log TransUNet performance vs baseline
        writer.add_scalar('val/transunet_vs_baseline', coarse_vs_baseline, epoch)
        writer.add_scalar('val/baseline_dice', baseline_dice, epoch)

        # Log learning rates (handle variable number of param groups)
        lrs = scheduler.get_last_lr()
        current_lr = lrs[0]  # Use first param group's LR for display
        if len(lrs) == 1:
            # Only one param group (either TransUNet or SAM is frozen)
            if transunet_frozen:
                writer.add_scalar('train/lr_sam', lrs[0], epoch)
            else:
                writer.add_scalar('train/lr_transunet', lrs[0], epoch)
        else:
            # Both param groups present
            writer.add_scalar('train/lr_transunet', lrs[0], epoch)
            writer.add_scalar('train/lr_sam', lrs[1], epoch)

        # Save checkpoint (based on refined dice)
        is_best = refined_dice > best_dice
        if is_best:
            best_dice = refined_dice
            best_epoch = epoch

        # Print beautiful epoch summary (using refined metrics as the primary)
        logger.print_epoch_summary(
            epoch=epoch,
            train_metrics=train_metrics['refined'],
            val_metrics=val_metrics['refined'],
            lr=current_lr,
            is_best=is_best
        )

        print(f"\n  Performance Comparison:")
        print(f"  ├── Baseline (Phase 1):  {baseline_dice:.4f}")
        print(f"  ├── Coarse (TransUNet):  {coarse_dice:.4f} ({coarse_vs_baseline:+.4f} vs baseline)")
        print(f"  └── Refined (SAM):       {refined_dice:.4f} ({refined_vs_coarse:+.4f} vs coarse)")

        # Warning if TransUNet performance drops significantly
        if coarse_vs_baseline < -0.02:
            print(f"\n  ⚠️  WARNING: TransUNet performance dropped by {-coarse_vs_baseline:.4f}!")
            print(f"      Consider: --transunet_grad_scale 0.1 or --transunet_weight_reg 0.01")

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_dice': best_dice,
            'baseline_metrics': baseline_metrics,
            'config': {
                'vit_name': args.vit_name,
                'sam_model_type': args.sam_model_type,
                'img_size': args.img_size,
                'num_classes': args.num_classes,
                'fold': args.fold,
                'n_splits': args.n_splits,
                'datasets': args.datasets,
            }
        }

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))
            logging.info(f'New best model saved with Dice: {best_dice:.4f}')

        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, os.path.join(snapshot_path, f'epoch_{epoch}.pth'))

    writer.close()
    logger.print_training_complete(best_dice, best_epoch)

    # Final summary with baseline comparison
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  TransUNet Baseline (Phase 1): {baseline_metrics['dice']:.4f}")
    print(f"  Best Refined Dice (E2E):      {best_dice:.4f}")
    print(f"  Improvement over baseline:    {best_dice - baseline_metrics['dice']:+.4f}")
    print("=" * 70)

    logging.info(f'Training finished. Best Dice: {best_dice:.4f}')
    logging.info(f'Baseline Dice: {baseline_metrics["dice"]:.4f}, Improvement: {best_dice - baseline_metrics["dice"]:+.4f}')


if __name__ == '__main__':
    main()
