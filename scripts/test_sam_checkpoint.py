#!/usr/bin/env python3
"""
Quick test script to verify SAM checkpoint works on specific dataset.
This helps debug Phase 2 → Phase 3 transfer issues.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sam import sam_model_registry
from models.sam_refiner import DifferentiableSAMRefiner
from data import get_kfold_dataloaders


def compute_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.item()


def main():
    parser = argparse.ArgumentParser(description='Test SAM checkpoint on dataset')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--sam_checkpoint', type=str, required=True)
    parser.add_argument('--transunet_checkpoint', type=str, required=True)
    parser.add_argument('--mask_prompt_style', type=str, default='direct')
    parser.add_argument('--use_roi_crop', action='store_true')
    parser.add_argument('--roi_expand_ratio', type=float, default=0.2)
    parser.add_argument('--sam_model_type', type=str, default='vit_b')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load TransUNet for coarse predictions
    from models import VisionTransformer, CONFIGS
    config = CONFIGS['R50-ViT-B_16']
    config.n_classes = 2
    config.n_skip = 3
    # Set grid size based on image size (for R50-ViT hybrid model)
    if hasattr(config.patches, 'grid'):
        config.patches.grid = (224 // 16, 224 // 16)  # (14, 14) for 224x224
    transunet = VisionTransformer(config=config, img_size=(224, 224), num_classes=2)

    ckpt = torch.load(args.transunet_checkpoint, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    transunet.load_state_dict(ckpt, strict=False)
    transunet = transunet.to(device).eval()
    print(f"Loaded TransUNet from {args.transunet_checkpoint}")

    # Load SAM
    sam_builder = sam_model_registry[args.sam_model_type]
    sam = sam_builder(checkpoint=args.sam_checkpoint)
    print(f"Loaded SAM from {args.sam_checkpoint}")

    # Create SAM refiner
    sam_refiner = DifferentiableSAMRefiner(
        sam_model=sam,
        use_point_prompt=True,
        use_box_prompt=True,
        use_mask_prompt=True,
        mask_prompt_style=args.mask_prompt_style,
        use_roi_crop=args.use_roi_crop,
        roi_expand_ratio=args.roi_expand_ratio,
    ).to(device).eval()

    # SAM normalization
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)

    # Get dataloader
    _, val_loader = get_kfold_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset,
        fold_idx=args.fold,
        batch_size=1,
        img_size=224,
    )

    coarse_dices = []
    refined_dices = []

    print(f"\nEvaluating on {args.dataset} fold {args.fold}...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch['image'].to(device)  # (B, 1, 224, 224)
            label = batch['label'].to(device)  # (B, 224, 224)

            # Get TransUNet coarse prediction
            transunet_output = transunet(image)
            coarse_mask = torch.softmax(transunet_output, dim=1)[:, 1]  # (B, 224, 224)

            # Prepare image for SAM (224 -> 1024)
            if image.shape[1] == 1:
                image_rgb = image.repeat(1, 3, 1, 1)
            else:
                image_rgb = image
            image_rgb = image_rgb * 255.0
            image_1024 = F.interpolate(image_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
            image_sam = (image_1024 - pixel_mean) / pixel_std

            # Resize coarse mask for SAM
            coarse_1024 = F.interpolate(
                coarse_mask.unsqueeze(1), size=(1024, 1024), mode='bilinear', align_corners=False
            ).squeeze(1)

            # SAM refinement
            sam_output = sam_refiner(image_sam, coarse_1024, image_already_normalized=True)
            refined_mask_1024 = torch.sigmoid(sam_output['masks'])

            # Resize back to 224 for comparison
            refined_mask = F.interpolate(
                refined_mask_1024.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze(1)

            # Compute Dice
            coarse_dice = compute_dice(coarse_mask, label)
            refined_dice = compute_dice(refined_mask, label)

            coarse_dices.append(coarse_dice)
            refined_dices.append(refined_dice)

    avg_coarse = np.mean(coarse_dices)
    avg_refined = np.mean(refined_dices)
    delta = avg_refined - avg_coarse

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Coarse (TransUNet):  {avg_coarse:.4f}")
    print(f"Refined (SAM):       {avg_refined:.4f}")
    print(f"Delta:               {delta:+.4f} {'↑' if delta > 0 else '↓'}")
    print("=" * 60)

    if delta < 0.01:
        print("\n⚠️  SAM is NOT improving the coarse masks!")
        print("   Possible issues:")
        print("   1. Checkpoint not loaded correctly")
        print("   2. Phase 2/3 settings mismatch")
        print("   3. Dataset-specific issue (BUSI might be different)")
    else:
        print(f"\n✓ SAM is improving masks by {delta:.2%}")


if __name__ == '__main__':
    main()
