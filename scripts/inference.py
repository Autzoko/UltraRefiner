"""
Inference script for UltraRefiner.

Supports inference using:
1. TransUNet only
2. TransUNet + SAM Refiner (full pipeline)

Usage:
    # Full pipeline inference
    python scripts/inference.py \
        --model_checkpoint ./checkpoints/ultra_refiner/best.pth \
        --image_path ./test_image.png \
        --output_dir ./results

    # TransUNet only
    python scripts/inference.py \
        --transunet_checkpoint ./checkpoints/transunet/best.pth \
        --mode transunet \
        --image_path ./test_image.png \
        --output_dir ./results
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_ultra_refiner, VisionTransformer, CONFIGS


def get_args():
    parser = argparse.ArgumentParser(description='UltraRefiner Inference')

    # Input arguments
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images')

    # Model arguments
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'transunet'],
                        help='Inference mode')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Path to full UltraRefiner checkpoint')
    parser.add_argument('--transunet_checkpoint', type=str, default=None,
                        help='Path to TransUNet checkpoint (for transunet mode)')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='TransUNet ViT model variant')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        help='SAM model variant')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--save_overlay', action='store_true', default=True,
                        help='Save overlay visualization')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')

    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    return parser.parse_args()


def preprocess_image(image_path, img_size):
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('L')  # Grayscale
    original_size = image.size[::-1]  # (H, W)

    # Resize
    image = image.resize((img_size, img_size), Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0

    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return image_tensor, original_size


def postprocess_mask(mask, original_size, threshold=0.5):
    """Postprocess mask to original size."""
    # Resize to original size
    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=original_size,
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Threshold
    mask_binary = (mask > threshold).cpu().numpy().astype(np.uint8) * 255

    return mask_binary


def create_overlay(image_path, mask, alpha=0.5):
    """Create overlay visualization."""
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        image = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)

    # Resize mask to image size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 1] = mask_resized  # Green channel

    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    # Add contour
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def inference_transunet(model, image_tensor, device):
    """Run TransUNet inference."""
    model.eval()
    with torch.no_grad():
        image = image_tensor.to(device)
        output = model(image)
        pred = torch.softmax(output, dim=1)[:, 1]  # Foreground probability
    return pred.squeeze()


def inference_full(model, image_tensor, device):
    """Run full UltraRefiner inference."""
    model.eval()
    with torch.no_grad():
        image = image_tensor.to(device)
        outputs = model(image)
        coarse_mask = outputs['coarse_mask']
        refined_mask = torch.sigmoid(outputs['refined_mask'])
    return coarse_mask.squeeze(), refined_mask.squeeze()


def main():
    args = get_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get image paths
    image_paths = []
    if args.image_path:
        image_paths.append(args.image_path)
    elif args.image_dir:
        for f in os.listdir(args.image_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(args.image_dir, f))
    else:
        print("Please provide --image_path or --image_dir")
        return

    print(f'Found {len(image_paths)} images')

    # Load model
    if args.mode == 'full':
        if args.model_checkpoint is None:
            print("Please provide --model_checkpoint for full mode")
            return

        # Load checkpoint to get config
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        config = checkpoint.get('config', {})

        model = build_ultra_refiner(
            vit_name=config.get('vit_name', args.vit_name),
            img_size=config.get('img_size', args.img_size),
            num_classes=config.get('num_classes', args.num_classes),
            sam_model_type=config.get('sam_model_type', args.sam_model_type),
        ).to(device)

        model.load_state_dict(checkpoint['model'])
        print(f'Loaded UltraRefiner from {args.model_checkpoint}')

    else:  # transunet mode
        if args.transunet_checkpoint is None:
            print("Please provide --transunet_checkpoint for transunet mode")
            return

        config = CONFIGS[args.vit_name]
        config.n_classes = args.num_classes
        config.n_skip = 3

        if args.vit_name.startswith('R50'):
            config.patches.grid = (args.img_size // 16, args.img_size // 16)

        model = VisionTransformer(
            config=config,
            img_size=args.img_size,
            num_classes=args.num_classes
        ).to(device)

        checkpoint = torch.load(args.transunet_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f'Loaded TransUNet from {args.transunet_checkpoint}')

    # Run inference
    for image_path in tqdm(image_paths, desc='Processing'):
        image_name = os.path.basename(image_path).rsplit('.', 1)[0]

        # Preprocess
        image_tensor, original_size = preprocess_image(image_path, args.img_size)

        # Inference
        if args.mode == 'full':
            coarse_mask, refined_mask = inference_full(model, image_tensor, device)

            # Save coarse mask
            coarse_binary = postprocess_mask(coarse_mask, original_size, args.threshold)
            cv2.imwrite(
                os.path.join(args.output_dir, f'{image_name}_coarse.png'),
                coarse_binary
            )

            # Save refined mask
            refined_binary = postprocess_mask(refined_mask, original_size, args.threshold)
            cv2.imwrite(
                os.path.join(args.output_dir, f'{image_name}_refined.png'),
                refined_binary
            )

            # Save overlay
            if args.save_overlay:
                overlay = create_overlay(image_path, refined_binary)
                cv2.imwrite(
                    os.path.join(args.output_dir, f'{image_name}_overlay.png'),
                    overlay
                )

        else:  # transunet
            pred_mask = inference_transunet(model, image_tensor, device)

            # Save mask
            mask_binary = postprocess_mask(pred_mask, original_size, args.threshold)
            cv2.imwrite(
                os.path.join(args.output_dir, f'{image_name}_pred.png'),
                mask_binary
            )

            # Save overlay
            if args.save_overlay:
                overlay = create_overlay(image_path, mask_binary)
                cv2.imwrite(
                    os.path.join(args.output_dir, f'{image_name}_overlay.png'),
                    overlay
                )

    print(f'Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
