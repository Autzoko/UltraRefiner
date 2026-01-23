"""
Build SAM model with support for MedSAM checkpoint loading.
"""
import torch
from functools import partial

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .sam import Sam
from .transformer import TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        # Handle MedSAM checkpoint format
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # Handle DifferentiableSAMRefiner checkpoint format (has "sam." prefix)
        # This can happen if user accidentally passes the refiner checkpoint
        # instead of the SAM-specific checkpoint
        if any(k.startswith('sam.') for k in state_dict.keys()):
            print("Detected SAMRefiner checkpoint format, stripping 'sam.' prefix...")
            state_dict = {
                k.replace('sam.', '', 1): v
                for k, v in state_dict.items()
                if k.startswith('sam.')
            }

        sam.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint}")

    return sam


def build_sam_for_training(model_type='vit_b', checkpoint=None, freeze_image_encoder=True, freeze_prompt_encoder=False):
    """
    Build SAM model configured for training/finetuning.

    Args:
        model_type: One of 'vit_h', 'vit_l', 'vit_b'
        checkpoint: Path to checkpoint (e.g., MedSAM checkpoint)
        freeze_image_encoder: Whether to freeze the image encoder
        freeze_prompt_encoder: Whether to freeze the prompt encoder

    Returns:
        SAM model ready for training
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    if freeze_image_encoder:
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen")

    if freeze_prompt_encoder:
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False
        print("Prompt encoder frozen")

    # Always train the mask decoder
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    return sam


def build_sam_with_lora(
    model_type='vit_b',
    checkpoint=None,
    lora_rank=4,
    lora_alpha=4.0,
    lora_dropout=0.0,
    apply_lora_to_qkv=True,
    apply_lora_to_proj=False,
    apply_lora_to_mlp=False,
    freeze_prompt_encoder=False,
):
    """
    Build SAM model with LoRA adapters for parameter-efficient finetuning.

    This function builds SAM with LoRA layers injected into the image encoder,
    allowing adaptation of the frozen encoder with minimal trainable parameters.

    Args:
        model_type: One of 'vit_h', 'vit_l', 'vit_b'
        checkpoint: Path to checkpoint (e.g., MedSAM checkpoint)
        lora_rank: Rank of LoRA decomposition (higher = more capacity)
        lora_alpha: LoRA scaling factor (typically same as rank)
        lora_dropout: Dropout probability for LoRA layers
        apply_lora_to_qkv: Apply LoRA to QKV projections in attention
        apply_lora_to_proj: Apply LoRA to output projections in attention
        apply_lora_to_mlp: Apply LoRA to MLP layers
        freeze_prompt_encoder: Whether to freeze the prompt encoder

    Returns:
        Tuple of (SAM model with LoRA, number of LoRA parameters)
    """
    from ..lora import inject_lora_to_vit_sam, print_lora_info

    # Build base SAM
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    # Inject LoRA into image encoder
    sam.image_encoder, lora_param_count = inject_lora_to_vit_sam(
        image_encoder=sam.image_encoder,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        apply_to_qkv=apply_lora_to_qkv,
        apply_to_proj=apply_lora_to_proj,
        apply_to_mlp=apply_lora_to_mlp,
    )

    print(f"Injected LoRA with rank={lora_rank}, alpha={lora_alpha}")
    print(f"LoRA parameters: {lora_param_count:,}")

    # Handle prompt encoder
    if freeze_prompt_encoder:
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False
        print("Prompt encoder frozen")
    else:
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = True
        print("Prompt encoder trainable")

    # Always train the mask decoder
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    # Print LoRA info
    print_lora_info(sam)

    return sam, lora_param_count
