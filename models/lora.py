"""
LoRA (Low-Rank Adaptation) implementation for SAM.

This module provides LoRA layers and utilities for injecting LoRA into
SAM's image encoder, enabling parameter-efficient fine-tuning.

Reference: https://arxiv.org/abs/2106.09685
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.

    Implements the LoRA technique where a pre-trained weight matrix W is augmented
    with a low-rank decomposition: W' = W + BA, where B and A are low-rank matrices.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition (r)
        alpha: Scaling factor (alpha/r is the actual scale)
        dropout: Dropout probability for LoRA layers
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA weights."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # Initialize B with zeros (so LoRA starts as identity)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA output of shape (..., out_features)
        """
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps an existing linear layer and adds LoRA adaptation.
    The original weights are frozen, only LoRA weights are trained.

    Args:
        original_layer: The original nn.Linear layer to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Add LoRA
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + LoRA."""
        return self.original_layer(x) + self.lora(x)

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into the original layer for efficient inference.

        Returns:
            A new nn.Linear with merged weights
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.original_layer.bias is not None
        )

        # Compute merged weight: W + BA * scale
        with torch.no_grad():
            lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight
            merged.weight.copy_(self.original_layer.weight + lora_weight * self.lora.scaling)
            if self.original_layer.bias is not None:
                merged.bias.copy_(self.original_layer.bias)

        return merged


class LoRAMultiheadAttention(nn.Module):
    """
    Multi-head attention with LoRA on Q, K, V projections.

    This wraps an attention module and adds LoRA to the query, key, and/or value
    projections for parameter-efficient fine-tuning.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        apply_lora_to: List[str] = ['q', 'v'],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.apply_lora_to = apply_lora_to

        # LoRA for Q, K, V projections
        self.lora_q = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout) if 'q' in apply_lora_to else None
        self.lora_k = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout) if 'k' in apply_lora_to else None
        self.lora_v = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout) if 'v' in apply_lora_to else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply LoRA to Q, K, V.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Tuple of (q + lora_q, k + lora_k, v + lora_v)
        """
        if self.lora_q is not None:
            q = q + self.lora_q(q)
        if self.lora_k is not None:
            k = k + self.lora_k(k)
        if self.lora_v is not None:
            v = v + self.lora_v(v)
        return q, k, v


def inject_lora_to_sam_encoder(
    image_encoder: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: List[str] = ['qkv'],
) -> Tuple[nn.Module, List[nn.Parameter]]:
    """
    Inject LoRA layers into SAM's image encoder.

    This function modifies the image encoder in-place by adding LoRA adapters
    to the specified target modules (typically the QKV projections in attention).

    Args:
        image_encoder: SAM's image encoder module
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        target_modules: List of module names to apply LoRA to

    Returns:
        Tuple of (modified encoder, list of LoRA parameters)
    """
    lora_params = []
    lora_layers = {}

    # First, freeze all parameters
    for param in image_encoder.parameters():
        param.requires_grad = False

    # Find and wrap target modules with LoRA
    for name, module in image_encoder.named_modules():
        # Check if this module should have LoRA
        should_apply = any(target in name for target in target_modules)

        if should_apply and isinstance(module, nn.Linear):
            # Create LoRA wrapper
            lora_linear = LoRALinear(
                original_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            lora_layers[name] = lora_linear
            lora_params.extend(lora_linear.lora.parameters())

    # Replace modules with LoRA versions
    for name, lora_module in lora_layers.items():
        # Navigate to parent module and replace
        parts = name.split('.')
        parent = image_encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_module)

    return image_encoder, lora_params


def inject_lora_to_vit_sam(
    image_encoder: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    apply_to_qkv: bool = True,
    apply_to_proj: bool = False,
    apply_to_mlp: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Inject LoRA into SAM's ViT image encoder with fine-grained control.

    This is specifically designed for SAM's image encoder architecture.

    Args:
        image_encoder: SAM's image encoder
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        apply_to_qkv: Apply LoRA to QKV projections
        apply_to_proj: Apply LoRA to output projections
        apply_to_mlp: Apply LoRA to MLP layers

    Returns:
        Tuple of (modified encoder, number of LoRA parameters)
    """
    lora_param_count = 0

    # Freeze all parameters first
    for param in image_encoder.parameters():
        param.requires_grad = False

    # Iterate through transformer blocks
    if hasattr(image_encoder, 'blocks'):
        blocks = image_encoder.blocks
    elif hasattr(image_encoder, 'layers'):
        blocks = image_encoder.layers
    else:
        print("Warning: Could not find transformer blocks in image encoder")
        return image_encoder, 0

    for block_idx, block in enumerate(blocks):
        # Find attention module
        attn = None
        if hasattr(block, 'attn'):
            attn = block.attn
        elif hasattr(block, 'self_attn'):
            attn = block.self_attn

        if attn is not None:
            # Apply LoRA to QKV
            if apply_to_qkv and hasattr(attn, 'qkv'):
                original_qkv = attn.qkv
                lora_qkv = LoRALinear(original_qkv, rank=rank, alpha=alpha, dropout=dropout)
                attn.qkv = lora_qkv
                lora_param_count += sum(p.numel() for p in lora_qkv.lora.parameters())

            # Apply LoRA to output projection
            if apply_to_proj and hasattr(attn, 'proj'):
                original_proj = attn.proj
                lora_proj = LoRALinear(original_proj, rank=rank, alpha=alpha, dropout=dropout)
                attn.proj = lora_proj
                lora_param_count += sum(p.numel() for p in lora_proj.lora.parameters())

        # Apply LoRA to MLP
        if apply_to_mlp:
            mlp = None
            if hasattr(block, 'mlp'):
                mlp = block.mlp
            elif hasattr(block, 'ffn'):
                mlp = block.ffn

            if mlp is not None:
                # MLP typically has lin1/lin2 or fc1/fc2
                for lin_name in ['lin1', 'lin2', 'fc1', 'fc2']:
                    if hasattr(mlp, lin_name):
                        original_lin = getattr(mlp, lin_name)
                        if isinstance(original_lin, nn.Linear):
                            lora_lin = LoRALinear(original_lin, rank=rank, alpha=alpha, dropout=dropout)
                            setattr(mlp, lin_name, lora_lin)
                            lora_param_count += sum(p.numel() for p in lora_lin.lora.parameters())

    return image_encoder, lora_param_count


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            lora_params.extend(module.parameters())
        elif isinstance(module, LoRALinear):
            lora_params.extend(module.lora.parameters())
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.

    Args:
        model: Model with LoRA layers

    Returns:
        State dict with LoRA parameters only
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    Load LoRA parameters from state dict.

    Args:
        model: Model with LoRA layers
        state_dict: State dict with LoRA parameters
    """
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base model for efficient inference.

    This modifies the model in-place.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights (LoRA layers removed)
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Get parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            # Replace with merged linear
            merged = module.merge_weights()
            setattr(parent, parts[-1], merged)

    return model


def print_lora_info(model: nn.Module):
    """Print information about LoRA layers in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in get_lora_params(model))

    print(f"\n{'='*60}")
    print(f"{'LoRA Configuration':^60}")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA parameters:      {lora_params:,}")
    print(f"  LoRA ratio:           {lora_params/total_params*100:.2f}%")
    print(f"{'='*60}\n")
