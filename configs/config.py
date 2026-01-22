"""
Configuration file for UltraRefiner project.
Supports all breast ultrasound datasets: BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM
"""
import ml_collections


def get_transunet_config(vit_name='R50-ViT-B_16'):
    """Returns the TransUNet configuration."""
    config = ml_collections.ConfigDict()

    # Patches configuration
    config.patches = ml_collections.ConfigDict()

    if vit_name.startswith('R50'):
        config.patches.grid = (14, 14)  # Will be updated based on img_size
        config.resnet = ml_collections.ConfigDict()
        config.resnet.num_layers = (3, 4, 9)
        config.resnet.width_factor = 1
        config.skip_channels = [512, 256, 64, 16]
        config.n_skip = 3
    else:
        config.patches.size = (16, 16)
        config.n_skip = 0
        config.skip_channels = [0, 0, 0, 0]

    # ViT configuration based on model name
    if 'B' in vit_name:
        config.hidden_size = 768
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 3072
        config.transformer.num_heads = 12
        config.transformer.num_layers = 12
    elif 'L' in vit_name:
        config.hidden_size = 1024
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 4096
        config.transformer.num_heads = 16
        config.transformer.num_layers = 24

    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2  # Binary segmentation for breast ultrasound
    config.activation = 'softmax'

    return config


def get_sam_config(model_type='vit_b'):
    """Returns SAM model configuration."""
    config = ml_collections.ConfigDict()

    if model_type == 'vit_h':
        config.encoder_embed_dim = 1280
        config.encoder_depth = 32
        config.encoder_num_heads = 16
        config.encoder_global_attn_indexes = [7, 15, 23, 31]
    elif model_type == 'vit_l':
        config.encoder_embed_dim = 1024
        config.encoder_depth = 24
        config.encoder_num_heads = 16
        config.encoder_global_attn_indexes = [5, 11, 17, 23]
    else:  # vit_b
        config.encoder_embed_dim = 768
        config.encoder_depth = 12
        config.encoder_num_heads = 12
        config.encoder_global_attn_indexes = [2, 5, 8, 11]

    config.prompt_embed_dim = 256
    config.image_size = 1024
    config.vit_patch_size = 16
    config.mask_in_chans = 16

    return config


def get_training_config():
    """Returns training configuration."""
    config = ml_collections.ConfigDict()

    # Common settings
    config.seed = 1234
    config.num_workers = 8
    config.pin_memory = True

    # TransUNet training (Phase 1)
    config.transunet = ml_collections.ConfigDict()
    config.transunet.img_size = 224
    config.transunet.batch_size = 24
    config.transunet.max_epochs = 150
    config.transunet.base_lr = 0.01
    config.transunet.weight_decay = 0.0001
    config.transunet.momentum = 0.9

    # SAM finetuning (Phase 2)
    config.sam_finetune = ml_collections.ConfigDict()
    config.sam_finetune.img_size = 1024
    config.sam_finetune.batch_size = 4
    config.sam_finetune.max_epochs = 100
    config.sam_finetune.base_lr = 1e-4
    config.sam_finetune.weight_decay = 0.01
    config.sam_finetune.freeze_image_encoder = True
    config.sam_finetune.freeze_prompt_encoder = False

    # End-to-end training (Phase 3)
    config.e2e = ml_collections.ConfigDict()
    config.e2e.img_size = 224  # TransUNet input size
    config.e2e.sam_img_size = 1024  # SAM input size
    config.e2e.batch_size = 8
    config.e2e.max_epochs = 100
    config.e2e.transunet_lr = 1e-4
    config.e2e.sam_lr = 1e-5
    config.e2e.weight_decay = 0.01
    config.e2e.loss_weights = ml_collections.ConfigDict()
    config.e2e.loss_weights.transunet = 0.3
    config.e2e.loss_weights.sam_refiner = 0.7

    return config


def get_dataset_config():
    """Returns dataset configuration for all breast ultrasound datasets."""
    config = ml_collections.ConfigDict()

    # Dataset names
    config.datasets = ['BUSI', 'BUSBRA', 'BUS', 'BUS_UC', 'BUS_UCLM']

    # Will be set by user - placeholder paths
    config.data_root = './data'

    # Common settings
    config.num_classes = 2  # Background + lesion
    config.img_size = 224

    # Data augmentation
    config.augmentation = ml_collections.ConfigDict()
    config.augmentation.random_rotate = True
    config.augmentation.rotate_range = 20
    config.augmentation.random_flip = True
    config.augmentation.random_scale = True
    config.augmentation.scale_range = (0.8, 1.2)

    return config


def get_config():
    """Returns the complete configuration."""
    config = ml_collections.ConfigDict()

    config.transunet = get_transunet_config()
    config.sam = get_sam_config()
    config.training = get_training_config()
    config.dataset = get_dataset_config()

    return config
