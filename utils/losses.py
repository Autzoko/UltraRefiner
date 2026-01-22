"""
Loss functions for segmentation training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary or multi-class segmentation.
    """
    def __init__(self, n_classes=2, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape mismatch'
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation with sigmoid activation.
    """
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (after sigmoid or raw logits)
            target: Ground truth binary mask
        """
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss for binary segmentation.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, pred, target):
        """
        Args:
            pred: Raw logits (before sigmoid)
            target: Ground truth binary mask
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: Raw logits (before sigmoid)
            target: Ground truth binary mask
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function supporting multiple loss types.
    """
    def __init__(
        self,
        loss_types=['bce', 'dice'],
        weights=None,
        n_classes=2,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super(CombinedLoss, self).__init__()
        self.loss_types = loss_types
        self.weights = weights if weights else [1.0] * len(loss_types)
        self.n_classes = n_classes

        self.losses = nn.ModuleDict()
        for lt in loss_types:
            if lt == 'bce':
                self.losses['bce'] = nn.BCEWithLogitsLoss()
            elif lt == 'dice':
                self.losses['dice'] = BinaryDiceLoss()
            elif lt == 'focal':
                self.losses['focal'] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            elif lt == 'ce':
                self.losses['ce'] = nn.CrossEntropyLoss()
            elif lt == 'multi_dice':
                self.losses['multi_dice'] = DiceLoss(n_classes=n_classes)

    def forward(self, pred, target, softmax=False):
        """
        Args:
            pred: Predictions
            target: Ground truth
            softmax: Whether to apply softmax (for multi-class)
        """
        total_loss = 0.0
        loss_dict = {}

        for lt, w in zip(self.loss_types, self.weights):
            if lt == 'bce':
                loss = self.losses['bce'](pred, target)
            elif lt == 'dice':
                loss = self.losses['dice'](torch.sigmoid(pred), target)
            elif lt == 'focal':
                loss = self.losses['focal'](pred, target)
            elif lt == 'ce':
                loss = self.losses['ce'](pred, target.long())
            elif lt == 'multi_dice':
                loss = self.losses['multi_dice'](pred, target, softmax=softmax)

            loss_dict[lt] = loss.item()
            total_loss += w * loss

        return total_loss, loss_dict


class SAMLoss(nn.Module):
    """
    Loss function for SAM refinement training.
    Combines mask loss and IoU prediction loss.
    """
    def __init__(self, mask_weight=1.0, iou_weight=1.0):
        super(SAMLoss, self).__init__()
        self.mask_weight = mask_weight
        self.iou_weight = iou_weight
        self.bce_dice = BCEDiceLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred_masks, pred_ious, target_masks):
        """
        Args:
            pred_masks: Predicted masks from SAM (B, 3, H, W) for multimask output
            pred_ious: Predicted IoU scores (B, 3)
            target_masks: Ground truth masks (B, H, W)
        """
        # Select best mask based on IoU prediction
        best_idx = pred_ious.argmax(dim=1)
        batch_size = pred_masks.size(0)

        selected_masks = torch.stack([
            pred_masks[i, best_idx[i]] for i in range(batch_size)
        ])

        # Mask loss
        mask_loss = self.bce_dice(selected_masks.unsqueeze(1), target_masks.unsqueeze(1))

        # IoU prediction loss
        with torch.no_grad():
            true_ious = self._compute_iou(selected_masks > 0, target_masks > 0.5)

        selected_pred_ious = torch.stack([
            pred_ious[i, best_idx[i]] for i in range(batch_size)
        ])
        iou_loss = self.mse(selected_pred_ious, true_ious)

        total_loss = self.mask_weight * mask_loss + self.iou_weight * iou_loss

        return total_loss, {
            'mask_loss': mask_loss.item(),
            'iou_loss': iou_loss.item()
        }

    def _compute_iou(self, pred, target):
        pred = pred.float()
        target = target.float()
        intersection = (pred * target).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou
