"""
Evaluation metrics for segmentation.
"""
import numpy as np
import torch


def dice_score(pred, target, smooth=1e-5):
    """
    Calculate Dice score.

    Args:
        pred: Predictions (binary or probability)
        target: Ground truth (binary)
        smooth: Smoothing factor

    Returns:
        Dice score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def iou_score(pred, target, smooth=1e-5):
    """
    Calculate IoU (Intersection over Union) score.

    Args:
        pred: Predictions (binary or probability)
        target: Ground truth (binary)
        smooth: Smoothing factor

    Returns:
        IoU score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou


def precision_score(pred, target, smooth=1e-5):
    """
    Calculate Precision.

    Args:
        pred: Predictions (binary or probability)
        target: Ground truth (binary)
        smooth: Smoothing factor

    Returns:
        Precision score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    tp = (pred * target).sum()
    precision = (tp + smooth) / (pred.sum() + smooth)

    return precision


def recall_score(pred, target, smooth=1e-5):
    """
    Calculate Recall (Sensitivity).

    Args:
        pred: Predictions (binary or probability)
        target: Ground truth (binary)
        smooth: Smoothing factor

    Returns:
        Recall score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    tp = (pred * target).sum()
    recall = (tp + smooth) / (target.sum() + smooth)

    return recall


def specificity_score(pred, target, smooth=1e-5):
    """
    Calculate Specificity.

    Args:
        pred: Predictions (binary or probability)
        target: Ground truth (binary)
        smooth: Smoothing factor

    Returns:
        Specificity score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    tn = ((1 - pred) * (1 - target)).sum()
    specificity = (tn + smooth) / ((1 - target).sum() + smooth)

    return specificity


def hausdorff_distance_95(pred, target):
    """
    Calculate 95th percentile Hausdorff distance.

    Args:
        pred: Predictions (binary)
        target: Ground truth (binary)

    Returns:
        HD95 distance
    """
    try:
        from scipy.ndimage import distance_transform_edt
        from scipy.spatial.distance import directed_hausdorff

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred = (pred > 0.5).astype(np.bool_)
        target = (target > 0.5).astype(np.bool_)

        if pred.sum() == 0 or target.sum() == 0:
            return 0.0

        # Get surface points
        pred_surface = pred ^ distance_transform_edt(pred) <= 1
        target_surface = target ^ distance_transform_edt(target) <= 1

        pred_coords = np.array(np.where(pred_surface)).T
        target_coords = np.array(np.where(target_surface)).T

        if len(pred_coords) == 0 or len(target_coords) == 0:
            return 0.0

        # Calculate distances
        from scipy.spatial import distance
        distances1 = distance.cdist(pred_coords, target_coords, 'euclidean')
        distances2 = distance.cdist(target_coords, pred_coords, 'euclidean')

        dist1 = distances1.min(axis=1)
        dist2 = distances2.min(axis=1)

        all_distances = np.concatenate([dist1, dist2])
        hd95 = np.percentile(all_distances, 95)

        return hd95
    except:
        return 0.0


def calculate_metrics(pred, target):
    """
    Calculate all metrics at once.

    Args:
        pred: Predictions
        target: Ground truth

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'dice': dice_score(pred, target),
        'iou': iou_score(pred, target),
        'precision': precision_score(pred, target),
        'recall': recall_score(pred, target),
        'specificity': specificity_score(pred, target),
    }

    try:
        metrics['hd95'] = hausdorff_distance_95(pred, target)
    except:
        metrics['hd95'] = 0.0

    return metrics


class MetricTracker:
    """
    Track metrics over multiple batches/epochs.
    """
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = ['dice', 'iou', 'precision', 'recall', 'specificity']
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.values = {m: [] for m in self.metrics}
        self.values['loss'] = []

    def update(self, pred, target, loss=None):
        """Update metrics with a batch of predictions."""
        batch_metrics = calculate_metrics(pred, target)
        for m in self.metrics:
            if m in batch_metrics:
                self.values[m].append(batch_metrics[m])
        if loss is not None:
            self.values['loss'].append(loss)

    def get_average(self):
        """Get average of all tracked metrics."""
        return {m: np.mean(v) if v else 0.0 for m, v in self.values.items()}

    def get_std(self):
        """Get standard deviation of all tracked metrics."""
        return {m: np.std(v) if v else 0.0 for m, v in self.values.items()}

    def summary(self):
        """Get summary string of metrics."""
        avg = self.get_average()
        parts = []
        for m, v in avg.items():
            parts.append(f"{m}: {v:.4f}")
        return " | ".join(parts)
