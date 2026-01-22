"""
Evaluation metrics for segmentation.
"""
import numpy as np
import torch
import sys
from typing import Dict, List, Optional


def dice_score(pred, target, smooth=1e-5):
    """
    Calculate Dice score (F1 Score).

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
    Also known as Jaccard Index.

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


# Jaccard is same as IoU
jaccard_score = iou_score


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


def accuracy_score(pred, target):
    """
    Calculate Pixel Accuracy.

    Args:
        pred: Predictions (binary or probability)
        target: Ground truth (binary)

    Returns:
        Accuracy score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    correct = (pred == target).sum()
    total = target.size
    accuracy = correct / total

    return accuracy


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
        'jaccard': iou_score(pred, target),  # Same as IoU
        'precision': precision_score(pred, target),
        'recall': recall_score(pred, target),
        'accuracy': accuracy_score(pred, target),
        'specificity': specificity_score(pred, target),
    }

    return metrics


class MetricTracker:
    """
    Track metrics over multiple batches/epochs with beautiful console output.
    """
    def __init__(self, metrics: Optional[List[str]] = None):
        if metrics is None:
            metrics = ['dice', 'iou', 'jaccard', 'precision', 'recall', 'accuracy']
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

    def get_average(self) -> Dict[str, float]:
        """Get average of all tracked metrics."""
        return {m: np.mean(v) if v else 0.0 for m, v in self.values.items()}

    def get_std(self) -> Dict[str, float]:
        """Get standard deviation of all tracked metrics."""
        return {m: np.std(v) if v else 0.0 for m, v in self.values.items()}

    def summary(self) -> str:
        """Get summary string of metrics."""
        avg = self.get_average()
        parts = []
        for m, v in avg.items():
            parts.append(f"{m}: {v:.4f}")
        return " | ".join(parts)


class TrainingLogger:
    """
    Beautiful console logger for training progress.
    """

    # Color codes
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self, experiment_name: str, total_epochs: int):
        self.experiment_name = experiment_name
        self.total_epochs = total_epochs
        self.best_metric = 0.0
        self.best_epoch = 0

    def print_header(self, dataset: str, fold: int, n_splits: int,
                     train_samples: int, val_samples: int):
        """Print beautiful training header."""
        width = 70
        print("\n" + "=" * width)
        print(f"{self.BOLD}{self.CYAN}{'ULTRAREFINER TRAINING':^{width}}{self.ENDC}")
        print("=" * width)
        print(f"{self.BOLD}Experiment:{self.ENDC} {self.experiment_name}")
        print(f"{self.BOLD}Dataset:{self.ENDC}    {dataset}")
        print(f"{self.BOLD}Fold:{self.ENDC}       {fold + 1}/{n_splits}")
        print(f"{self.BOLD}Samples:{self.ENDC}    Train: {train_samples} | Val: {val_samples}")
        print("=" * width + "\n")

    def print_epoch_header(self, epoch: int):
        """Print epoch header."""
        width = 70
        print(f"\n{self.BOLD}{self.BLUE}{'─' * width}{self.ENDC}")
        print(f"{self.BOLD}{self.BLUE}  EPOCH {epoch + 1}/{self.total_epochs}{self.ENDC}")
        print(f"{self.BOLD}{self.BLUE}{'─' * width}{self.ENDC}")

    def print_train_metrics(self, metrics: Dict[str, float], lr: float):
        """Print training metrics in a beautiful format."""
        print(f"\n  {self.BOLD}{self.GREEN}[TRAIN]{self.ENDC}")
        print(f"  ├── Loss:      {metrics.get('loss', 0):.4f}")
        print(f"  ├── Dice:      {metrics.get('dice', 0):.4f}")
        print(f"  ├── IoU:       {metrics.get('iou', 0):.4f}")
        print(f"  └── LR:        {lr:.2e}")

    def print_val_metrics(self, metrics: Dict[str, float], is_best: bool = False):
        """Print validation metrics in a beautiful format."""
        best_marker = f" {self.YELLOW}★ BEST{self.ENDC}" if is_best else ""
        print(f"\n  {self.BOLD}{self.CYAN}[VALIDATION]{best_marker}{self.ENDC}")
        print(f"  ┌{'─' * 40}")
        print(f"  │  {'Metric':<15} {'Value':>10}")
        print(f"  ├{'─' * 40}")
        print(f"  │  {'Dice':<15} {metrics.get('dice', 0):>10.4f}")
        print(f"  │  {'IoU (Jaccard)':<15} {metrics.get('iou', 0):>10.4f}")
        print(f"  │  {'Precision':<15} {metrics.get('precision', 0):>10.4f}")
        print(f"  │  {'Recall':<15} {metrics.get('recall', 0):>10.4f}")
        print(f"  │  {'Accuracy':<15} {metrics.get('accuracy', 0):>10.4f}")
        print(f"  │  {'Loss':<15} {metrics.get('loss', 0):>10.4f}")
        print(f"  └{'─' * 40}")

        if is_best:
            self.best_metric = metrics.get('dice', 0)
            self.best_epoch = metrics.get('epoch', 0)

    def print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                           lr: float, is_best: bool = False):
        """Print complete epoch summary."""
        self.print_epoch_header(epoch)
        self.print_train_metrics(train_metrics, lr)
        val_metrics['epoch'] = epoch
        self.print_val_metrics(val_metrics, is_best)

    def print_training_complete(self, best_dice: float, best_epoch: int):
        """Print training completion message."""
        width = 70
        print(f"\n{'=' * width}")
        print(f"{self.BOLD}{self.GREEN}{'TRAINING COMPLETE':^{width}}{self.ENDC}")
        print(f"{'=' * width}")
        print(f"  Best Dice Score: {self.BOLD}{best_dice:.4f}{self.ENDC}")
        print(f"  Best Epoch:      {best_epoch + 1}")
        print(f"{'=' * width}\n")

    def print_progress_bar(self, current: int, total: int, prefix: str = '',
                          suffix: str = '', length: int = 40):
        """Print a progress bar."""
        percent = current / total
        filled = int(length * percent)
        bar = '█' * filled + '░' * (length - filled)
        print(f'\r  {prefix} |{bar}| {percent*100:.1f}% {suffix}', end='')
        if current == total:
            print()


class ProgressBar:
    """
    Custom progress bar for training iterations.
    """
    def __init__(self, total: int, desc: str = '', width: int = 30):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.metrics = {}

    def update(self, n: int = 1, **kwargs):
        """Update progress bar."""
        self.current += n
        self.metrics.update(kwargs)
        self._display()

    def _display(self):
        """Display the progress bar."""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)

        # Format metrics
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in self.metrics.items()])

        # Print
        sys.stdout.write(f'\r  {self.desc} |{bar}| {self.current}/{self.total} [{percent*100:.1f}%] {metrics_str}')
        sys.stdout.flush()

        if self.current >= self.total:
            print()

    def close(self):
        """Close the progress bar."""
        if self.current < self.total:
            print()


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a beautiful table format.

    Args:
        metrics: Dictionary of metric names and values
        title: Table title
    """
    width = 45
    print(f"\n  ┌{'─' * width}┐")
    print(f"  │{title:^{width}}│")
    print(f"  ├{'─' * width}┤")
    print(f"  │  {'Metric':<20} {'Value':>20}  │")
    print(f"  ├{'─' * width}┤")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  │  {name:<20} {value:>20.4f}  │")
        else:
            print(f"  │  {name:<20} {str(value):>20}  │")

    print(f"  └{'─' * width}┘")


def format_metrics_inline(metrics: Dict[str, float], keys: List[str] = None) -> str:
    """
    Format metrics as inline string.

    Args:
        metrics: Dictionary of metrics
        keys: Specific keys to include (None for all)

    Returns:
        Formatted string
    """
    if keys is None:
        keys = list(metrics.keys())

    parts = []
    for k in keys:
        if k in metrics:
            parts.append(f"{k}: {metrics[k]:.4f}")

    return " | ".join(parts)
