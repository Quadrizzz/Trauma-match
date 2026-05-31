"""
Evaluation metrics for face matching.
"""
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score


def best_threshold_accuracy(similarities, labels):
    """
    Find the threshold that maximizes accuracy on this set.
    Returns the best threshold and the accuracy at that threshold.
    """
    thresholds = np.linspace(-1, 1, 200)
    best_acc = 0
    best_thresh = 0
    
    for t in thresholds:
        preds = (similarities >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    
    return best_thresh, best_acc


def evaluate_pairs(similarities, labels, n_skipped=0):
    """
    Compute face verification metrics.
    
    Args:
        similarities: similarity scores for successfully processed pairs
        labels: ground truth labels for those pairs
        n_skipped: number of pairs where detection failed
    """
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    fpr, tpr, _ = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)
    
    best_thresh, best_acc = best_threshold_accuracy(similarities, labels)
    
    target_far = 0.001
    idx = np.argmin(np.abs(fpr - target_far))
    tar_at_far = tpr[idx]
    
    # effective metrics counting skipped pairs as failures
    n_total = len(labels) + n_skipped
    n_correct = int(best_acc * len(labels))
    effective_accuracy = n_correct / n_total if n_total > 0 else 0
    
    return {
        "auc": float(auc_score),
        "accuracy": float(best_acc),
        "effective_accuracy": float(effective_accuracy),
        "best_threshold": float(best_thresh),
        "tar_at_far_0.1%": float(tar_at_far),
        "n_pairs_evaluated": len(labels),
        "n_skipped": n_skipped,
        "n_total": n_total,
        "detection_rate": len(labels) / n_total if n_total > 0 else 0,
    }