import numpy as np
from typing import Dict, Any, Optional


def compute_recall_at_k(
    pred_indices: np.ndarray,
    gt_indices: np.ndarray,
    k: Optional[int] = None,
) -> float:
    """Compute Recall@K: proportion of true top-k neighbors retrieved in predictions.

    Args:
        pred_indices: Predicted indices of shape (Q, K_pred).
        gt_indices: Ground-truth nearest neighbor indices of shape (Q, K_gt).
        k: Number of neighbors to evaluate (defaults to min of widths).

    Returns:
        recall: Float in [0.0, 1.0].
    """
    Q = pred_indices.shape[0]
    if k is None:
        k = min(pred_indices.shape[1], gt_indices.shape[1])

    hits = 0
    for i in range(Q):
        pred_set = set(pred_indices[i, :k])
        gt_set = set(gt_indices[i, :k])
        hits += len(pred_set.intersection(gt_set))

    return float(hits / (Q * k))


def compute_recall_at_1(
    pred_indices: np.ndarray,
    gt_indices: np.ndarray,
) -> float:
    """Compute Recall@1: accuracy of retrieving the exact 1st nearest neighbor."""
    return float(np.mean(pred_indices[:, 0] == gt_indices[:, 0]))


def compute_distance_errors(
    pred_dists: np.ndarray,
    gt_dists: np.ndarray,
) -> Dict[str, float]:
    """Compute distance estimation error metrics (MSE, MARE, Signed Relative Error, Max Rel Error).

    Args:
        pred_dists: Estimated distances of shape (Q, K) or (N,).
        gt_dists: Ground-truth distances of matching shape.

    Returns:
        dict containing 'mse', 'mean_rel_err', 'signed_rel_err', 'max_rel_err'.
    """
    p = pred_dists.flatten()
    g = gt_dists.flatten()

    # Prevent division by zero
    safe_gt = np.maximum(g, 1e-7)

    # Errors
    diff = p - g
    mse = float(np.mean(diff**2))

    signed_rel = diff / safe_gt
    signed_mean_rel_err = float(np.mean(signed_rel))
    mean_abs_rel_err = float(np.mean(np.abs(signed_rel)))
    max_rel_err = float(np.max(np.abs(signed_rel)))

    return {
        "mse": mse,
        "mean_abs_rel_err": mean_abs_rel_err,
        "signed_rel_err": signed_mean_rel_err,
        "max_rel_err": max_rel_err,
    }


def compute_compression_ratio(
    uncompressed_bytes: float,
    compressed_bytes: float,
) -> float:
    """Compute compression ratio: uncompressed / compressed."""
    if compressed_bytes <= 0:
        return 1.0
    return float(uncompressed_bytes / compressed_bytes)


def compute_qps(num_queries: int, total_time_sec: float) -> float:
    """Compute queries per second (QPS)."""
    if total_time_sec <= 0:
        return 0.0
    return float(num_queries / total_time_sec)
