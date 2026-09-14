import time
import numpy as np
from typing import Tuple, Optional


class FP32FlatIndex:
    """Exact Flat FP32 Index Baseline.

    Stores all vectors in 32-bit floating point precision and performs
    exhaustive linear scans for exact nearest neighbor search.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: Optional[np.ndarray] = None
        self.num_vectors: int = 0
        self.norms_sq: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FP32FlatIndex":
        """Index dataset vectors."""
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected array of shape (N, {self.dim}), got {X.shape}")
        self.vectors = np.ascontiguousarray(X, dtype=np.float32)
        self.num_vectors = X.shape[0]
        self.norms_sq = np.sum(self.vectors**2, axis=1)
        return self

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Exhaustive search for k nearest neighbors.

        Args:
            queries: Query vectors of shape (Q, D).
            k: Number of nearest neighbors.
            batch_size: Batch size for linear scan.

        Returns:
            indices: Neighbor indices of shape (Q, k).
            distances: Euclidean distances of shape (Q, k).
            latency_ms: Average search latency per query in milliseconds.
        """
        if self.vectors is None or self.norms_sq is None:
            raise RuntimeError("Index has not been fitted.")

        Q = queries.shape[0]
        indices = np.zeros((Q, k), dtype=np.int64)
        distances = np.zeros((Q, k), dtype=np.float32)

        start_time = time.perf_counter()

        for start_idx in range(0, Q, batch_size):
            end_idx = min(start_idx + batch_size, Q)
            q_batch = queries[start_idx:end_idx]
            q_norms_sq = np.sum(q_batch**2, axis=1, keepdims=True)

            # dist^2 = ||x||^2 + ||q||^2 - 2 * <x, q>
            dists_sq = q_norms_sq + self.norms_sq - 2.0 * np.matmul(q_batch, self.vectors.T)
            dists_sq = np.maximum(dists_sq, 0.0)

            if k < dists_sq.shape[1]:
                partition_idx = np.argpartition(dists_sq, k, axis=1)[:, :k]
                batch_dists = np.take_along_axis(dists_sq, partition_idx, axis=1)
                sorted_order = np.argsort(batch_dists, axis=1)
                batch_indices = np.take_along_axis(partition_idx, sorted_order, axis=1)
                batch_sorted_dists = np.sqrt(np.take_along_axis(batch_dists, sorted_order, axis=1))
            else:
                batch_indices = np.argsort(dists_sq, axis=1)
                batch_sorted_dists = np.sqrt(np.take_along_axis(dists_sq, batch_indices, axis=1))

            indices[start_idx:end_idx] = batch_indices
            distances[start_idx:end_idx] = batch_sorted_dists

        total_time = time.perf_counter() - start_time
        latency_ms = (total_time / Q) * 1000.0

        return indices, distances, latency_ms

    @property
    def memory_bytes_per_vector(self) -> float:
        """Memory footprint per vector in bytes (FP32 = 4 bytes/dim)."""
        return float(self.dim * 4)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory footprint of vector storage in bytes."""
        return self.num_vectors * self.dim * 4
