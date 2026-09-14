import time
import numpy as np
from typing import Tuple, Optional


class BinaryFlatIndex:
    """Standard Sign Binarization Flat Baseline (1-bit LSH / Hamming).

    Binarizes vectors by taking the sign of each dimension:
        b[i] = 1 if x[i] >= 0 else 0
    Packs bits into uint64 words and searches via Hamming distance:
        d_H(x_b, q_b) = popcount(x_b ^ q_b)
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.num_words = (dim + 63) // 64
        self.binary_codes: Optional[np.ndarray] = None  # shape: (N, num_words), uint64
        self.raw_vectors: Optional[np.ndarray] = None
        self.num_vectors: int = 0

    def fit(self, X: np.ndarray) -> "BinaryFlatIndex":
        """Quantize and pack dataset vectors."""
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected array of shape (N, {self.dim}), got {X.shape}")
        
        self.num_vectors = X.shape[0]
        self.raw_vectors = np.ascontiguousarray(X, dtype=np.float32)
        self.binary_codes = self._pack_vectors(X >= 0)
        return self

    def _pack_vectors(self, bool_array: np.ndarray) -> np.ndarray:
        """Pack boolean matrix (N, D) into uint64 array (N, num_words)."""
        N, D = bool_array.shape
        # Pad to multiple of 64
        pad_width = (64 - (D % 64)) % 64
        if pad_width > 0:
            padded = np.pad(bool_array, ((0, 0), (0, pad_width)), mode="constant", constant_values=False)
        else:
            padded = bool_array

        # Pack into uint64
        packed = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little"))
        # View as uint64
        return np.ascontiguousarray(packed.view(np.uint64))

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        batch_size: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search nearest neighbors by Hamming distance.

        Args:
            queries: Query vectors of shape (Q, D).
            k: Number of nearest neighbors.

        Returns:
            indices: Neighbor indices of shape (Q, k).
            distances: Hamming distance / angular distance approximation (Q, k).
            latency_ms: Average search latency per query in milliseconds.
        """
        if self.binary_codes is None:
            raise RuntimeError("Index has not been fitted.")

        Q = queries.shape[0]
        indices = np.zeros((Q, k), dtype=np.int64)
        hamming_dists = np.zeros((Q, k), dtype=np.float32)

        start_time = time.perf_counter()
        q_packed = self._pack_vectors(queries >= 0)  # (Q, num_words)

        for start_idx in range(0, Q, batch_size):
            end_idx = min(start_idx + batch_size, Q)
            batch_q = q_packed[start_idx:end_idx]  # (B, num_words)

            # XOR between batch_q and self.binary_codes
            # batch_q shape: (B, 1, num_words)
            # codes shape: (1, N, num_words)
            xor_res = np.bitwise_xor(batch_q[:, np.newaxis, :], self.binary_codes[np.newaxis, :, :])
            # Popcount across uint64 words: python/numpy popcount
            # In numpy >= 2.0 or via bit_count:
            popcounts = np.zeros((xor_res.shape[0], xor_res.shape[1]), dtype=np.int32)
            for w in range(self.num_words):
                # Efficient bit count
                w_xor = xor_res[:, :, w]
                # Fast parallel bitcount in numpy
                w_xor = w_xor - ((w_xor >> np.uint64(1)) & np.uint64(0x5555555555555555))
                w_xor = (w_xor & np.uint64(0x3333333333333333)) + ((w_xor >> np.uint64(2)) & np.uint64(0x3333333333333333))
                w_xor = (w_xor + (w_xor >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
                w_xor = (w_xor * np.uint64(0x0101010101010101)) >> np.uint64(56)
                popcounts += w_xor.astype(np.int32)

            # Retrieve top k smallest Hamming distances
            if k < popcounts.shape[1]:
                part_idx = np.argpartition(popcounts, k, axis=1)[:, :k]
                part_dists = np.take_along_axis(popcounts, part_idx, axis=1)
                sorted_order = np.argsort(part_dists, axis=1)
                batch_indices = np.take_along_axis(part_idx, sorted_order, axis=1)
                batch_dists = np.take_along_axis(part_dists, sorted_order, axis=1)
            else:
                batch_indices = np.argsort(popcounts, axis=1)
                batch_dists = np.take_along_axis(popcounts, batch_indices, axis=1)

            indices[start_idx:end_idx] = batch_indices
            hamming_dists[start_idx:end_idx] = batch_dists.astype(np.float32)

        total_time = time.perf_counter() - start_time
        latency_ms = (total_time / Q) * 1000.0

        return indices, hamming_dists, latency_ms

    @property
    def memory_bytes_per_vector(self) -> float:
        """Memory footprint per vector in bytes (D / 8 bytes)."""
        return float(self.num_words * 8)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory footprint of binary code storage in bytes."""
        return self.num_vectors * self.num_words * 8
