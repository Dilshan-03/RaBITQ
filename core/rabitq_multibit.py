import time
import numpy as np
from typing import Tuple, Optional
from core.transform import RandomOrthogonalTransform
from dataset.preprocessor import VectorPreprocessor
from core.rabitq_1bit import RaBitQ1Bit


class RaBitQMultiBit:
    """SIGMOD 2025 Uniform Multi-Bit RaBitQ Vector Quantizer.

    Allocates uniform B bits (B in {1, 2, 4, 8}) to every dimension.
    Constructs an unbiased estimator calibrated by <\bar{o}, o>.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 2,
        query_bits: int = 4,
        seed: Optional[int] = 42,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.query_bits = query_bits
        self.seed = seed

        # If 1-bit, delegate internally or use dedicated 1-bit logic
        if self.bit_width == 1:
            self.base_1bit = RaBitQ1Bit(dim, query_bits=query_bits, seed=seed)
        else:
            self.base_1bit = None

        self.transform = RandomOrthogonalTransform(dim, seed=seed)
        self.preprocessor = VectorPreprocessor()

        # Indexed state for B > 1
        self.num_vectors: int = 0
        self.quantized_codes: Optional[np.ndarray] = None  # (N, D) uint8
        self.data_norms: Optional[np.ndarray] = None       # (N,) float32
        self.inner_products_oo: Optional[np.ndarray] = None  # (N,) float32, <\bar{o}, o>
        self.scale_offsets: Optional[np.ndarray] = None     # (N, 2) float32: [v_min, delta]
        self.is_fitted: bool = False

    def fit(self, X_raw: np.ndarray) -> "RaBitQMultiBit":
        """Index dataset vectors."""
        if X_raw.ndim != 2 or X_raw.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {X_raw.shape}")

        if self.bit_width == 1 and self.base_1bit is not None:
            self.base_1bit.fit(X_raw)
            self.is_fitted = True
            return self

        self.num_vectors = X_raw.shape[0]

        # 1. Hypersphere normalization & centroid subtraction
        o_norm, norms = self.preprocessor.fit_transform(X_raw)
        self.data_norms = norms

        # 2. Orthogonal rotation: o' = o @ P
        o_prime = self.transform.forward(o_norm)  # (N, D)

        # 3. Scalar quantization with B bits (levels = 2^B - 1)
        levels = (1 << self.bit_width) - 1
        v_min = np.min(o_prime, axis=1, keepdims=True)  # (N, 1)
        v_max = np.max(o_prime, axis=1, keepdims=True)  # (N, 1)
        delta = np.maximum((v_max - v_min) / levels, 1e-12)  # (N, 1)

        # Quantized codes in {0, ..., levels}
        scaled = (o_prime - v_min) / delta
        q_codes = np.clip(np.round(scaled), 0, levels).astype(np.uint8)

        # 4. Dequantized representation \bar{x}
        x_bar = v_min + q_codes.astype(np.float32) * delta  # (N, D)

        # 5. Pre-compute calibration factor <\bar{o}, o> = <\bar{x}, o'>
        oo_dot = np.sum(x_bar * o_prime, axis=1)
        self.inner_products_oo = np.maximum(oo_dot, 1e-7).astype(np.float32)

        self.quantized_codes = q_codes
        self.scale_offsets = np.hstack([v_min, delta]).astype(np.float32)
        self.is_fitted = True
        return self

    def estimate_inner_products(self, q_norm: np.ndarray) -> np.ndarray:
        """Estimate inner products <o, q> for all vectors for single normalized query.

        Args:
            q_norm: Normalized query vector of shape (D,).

        Returns:
            est_ip: Estimated inner products of shape (N,).
        """
        if not self.is_fitted:
            raise RuntimeError("RaBitQMultiBit must be fitted before estimation.")

        if self.bit_width == 1 and self.base_1bit is not None:
            return self.base_1bit.estimate_inner_products(q_norm)

        # Transform query: q' = q @ P
        q_prime = self.transform.forward(q_norm)  # (D,)
        q_sum = float(np.sum(q_prime))

        # Reconstructed <\bar{x}, q'> = v_min * sum(q') + delta * <q_codes, q'>
        # Fast matrix-vector dot: q_codes (N, D) @ q' (D,)
        codes_dot = np.matmul(self.quantized_codes.astype(np.float32), q_prime)  # (N,)

        v_min = self.scale_offsets[:, 0]
        delta = self.scale_offsets[:, 1]
        xbar_qp_dot = v_min * q_sum + delta * codes_dot

        # Unbiased estimator: <\bar{x}, q'> / <\bar{o}, o>
        est_ip = xbar_qp_dot / self.inner_products_oo
        return np.clip(est_ip, -1.0, 1.0).astype(np.float32)

    def search(
        self,
        queries_raw: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search k nearest neighbors."""
        if self.bit_width == 1 and self.base_1bit is not None:
            return self.base_1bit.search(queries_raw, k=k)

        Q = queries_raw.shape[0]
        indices = np.zeros((Q, k), dtype=np.int64)
        distances = np.zeros((Q, k), dtype=np.float32)

        start_time = time.perf_counter()

        for q_idx in range(Q):
            q_raw = queries_raw[q_idx : q_idx + 1]
            q_norm, q_norm_val = self.preprocessor.transform(q_raw)
            q_norm_scalar = float(q_norm_val[0])

            est_ip = self.estimate_inner_products(q_norm[0])
            est_dist_sq = self.preprocessor.reconstruct_distance_sq(
                est_ip, self.data_norms, q_norm_scalar
            )

            if k < self.num_vectors:
                part_idx = np.argpartition(est_dist_sq, k)[:k]
                part_dists = est_dist_sq[part_idx]
                sort_order = np.argsort(part_dists)
                indices[q_idx] = part_idx[sort_order]
                distances[q_idx] = np.sqrt(part_dists[sort_order])
            else:
                sort_order = np.argsort(est_dist_sq)
                indices[q_idx] = sort_order
                distances[q_idx] = np.sqrt(est_dist_sq[sort_order])

        total_time = time.perf_counter() - start_time
        latency_ms = (total_time / Q) * 1000.0

        return indices, distances, latency_ms

    @property
    def memory_bytes_per_vector(self) -> float:
        """Memory footprint per vector in bytes.

        B bits per dimension + metadata.
        """
        if self.bit_width == 1 and self.base_1bit is not None:
            return self.base_1bit.memory_bytes_per_vector

        # (D * bit_width) / 8 bytes + 8 bytes (v_min, delta) + 4 bytes (<o_bar, o>) + 4 bytes (||o_r - c||)
        code_bytes = (self.dim * self.bit_width + 7) // 8
        metadata_bytes = 16
        return float(code_bytes + metadata_bytes)

    @property
    def total_memory_bytes(self) -> int:
        if self.bit_width == 1 and self.base_1bit is not None:
            return self.base_1bit.total_memory_bytes
        return int(self.num_vectors * self.memory_bytes_per_vector)
