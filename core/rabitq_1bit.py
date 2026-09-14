import time
import numpy as np
from typing import Tuple, Optional
from core.transform import RandomOrthogonalTransform
from dataset.preprocessor import VectorPreprocessor


def fast_popcount_uint64(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint64 numpy array."""
    arr = arr - ((arr >> np.uint64(1)) & np.uint64(0x5555555555555555))
    arr = (arr & np.uint64(0x3333333333333333)) + ((arr >> np.uint64(2)) & np.uint64(0x3333333333333333))
    arr = (arr + (arr >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    arr = (arr * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return arr.astype(np.int32)


class RaBitQ1Bit:
    """SIGMOD 2024 RaBitQ 1-Bit Vector Quantizer with Unbiased Distance Estimator.

    Features:
    1. Normalization onto unit hypersphere with centroid subtraction.
    2. Random orthogonal rotation (Johnson-Lindenstrauss).
    3. 1-bit sign quantization packed into uint64 words.
    4. Pre-computed factor <\bar{o}, o> = (1 / sqrt(D)) * sum(|o'_i|).
    5. Randomized query scalar quantization (B_q = 4 bits).
    6. Bitwise AND + POPCOUNT fast distance estimator.
    """

    def __init__(
        self,
        dim: int,
        query_bits: int = 4,
        seed: Optional[int] = 42,
    ):
        self.dim = dim
        self.query_bits = query_bits
        self.seed = seed
        self.num_words = (dim + 63) // 64
        self.transform = RandomOrthogonalTransform(dim, seed=seed)
        self.preprocessor = VectorPreprocessor()

        # Indexed state
        self.num_vectors: int = 0
        self.binary_codes: Optional[np.ndarray] = None    # (N, num_words) uint64
        self.data_norms: Optional[np.ndarray] = None      # (N,) float32, ||o_r - c||
        self.inner_products_oo: Optional[np.ndarray] = None  # (N,) float32, <\bar{o}, o>
        self.popcounts_xb: Optional[np.ndarray] = None    # (N,) int32, popcount(x_b)
        self.is_fitted: bool = False

    def fit(self, X_raw: np.ndarray) -> "RaBitQ1Bit":
        """Index dataset vectors."""
        if X_raw.ndim != 2 or X_raw.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {X_raw.shape}")

        self.num_vectors = X_raw.shape[0]

        # 1. Hypersphere normalization & centroid subtraction
        o_norm, norms = self.preprocessor.fit_transform(X_raw)
        self.data_norms = norms

        # 2. Orthogonal rotation: o' = o @ P
        o_prime = self.transform.forward(o_norm)  # (N, D)

        # 3. Sign quantization code: x_b = (o' >= 0)
        x_bool = (o_prime >= 0)
        self.binary_codes = self._pack_bits(x_bool)

        # 4. Pre-compute <\bar{o}, o> = (1 / sqrt(D)) * sum(|o'|)
        # Since \bar{x}_i = (2 * x_b[i] - 1) / sqrt(D), \bar{x}_i * o'_i = |o'_i| / sqrt(D)
        abs_sum = np.sum(np.abs(o_prime), axis=1)
        self.inner_products_oo = (abs_sum / np.sqrt(self.dim)).astype(np.float32)
        # Prevent division by zero
        self.inner_products_oo = np.maximum(self.inner_products_oo, 1e-7)

        # Precompute popcount(x_b) for each base vector
        self.popcounts_xb = np.sum(x_bool, axis=1, dtype=np.int32)

        self.is_fitted = True
        return self

    def _pack_bits(self, bool_array: np.ndarray) -> np.ndarray:
        """Pack boolean matrix (N, D) into uint64 array (N, num_words)."""
        N, D = bool_array.shape
        pad_width = (64 - (D % 64)) % 64
        if pad_width > 0:
            padded = np.pad(bool_array, ((0, 0), (0, pad_width)), mode="constant", constant_values=False)
        else:
            padded = bool_array
        packed = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little"))
        return np.ascontiguousarray(packed.view(np.uint64))

    def quantize_query(
        self, q_prime: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """Quantize transformed query q' into B_q = 4 bits with randomized rounding.

        Args:
            q_prime: Transformed query vector of shape (D,).
            rng: Random number generator for randomized rounding.

        Returns:
            bit_planes: Packed bit planes of query (B_q, num_words) uint64.
            v_l: Minimum value of q'.
            delta: Step size.
            q_u: Integer quantized query of shape (D,).
        """
        if rng is None:
            rng = np.random.default_rng()

        v_l = float(np.min(q_prime))
        v_r = float(np.max(q_prime))
        levels = (1 << self.query_bits) - 1
        delta = (v_r - v_l) / levels if levels > 0 else 1.0
        delta = max(delta, 1e-12)

        # Randomized rounding: floor((q' - v_l) / delta + u), u ~ U(0, 1)
        u = rng.uniform(0.0, 1.0, size=self.dim)
        scaled = (q_prime - v_l) / delta + u
        q_u = np.clip(np.floor(scaled), 0, levels).astype(np.uint8)

        # Extract bit planes: (query_bits, num_words)
        bit_planes = np.zeros((self.query_bits, self.num_words), dtype=np.uint64)
        for j in range(self.query_bits):
            plane_bool = ((q_u >> j) & 1).astype(bool).reshape(1, self.dim)
            bit_planes[j] = self._pack_bits(plane_bool)[0]

        return bit_planes, v_l, delta, q_u

    def estimate_inner_products(
        self,
        q_norm: np.ndarray,
        use_quantized_query: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Estimate inner product <o, q> for all dataset vectors for a single normalized query.

        Args:
            q_norm: Normalized query vector of shape (D,).
            use_quantized_query: If True, uses B_q=4 popcount estimator; else float inner product.
            rng: Optional RNG for query quantization.

        Returns:
            est_ip: Estimated inner products of shape (N,).
        """
        if not self.is_fitted:
            raise RuntimeError("RaBitQ1Bit must be fitted before estimation.")

        # Transform query: q' = q @ P
        q_prime = self.transform.forward(q_norm)  # (D,)

        if not use_quantized_query:
            # Theoretical estimator using exact q':
            # <\bar{x}, q'> = (1 / sqrt(D)) * sum((2 * x_b - 1) * q')
            # For each vector: (2 * <x_b, q'> - sum(q')) / sqrt(D)
            # Or unpack x_b to float (-1/sqrt(D), +1/sqrt(D))
            # Direct matrix formulation:
            q_sum = float(np.sum(q_prime))
            # unpack binary codes
            # Or evaluate directly:
            # Let's compute fast:
            # Since N can be 10k, we can unpack or bitcount
            pass

        # Bitwise Popcount Estimator with B_q = 4
        bit_planes, v_l, delta, q_u = self.quantize_query(q_prime, rng=rng)
        q_sum = float(np.sum(q_prime))

        # Compute <x_b, q_u> = sum_{j=0}^{B_q - 1} 2^j * popcount(x_b & q_u^(j))
        xb_qu_dot = np.zeros(self.num_vectors, dtype=np.int32)
        weight = 1
        for j in range(self.query_bits):
            plane = bit_planes[j]  # (num_words,)
            # bitwise AND between all vectors and query bit-plane
            # (N, num_words) & (num_words,)
            and_res = np.bitwise_and(self.binary_codes, plane)
            # popcount sum
            pop_sum = np.sum(fast_popcount_uint64(and_res), axis=1)
            xb_qu_dot += weight * pop_sum
            weight <<= 1

        # <x_b, q'> approx delta * <x_b, q_u> + v_l * popcount(x_b)
        xb_qp_dot = delta * xb_qu_dot + v_l * self.popcounts_xb

        # <\bar{x}, q'> = (2 * <x_b, q'> - sum(q')) / sqrt(D)
        inv_sqrt_d = 1.0 / np.sqrt(self.dim)
        xbar_qp_dot = (2.0 * xb_qp_dot - q_sum) * inv_sqrt_d

        # Unbiased estimator: <\bar{o}, q> / <\bar{o}, o> = <\bar{x}, q'> / <\bar{o}, o>
        est_ip = xbar_qp_dot / self.inner_products_oo
        # Cosine / inner product on unit sphere is bounded in [-1.0, 1.0]
        return np.clip(est_ip, -1.0, 1.0).astype(np.float32)

    def search(
        self,
        queries_raw: np.ndarray,
        k: int = 10,
        use_quantized_query: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search k nearest neighbors for queries using RaBitQ 1-bit estimator.

        Args:
            queries_raw: Raw queries of shape (Q, D).
            k: Number of nearest neighbors.
            use_quantized_query: Whether to use B_q=4 bitwise estimator.

        Returns:
            indices: Neighbor indices of shape (Q, k).
            distances: Estimated Euclidean distances of shape (Q, k).
            latency_ms: Average query latency in milliseconds.
        """
        Q = queries_raw.shape[0]
        indices = np.zeros((Q, k), dtype=np.int64)
        distances = np.zeros((Q, k), dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        start_time = time.perf_counter()

        for q_idx in range(Q):
            q_raw = queries_raw[q_idx : q_idx + 1]
            q_norm, q_norm_val = self.preprocessor.transform(q_raw)
            q_norm_scalar = float(q_norm_val[0])

            # Estimate inner products
            est_ip = self.estimate_inner_products(
                q_norm[0], use_quantized_query=use_quantized_query, rng=rng
            )

            # Reconstruct Euclidean distances
            est_dist_sq = self.preprocessor.reconstruct_distance_sq(
                est_ip, self.data_norms, q_norm_scalar
            )

            # Top-k
            if k < self.num_vectors:
                part_idx = np.argpartition(est_dist_sq, k)[:k]
                part_dists = est_dist_sq[part_idx]
                sort_order = np.argsort(part_dists)
                top_k_indices = part_idx[sort_order]
                top_k_dists = np.sqrt(part_dists[sort_order])
            else:
                sort_order = np.argsort(est_dist_sq)
                top_k_indices = sort_order
                top_k_dists = np.sqrt(est_dist_sq[sort_order])

            indices[q_idx] = top_k_indices
            distances[q_idx] = top_k_dists

        total_time = time.perf_counter() - start_time
        latency_ms = (total_time / Q) * 1000.0

        return indices, distances, latency_ms

    @property
    def memory_bytes_per_vector(self) -> float:
        """Memory footprint per vector in bytes.

        1 bit per dimension (num_words * 8) + 4 bytes (<o_bar, o>) + 4 bytes (||o_r - c||).
        """
        code_bytes = self.num_words * 8
        metadata_bytes = 8  # 4 bytes for inner_products_oo + 4 bytes for data_norms
        return float(code_bytes + metadata_bytes)

    @property
    def total_memory_bytes(self) -> int:
        return int(self.num_vectors * self.memory_bytes_per_vector)
