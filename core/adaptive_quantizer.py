import time
import numpy as np
from typing import Tuple, Dict, List, Optional
from core.transform import RandomOrthogonalTransform
from dataset.preprocessor import VectorPreprocessor
from core.rabitq_1bit import fast_popcount_uint64


class ABVQuantizer:
    """Adaptive Bit-Width Vector Quantizer (ABV-Quant).

    Dynamically allocates bit widths b_i in {1, 2, 4, 8} per dimension based on
    feature importance (variance, PCA eigenvalues, or perturbation sensitivity)
    under a strict target budget: sum_{i=1}^D b_i <= D * B_target.
    """

    def __init__(
        self,
        dim: int,
        target_bpd: float = 2.0,
        importance_metric: str = "variance",
        query_bits: int = 4,
        allowed_bits: Optional[List[int]] = None,
        seed: Optional[int] = 42,
    ):
        self.dim = dim
        self.target_bpd = target_bpd
        self.importance_metric = importance_metric.lower()
        self.query_bits = query_bits
        self.allowed_bits = allowed_bits or [1, 2, 4, 8]
        self.seed = seed

        self.transform = RandomOrthogonalTransform(dim, seed=seed)
        self.preprocessor = VectorPreprocessor()

        # Dimension bit allocation
        self.bit_allocation: Optional[np.ndarray] = None  # (D,) int
        self.importance_scores: Optional[np.ndarray] = None  # (D,) float
        self.dim_order: Optional[np.ndarray] = None  # (D,) sorted dimension indices

        # Group indices by bit width
        self.dim_groups: Dict[int, np.ndarray] = {}

        # 1-bit packed data
        self.num_1bit_dims: int = 0
        self.num_1bit_words: int = 0
        self.codes_1bit: Optional[np.ndarray] = None  # (N, num_1bit_words) uint64
        self.popcounts_1bit: Optional[np.ndarray] = None  # (N,) int32
        self.mean_abs_1bit: float = 1.0

        # Multi-bit data
        self.multi_bit_codes: Dict[int, np.ndarray] = {}  # bit_width -> (N, D_b) uint8
        self.multi_bit_scales: Dict[int, np.ndarray] = {}  # bit_width -> (N, 2) [v_min, delta]

        # Metadata
        self.inner_products_oo: Optional[np.ndarray] = None  # (N,) float32
        self.data_norms: Optional[np.ndarray] = None  # (N,) float32
        self.num_vectors: int = 0
        self.is_fitted: bool = False

    def _compute_importance(self, O_prime: np.ndarray) -> np.ndarray:
        """Compute importance score S(i) for each dimension of rotated dataset O'."""
        if self.importance_metric == "variance":
            scores = np.var(O_prime, axis=0)
        elif self.importance_metric == "pca":
            cov = np.cov(O_prime, rowvar=False)
            # Diagonal of covariance or PCA eigenvalues
            eigenvalues, _ = np.linalg.eigh(cov)
            scores = np.abs(eigenvalues)[::-1]
            if len(scores) != self.dim:
                scores = np.var(O_prime, axis=0)
        elif self.importance_metric == "sensitivity":
            # Perturbation sensitivity: average magnitude of coordinates
            scores = np.mean(np.abs(O_prime), axis=0)
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")

        # Normalize scores
        scores = np.maximum(scores, 1e-12)
        return scores.astype(np.float32)

    def _allocate_bits(self, importance_scores: np.ndarray) -> np.ndarray:
        """Non-linear binning and adaptive budget allocation.

        Ranks dimensions by importance S(i) and assigns bit widths b_i in {1, 2, 4, 8}
        using non-linear quantile binning such that high-importance dimensions
        receive higher bit-widths (e.g. 4 or 8 bits) and low-importance receive
        1 bit, strictly complying with sum(b_i) <= D * B_target.
        """
        total_budget = int(np.floor(self.dim * self.target_bpd))
        sorted_dims = np.argsort(importance_scores)[::-1]
        allocation = np.ones(self.dim, dtype=np.int32)

        B = self.target_bpd
        if B <= 1.0:
            return allocation

        if B <= 2.0:
            # Target B in (1.0, 2.0]: boost top dimensions to 4-bit, bottom to 1-bit, middle to 2-bit
            alpha = B - 1.0
            n4 = int(round(0.06 * alpha * self.dim)) if 4 in self.allowed_bits else 0
            n1 = 2 * n4
            n2 = max(0, min(self.dim - n4 - n1, total_budget - self.dim - 3 * n4)) if 2 in self.allowed_bits else 0

            allocation[sorted_dims[:n4]] = 4
            allocation[sorted_dims[n4 : n4 + n2]] = 2
            # remaining sorted_dims[n4+n2:] remain 1-bit

        elif B <= 4.0:
            # Target B in (2.0, 4.0]: boost top dimensions to 8-bit, middle to 4-bit, lower to 2-bit
            alpha = (B - 2.0) / 2.0
            n8 = int(round(0.05 * alpha * self.dim)) if 8 in self.allowed_bits else 0
            n4 = int(round((0.50 + 0.35 * alpha) * self.dim)) if 4 in self.allowed_bits else 0
            budget_used = self.dim + (7 * n8) + (3 * n4)
            n2 = max(0, min(self.dim - n8 - n4, total_budget - budget_used)) if 2 in self.allowed_bits else 0

            allocation[sorted_dims[:n8]] = 8
            allocation[sorted_dims[n8 : n8 + n4]] = 4
            allocation[sorted_dims[n8 + n4 : n8 + n4 + n2]] = 2
        else:
            # B > 4.0: mix 8-bit and 4-bit
            n8 = int(round((B - 4.0) / 4.0 * self.dim)) if 8 in self.allowed_bits else 0
            allocation[sorted_dims[:n8]] = 8
            allocation[sorted_dims[n8:]] = 4

        # Enforce exact budget compliance: downgrade if rounding caused overflow
        prev_b = {8: 4, 4: 2, 2: 1}
        while np.sum(allocation) > total_budget:
            for d in sorted_dims[::-1]:
                cur_b = allocation[d]
                if cur_b in prev_b:
                    allocation[d] = prev_b[cur_b]
                    if np.sum(allocation) <= total_budget:
                        break

        # If any spare budget, greedily upgrade top dimensions
        upgrades = [(1, 2, 1), (2, 4, 2), (4, 8, 4)]
        for from_b, to_b, cost in upgrades:
            if to_b not in self.allowed_bits:
                continue
            for d in sorted_dims:
                if allocation[d] == from_b and np.sum(allocation) + cost <= total_budget:
                    allocation[d] = to_b

        # Verify strict compliance with budget
        assert np.sum(allocation) <= total_budget, f"Bit budget exceeded: {np.sum(allocation)} > {total_budget}"
        return allocation

    def fit(self, X_raw: np.ndarray) -> "ABVQuantizer":
        """Index dataset vectors with adaptive bit-width allocation."""
        if X_raw.ndim != 2 or X_raw.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {X_raw.shape}")

        self.num_vectors = X_raw.shape[0]

        # 1. Hypersphere normalization & centroid subtraction
        o_norm, norms = self.preprocessor.fit_transform(X_raw)
        self.data_norms = norms

        # 2. Orthogonal rotation: O' = O @ P
        o_prime = self.transform.forward(o_norm)  # (N, D)

        # 3. Compute importance scores and bit allocation
        self.importance_scores = self._compute_importance(o_prime)
        self.bit_allocation = self._allocate_bits(self.importance_scores)

        # Group dimensions by allocated bit width
        self.dim_groups = {}
        for b in self.allowed_bits:
            idx = np.where(self.bit_allocation == b)[0]
            if len(idx) > 0:
                self.dim_groups[b] = idx

        # Reconstructed vectors in rotated space \bar{x}
        x_bar = np.zeros_like(o_prime, dtype=np.float32)

        # 4. Quantize 1-bit dimensions
        if 1 in self.dim_groups:
            dims_1 = self.dim_groups[1]
            self.num_1bit_dims = len(dims_1)
            self.num_1bit_words = (self.num_1bit_dims + 63) // 64

            o_prime_1 = o_prime[:, dims_1]  # (N, D_1)
            bool_1 = (o_prime_1 >= 0)

            # Pack 1-bit dimensions into uint64
            pad_w = (64 - (self.num_1bit_dims % 64)) % 64
            padded = np.pad(bool_1, ((0, 0), (0, pad_w)), mode="constant", constant_values=False) if pad_w > 0 else bool_1
            packed = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little"))
            self.codes_1bit = np.ascontiguousarray(packed.view(np.uint64))
            self.popcounts_1bit = np.sum(bool_1, axis=1, dtype=np.int32)

            # Mean absolute value per vector for optimal calibration
            if self.num_1bit_dims > 0:
                self.mean_abs_1bit = np.mean(np.abs(o_prime_1), axis=1).astype(np.float32)
                x_bar[:, dims_1] = np.where(bool_1, self.mean_abs_1bit[:, None], -self.mean_abs_1bit[:, None])
            else:
                self.mean_abs_1bit = np.ones(self.num_vectors, dtype=np.float32)

        # 5. Quantize multi-bit dimensions (b in {2, 4, 8})
        self.multi_bit_codes = {}
        self.multi_bit_scales = {}

        for b in [2, 4, 8]:
            if b in self.dim_groups:
                dims_b = self.dim_groups[b]
                o_prime_b = o_prime[:, dims_b]  # (N, D_b)

                levels = (1 << b) - 1
                v_min = np.min(o_prime_b, axis=1, keepdims=True)
                v_max = np.max(o_prime_b, axis=1, keepdims=True)
                delta = np.maximum((v_max - v_min) / levels, 1e-12)

                scaled = (o_prime_b - v_min) / delta
                q_b = np.clip(np.round(scaled), 0, levels).astype(np.uint8)

                self.multi_bit_codes[b] = q_b
                self.multi_bit_scales[b] = np.hstack([v_min, delta]).astype(np.float32)

                # Dequantize for calibration
                x_bar[:, dims_b] = v_min + q_b.astype(np.float32) * delta

        # 6. Pre-compute calibration factor <\bar{o}, o> = <\bar{x}, o'>
        oo_dot = np.sum(x_bar * o_prime, axis=1)
        self.inner_products_oo = np.maximum(oo_dot, 1e-7).astype(np.float32)

        self.is_fitted = True
        return self

    def estimate_inner_products(self, q_norm: np.ndarray) -> np.ndarray:
        """Estimate inner product <o, q> using hybrid slice-wise estimator."""
        if not self.is_fitted:
            raise RuntimeError("ABVQuantizer must be fitted before estimation.")

        # Transform query: q' = q @ P
        q_prime = self.transform.forward(q_norm)  # (D,)

        # Accumulator for <\bar{x}, q'> across all slices
        xbar_qp_dot = np.zeros(self.num_vectors, dtype=np.float32)

        # 1-bit slice inner product
        if 1 in self.dim_groups:
            dims_1 = self.dim_groups[1]
            q_prime_1 = q_prime[dims_1]
            q_sum_1 = float(np.sum(q_prime_1))

            # Query quantization with B_q = 4 for 1-bit slice
            v_l = float(np.min(q_prime_1))
            v_r = float(np.max(q_prime_1))
            levels = (1 << self.query_bits) - 1
            delta_q = max((v_r - v_l) / levels, 1e-12)

            q_u_1 = np.clip(np.round((q_prime_1 - v_l) / delta_q), 0, levels).astype(np.uint8)

            # Bit planes
            xb_qu_dot = np.zeros(self.num_vectors, dtype=np.int32)
            weight = 1
            for j in range(self.query_bits):
                plane_bool = ((q_u_1 >> j) & 1).astype(bool).reshape(1, self.num_1bit_dims)
                pad_w = (64 - (self.num_1bit_dims % 64)) % 64
                if pad_w > 0:
                    plane_bool = np.pad(plane_bool, ((0, 0), (0, pad_w)), mode="constant", constant_values=False)
                plane_packed = np.ascontiguousarray(np.packbits(plane_bool, axis=1, bitorder="little")).view(np.uint64)[0]

                and_res = np.bitwise_and(self.codes_1bit, plane_packed)
                pop_sum = np.sum(fast_popcount_uint64(and_res), axis=1)
                xb_qu_dot += weight * pop_sum
                weight <<= 1

            xb_qp_dot = delta_q * xb_qu_dot + v_l * self.popcounts_1bit
            # \bar{x}_1 = mean_abs * (2 * x_b - 1)
            # <\bar{x}_1, q'_1> = mean_abs * (2 * <x_b, q'_1> - sum(q'_1))
            slice_dot_1 = self.mean_abs_1bit * (2.0 * xb_qp_dot - q_sum_1)
            xbar_qp_dot += slice_dot_1.astype(np.float32)

        # Multi-bit slices inner product (b in {2, 4, 8})
        for b in [2, 4, 8]:
            if b in self.dim_groups:
                dims_b = self.dim_groups[b]
                q_prime_b = q_prime[dims_b]
                q_sum_b = float(np.sum(q_prime_b))

                codes = self.multi_bit_codes[b]  # (N, D_b) uint8
                scales = self.multi_bit_scales[b]  # (N, 2) [v_min, delta]

                codes_dot = np.matmul(codes.astype(np.float32), q_prime_b)  # (N,)
                v_min = scales[:, 0]
                delta = scales[:, 1]
                slice_dot_b = v_min * q_sum_b + delta * codes_dot
                xbar_qp_dot += slice_dot_b

        # Unbiased estimator: <\bar{x}, q'> / <\bar{o}, o>
        est_ip = xbar_qp_dot / self.inner_products_oo
        return np.clip(est_ip, -1.0, 1.0).astype(np.float32)

    def search(
        self,
        queries_raw: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search k nearest neighbors."""
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
    def total_allocated_bits(self) -> int:
        if self.bit_allocation is None:
            return 0
        return int(np.sum(self.bit_allocation))

    @property
    def average_bpd(self) -> float:
        return float(self.total_allocated_bits / self.dim)

    @property
    def memory_bytes_per_vector(self) -> float:
        """Total memory footprint per vector in bytes."""
        # Code bytes = sum(b_i) / 8
        code_bytes = (self.total_allocated_bits + 7) // 8
        # Multi-bit scales metadata: 8 bytes per active multi-bit slice
        num_multi_slices = sum(1 for b in [2, 4, 8] if b in self.dim_groups)
        metadata_bytes = 8 + (num_multi_slices * 8)  # 4 (<o_bar, o>) + 4 (norm) + slice scales
        return float(code_bytes + metadata_bytes)

    @property
    def total_memory_bytes(self) -> int:
        return int(self.num_vectors * self.memory_bytes_per_vector)
