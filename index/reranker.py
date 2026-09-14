import numpy as np
from typing import Tuple, Optional


class ErrorBoundReRanker:
    """Theoretical Error-Bound Pruning and Exact FP32 Re-Ranker.

    Implements the RaBitQ confidence bound:
        Confidence Interval = +/- sqrt((1 - <o_bar, o>^2) / <o_bar, o>^2) * (epsilon_0 / sqrt(D - 1))
    where epsilon_0 = 1.9 ensures 95%+ theoretical confidence.

    Prunes non-viable candidate vectors whose lower distance bounds exceed the
    K-th best upper bound, then computes exact FP32 Euclidean distances on the
    surviving candidates.
    """

    def __init__(self, dim: int, epsilon_0: float = 1.9):
        self.dim = dim
        self.epsilon_0 = epsilon_0

    def compute_bound_radius(self, inner_products_oo: np.ndarray) -> np.ndarray:
        """Compute the confidence interval half-width Delta_i for each vector.

        Args:
            inner_products_oo: <o_bar, o> factor of shape (N,) in (0, 1].

        Returns:
            delta: Half-width array of shape (N,).
        """
        oo = np.clip(inner_products_oo, 1e-6, 0.999999)
        ratio = (1.0 - oo**2) / (oo**2)
        ratio = np.maximum(ratio, 0.0)
        denom = np.sqrt(max(self.dim - 1, 1))
        return (np.sqrt(ratio) * (self.epsilon_0 / denom)).astype(np.float32)

    def prune_and_rerank(
        self,
        candidate_indices: np.ndarray,
        candidate_est_ips: np.ndarray,
        candidate_oo: np.ndarray,
        candidate_norms: np.ndarray,
        query_norm: float,
        query_raw: np.ndarray,
        X_base_raw: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Prune candidates using theoretical error bounds and re-rank survivors with FP32.

        Args:
            candidate_indices: Indices of candidates in X_base_raw of shape (M,).
            candidate_est_ips: Estimated <o, q> of shape (M,).
            candidate_oo: <o_bar, o> of shape (M,).
            candidate_norms: ||o_r - c|| of shape (M,).
            query_norm: ||q_r - c|| (scalar float).
            query_raw: Raw query vector of shape (D,).
            X_base_raw: Full precision base vectors of shape (N, D).
            k: Desired number of top nearest neighbors.

        Returns:
            final_indices: Top-k neighbor indices of shape (k,).
            final_distances: Exact Euclidean distances of shape (k,).
            num_survivors: Number of candidates that survived bound pruning.
        """
        M = len(candidate_indices)
        if M <= k:
            # Not enough candidates to prune, re-rank all
            survivor_indices = candidate_indices
        else:
            # 1. Compute bound radii Delta_i
            delta = self.compute_bound_radius(candidate_oo)

            # 2. Inner product upper/lower bounds:
            # LB_ip = est_ip - delta, UB_ip = est_ip + delta
            ub_ip = np.clip(candidate_est_ips + delta, -1.0, 1.0)
            lb_ip = np.clip(candidate_est_ips - delta, -1.0, 1.0)

            # 3. Distance bounds:
            # Dist^2 = ||o||^2 + ||q||^2 - 2 * ||o|| * ||q|| * <o, q>
            # Minimum distance (LB_dist) corresponds to maximum inner product (UB_ip)
            # Maximum distance (UB_dist) corresponds to minimum inner product (LB_ip)
            norm_prod = 2.0 * candidate_norms * query_norm
            base_sq_sum = candidate_norms**2 + query_norm**2

            dist_sq_lb = np.maximum(base_sq_sum - norm_prod * ub_ip, 0.0)
            dist_sq_ub = np.maximum(base_sq_sum - norm_prod * lb_ip, 0.0)

            # Find the k-th smallest upper bound
            k_th_ub = np.partition(dist_sq_ub, k - 1)[k - 1]

            # Keep candidates whose lower bound <= k-th upper bound
            survivor_mask = dist_sq_lb <= k_th_ub
            survivor_indices = candidate_indices[survivor_mask]

            # Guarantee at least k candidates survive
            if len(survivor_indices) < k:
                survivor_indices = candidate_indices[np.argpartition(dist_sq_lb, k)[:k]]

        # 4. FP32 Re-ranking on survivors
        num_survivors = len(survivor_indices)
        survivor_vectors = X_base_raw[survivor_indices]  # (S, D)
        diff = survivor_vectors - query_raw.reshape(1, -1)
        exact_dists = np.linalg.norm(diff, axis=1)

        # Retrieve top-k
        if k < num_survivors:
            top_k_sub = np.argpartition(exact_dists, k)[:k]
            sort_order = np.argsort(exact_dists[top_k_sub])
            final_sub_indices = top_k_sub[sort_order]
        else:
            final_sub_indices = np.argsort(exact_dists)[:k]

        final_indices = survivor_indices[final_sub_indices]
        final_distances = exact_dists[final_sub_indices].astype(np.float32)

        return final_indices, final_distances, num_survivors
