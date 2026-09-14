import time
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.cluster import KMeans
from index.reranker import ErrorBoundReRanker


class IVFIndex:
    """Inverted File Index (IVF) with Quantization and Theoretical Re-ranking.

    Partitions dataset vectors into N_list clusters via K-Means. During query
    time, probes the nearest n_probe clusters and uses the underlying vector
    quantizer (RaBitQ or ABV-Quant) for high-speed candidate filtering.
    """

    def __init__(
        self,
        quantizer: Any,
        n_list: int = 32,
        n_probe: int = 4,
        use_reranking: bool = True,
        epsilon_0: float = 1.9,
        rerank_factor: int = 5,
        seed: Optional[int] = 42,
    ):
        self.quantizer = quantizer
        self.n_list = n_list
        self.n_probe = min(n_probe, n_list)
        self.use_reranking = use_reranking
        self.epsilon_0 = epsilon_0
        self.rerank_factor = rerank_factor
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None  # (n_list, D)
        self.inverted_lists: Dict[int, np.ndarray] = {}  # cluster_id -> array of vector indices
        self.X_raw: Optional[np.ndarray] = None
        self.reranker: Optional[ErrorBoundReRanker] = None
        self.is_fitted: bool = False

    def fit(self, X_raw: np.ndarray) -> "IVFIndex":
        """Cluster vectors and fit quantizer."""
        self.X_raw = np.ascontiguousarray(X_raw, dtype=np.float32)
        N, D = X_raw.shape

        # 1. Fit quantizer on the full dataset
        self.quantizer.fit(X_raw)

        # 2. Run K-Means clustering
        n_clusters = min(self.n_list, N)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init="auto")
        labels = kmeans.fit_predict(X_raw)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)

        # 3. Populate inverted lists
        self.inverted_lists = {}
        for c_id in range(n_clusters):
            members = np.where(labels == c_id)[0]
            self.inverted_lists[c_id] = members.astype(np.int64)

        if self.use_reranking:
            self.reranker = ErrorBoundReRanker(dim=D, epsilon_0=self.epsilon_0)

        self.is_fitted = True
        return self

    def search(
        self,
        queries_raw: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search k nearest neighbors via multi-probe IVF."""
        if not self.is_fitted or self.centroids is None or self.X_raw is None:
            raise RuntimeError("IVFIndex must be fitted before search.")

        Q = queries_raw.shape[0]
        indices = np.zeros((Q, k), dtype=np.int64)
        distances = np.zeros((Q, k), dtype=np.float32)

        start_time = time.perf_counter()

        for q_idx in range(Q):
            q_raw = queries_raw[q_idx : q_idx + 1]
            q_raw_1d = q_raw[0]

            # 1. Probe n_probe nearest centroids
            c_dists_sq = np.sum((self.centroids - q_raw_1d) ** 2, axis=1)
            probed_clusters = np.argpartition(c_dists_sq, self.n_probe)[: self.n_probe]

            # 2. Gather candidate indices from probed inverted lists
            candidate_list = [self.inverted_lists[c] for c in probed_clusters if len(self.inverted_lists[c]) > 0]
            if len(candidate_list) == 0:
                candidate_ids = np.arange(k, dtype=np.int64)
            else:
                candidate_ids = np.concatenate(candidate_list)

            # 3. Quantizer distance estimation on candidate subset
            q_norm, q_norm_val = self.quantizer.preprocessor.transform(q_raw)
            q_norm_scalar = float(q_norm_val[0])

            all_est_ips = self.quantizer.estimate_inner_products(q_norm[0])
            cand_est_ips = all_est_ips[candidate_ids]
            cand_norms = self.quantizer.data_norms[candidate_ids]

            if self.use_reranking and self.reranker is not None:
                # Retrieve candidate oo factors
                cand_oo = self.quantizer.inner_products_oo[candidate_ids]
                # Candidates to retrieve before bound pruning
                num_to_consider = min(len(candidate_ids), k * self.rerank_factor)
                cand_dists_sq = self.quantizer.preprocessor.reconstruct_distance_sq(
                    cand_est_ips, cand_norms, q_norm_scalar
                )
                top_cand_idx = np.argpartition(cand_dists_sq, min(num_to_consider - 1, len(cand_dists_sq) - 1))[:num_to_consider]

                final_cand_ids = candidate_ids[top_cand_idx]
                final_cand_ips = cand_est_ips[top_cand_idx]
                final_cand_oo = cand_oo[top_cand_idx]
                final_cand_norms = cand_norms[top_cand_idx]

                top_k_indices, top_k_dists, _ = self.reranker.prune_and_rerank(
                    candidate_indices=final_cand_ids,
                    candidate_est_ips=final_cand_ips,
                    candidate_oo=final_cand_oo,
                    candidate_norms=final_cand_norms,
                    query_norm=q_norm_scalar,
                    query_raw=q_raw_1d,
                    X_base_raw=self.X_raw,
                    k=k,
                )
            else:
                cand_dists_sq = self.quantizer.preprocessor.reconstruct_distance_sq(
                    cand_est_ips, cand_norms, q_norm_scalar
                )
                if k < len(candidate_ids):
                    part_idx = np.argpartition(cand_dists_sq, k)[:k]
                    sort_order = np.argsort(cand_dists_sq[part_idx])
                    top_k_indices = candidate_ids[part_idx[sort_order]]
                    top_k_dists = np.sqrt(cand_dists_sq[part_idx[sort_order]])
                else:
                    sort_order = np.argsort(cand_dists_sq)[:k]
                    top_k_indices = candidate_ids[sort_order]
                    top_k_dists = np.sqrt(cand_dists_sq[sort_order])

            indices[q_idx] = top_k_indices
            distances[q_idx] = top_k_dists

        total_time = time.perf_counter() - start_time
        latency_ms = (total_time / Q) * 1000.0

        return indices, distances, latency_ms

    @property
    def memory_bytes_per_vector(self) -> float:
        """Memory footprint per vector including inverted list pointers and quantizer."""
        # 4 bytes for inverted list indexing + quantizer storage
        return self.quantizer.memory_bytes_per_vector + 4.0
