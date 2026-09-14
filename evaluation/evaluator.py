import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from evaluation.metrics import (
    compute_recall_at_k,
    compute_recall_at_1,
    compute_distance_errors,
    compute_compression_ratio,
    compute_qps,
)
from core.baseline_fp32 import FP32FlatIndex
from core.baseline_binary import BinaryFlatIndex
from core.rabitq_1bit import RaBitQ1Bit
from core.rabitq_multibit import RaBitQMultiBit
from core.adaptive_quantizer import ABVQuantizer
from index.ivf_index import IVFIndex


class BenchmarkEvaluator:
    """End-to-End Evaluation Harness for Vector Quantization Methods.

    Benchmarks:
    - FP32 Flat Baseline
    - Binary Sign Baseline (Hamming)
    - RaBitQ 1-bit (SIGMOD 2024)
    - Uniform Multi-bit RaBitQ (SIGMOD 2025)
    - Adaptive Bit-Width Vector Quantizer (ABV-Quant)
    """

    def __init__(
        self,
        dim: int,
        k: int = 10,
        target_bpd: float = 2.0,
        importance_metric: str = "variance",
        query_bits: int = 4,
        use_ivf: bool = False,
        n_list: int = 32,
        n_probe: int = 4,
        use_reranking: bool = True,
        seed: Optional[int] = 42,
    ):
        self.dim = dim
        self.k = k
        self.target_bpd = target_bpd
        self.importance_metric = importance_metric
        self.query_bits = query_bits
        self.use_ivf = use_ivf
        self.n_list = n_list
        self.n_probe = n_probe
        self.use_reranking = use_reranking
        self.seed = seed

    def evaluate_all(
        self,
        X_base: np.ndarray,
        X_query: np.ndarray,
        gt_indices: np.ndarray,
        gt_distances: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run comprehensive benchmark across all algorithms.

        Returns:
            results: List of metric dictionaries per algorithm.
            extras: Additional data (bit allocations, error distributions) for plotting.
        """
        N, D = X_base.shape
        Q = X_query.shape[0]
        fp32_bytes_per_vector = D * 4.0

        results = []
        extras = {
            "error_distributions": {},
            "bit_allocation": None,
            "importance_scores": None,
        }

        # 1. FP32 Flat Baseline
        print("[1/5] Evaluating Flat FP32 Baseline...")
        t0 = time.perf_counter()
        fp32_idx = FP32FlatIndex(dim=D)
        fp32_idx.fit(X_base)
        t_build_fp32 = time.perf_counter() - t0

        pred_idx_fp32, pred_dist_fp32, lat_fp32 = fp32_idx.search(X_query, k=self.k)
        rec1_fp32 = compute_recall_at_1(pred_idx_fp32, gt_indices)
        recK_fp32 = compute_recall_at_k(pred_idx_fp32, gt_indices, k=self.k)
        err_fp32 = compute_distance_errors(pred_dist_fp32, gt_distances[:, :self.k])

        results.append({
            "Method": "Flat FP32",
            "Recall@1": rec1_fp32,
            f"Recall@{self.k}": recK_fp32,
            "MSE": err_fp32["mse"],
            "RelError": err_fp32["mean_abs_rel_err"],
            "SignedRelErr": err_fp32["signed_rel_err"],
            "Bytes/Vec": fp32_idx.memory_bytes_per_vector,
            "Compression": 1.0,
            "Latency (ms)": lat_fp32,
            "QPS": compute_qps(Q, (lat_fp32 * Q) / 1000.0),
            "BuildTime (s)": t_build_fp32,
        })

        # 2. Binary Sign Baseline
        print("[2/5] Evaluating Binary Sign Baseline (Hamming)...")
        t0 = time.perf_counter()
        bin_idx = BinaryFlatIndex(dim=D)
        bin_idx.fit(X_base)
        t_build_bin = time.perf_counter() - t0

        pred_idx_bin, pred_dist_bin, lat_bin = bin_idx.search(X_query, k=self.k)
        rec1_bin = compute_recall_at_1(pred_idx_bin, gt_indices)
        recK_bin = compute_recall_at_k(pred_idx_bin, gt_indices, k=self.k)

        results.append({
            "Method": "Binary Sign",
            "Recall@1": rec1_bin,
            f"Recall@{self.k}": recK_bin,
            "MSE": float("nan"),  # Hamming distance scale
            "RelError": float("nan"),
            "SignedRelErr": float("nan"),
            "Bytes/Vec": bin_idx.memory_bytes_per_vector,
            "Compression": compute_compression_ratio(fp32_bytes_per_vector, bin_idx.memory_bytes_per_vector),
            "Latency (ms)": lat_bin,
            "QPS": compute_qps(Q, (lat_bin * Q) / 1000.0),
            "BuildTime (s)": t_build_bin,
        })

        # 3. RaBitQ 1-bit (SIGMOD 2024)
        print("[3/5] Evaluating RaBitQ 1-Bit (SIGMOD 2024)...")
        t0 = time.perf_counter()
        rabitq_1 = RaBitQ1Bit(dim=D, query_bits=self.query_bits, seed=self.seed)
        rabitq_1.fit(X_base)
        t_build_rab1 = time.perf_counter() - t0

        pred_idx_r1, pred_dist_r1, lat_r1 = rabitq_1.search(X_query, k=self.k)
        rec1_r1 = compute_recall_at_1(pred_idx_r1, gt_indices)
        recK_r1 = compute_recall_at_k(pred_idx_r1, gt_indices, k=self.k)
        err_r1 = compute_distance_errors(pred_dist_r1, gt_distances[:, :self.k])
        extras["error_distributions"]["RaBitQ 1-bit"] = (pred_dist_r1 - gt_distances[:, :self.k]).flatten()

        results.append({
            "Method": "RaBitQ 1-bit",
            "Recall@1": rec1_r1,
            f"Recall@{self.k}": recK_r1,
            "MSE": err_r1["mse"],
            "RelError": err_r1["mean_abs_rel_err"],
            "SignedRelErr": err_r1["signed_rel_err"],
            "Bytes/Vec": rabitq_1.memory_bytes_per_vector,
            "Compression": compute_compression_ratio(fp32_bytes_per_vector, rabitq_1.memory_bytes_per_vector),
            "Latency (ms)": lat_r1,
            "QPS": compute_qps(Q, (lat_r1 * Q) / 1000.0),
            "BuildTime (s)": t_build_rab1,
        })

        # 4. Uniform Multi-Bit RaBitQ (SIGMOD 2025)
        uniform_bit = int(round(self.target_bpd))
        print(f"[4/5] Evaluating Uniform {uniform_bit}-Bit RaBitQ (SIGMOD 2025)...")
        t0 = time.perf_counter()
        rabitq_mb = RaBitQMultiBit(dim=D, bit_width=uniform_bit, query_bits=self.query_bits, seed=self.seed)
        rabitq_mb.fit(X_base)
        t_build_rmb = time.perf_counter() - t0

        pred_idx_rmb, pred_dist_rmb, lat_rmb = rabitq_mb.search(X_query, k=self.k)
        rec1_rmb = compute_recall_at_1(pred_idx_rmb, gt_indices)
        recK_rmb = compute_recall_at_k(pred_idx_rmb, gt_indices, k=self.k)
        err_rmb = compute_distance_errors(pred_dist_rmb, gt_distances[:, :self.k])
        extras["error_distributions"][f"Uniform {uniform_bit}-bit RaBitQ"] = (pred_dist_rmb - gt_distances[:, :self.k]).flatten()

        results.append({
            "Method": f"Uniform {uniform_bit}-bit",
            "Recall@1": rec1_rmb,
            f"Recall@{self.k}": recK_rmb,
            "MSE": err_rmb["mse"],
            "RelError": err_rmb["mean_abs_rel_err"],
            "SignedRelErr": err_rmb["signed_rel_err"],
            "Bytes/Vec": rabitq_mb.memory_bytes_per_vector,
            "Compression": compute_compression_ratio(fp32_bytes_per_vector, rabitq_mb.memory_bytes_per_vector),
            "Latency (ms)": lat_rmb,
            "QPS": compute_qps(Q, (lat_rmb * Q) / 1000.0),
            "BuildTime (s)": t_build_rmb,
        })

        # 5. ABV-Quant (Adaptive Bit-Width Vector Quantization)
        print(f"[5/5] Evaluating Novel ABV-Quant (Target BPD={self.target_bpd}, Metric={self.importance_metric})...")
        t0 = time.perf_counter()
        abv = ABVQuantizer(
            dim=D,
            target_bpd=self.target_bpd,
            importance_metric=self.importance_metric,
            query_bits=self.query_bits,
            seed=self.seed,
        )
        abv.fit(X_base)
        t_build_abv = time.perf_counter() - t0

        pred_idx_abv, pred_dist_abv, lat_abv = abv.search(X_query, k=self.k)
        rec1_abv = compute_recall_at_1(pred_idx_abv, gt_indices)
        recK_abv = compute_recall_at_k(pred_idx_abv, gt_indices, k=self.k)
        err_abv = compute_distance_errors(pred_dist_abv, gt_distances[:, :self.k])
        extras["error_distributions"]["ABV-Quant (Ours)"] = (pred_dist_abv - gt_distances[:, :self.k]).flatten()
        extras["bit_allocation"] = abv.bit_allocation
        extras["importance_scores"] = abv.importance_scores

        results.append({
            "Method": "ABV-Quant (Ours)",
            "Recall@1": rec1_abv,
            f"Recall@{self.k}": recK_abv,
            "MSE": err_abv["mse"],
            "RelError": err_abv["mean_abs_rel_err"],
            "SignedRelErr": err_abv["signed_rel_err"],
            "Bytes/Vec": abv.memory_bytes_per_vector,
            "Compression": compute_compression_ratio(fp32_bytes_per_vector, abv.memory_bytes_per_vector),
            "Latency (ms)": lat_abv,
            "QPS": compute_qps(Q, (lat_abv * Q) / 1000.0),
            "BuildTime (s)": t_build_abv,
        })

        # 6. Optional: ABV-Quant with IVF + Theoretical Re-ranking
        if self.use_ivf:
            print(f"[Bonus] Evaluating ABV-Quant with IVF (n_list={self.n_list}, n_probe={self.n_probe}, rerank={self.use_reranking})...")
            abv_inner = ABVQuantizer(
                dim=D,
                target_bpd=self.target_bpd,
                importance_metric=self.importance_metric,
                query_bits=self.query_bits,
                seed=self.seed,
            )
            t0 = time.perf_counter()
            ivf_idx = IVFIndex(
                quantizer=abv_inner,
                n_list=self.n_list,
                n_probe=self.n_probe,
                use_reranking=self.use_reranking,
                seed=self.seed,
            )
            ivf_idx.fit(X_base)
            t_build_ivf = time.perf_counter() - t0

            pred_idx_ivf, pred_dist_ivf, lat_ivf = ivf_idx.search(X_query, k=self.k)
            rec1_ivf = compute_recall_at_1(pred_idx_ivf, gt_indices)
            recK_ivf = compute_recall_at_k(pred_idx_ivf, gt_indices, k=self.k)
            err_ivf = compute_distance_errors(pred_dist_ivf, gt_distances[:, :self.k])

            results.append({
                "Method": "ABV-Quant + IVF",
                "Recall@1": rec1_ivf,
                f"Recall@{self.k}": recK_ivf,
                "MSE": err_ivf["mse"],
                "RelError": err_ivf["mean_abs_rel_err"],
                "SignedRelErr": err_ivf["signed_rel_err"],
                "Bytes/Vec": ivf_idx.memory_bytes_per_vector,
                "Compression": compute_compression_ratio(fp32_bytes_per_vector, ivf_idx.memory_bytes_per_vector),
                "Latency (ms)": lat_ivf,
                "QPS": compute_qps(Q, (lat_ivf * Q) / 1000.0),
                "BuildTime (s)": t_build_ivf,
            })

        return results, extras
