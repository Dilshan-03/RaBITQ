import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional


class PublicationPlotter:
    """Publication-quality figure generator for ABV-Quant and RaBitQ benchmarks."""

    def __init__(self, output_dir: str = "results", dpi: int = 300):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set modern research style
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        })

    def plot_recall_vs_qps(
        self,
        results: List[Dict[str, Any]],
        k: int = 10,
        filename: str = "recall_vs_qps.png",
    ) -> str:
        """Plot Recall@K vs QPS trade-off."""
        plt.figure(figsize=(8, 5.5))
        
        markers = {"Flat FP32": "s", "Binary Sign": "X", "RaBitQ 1-bit": "^", "Uniform 2-bit": "D", "ABV-Quant (Ours)": "o", "ABV-Quant + IVF": "*"}
        colors = {"Flat FP32": "#7f7f7f", "Binary Sign": "#d62728", "RaBitQ 1-bit": "#ff7f0e", "Uniform 2-bit": "#2ca02c", "ABV-Quant (Ours)": "#1f77b4", "ABV-Quant + IVF": "#9467bd"}

        for r in results:
            method = r["Method"]
            qps = r["QPS"]
            rec = r[f"Recall@{k}"]
            marker = markers.get(method, "o")
            color = colors.get(method, "#333333")
            size = 140 if "Ours" in method else 100

            plt.scatter(qps, rec, s=size, marker=marker, color=color, label=method, edgecolors="black", linewidths=1.2, zorder=5)
            # Add text label offset
            plt.annotate(
                f" {method}\n ({rec:.3f}, {int(qps)} QPS)",
                (qps, rec),
                textcoords="offset points",
                xytext=(8, -4),
                fontsize=9,
                fontweight="bold" if "Ours" in method else "normal"
            )

        plt.xlabel("Throughput (Queries Per Second, QPS) [Higher is Better]")
        plt.ylabel(f"Search Accuracy (Recall@{k}) [Higher is Better]")
        plt.title(f"Recall@{k} vs. Throughput (QPS) Benchmark")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=self.dpi)
        plt.close()
        return out_path

    def plot_recall_vs_memory(
        self,
        results: List[Dict[str, Any]],
        k: int = 10,
        filename: str = "recall_vs_memory.png",
    ) -> str:
        """Plot Recall@K vs Memory Footprint (Bytes per Vector)."""
        plt.figure(figsize=(8, 5.5))
        
        markers = {"Flat FP32": "s", "Binary Sign": "X", "RaBitQ 1-bit": "^", "Uniform 2-bit": "D", "ABV-Quant (Ours)": "o", "ABV-Quant + IVF": "*"}
        colors = {"Flat FP32": "#7f7f7f", "Binary Sign": "#d62728", "RaBitQ 1-bit": "#ff7f0e", "Uniform 2-bit": "#2ca02c", "ABV-Quant (Ours)": "#1f77b4", "ABV-Quant + IVF": "#9467bd"}

        for r in results:
            method = r["Method"]
            mem = r["Bytes/Vec"]
            rec = r[f"Recall@{k}"]
            marker = markers.get(method, "o")
            color = colors.get(method, "#333333")
            size = 140 if "Ours" in method else 100

            plt.scatter(mem, rec, s=size, marker=marker, color=color, label=method, edgecolors="black", linewidths=1.2, zorder=5)
            plt.annotate(
                f" {method}\n ({mem:.0f} B, {rec:.3f})",
                (mem, rec),
                textcoords="offset points",
                xytext=(8, -4),
                fontsize=9,
                fontweight="bold" if "Ours" in method else "normal"
            )

        plt.xlabel("Memory Footprint (Bytes per Vector) [Lower is Better]")
        plt.ylabel(f"Search Accuracy (Recall@{k}) [Higher is Better]")
        plt.title(f"Recall@{k} vs. Memory Footprint Comparison")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=self.dpi)
        plt.close()
        return out_path

    def plot_bit_allocation_heatmap(
        self,
        bit_allocation: np.ndarray,
        importance_scores: np.ndarray,
        filename: str = "bit_allocation_heatmap.png",
    ) -> str:
        """Plot dimension importance ranking and allocated bit widths."""
        D = len(bit_allocation)
        dim_indices = np.arange(D)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        # Top: Importance Scores
        ax1.plot(dim_indices, importance_scores, color="#1f77b4", lw=1.8, label="Dimension Importance")
        ax1.set_ylabel("Importance Score S(i)")
        ax1.set_title("ABV-Quant Dimension Importance Ranking & Adaptive Bit Allocation")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Bottom: Bit Widths as step / bar
        color_map = {1: "#a1d99b", 2: "#41ab5d", 4: "#238b45", 8: "#00441b"}
        bar_colors = [color_map.get(b, "#333333") for b in bit_allocation]

        ax2.bar(dim_indices, bit_allocation, color=bar_colors, width=1.0, edgecolor="none")
        ax2.set_ylabel("Bits Allocated (b_i)")
        ax2.set_xlabel("Dimension Index (i)")
        ax2.set_yticks([1, 2, 4, 8])
        ax2.set_ylim(0, 9)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=self.dpi)
        plt.close()
        return out_path

    def plot_error_distribution(
        self,
        error_distributions: Dict[str, np.ndarray],
        filename: str = "error_distribution.png",
    ) -> str:
        """Plot histogram / KDE of distance estimation errors demonstrating unbiasedness."""
        plt.figure(figsize=(8.5, 5.5))

        palette = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
        for idx, (method, errors) in enumerate(error_distributions.items()):
            # Filter finite errors
            clean_errors = errors[np.isfinite(errors)]
            if len(clean_errors) > 0:
                color = palette[idx % len(palette)]
                mean_err = np.mean(clean_errors)
                sns.kdeplot(clean_errors, label=f"{method} (mean={mean_err:+.3e})", color=color, lw=2.0)

        plt.axvline(0.0, color="red", linestyle="--", lw=1.5, alpha=0.8, label="Zero Bias Line")
        plt.xlabel("Distance Estimation Error (Estimated - Exact)")
        plt.ylabel("Probability Density")
        plt.title("Distance Estimation Error Distribution (Unbiased Estimator Verification)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=self.dpi)
        plt.close()
        return out_path
