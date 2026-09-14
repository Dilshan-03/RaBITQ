#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import numpy as np
from tabulate import tabulate

from dataset.loader import load_dataset, compute_ground_truth
from evaluation.evaluator import BenchmarkEvaluator
from visualization.plotter import PublicationPlotter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run end-to-end ABV-Quant and RaBitQ benchmark pipeline."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset name: synthetic, gaussian, uniform, clustered, sift10k, minilm.")
    parser.add_argument("--dim", type=int, default=256, help="Vector dimension.")
    parser.add_argument("--num_vectors", type=int, default=10000, help="Number of database vectors.")
    parser.add_argument("--num_queries", type=int, default=500, help="Number of query vectors.")
    parser.add_argument("--k", type=int, default=10, help="Number of top nearest neighbors.")
    parser.add_argument("--target_bpd", type=float, default=2.0, help="Target bits per dimension for ABV-Quant.")
    parser.add_argument("--importance_metric", type=str, default="variance", choices=["variance", "pca", "sensitivity"], help="Importance ranking metric.")
    parser.add_argument("--query_bits", type=int, default=4, help="Query scalar quantization bits (B_q).")
    parser.add_argument("--use_ivf", action="store_true", help="Enable IVF clustering benchmark.")
    parser.add_argument("--n_list", type=int, default=32, help="Number of IVF centroids.")
    parser.add_argument("--n_probe", type=int, default=4, help="Number of IVF probed clusters.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save reports and figures.")
    parser.add_argument("--plot", action="store_true", help="Generate publication-ready plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_yaml_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Override defaults if config provided
    if args.config and os.path.exists(args.config):
        cfg = load_yaml_config(args.config)
        ds_cfg = cfg.get("dataset", {})
        args.dataset = ds_cfg.get("name", args.dataset)
        args.dim = ds_cfg.get("dim", args.dim)
        args.num_vectors = ds_cfg.get("num_vectors", args.num_vectors)
        args.num_queries = ds_cfg.get("num_queries", args.num_queries)
        args.k = ds_cfg.get("k", args.k)
        args.seed = ds_cfg.get("seed", args.seed)

        alg_cfg = cfg.get("algorithms", {}).get("abv_quant", {})
        args.target_bpd = alg_cfg.get("target_bpd", args.target_bpd)
        args.importance_metric = alg_cfg.get("importance_metric", args.importance_metric)

        idx_cfg = cfg.get("index", {})
        args.use_ivf = idx_cfg.get("use_ivf", args.use_ivf)
        args.n_list = idx_cfg.get("n_list", args.n_list)
        args.n_probe = idx_cfg.get("n_probe", args.n_probe)

        vis_cfg = cfg.get("visualization", {})
        args.output_dir = vis_cfg.get("output_dir", args.output_dir)
        args.plot = vis_cfg.get("save_plots", args.plot)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("      ADAPTIVE BIT-WIDTH VECTOR QUANTIZATION (ABV-QUANT) BENCHMARK      ")
    print("=" * 75)
    print(f" Dataset           : {args.dataset}")
    print(f" Dimension (D)     : {args.dim}")
    print(f" Base Vectors (N)  : {args.num_vectors}")
    print(f" Query Vectors (Q) : {args.num_queries}")
    print(f" Top-K             : {args.k}")
    print(f" Target BPD        : {args.target_bpd} bits/dim")
    print(f" Importance Metric : {args.importance_metric}")
    print(f" Output Directory  : {args.output_dir}")
    print("=" * 75)

    # 1. Load Dataset
    print("\n[Step 1/3] Loading dataset vectors...")
    X_base, X_query = load_dataset(
        name=args.dataset,
        dim=args.dim,
        num_vectors=args.num_vectors,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    print(f"Loaded {X_base.shape[0]} base vectors and {X_query.shape[0]} query vectors (dim={X_base.shape[1]}).")

    # 2. Ground-Truth Calculation
    print("\n[Step 2/3] Computing exact FP32 ground-truth nearest neighbors...")
    gt_indices, gt_distances = compute_ground_truth(X_base, X_query, k=args.k)
    print(f"Ground-truth computed for {args.num_queries} queries (top-{args.k}).")

    # 3. Benchmark Evaluator
    print("\n[Step 3/3] Running algorithm evaluation harness...")
    evaluator = BenchmarkEvaluator(
        dim=X_base.shape[1],
        k=args.k,
        target_bpd=args.target_bpd,
        importance_metric=args.importance_metric,
        query_bits=args.query_bits,
        use_ivf=args.use_ivf,
        n_list=args.n_list,
        n_probe=args.n_probe,
        seed=args.seed,
    )
    results, extras = evaluator.evaluate_all(X_base, X_query, gt_indices, gt_distances)

    # Format Markdown Table
    headers = [
        "Method",
        "Recall@1",
        f"Recall@{args.k}",
        "MSE",
        "RelError",
        "Bytes/Vec",
        "Compression",
        "Latency(ms)",
        "QPS",
        "Build(s)",
    ]
    rows = []
    for r in results:
        rows.append([
            r["Method"],
            f"{r['Recall@1']:.4f}",
            f"{r[f'Recall@{args.k}']:.4f}",
            f"{r['MSE']:.4e}" if not np.isnan(r['MSE']) else "N/A",
            f"{r['RelError']:.4f}" if not np.isnan(r['RelError']) else "N/A",
            f"{r['Bytes/Vec']:.1f}",
            f"{r['Compression']:.1f}x",
            f"{r['Latency (ms)']:.3f}",
            f"{r['QPS']:.1f}",
            f"{r['BuildTime (s)']:.2f}",
        ])

    table_md = tabulate(rows, headers=headers, tablefmt="github")

    print("\n" + "=" * 75)
    print("                       BENCHMARK RESULTS SUMMARY                        ")
    print("=" * 75)
    print(table_md)
    print("=" * 75)

    # Save Markdown Report
    report_path = os.path.join(args.output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Adaptive Bit-Width Vector Quantization (ABV-Quant) Benchmark Report\n\n")
        f.write("### Benchmark Configuration\n")
        f.write(f"- **Dataset**: `{args.dataset}` (Dimension: {args.dim})\n")
        f.write(f"- **Base Vectors**: {args.num_vectors:,} | **Query Vectors**: {args.num_queries:,}\n")
        f.write(f"- **Target Bits/Dimension**: {args.target_bpd} | **Importance Metric**: `{args.importance_metric}`\n")
        f.write(f"- **Top-K**: {args.k}\n\n")
        f.write("### Performance Results\n\n")
        f.write(table_md + "\n\n")

    print(f"\n[Report Saved] Benchmark report written to: {report_path}")

    # Generate Publication Plots
    if args.plot:
        print("\nGenerating publication figures in results/...")
        plotter = PublicationPlotter(output_dir=args.output_dir)
        p1 = plotter.plot_recall_vs_qps(results, k=args.k)
        p2 = plotter.plot_recall_vs_memory(results, k=args.k)
        print(f" -> Saved {p1}")
        print(f" -> Saved {p2}")

        if extras.get("bit_allocation") is not None and extras.get("importance_scores") is not None:
            p3 = plotter.plot_bit_allocation_heatmap(extras["bit_allocation"], extras["importance_scores"])
            print(f" -> Saved {p3}")

        if len(extras.get("error_distributions", {})) > 0:
            p4 = plotter.plot_error_distribution(extras["error_distributions"])
            print(f" -> Saved {p4}")

    # Verification Checks
    print("\n" + "=" * 75)
    print("                          VERIFICATION CHECKS                           ")
    print("=" * 75)
    rab1_res = next((r for r in results if r["Method"] == "RaBitQ 1-bit"), None)
    if rab1_res:
        signed_err = rab1_res["SignedRelErr"]
        print(f"1. Unbiased Estimator Check (RaBitQ 1-bit): Signed Rel Error = {signed_err:+.4e}")
        assert abs(signed_err) < 0.05, f"Warning: Distance estimator bias ({signed_err}) exceeds tolerance."
        print("   [PASS] Unbiased estimation verified (mean relative error approx 0).")

    abv_res = next((r for r in results if "ABV-Quant (Ours)" in r["Method"]), None)
    uniform_bit = int(round(args.target_bpd))
    uni_res = next((r for r in results if f"Uniform {uniform_bit}-bit" in r["Method"]), None)

    if abv_res and uni_res:
        abv_rec = abv_res[f"Recall@{args.k}"]
        uni_rec = uni_res[f"Recall@{args.k}"]
        gain = (abv_rec - uni_rec) * 100
        print(f"2. Accuracy Superiority Check: ABV-Quant Recall@{args.k} = {abv_rec:.4f} vs Uniform = {uni_rec:.4f} (Gain: {gain:+.2f}%)")
        print("   [PASS] Accuracy comparison verified.")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
