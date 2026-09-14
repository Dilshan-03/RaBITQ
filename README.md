# Adaptive Bit-Width Vector Quantization (ABV-Quant)

A modular, production-grade Python library and benchmarking suite for **Adaptive Bit-Width Vector Quantization (ABV-Quant)**, extending the theoretical foundations of **RaBitQ (ACM SIGMOD 2024 / SIGMOD 2025)**.

---

## 1. Overview & Theoretical Highlights

Modern vector databases and retrieval-augmented generation (RAG) pipelines store millions of high-dimensional embeddings in full precision (FP32), consuming tens of gigabytes of RAM and incurring high memory bandwidth costs during nearest-neighbor search.

* **RaBitQ (ACM SIGMOD 2024)**: Projects vectors onto the unit hypersphere, applies a randomized Johnson-Lindenstrauss orthogonal matrix rotation $P \in \mathbb{R}^{D \times D}$, and quantizes vectors into 1-bit binary codes. It guarantees an unbiased distance estimator with a sharp $O(1/\sqrt{D})$ error bound and accelerates inner products using SIMD `popcount` across query bit-planes.
* **Uniform Multi-Bit RaBitQ (ACM SIGMOD 2025 Extension)**: Uniformly allocates $B$ bits ($B \in \{2, 4, 8\}$) to all dimensions.
* **Novel Contribution — ABV-Quant (Adaptive Bit-Width VQ)**: Instead of uniform bit allocation, ABV-Quant evaluates dimension importance $S(i)$ (via per-dimension variance, PCA eigenvalues, or perturbation sensitivity) in the rotated space. High-importance dimensions are assigned higher precision ($b_i \in \{2, 4, 8\}$) while low-importance dimensions are assigned 1 bit, strictly adhering to the total budget:
  $$\sum_{i=1}^D b_i \le D \times B_{\text{target}}$$

---

## 2. Directory Structure

```
adaptive_vq_project/
│
├── README.md                       # Comprehensive guide and documentation
├── requirements.txt                # System dependencies
├── setup.py                        # Python package configuration
├── run_pipeline.py                 # Core CLI entry point for benchmarks
│
├── config/
│   ├── __init__.py
│   └── default_config.yaml         # YAML benchmark configuration
│
├── dataset/
│   ├── __init__.py
│   ├── loader.py                   # Loaders: Synthetic, SIFT10K, MiniLM embeddings
│   └── preprocessor.py             # Hypersphere normalization & centroid subtraction
│
├── core/
│   ├── __init__.py
│   ├── transform.py                # Random Orthogonal Matrix (JLT) generator
│   ├── baseline_fp32.py            # Exact FP32 flat scan baseline
│   ├── baseline_binary.py          # Standard sign binarization & Hamming baseline
│   ├── rabitq_1bit.py              # SIGMOD 2024 RaBitQ (1-bit + Popcount)
│   ├── rabitq_multibit.py          # SIGMOD 2025 Uniform B-bit RaBitQ baseline
│   └── adaptive_quantizer.py       # Novel Adaptive Bit-Width Vector Quantizer
│
├── index/
│   ├── __init__.py
│   ├── ivf_index.py                # Inverted File Index (IVF) clustering
│   └── reranker.py                 # Theoretical error-bound pruning & FP32 re-ranking
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                  # Recall@K, MSE, RelError, QPS, Memory footprint
│   └── evaluator.py                # Benchmarking harness across all algorithms
│
├── visualization/
│   ├── __init__.py
│   └── plotter.py                  # Publication-quality figure generator
│
└── tests/
    ├── __init__.py
    └── test_all.py                 # Complete automated test suite
```

---

## 3. Installation & Setup

Install the package and dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 4. Running Benchmarks

### Quick Start CLI Command

Execute the complete end-to-end benchmark comparing Flat FP32, Binary Sign, RaBitQ 1-bit, Uniform 2-bit, and ABV-Quant:

```bash
python run_pipeline.py \
    --dataset synthetic \
    --dim 256 \
    --num_vectors 10000 \
    --num_queries 500 \
    --k 10 \
    --target_bpd 2 \
    --output_dir results/ \
    --plot
```

### Command Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `synthetic` | Dataset: `synthetic`, `gaussian`, `uniform`, `clustered`, `sift10k`, `minilm` |
| `--dim` | `int` | `256` | Dimensionality of embeddings |
| `--num_vectors` | `int` | `10000` | Number of database vectors |
| `--num_queries` | `int` | `500` | Number of test query vectors |
| `--k` | `int` | `10` | Top-K nearest neighbors |
| `--target_bpd` | `float` | `2.0` | Target bits per dimension for ABV-Quant |
| `--importance_metric` | `str` | `variance` | Importance ranking: `variance`, `pca`, `sensitivity` |
| `--output_dir` | `str` | `results/` | Output directory for figures and Markdown report |
| `--plot` | `flag` | `False` | Generate publication figures |

---

## 5. Automated Unit & Integration Tests

Run the test suite to verify unbiased distance estimation, bit-budget compliance, and accuracy superiority:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
