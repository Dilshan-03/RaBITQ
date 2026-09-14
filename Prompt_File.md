# SYSTEM INSTRUCTION FOR ANTIGRAVITY AI AGENT

**ROLE:** You are an expert AI Software Engineer and Computer Science Researcher.
**TASK:** Build the complete, modular, production-grade Python library and benchmarking suite for **Adaptive Bit-Width Vector Quantization (ABV-Quant)** based on the theoretical specification below.

---

## AGENT INSTRUCTIONS & EXECUTION WORKFLOW

Follow these execution steps sequentially. Do not stop until all modules, tests, benchmarker, visualizer, and CLI entry points are completely implemented and verified.

1. **Setup Directory Structure**: Create all required folders (`config`, `dataset`, `core`, `index`, `evaluation`, `visualization`) and populate them with valid `__init__.py` files.
2. **Environment Dependencies**: Write `requirements.txt` and `setup.py`.
3. **Data Pipeline**: Implement `dataset/loader.py` and `dataset/preprocessor.py`. Include support for synthetic Gaussian/Uniform data, SIFT/GIST vectors, and HuggingFace transformer embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
4. **Core Mathematics & Algorithms**:
   * Implement `core/transform.py` for Johnson-Lindenstrauss orthogonal matrix rotation ($P \in \mathbb{R}^{D \times D}$).
   * Implement `core/baseline_fp32.py` and `core/baseline_binary.py`.
   * Implement `core/rabitq_1bit.py` (SIGMOD 2024 RaBitQ with 1-bit quantization and $B_q=4$ bitwise `popcount` distance estimator).
   * Implement `core/rabitq_multibit.py` (SIGMOD 2025 Uniform $B$-bit RaBitQ).
   * Implement `core/adaptive_quantizer.py` (Proposed ABV-Quant method with Variance/PCA dimension ranking and non-uniform bit allocation $b_i \in \{1, 2, 4, 8\}$).
5. **Indexing & Re-ranking**: Implement `index/ivf_index.py` (IVF $K$-means clustering) and `index/reranker.py` (theoretical error-bound filtering with $\epsilon_0 = 1.9$ and FP32 re-ranking).
6. **Evaluation & Visualization**: Implement `evaluation/metrics.py`, `evaluation/evaluator.py`, and `visualization/plotter.py`.
7. **CLI Pipeline**: Create `run_pipeline.py` to run end-to-end benchmarks and output formatted Markdown tables and publication-ready plots (`recall_vs_qps.png`, `recall_vs_memory.png`, `bit_allocation_heatmap.png`).
8. **Verification**: Run tests to confirm unbiased distance estimation ($	ext{mean relative error} \approx 0$) and ensure ABV-Quant outperforms Uniform RaBitQ at matching memory budgets.

---

# PROJECT SPECIFICATION

## 1. Executive Summary & Research Context

High-dimensional dense vector embeddings are critical for modern artificial intelligence applications such as semantic search, recommendation systems, image retrieval, and retrieval-augmented generation (RAG). As the number of stored embeddings grows into millions or billions, storing every vector in full precision (FP32) creates severe memory footprint constraints and high computational latency during nearest-neighbor search.

* **Base Paper — RaBitQ (ACM SIGMOD 2024)**: RaBitQ normalizes high-dimensional vectors onto the unit hypersphere and applies a randomized orthogonal rotation matrix $P \in \mathbb{R}^{D \times D}$ (Johnson-Lindenstrauss Transformation). It quantizes $D$-dimensional vectors into compact $D$-bit binary strings. RaBitQ provides an unbiased distance estimator with a sharp theoretical error bound of $O(1/\sqrt{D})$ and enables fast distance estimation using SIMD or bitwise `AND` + `POPCOUNT` operations.
* **Uniform Multi-Bit RaBitQ (ACM SIGMOD 2025 Extension)**: Extends 1-bit RaBitQ by uniformly allocating $B$ bits per dimension across all $D$ dimensions, yielding flexible trade-offs between memory and accuracy.
* **Proposed Novel Extension — Adaptive Bit-Width Vector Quantization (ABV-Quant)**: Instead of assigning an identical bit width $B$ to every dimension, ABV-Quant dynamically estimates the information importance of each dimension (or dimension group) using measurable criteria (e.g., variance, PCA eigenvalues, or perturbation sensitivity). High-importance dimensions receive higher bit precision ($b_i \in \{2, 4, 8\}$), while lower-importance dimensions receive lower precision ($b_i = 1$). The total bit budget is strictly constrained to $\sum_{i=1}^D b_i \le D \times B_{\text{target}}$ to ensure fair performance comparison against uniform quantization.

---

## 2. Theoretical Framework & Mathematical Formulations

### 2.1 Vector Normalization & Distance Decomposition
Let $o_r, q_r \in \mathbb{R}^D$ represent raw data and query vectors, respectively, and $c \in \mathbb{R}^D$ be the dataset or cluster centroid. Vectors are normalized onto the unit hypersphere:
$$o = \frac{o_r - c}{||o_r - c||_2}, \quad q = \frac{q_r - c}{||q_r - c||_2}$$

The squared Euclidean distance between raw vectors is decomposed as:
$$||o_r - q_r||^2 = ||o_r - c||^2 + ||q_r - c||^2 - 2 ||o_r - c|| \cdot ||q_r - c|| \cdot \langle o, q \rangle$$

Where $||o_r - c||$ is pre-computed during indexing and $||q_r - c||$ is computed once per query.

### 2.2 Randomized Rotation & Codebook Construction
1. Sample a random orthogonal matrix $P \in \mathbb{R}^{D \times D}$ via QR decomposition of a Gaussian random matrix ($G = QR$).
2. Compute the inversely transformed vector: $o' = P^{-1} o = P^T o$.
3. Generate the binary quantization code $\bar{x}_b \in \{0, 1\}^D$:
   $$\bar{x}_b[i] = \begin{cases} 1 & \text{if } o'[i] \ge 0 \\ 0 & \text{if } o'[i] < 0 \end{cases}$$
4. Reconstruct the bi-valued vector $\bar{x} \in \{-1/\sqrt{D}, +1/\sqrt{D}\}^D$:
   $$\bar{x} = \frac{2 \bar{x}_b - 1_D}{\sqrt{D}}$$
5. The quantized vector in the original space is $\bar{o} = P \bar{x}$.

### 2.3 Unbiased Distance Estimator
RaBitQ constructs an unbiased estimator for the inner product $\langle o, q \rangle$:
$$\hat{\langle o, q \rangle} = \frac{\langle \bar{o}, q \rangle}{\langle \bar{o}, o \rangle} = \frac{\langle \bar{x}, P^T q \rangle}{\langle \bar{o}, o \rangle}$$

Where $\langle \bar{o}, o \rangle$ is a scalar pre-computed and stored for each vector during index construction. The estimator guarantees an absolute theoretical error bound of:
$$\left| \frac{\langle \bar{o}, q \rangle}{\langle \bar{o}, o \rangle} - \langle o, q \rangle \right| = O\left(\frac{1}{\sqrt{D}}\right) \quad \text{with high probability}$$

### 2.4 Query Scalar Quantization ($B_q = 4$)
To accelerate computation without floating-point overhead, the transformed query $q' = P^T q$ is scalar-quantized into $B_q = 4$ bit unsigned integers using randomized rounding:
$$\bar{q}_u[i] = \left\lfloor \frac{q'[i] - v_l}{\Delta} + u_i \right\rfloor, \quad u_i \sim U(0, 1)$$
where $v_l = \min(q')$, $v_r = \max(q')$, and $\Delta = \frac{v_r - v_l}{2^{B_q} - 1}$.

Fast bitwise inner product evaluation is computed as:
$$\langle \bar{x}_b, \bar{q}_u \rangle = \sum_{j=0}^{B_q-1} 2^j \cdot \text{popcount}\left(\bar{x}_b \,\&\, \bar{q}_u^{(j)}\right)$$

### 2.5 Adaptive Bit-Width Allocation (ABV-Quant)
In ABV-Quant, dimensions of the rotated space $O' = X P^T$ are ranked by an importance criterion $S(i)$:
1. **Importance Metrics**:
   * **Variance**: $S_{\text{var}}(i) = \text{Var}_o(o'[i])$
   * **PCA Eigenvalues**: $S_{\text{pca}}(i) = \lambda_i$ from covariance matrix of $O'$
   * **Perturbation Sensitivity**: $S_{\text{sens}}(i) = \mathbb{E}_{q'} [ |q'[i] \cdot o'[i]| ]$
2. **Bit Allocation**:
   Sort dimensions by $S(i)$ and assign bit-widths $b_i \in \{1, 2, 4, 8\}$ using non-linear binning or greedy budget allocation such that $\sum_{i=1}^D b_i \le D \times B_{\text{target}}$.
3. **Hybrid Representation**:
   Low-importance dimensions ($b_i = 1$) use 1-bit binary codes and popcount distance evaluation. High-importance dimensions ($b_i \ge 2$) store multi-bit scalar quantized representations.

---

## 3. Complete Codebase Architecture & File Layout

```
adaptive_vq_project/
│
├── README.md                       # Setup instructions and execution guide
├── requirements.txt                # System dependencies
├── setup.py                        # Python package configuration
├── run_pipeline.py                 # Core CLI entry point for full benchmark evaluation
│
├── config/
│   └── default_config.yaml         # Dataset and algorithm benchmark configurations
│
├── dataset/
│   ├── __init__.py
│   ├── loader.py                   # Data loaders (Synthetic, SIFT10K, GIST, HF embeddings)
│   └── preprocessor.py             # Hypersphere normalization & centroid subtraction
│
├── core/
│   ├── __init__.py
│   ├── transform.py                # Random Orthogonal Matrix (JLT) generator
│   ├── baseline_fp32.py            # Flat FP32 index baseline
│   ├── baseline_binary.py          # Standard sign binarization baseline
│   ├── rabitq_1bit.py              # SIGMOD 2024 RaBitQ (1-bit + Popcount)
│   ├── rabitq_multibit.py          # SIGMOD 2025 Uniform B-bit RaBitQ baseline
│   └── adaptive_quantizer.py       # Novel Adaptive Bit-Width Vector Quantizer
│
├── index/
│   ├── __init__.py
│   ├── ivf_index.py                # Inverted File Index (IVF) implementation
│   └── reranker.py                 # Theoretical error-bound pruning & FP32 re-ranking
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                  # Recall@K, MSE, RelError, QPS, Memory footprint
│   └── evaluator.py                # Benchmarking harness and stats aggregator
│
└── visualization/
    ├── __init__.py
    └── plotter.py                  # Publication-quality figure generator
```

---

## 4. Detailed Component Specifications for Development

### 4.1 Dataset Module (`dataset/loader.py`, `dataset/preprocessor.py`)
* Load synthetic high-dimensional Gaussian/Uniform datasets ($D \in \{128, 256, 512, 960\}$).
* Load benchmark datasets (SIFT, GIST, Deep, MSong, Word2Vec) or HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
* Compute ground-truth $K$-NN indices and exact distances using FP32 Euclidean norm.
* Subtract global or IVF cluster centroids $c$ and normalize vectors onto the unit hypersphere.

### 4.2 Orthogonal Transformation (`core/transform.py`)
* Instantiate `RandomOrthogonalTransform(dim, seed)` using QR decomposition of a standard Gaussian matrix $G = QR$.
* Provide `.forward(X)` ($X P$) and `.inverse(X)` ($X P^T$) operations.

### 4.3 Base RaBitQ (`core/rabitq_1bit.py`)
* Implement sign-bit extraction $\text{sign}(P^T o)$ packed into `uint64` bitstrings.
* Pre-compute vector magnitudes $||o_r - c||$ and inner products $\langle \bar{o}, o \rangle$ during indexing.
* Perform randomized scalar query quantization into $B_q = 4$ bit unsigned integers.
* Compute inner product estimates using bitwise `AND` and `POPCOUNT` operations across query bit planes.

### 4.4 Adaptive Quantizer (`core/adaptive_quantizer.py`)
* Calculate dimension importance scores $S(i)$ using variance, PCA eigenvalues, or perturbation sensitivity.
* Execute bit allocation mapping dimensions to bit widths $b_i \in \{1, 2, 4, 8\}$ subject to total target budget $D \times B_{\text{target}}$.
* Maintain packed bitstrings for 1-bit slices and scalar quantized arrays for multi-bit slices.
* Combine slice-wise inner product estimates into a unified unbiased distance estimate.

### 4.5 Indexing & Theoretical Bound Re-Ranker (`index/ivf_index.py`, `index/reranker.py`)
* Partition dataset into $N_{\text{list}}$ clusters using $K$-Means clustering.
* Apply RaBitQ confidence bound during search to prune non-viable candidates:
  $$\text{Confidence Interval} = \pm \sqrt{\frac{1 - \langle \bar{o}, o \rangle^2}{\langle \bar{o}, o \rangle^2}} \cdot \frac{\epsilon_0}{\sqrt{D - 1}} \quad (\epsilon_0 = 1.9)$$
* Re-rank surviving candidates using full-precision FP32 vectors.

### 4.6 Benchmarking & Visualization (`evaluation/metrics.py`, `visualization/plotter.py`)
* Measure Recall@1 and Recall@K.
* Measure Mean Squared Error (MSE) and Maximum/Average Relative Error on estimated distances.
* Measure Memory per Vector (bytes), Compression Ratio, Query Latency (ms), Queries Per Second (QPS), and Index Construction Time.
* Generate comparative plots (`recall_vs_qps.png`, `recall_vs_memory.png`, `error_distribution.png`, `bit_allocation_heatmap.png`).

---

## 5. Execution Pipeline & Dependencies

### `requirements.txt`
```txt
numpy>=1.22.0
scipy>=1.8.0
scikit-learn>=1.0.0
torch>=1.11.0
matplotlib>=3.5.0
seaborn>=0.11.2
pyyaml>=6.0
tqdm>=4.64.0
tabulate>=0.8.9
```

### Main CLI Command Syntax (`run_pipeline.py`)
```bash
# Execute benchmark comparing FP32, Binary, RaBitQ 1-bit, Uniform Multi-bit, and ABV-Quant
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

---

## 6. Implementation Checklist & Verification Requirements

1. **Environment Setup**: Construct folder structure and verify all `__init__.py` modules exist.
2. **Unbiased Estimation Test**: Verify that the mean relative error of `rabitq_1bit.py` on synthetic data approaches 0.
3. **Budget Compliance**: Enforce $\sum b_i \le D \times B_{\text{target}}$ in `adaptive_quantizer.py`.
4. **Accuracy Verification**: Confirm that ABV-Quant achieves higher Recall@K than Uniform RaBitQ at identical memory budgets.
5. **Output Generation**: Produce structured benchmark tables in Markdown format and export figure graphics to `results/`.
