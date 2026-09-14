import os
import urllib.request
import gzip
import tarfile
import numpy as np
from typing import Tuple, Dict, Any, Optional


def load_synthetic_data(
    num_vectors: int = 10000,
    num_queries: int = 500,
    dim: int = 256,
    distribution: str = "gaussian",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-dimensional synthetic dataset and query vectors.

    Args:
        num_vectors: Number of dataset base vectors.
        num_queries: Number of query vectors.
        dim: Dimensionality of vectors (e.g. 128, 256, 512, 960).
        distribution: 'gaussian' (standard normal) or 'uniform'.
        seed: Random seed for reproducibility.

    Returns:
        X_base: Array of shape (num_vectors, dim), float32.
        X_query: Array of shape (num_queries, dim), float32.
    """
    rng = np.random.default_rng(seed)

    if distribution.lower() == "gaussian":
        # Generate Gaussian vectors with non-uniform covariance to create realistic variance variance across dimensions
        # This makes adaptive quantization evaluation realistic and meaningful
        scales = np.exp(-np.linspace(0, 2.0, dim))  # decaying variance across dimensions
        base_raw = rng.standard_normal((num_vectors, dim), dtype=np.float32) * scales
        query_raw = rng.standard_normal((num_queries, dim), dtype=np.float32) * scales
    elif distribution.lower() == "uniform":
        base_raw = rng.uniform(-1.0, 1.0, size=(num_vectors, dim)).astype(np.float32)
        query_raw = rng.uniform(-1.0, 1.0, size=(num_queries, dim)).astype(np.float32)
    elif distribution.lower() == "clustered":
        # Clustered dataset mimicking real-world embedding manifolds
        num_clusters = max(10, num_vectors // 500)
        centers = rng.standard_normal((num_clusters, dim), dtype=np.float32)
        cluster_assignments = rng.integers(0, num_clusters, size=num_vectors)
        base_raw = centers[cluster_assignments] + 0.3 * rng.standard_normal(
            (num_vectors, dim), dtype=np.float32
        )
        query_clusters = rng.integers(0, num_clusters, size=num_queries)
        query_raw = centers[query_clusters] + 0.3 * rng.standard_normal(
            (num_queries, dim), dtype=np.float32
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return base_raw, query_raw


def read_fvecs(file_path: str, max_vectors: Optional[int] = None) -> np.ndarray:
    """Read .fvecs binary file format commonly used in ANN benchmarks (SIFT/GIST)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            return np.empty((0, 0), dtype=np.float32)
        dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
        f.seek(0)
        
        record_bytes = 4 + dim * 4
        file_size = os.path.getsize(file_path)
        total_vectors = file_size // record_bytes
        
        count = total_vectors if max_vectors is None else min(total_vectors, max_vectors)
        raw_data = np.fromfile(f, dtype=np.float32, count=count * (dim + 1))
        vectors = raw_data.reshape((count, dim + 1))[:, 1:]
        return np.ascontiguousarray(vectors, dtype=np.float32)


def load_sift10k(
    cache_dir: str = "data/sift10k",
    max_vectors: int = 10000,
    max_queries: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load SIFT10K dataset, downloading and caching if not already present."""
    os.makedirs(cache_dir, exist_ok=True)
    base_file = os.path.join(cache_dir, "siftsmall_base.fvecs")
    query_file = os.path.join(cache_dir, "siftsmall_query.fvecs")

    if not (os.path.exists(base_file) and os.path.exists(query_file)):
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
        archive_path = os.path.join(cache_dir, "siftsmall.tar.gz")
        try:
            print(f"Downloading SIFT10K from {url}...")
            urllib.request.urlretrieve(url, archive_path)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=cache_dir)
            
            # Siftsmall extracts into siftsmall/ folder
            extracted_base = os.path.join(cache_dir, "siftsmall", "siftsmall_base.fvecs")
            extracted_query = os.path.join(cache_dir, "siftsmall", "siftsmall_query.fvecs")
            if os.path.exists(extracted_base):
                os.rename(extracted_base, base_file)
            if os.path.exists(extracted_query):
                os.rename(extracted_query, query_file)
        except Exception as e:
            print(f"Could not download SIFT10K ({e}). Falling back to realistic synthetic SIFT-like vectors (dim=128).")
            return load_synthetic_data(num_vectors=max_vectors, num_queries=max_queries, dim=128, distribution="clustered")

    X_base = read_fvecs(base_file, max_vectors=max_vectors)
    X_query = read_fvecs(query_file, max_vectors=max_queries)
    return X_base, X_query


def load_dataset(
    name: str = "synthetic",
    dim: int = 256,
    num_vectors: int = 10000,
    num_queries: int = 500,
    seed: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Universal dataset loader supporting synthetic, SIFT10K, and custom data."""
    name = name.lower()
    if name in ["synthetic", "gaussian"]:
        return load_synthetic_data(num_vectors, num_queries, dim, distribution="gaussian", seed=seed)
    elif name == "uniform":
        return load_synthetic_data(num_vectors, num_queries, dim, distribution="uniform", seed=seed)
    elif name == "clustered":
        return load_synthetic_data(num_vectors, num_queries, dim, distribution="clustered", seed=seed)
    elif name in ["sift", "sift10k"]:
        return load_sift10k(max_vectors=num_vectors, max_queries=num_queries)
    elif name in ["minilm", "huggingface"]:
        # Fallback or synthetic representation if transformers not available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("Loaded all-MiniLM-L6-v2 model for embedding extraction.")
            # Synthesize sentences for benchmark
            sentences = [f"This is synthetic benchmark sentence number {i} for vector retrieval evaluation." for i in range(num_vectors + num_queries)]
            all_embs = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            return all_embs[:num_vectors], all_embs[num_vectors:]
        except Exception as e:
            print(f"HuggingFace sentence-transformers unavailable ({e}). Generating realistic 384-dim semantic embeddings.")
            return load_synthetic_data(num_vectors, num_queries, dim=384, distribution="clustered", seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def compute_ground_truth(
    X_base: np.ndarray,
    X_query: np.ndarray,
    k: int = 10,
    batch_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact ground-truth nearest neighbors using full precision Euclidean distance.

    Args:
        X_base: Dataset vectors of shape (N, D).
        X_query: Query vectors of shape (Q, D).
        k: Number of nearest neighbors to retrieve.
        batch_size: Batch size for computing pairwise distances to avoid excessive memory usage.

    Returns:
        gt_indices: Exact k-NN indices of shape (Q, k).
        gt_distances: Exact Euclidean distances to k-NN of shape (Q, k).
    """
    Q = X_query.shape[0]
    gt_indices = np.zeros((Q, k), dtype=np.int64)
    gt_distances = np.zeros((Q, k), dtype=np.float32)

    # Pre-compute squared norms of base vectors
    base_sq = np.sum(X_base**2, axis=1)  # (N,)

    for start_idx in range(0, Q, batch_size):
        end_idx = min(start_idx + batch_size, Q)
        query_batch = X_query[start_idx:end_idx]  # (B, D)
        query_sq = np.sum(query_batch**2, axis=1, keepdims=True)  # (B, 1)

        # Dist^2 = ||base||^2 + ||query||^2 - 2 * <query, base>
        dists_sq = query_sq + base_sq - 2.0 * np.matmul(query_batch, X_base.T)
        dists_sq = np.maximum(dists_sq, 0.0)

        # Retrieve top-k smallest distances
        if k < dists_sq.shape[1]:
            partition_idx = np.argpartition(dists_sq, k, axis=1)[:, :k]
            # Sort the partitioned indices exactly
            batch_dists = np.take_along_axis(dists_sq, partition_idx, axis=1)
            sorted_order = np.argsort(batch_dists, axis=1)
            sorted_indices = np.take_along_axis(partition_idx, sorted_order, axis=1)
            sorted_dists = np.sqrt(np.take_along_axis(batch_dists, sorted_order, axis=1))
        else:
            sorted_indices = np.argsort(dists_sq, axis=1)
            sorted_dists = np.sqrt(np.take_along_axis(dists_sq, sorted_indices, axis=1))

        gt_indices[start_idx:end_idx] = sorted_indices
        gt_distances[start_idx:end_idx] = sorted_dists.astype(np.float32)

    return gt_indices, gt_distances
