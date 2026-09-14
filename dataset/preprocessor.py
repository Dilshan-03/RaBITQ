import numpy as np
from typing import Tuple, Optional


class VectorPreprocessor:
    """Preprocessor for hypersphere normalization and centroid alignment.

    Implements the RaBitQ distance decomposition:
        ||o_r - q_r||^2 = ||o_r - c||^2 + ||q_r - c||^2 - 2 * ||o_r - c|| * ||q_r - c|| * <o, q>
    where:
        o = (o_r - c) / ||o_r - c||_2
        q = (q_r - c) / ||q_r - c||_2
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.centroid: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "VectorPreprocessor":
        """Compute the dataset centroid c = mean(X, axis=0)."""
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        self.centroid = np.mean(X, axis=0, dtype=np.float64).astype(np.float32)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize vectors onto the unit hypersphere after centroid subtraction.

        Args:
            X: Input vectors of shape (N, D).

        Returns:
            normalized_X: Unit vectors on hypersphere of shape (N, D).
            norms: Raw distance from centroid ||X - c|| of shape (N,).
        """
        if not self.is_fitted or self.centroid is None:
            raise RuntimeError("VectorPreprocessor must be fitted before transforming.")

        diff = X - self.centroid
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.maximum(norms, self.eps)
        normalized_X = (diff / safe_norms).astype(np.float32)
        return normalized_X, norms.squeeze(axis=-1).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit centroid and transform vectors."""
        return self.fit(X).transform(X)

    def reconstruct_distance_sq(
        self,
        estimated_inner_products: np.ndarray,
        data_norms: np.ndarray,
        query_norm: float,
    ) -> np.ndarray:
        """Reconstruct squared Euclidean distances from estimated inner products.

        Args:
            estimated_inner_products: Estimated <o, q> of shape (N,).
            data_norms: Pre-computed ||o_r - c|| of shape (N,).
            query_norm: Pre-computed ||q_r - c|| (scalar float).

        Returns:
            dist_sq: Estimated squared Euclidean distances of shape (N,).
        """
        dist_sq = (
            data_norms**2
            + query_norm**2
            - 2.0 * data_norms * query_norm * estimated_inner_products
        )
        return np.maximum(dist_sq, 0.0)

    def reconstruct_distance(
        self,
        estimated_inner_products: np.ndarray,
        data_norms: np.ndarray,
        query_norm: float,
    ) -> np.ndarray:
        """Reconstruct Euclidean distances from estimated inner products."""
        return np.sqrt(
            self.reconstruct_distance_sq(
                estimated_inner_products, data_norms, query_norm
            )
        )
