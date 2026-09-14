import numpy as np
from typing import Optional, Union


class RandomOrthogonalTransform:
    """Random Orthogonal Transformation via Johnson-Lindenstrauss Theorem.

    Generates an orthogonal matrix P in R^{D x D} sampled uniformly from the
    Haar measure on the orthogonal group O(D) via QR decomposition of a standard
    Gaussian matrix:
        G = Q * R
        P = Q * diag(sign(diag(R)))
    """

    def __init__(self, dim: int, seed: Optional[int] = 42):
        self.dim = dim
        self.seed = seed
        self.P: np.ndarray = self._generate_orthogonal_matrix()

    def _generate_orthogonal_matrix(self) -> np.ndarray:
        """Sample a uniform random orthogonal matrix from O(D)."""
        rng = np.random.default_rng(self.seed)
        # Standard Gaussian random matrix
        G = rng.standard_normal((self.dim, self.dim), dtype=np.float64)
        Q, R = np.linalg.qr(G)
        
        # Ensure uniform Haar measure distribution
        d = np.diagonal(R)
        ph = d / np.abs(d)
        P = Q * ph
        return P.astype(np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply orthogonal transformation: X' = X @ P.

        Args:
            X: Input vectors of shape (N, D) or (D,).

        Returns:
            Rotated vectors of matching shape.
        """
        if X.ndim == 1:
            return np.dot(X, self.P).astype(np.float32)
        elif X.ndim == 2:
            return np.matmul(X, self.P).astype(np.float32)
        else:
            raise ValueError(f"Expected 1D or 2D array, got ndim={X.ndim}")

    def inverse(self, X: np.ndarray) -> np.ndarray:
        """Apply inverse orthogonal transformation: X = X' @ P^T.

        Args:
            X: Rotated vectors of shape (N, D) or (D,).

        Returns:
            Inversely rotated vectors of matching shape.
        """
        if X.ndim == 1:
            return np.dot(X, self.P.T).astype(np.float32)
        elif X.ndim == 2:
            return np.matmul(X, self.P.T).astype(np.float32)
        else:
            raise ValueError(f"Expected 1D or 2D array, got ndim={X.ndim}")

    @property
    def matrix(self) -> np.ndarray:
        """Return the D x D orthogonal matrix P."""
        return self.P
