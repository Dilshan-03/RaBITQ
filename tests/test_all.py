import unittest
import numpy as np

from dataset.preprocessor import VectorPreprocessor
from dataset.loader import load_synthetic_data, compute_ground_truth
from core.transform import RandomOrthogonalTransform
from core.baseline_fp32 import FP32FlatIndex
from core.baseline_binary import BinaryFlatIndex
from core.rabitq_1bit import RaBitQ1Bit
from core.rabitq_multibit import RaBitQMultiBit
from core.adaptive_quantizer import ABVQuantizer
from index.reranker import ErrorBoundReRanker
from index.ivf_index import IVFIndex
from evaluation.metrics import compute_recall_at_k, compute_distance_errors


class TestABVQuantSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dim = 128
        cls.num_vectors = 1000
        cls.num_queries = 50
        cls.k = 10
        cls.X_base, cls.X_query = load_synthetic_data(
            num_vectors=cls.num_vectors,
            num_queries=cls.num_queries,
            dim=cls.dim,
            distribution="gaussian",
            seed=42,
        )
        cls.gt_indices, cls.gt_distances = compute_ground_truth(
            cls.X_base, cls.X_query, k=cls.k
        )

    def test_orthogonal_transform(self):
        """Verify orthogonal matrix satisfies P^T P = I."""
        transform = RandomOrthogonalTransform(dim=self.dim, seed=42)
        P = transform.matrix
        identity = np.eye(self.dim)
        diff = np.matmul(P.T, P) - identity
        self.assertLess(np.max(np.abs(diff)), 1e-5)

        # Test forward and inverse
        x = np.random.randn(5, self.dim).astype(np.float32)
        x_rot = transform.forward(x)
        x_rec = transform.inverse(x_rot)
        self.assertLess(np.max(np.abs(x - x_rec)), 1e-5)

    def test_preprocessor(self):
        """Verify hypersphere normalization and norm reconstruction."""
        prep = VectorPreprocessor()
        normed, norms = prep.fit_transform(self.X_base)
        # Verify unit norm
        vector_norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(vector_norms, 1.0, atol=1e-5)

    def test_unbiased_estimation_rabitq_1bit(self):
        """Verify RaBitQ 1-bit produces unbiased distance estimates (mean signed error approx 0)."""
        rab1 = RaBitQ1Bit(dim=self.dim, query_bits=4, seed=42)
        rab1.fit(self.X_base)
        pred_indices, pred_dists, _ = rab1.search(self.X_query, k=self.k)

        errors = compute_distance_errors(pred_dists, self.gt_distances[:, :self.k])
        signed_rel_err = errors["signed_rel_err"]
        print(f"\n[Test] RaBitQ 1-bit Signed Relative Error: {signed_rel_err:+.4e}")
        self.assertLess(abs(signed_rel_err), 0.05, "RaBitQ 1-bit estimator bias exceeds tolerance.")

    def test_budget_compliance_abv_quant(self):
        """Verify ABV-Quant strictly respects the bit budget: sum(b_i) <= D * B_target."""
        for target_bpd in [1.5, 2.0, 3.0]:
            abv = ABVQuantizer(dim=self.dim, target_bpd=target_bpd, seed=42)
            abv.fit(self.X_base)
            max_budget = int(np.floor(self.dim * target_bpd))
            actual_bits = abv.total_allocated_bits
            self.assertLessEqual(
                actual_bits,
                max_budget,
                f"ABV-Quant allocated {actual_bits} bits exceeding budget {max_budget}",
            )
            # Verify valid bit widths
            for b in abv.bit_allocation:
                self.assertIn(b, [1, 2, 4, 8])

    def test_abv_quant_superiority(self):
        """Verify ABV-Quant achieves higher Recall than Uniform RaBitQ at identical target budget."""
        target_bpd = 2.0
        # Uniform 2-bit
        uni_2bit = RaBitQMultiBit(dim=self.dim, bit_width=2, seed=42)
        uni_2bit.fit(self.X_base)
        pred_idx_uni, _, _ = uni_2bit.search(self.X_query, k=self.k)
        rec_uni = compute_recall_at_k(pred_idx_uni, self.gt_indices, k=self.k)

        # ABV-Quant 2.0 BPD
        abv = ABVQuantizer(dim=self.dim, target_bpd=target_bpd, seed=42)
        abv.fit(self.X_base)
        pred_idx_abv, _, _ = abv.search(self.X_query, k=self.k)
        rec_abv = compute_recall_at_k(pred_idx_abv, self.gt_indices, k=self.k)

        print(f"\n[Test] Recall@{self.k} - ABV-Quant: {rec_abv:.4f} vs Uniform: {rec_uni:.4f}")
        self.assertGreaterEqual(
            rec_abv,
            rec_uni,
            "ABV-Quant should match or outperform Uniform RaBitQ at matching bit budgets.",
        )

    def test_error_bound_reranker(self):
        """Verify ErrorBoundReRanker successfully prunes and re-ranks."""
        reranker = ErrorBoundReRanker(dim=self.dim, epsilon_0=1.9)
        rab1 = RaBitQ1Bit(dim=self.dim, seed=42)
        rab1.fit(self.X_base)

        q_raw = self.X_query[0]
        q_norm, q_norm_val = rab1.preprocessor.transform(q_raw.reshape(1, -1))
        est_ips = rab1.estimate_inner_products(q_norm[0])

        cand_indices = np.arange(min(100, self.num_vectors))
        cand_ips = est_ips[cand_indices]
        cand_oo = rab1.inner_products_oo[cand_indices]
        cand_norms = rab1.data_norms[cand_indices]

        top_indices, top_dists, survivors = reranker.prune_and_rerank(
            candidate_indices=cand_indices,
            candidate_est_ips=cand_ips,
            candidate_oo=cand_oo,
            candidate_norms=cand_norms,
            query_norm=float(q_norm_val[0]),
            query_raw=q_raw,
            X_base_raw=self.X_base,
            k=self.k,
        )
        self.assertEqual(len(top_indices), self.k)
        self.assertGreaterEqual(survivors, self.k)


if __name__ == "__main__":
    unittest.main()
