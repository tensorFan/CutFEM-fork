"""Independent checks of analytic ball data and rotation-invariant diagnostics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

MODULE_PATH = Path(__file__).resolve().parents[1]/"mainFiles"/"poisson3D_eigen_analyze.py"
SPEC = importlib.util.spec_from_file_location("poisson_eigen", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class PoissonEigenAnalysisTests(unittest.TestCase):
    def test_ball_roots_multiplicities_and_radius(self):
        values = analysis.ball_spectrum(10, 1.)
        self.assertAlmostEqual(values[0], np.pi**2, places=11)
        self.assertAlmostEqual(values[9], 4*np.pi**2, places=11)
        np.testing.assert_allclose(values[1:4], 20.19072855642663, rtol=1e-12)
        np.testing.assert_allclose(values[4:9], 33.21746191426837, rtol=1e-12)
        np.testing.assert_allclose(analysis.ball_spectrum(10, 2.), values/4., rtol=1e-13)

    def test_eigenspace_ignores_rotation_scaling_and_sign(self):
        mass = sparse.diags([2., 3., 5., 7.])
        exact = np.diag(1./np.sqrt([2., 3., 5., 7.]))[:, :2]
        # Deliberately choose a nonorthogonal basis of the same exact space.
        exact = exact @ np.array([[2., 0.3], [-0.2, 1.]])
        gram = exact.T @ mass @ exact
        vectors = exact @ np.array([[-3., 2.], [1., 5.]])
        _, gap, _ = analysis.subspace_metrics(vectors, mass, mass @ exact, gram)
        self.assertLess(gap, 5e-8)

    def test_wrong_eigenspace_is_detected(self):
        mass = sparse.eye(4)
        exact = np.eye(4)[:, :2]
        wrong = np.eye(4)[:, [0, 2]]
        cosine, gap, angle = analysis.subspace_metrics(wrong, mass, exact, np.eye(2))
        self.assertAlmostEqual(cosine, 0.)
        self.assertAlmostEqual(gap, 1.)
        self.assertAlmostEqual(angle, 90.)

    def test_coercivity_audit_rejects_indefinite_stiffness(self):
        matrix = sparse.csc_matrix([[1., 2.], [2., 1.]])
        with self.assertRaisesRegex(ValueError, "nonpositive LDL pivots"):
            analysis.positive_factor(matrix, "A")
        analysis.positive_factor(sparse.csc_matrix([[3., 2.], [2., 3.]]), "A")


if __name__ == "__main__":
    unittest.main()
