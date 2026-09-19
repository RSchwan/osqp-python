import sys
import shutil as sh
import unittest

import numpy as np
import numpy.testing as nptest
import pytest
from scipy import sparse

import osqp
from osqp import PenaltyType


@pytest.mark.skipif(not osqp.algebra_available('builtin'), reason='Builtin Algebra not available')
class codegen_penalty_tests(unittest.TestCase):
    """
    The penalty is problem data, so it is baked into the generated workspace;
    the generated solver has no penalty API of its own.
    """

    @classmethod
    def setUpClass(cls):
        cls.P = sparse.diags([11.0, 4.0], format='csc')
        cls.q = np.array([3.0, 4.0])
        cls.A = sparse.csc_matrix([[-1.0, 0.0], [0.0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]])
        cls.u = np.array([0.0, 0.0, -15.0, 100.0, 80.0])
        cls.l = -np.inf * np.ones(len(cls.u))
        cls.opts = {
            'verbose': False,
            'eps_abs': 1e-08,
            'eps_rel': 1e-08,
            'max_iter': 10000,
            'warm_starting': True,
        }

        model = osqp.OSQP(algebra='builtin')
        if not model.has_capability('OSQP_CAPABILITY_CODEGEN'):
            pytest.skip('No codegen capability')
        model.setup(P=cls.P, q=cls.q, A=cls.A, l=cls.l, u=cls.u, **cls.opts)
        model.setup_penalty(PenaltyType.OSQP_PENALTY_L1L2, alpha1=0.5, alpha2=2.0)
        cls.res = model.solve(raise_error=True)

        model_dir = model.codegen(
            'codegen_penalty_out',
            extension_name='penalty_emosqp',
            include_codegen_src=True,
            force_rewrite=True,
            compile=True,
        )
        sys.path.append(model_dir)

    @classmethod
    def tearDownClass(cls):
        sh.rmtree('codegen_penalty_out', ignore_errors=True)

    def test_solve(self):
        import penalty_emosqp

        x, y, _, _, _ = penalty_emosqp.solve()

        nptest.assert_array_almost_equal(x, self.res.x, decimal=5)
        nptest.assert_array_almost_equal(y, self.res.y, decimal=5)

    def test_update_bounds(self):
        import penalty_emosqp

        u_new = self.u + 1.0
        penalty_emosqp.update_data_vec(u=u_new)
        x, y, _, _, _ = penalty_emosqp.solve()

        model = osqp.OSQP(algebra='builtin')
        model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=u_new, **self.opts)
        model.setup_penalty(PenaltyType.OSQP_PENALTY_L1L2, alpha1=0.5, alpha2=2.0)
        res = model.solve(raise_error=True)

        nptest.assert_array_almost_equal(x, res.x, decimal=5)
        nptest.assert_array_almost_equal(y, res.y, decimal=5)

        penalty_emosqp.update_data_vec(u=self.u)
