import numpy as np
from scipy import sparse
import pytest

import osqp
from osqp import PenaltyType


def lifted_norminf_reference(P, q, A, l, u, alpha, solver_opts, algebra):
    """
    alpha*||xi||_inf is LP-representable: introduce t with -t <= xi_i <= t and
    charge alpha*t. Solving that hard QP gives a reference that shares no code
    with the penalty path.
    """
    n = P.shape[0]
    m = A.shape[0]
    nl = n + m + 1

    P_lift = sparse.csc_matrix(
        (P.toarray()[np.ix_(range(n), range(n))].flatten(), (np.repeat(range(n), n), np.tile(range(n), n))),
        shape=(nl, nl),
    )
    q_lift = np.concatenate([q, np.zeros(m), [alpha]])

    rows, l_lift, u_lift = [], [], []

    A_dense = A.toarray()
    for i in range(m):  # l <= A x + xi <= u
        e = np.zeros(nl)
        e[:n] = A_dense[i]
        e[n + i] = 1.0
        rows.append(e)
        l_lift.append(l[i])
        u_lift.append(u[i])

    for i in range(m):  # xi_i - t <= 0 and -xi_i - t <= 0
        e = np.zeros(nl)
        e[n + i] = 1.0
        e[nl - 1] = -1.0
        rows.append(e)
        l_lift.append(-np.inf)
        u_lift.append(0.0)

        e = np.zeros(nl)
        e[n + i] = -1.0
        e[nl - 1] = -1.0
        rows.append(e)
        l_lift.append(-np.inf)
        u_lift.append(0.0)

    ref = osqp.OSQP(algebra=algebra)
    ref.setup(
        P_lift,
        q_lift,
        sparse.csc_matrix(np.array(rows)),
        np.array(l_lift),
        np.array(u_lift),
        **solver_opts,
    )
    return ref.solve()


class TestGroupPenalty:
    @pytest.fixture(autouse=True)
    def setup(self, algebra, solver_type, atol, rtol, decimal_tol):
        self.algebra = algebra
        self.n = 4
        self.m = 4

        self.P = sparse.identity(self.n, format='csc')
        self.q = np.array([-2.0, -2.5, -3.0, -3.5])
        self.A = sparse.identity(self.m, format='csc')
        self.l = -np.ones(self.m)
        self.u = np.ones(self.m)

        self.opts = {
            'verbose': False,
            'eps_abs': 1e-9,
            'eps_rel': 1e-9,
            'max_iter': 20000,
            'solver_type': solver_type,
        }
        self.decimal_tol = decimal_tol

    def _solve(self, **penalty):
        prob = osqp.OSQP(algebra=self.algebra)
        prob.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)
        prob.setup_penalty(**penalty)
        return prob.solve()

    def test_norminf_matches_epigraph_lift(self):
        alpha = 1.5

        res = self._solve(
            penalty_type=PenaltyType.OSQP_PENALTY_NORMINF,
            alpha1=alpha,
            group_id=np.zeros(self.m, dtype=np.int64),
        )
        ref = lifted_norminf_reference(
            self.P, self.q, self.A, self.l, self.u, alpha, self.opts, self.algebra
        )

        np.testing.assert_array_almost_equal(res.x, ref.x[: self.n], decimal=self.decimal_tol)
        np.testing.assert_almost_equal(res.info.obj_val, ref.info.obj_val, decimal=self.decimal_tol)

    def test_group_spellings_agree(self):
        """A label array, a slice and a list of index arrays name the same group."""
        common = dict(penalty_type=PenaltyType.OSQP_PENALTY_NORM2, alpha1=1.2)

        by_labels = self._solve(group_id=np.zeros(self.m, dtype=np.int64), **common)
        by_slice = self._solve(group_id=slice(0, self.m), **common)
        by_indices = self._solve(group_id=[list(range(self.m))], **common)

        np.testing.assert_array_almost_equal(by_labels.x, by_slice.x, decimal=self.decimal_tol)
        np.testing.assert_array_almost_equal(by_labels.x, by_indices.x, decimal=self.decimal_tol)

    def test_group_of_one_is_a_plain_l1_row(self):
        """Both norms of a scalar are the absolute value."""
        alpha = 0.8

        l1 = self._solve(penalty_type=PenaltyType.OSQP_PENALTY_L1L2, alpha1=alpha, alpha2=0.0)

        for norm in (PenaltyType.OSQP_PENALTY_NORM2, PenaltyType.OSQP_PENALTY_NORMINF):
            grouped = self._solve(
                penalty_type=np.full(self.m, norm, dtype=np.int64),
                alpha1=alpha,
                group_id=np.arange(self.m, dtype=np.int64),
            )
            np.testing.assert_array_almost_equal(grouped.x, l1.x, decimal=self.decimal_tol)

    def test_norm2_dual_lies_in_its_ball(self):
        """The conjugate of alpha*||.||_2 is the indicator of ||y||_2 <= alpha."""
        alpha = 1.5

        res = self._solve(
            penalty_type=PenaltyType.OSQP_PENALTY_NORM2,
            alpha1=alpha,
            group_id=np.zeros(self.m, dtype=np.int64),
        )

        assert np.linalg.norm(res.y, 2) <= alpha + 1e-6

    def test_ungrouped_norm_is_rejected(self):
        with pytest.raises(osqp.OSQPException):
            self._solve(penalty_type=PenaltyType.OSQP_PENALTY_NORM2, alpha1=1.0)

    def test_group_rows_must_agree_on_the_weight(self):
        with pytest.raises(osqp.OSQPException):
            self._solve(
                penalty_type=PenaltyType.OSQP_PENALTY_NORMINF,
                alpha1=np.array([1.0, 2.0, 1.0, 1.0]),
                group_id=np.zeros(self.m, dtype=np.int64),
            )

    def test_empty_group_is_rejected(self):
        with pytest.raises(osqp.OSQPException):
            # Labels 0 and 2 used, so group 1 is empty
            self._solve(
                penalty_type=PenaltyType.OSQP_PENALTY_NORMINF,
                alpha1=1.0,
                group_id=np.array([0, 0, 2, 2], dtype=np.int64),
            )
