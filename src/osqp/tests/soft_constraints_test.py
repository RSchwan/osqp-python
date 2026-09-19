from types import SimpleNamespace

import numpy as np
import numpy.testing as nptest
import pytest
from scipy import sparse

import osqp
from osqp import PenaltyType


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    self = SimpleNamespace()

    np.random.seed(2)

    self.n = 10
    self.m = 15
    P = sparse.random(self.n, self.n, density=0.4, format='csc')
    self.P = (P @ P.T + 0.1 * sparse.eye(self.n)).tocsc()
    self.q = np.random.randn(self.n)
    self.A = sparse.random(self.m, self.n, density=0.6, format='csc')
    # Bounds that the unconstrained minimizer violates, so softening matters
    self.l = -np.random.rand(self.m)
    self.u = np.random.rand(self.m)

    self.opts = {
        'verbose': False,
        'eps_abs': 1e-09,
        'eps_rel': 1e-09,
        'max_iter': 20000,
        'polishing': False,
        'solver_type': solver_type,
    }

    self.algebra = algebra
    self.rtol = rtol
    self.atol = atol
    self.decimal_tol = decimal_tol

    return self


def solve_qp(self, P, q, A, l, u, **kwargs):
    model = osqp.OSQP(algebra=self.algebra)
    model.setup(P=P, q=q, A=A, l=l, u=u, **{**self.opts, **kwargs})
    return model, model.solve(raise_error=True)


def soft_model(self, penalty_type, **weights):
    model = osqp.OSQP(algebra=self.algebra)
    model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    model.setup_penalty(penalty_type, **weights)
    return model


def l1l2_lifted(self, alpha1, alpha2):
    """
    The L1L2-softened problem written out with an explicit slack:
      variables [x; xi; t],  t >= |xi|,  l <= Ax + xi <= u
    """
    n, m = self.n, self.m
    alpha1 = np.broadcast_to(np.asarray(alpha1, dtype=float), (m,))
    alpha2 = np.broadcast_to(np.asarray(alpha2, dtype=float), (m,))

    P = sparse.block_diag([self.P, sparse.diags(alpha2), sparse.csc_matrix((m, m))], format='csc')
    q = np.hstack([self.q, np.zeros(m), alpha1])
    A = sparse.bmat(
        [
            [self.A, sparse.eye(m), None],
            [None, sparse.eye(m), -sparse.eye(m)],
            [None, sparse.eye(m), sparse.eye(m)],
        ],
        format='csc',
    )
    l = np.hstack([self.l, -np.inf * np.ones(m), np.zeros(m)])
    u = np.hstack([self.u, np.zeros(m), np.inf * np.ones(m)])

    _, res = solve_qp(self, P, q, A, l, u)
    return res.x[:n], res.info.obj_val


def huber_lifted(self, alpha1, delta):
    """
    The Huber-softened problem with the slack split as xi = v + w, |w| <= t,
    using h_delta(s) = min_{v+w=s} 0.5 v^2 + delta |w|
    """
    n, m = self.n, self.m
    alpha1 = np.broadcast_to(np.asarray(alpha1, dtype=float), (m,))
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (m,))

    P = sparse.block_diag([self.P, sparse.diags(alpha1), sparse.csc_matrix((2 * m, 2 * m))], format='csc')
    q = np.hstack([self.q, np.zeros(2 * m), alpha1 * delta])
    A = sparse.bmat(
        [
            [self.A, sparse.eye(m), sparse.eye(m), None],
            [None, None, sparse.eye(m), -sparse.eye(m)],
            [None, None, sparse.eye(m), sparse.eye(m)],
        ],
        format='csc',
    )
    l = np.hstack([self.l, -np.inf * np.ones(m), np.zeros(m)])
    u = np.hstack([self.u, np.zeros(m), np.inf * np.ones(m)])

    _, res = solve_qp(self, P, q, A, l, u)
    return res.x[:n], res.info.obj_val


def objective(self, x):
    return 0.5 * x @ self.P @ x + self.q @ x


def test_penalty_val_zero_without_penalty(self):
    _, res = solve_qp(self, self.P, self.q, self.A, self.l, self.u)
    assert res.info.penalty_val == 0.0


def test_quadratic_penalty(self):
    alpha2 = 0.7
    model = soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha2=alpha2)
    res = model.solve(raise_error=True)

    x_lifted, obj_lifted = l1l2_lifted(self, alpha1=0.0, alpha2=alpha2)
    nptest.assert_allclose(res.x, x_lifted, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.info.obj_val, obj_lifted, rtol=self.rtol, atol=self.atol)

    # The penalty part of the objective is the penalty evaluated at the slack
    xi = model.slack(res.x)
    nptest.assert_allclose(res.info.penalty_val, 0.5 * alpha2 * xi @ xi, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(
        res.info.obj_val - res.info.penalty_val, objective(self, res.x), rtol=self.rtol, atol=self.atol
    )


def test_l1_penalty(self):
    alpha1 = 0.3
    model = soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha1=alpha1)
    res = model.solve(raise_error=True)

    x_lifted, obj_lifted = l1l2_lifted(self, alpha1=alpha1, alpha2=0.0)
    nptest.assert_allclose(res.x, x_lifted, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.info.obj_val, obj_lifted, rtol=self.rtol, atol=self.atol)

    # The exact penalty clamps the dual variable to the L1 weight
    assert np.all(np.abs(res.y) <= alpha1 + self.atol)


def test_elastic_net_penalty(self):
    alpha1, alpha2 = 0.2, 1.5
    model = soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha1=alpha1, alpha2=alpha2)
    res = model.solve(raise_error=True)

    x_lifted, obj_lifted = l1l2_lifted(self, alpha1=alpha1, alpha2=alpha2)
    nptest.assert_allclose(res.x, x_lifted, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.info.obj_val, obj_lifted, rtol=self.rtol, atol=self.atol)


def test_huber_penalty(self):
    alpha1, delta = 2.0, 0.1
    model = soft_model(self, PenaltyType.OSQP_PENALTY_HUBER, alpha1=alpha1, delta=delta)
    res = model.solve(raise_error=True)

    x_lifted, obj_lifted = huber_lifted(self, alpha1=alpha1, delta=delta)
    nptest.assert_allclose(res.x, x_lifted, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.info.obj_val, obj_lifted, rtol=self.rtol, atol=self.atol)


def test_per_row_types_and_weights(self):
    types = np.full(self.m, PenaltyType.OSQP_PENALTY_NONE)
    types[::2] = PenaltyType.OSQP_PENALTY_L1L2
    alpha2 = np.linspace(0.5, 2.0, self.m)

    model = soft_model(self, types, alpha2=alpha2)
    res = model.solve(raise_error=True)

    # Rows left hard are satisfied; softened rows are free to be violated
    xi = model.slack(res.x)
    nptest.assert_allclose(xi[1::2], 0.0, atol=self.atol)

    # Equivalent lift: a hard row is alpha1 -> "infinite", so lift only the soft rows
    n, m = self.n, self.m
    soft = np.flatnonzero(types == PenaltyType.OSQP_PENALTY_L1L2)
    S = sparse.csc_matrix((np.ones(len(soft)), (soft, np.arange(len(soft)))), shape=(m, len(soft)))
    P = sparse.block_diag([self.P, sparse.diags(alpha2[soft])], format='csc')
    q = np.hstack([self.q, np.zeros(len(soft))])
    A = sparse.hstack([self.A, S], format='csc')
    _, res_lifted = solve_qp(self, P, q, A, self.l, self.u)

    nptest.assert_allclose(res.x, res_lifted.x[:n], rtol=self.rtol, atol=self.atol)


def test_hard_rows_match_unsoftened(self):
    _, res_hard = solve_qp(self, self.P, self.q, self.A, self.l, self.u)

    model = soft_model(self, PenaltyType.OSQP_PENALTY_NONE, alpha2=1.0)
    res = model.solve(raise_error=True)

    nptest.assert_allclose(res.x, res_hard.x, rtol=self.rtol, atol=self.atol)
    assert res.info.penalty_val == 0.0


def test_update_penalty_params(self):
    model = soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha2=1.0)
    model.solve(raise_error=True)

    model.update_penalty_params(alpha2=5.0)
    res = model.solve(raise_error=True)

    x_lifted, _ = l1l2_lifted(self, alpha1=0.0, alpha2=5.0)
    nptest.assert_allclose(res.x, x_lifted, rtol=self.rtol, atol=self.atol)

    # A weight left unspecified is unchanged
    model.update_penalty_params(alpha1=0.4)
    res = model.solve(raise_error=True)

    x_lifted, _ = l1l2_lifted(self, alpha1=0.4, alpha2=5.0)
    nptest.assert_allclose(res.x, x_lifted, rtol=self.rtol, atol=self.atol)


def test_update_penalty_types(self):
    model = soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha2=1.0)
    model.solve(raise_error=True)

    # Hardening every row recovers the original problem
    model.update_penalty_types(PenaltyType.OSQP_PENALTY_NONE)
    res = model.solve(raise_error=True)

    _, res_hard = solve_qp(self, self.P, self.q, self.A, self.l, self.u)
    nptest.assert_allclose(res.x, res_hard.x, rtol=self.rtol, atol=self.atol)

    types = np.full(self.m, PenaltyType.OSQP_PENALTY_NONE)
    types[0] = PenaltyType.OSQP_PENALTY_L1L2
    model.update_penalty_types(types)
    res = model.solve(raise_error=True)

    xi = model.slack(res.x)
    nptest.assert_allclose(xi[1:], 0.0, atol=self.atol)


def test_scaling_is_transparent(self):
    alpha1, delta = 1.0, 0.25

    results = []
    for scaling in (0, 10):
        model = osqp.OSQP(algebra=self.algebra)
        model.setup(
            P=self.P,
            q=self.q,
            A=self.A,
            l=self.l,
            u=self.u,
            **{**self.opts, 'scaling': scaling},
        )
        model.setup_penalty(PenaltyType.OSQP_PENALTY_HUBER, alpha1=alpha1, delta=delta)
        results.append(model.solve(raise_error=True))

    nptest.assert_allclose(results[0].x, results[1].x, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(results[0].info.obj_val, results[1].info.obj_val, rtol=self.rtol, atol=self.atol)


def test_polishing(self):
    alpha1 = 0.3
    model = osqp.OSQP(algebra=self.algebra)
    model.setup(
        P=self.P,
        q=self.q,
        A=self.A,
        l=self.l,
        u=self.u,
        **{**self.opts, 'polishing': True},
    )
    model.setup_penalty(PenaltyType.OSQP_PENALTY_L1L2, alpha1=alpha1)
    res = model.solve(raise_error=True)

    x_lifted, obj_lifted = l1l2_lifted(self, alpha1=alpha1, alpha2=0.0)
    nptest.assert_allclose(res.x, x_lifted, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.info.obj_val, obj_lifted, rtol=self.rtol, atol=self.atol)


def test_invalid_weights(self):
    with pytest.raises(osqp.OSQPException):
        soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha2=-1.0)

    with pytest.raises(osqp.OSQPException):
        soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha1=np.inf)

    # Huber requires a positive weight and a positive transition point
    with pytest.raises(osqp.OSQPException):
        soft_model(self, PenaltyType.OSQP_PENALTY_HUBER, alpha1=1.0)

    with pytest.raises(osqp.OSQPException):
        soft_model(self, PenaltyType.OSQP_PENALTY_HUBER, delta=1.0)


def test_invalid_type(self):
    with pytest.raises(osqp.OSQPException):
        soft_model(self, 42, alpha2=1.0)

    with pytest.raises(ValueError):
        soft_model(self, np.zeros(self.m + 1), alpha2=1.0)


def test_weight_dimension_mismatch(self):
    with pytest.raises(ValueError):
        soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha2=np.ones(self.m + 1))


def test_noncontiguous_types_and_weights(self):
    probe = osqp.OSQP(algebra=self.algebra)

    types = np.full(self.m, PenaltyType.OSQP_PENALTY_NONE, dtype=probe._itype)
    types[::2] = PenaltyType.OSQP_PENALTY_L1L2
    type_storage = np.empty(2 * self.m, dtype=probe._itype)
    type_storage[::2] = types
    type_storage[1::2] = PenaltyType.OSQP_PENALTY_L1L2

    alpha2 = np.linspace(0.5, 2.0, self.m, dtype=probe._dtype)
    alpha2_storage = np.empty(2 * self.m, dtype=probe._dtype)
    alpha2_storage[::2] = alpha2
    alpha2_storage[1::2] = 100.0

    assert not type_storage[::2].flags.c_contiguous
    assert not alpha2_storage[::2].flags.c_contiguous

    reference = soft_model(self, types.copy(), alpha2=alpha2.copy()).solve(raise_error=True)
    result = soft_model(self, type_storage[::2], alpha2=alpha2_storage[::2]).solve(raise_error=True)

    nptest.assert_allclose(result.x, reference.x, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(result.info.obj_val, reference.info.obj_val, rtol=self.rtol, atol=self.atol)


def test_penalty_setup_once(self):
    model = soft_model(self, PenaltyType.OSQP_PENALTY_L1L2, alpha2=1.0)
    with pytest.raises(osqp.OSQPException):
        model.setup_penalty(PenaltyType.OSQP_PENALTY_L1L2, alpha2=1.0)


def test_update_requires_setup(self):
    model = osqp.OSQP(algebra=self.algebra)
    model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    with pytest.raises(osqp.OSQPException):
        model.update_penalty_params(alpha2=1.0)

    with pytest.raises(osqp.OSQPException):
        model.update_penalty_types(PenaltyType.OSQP_PENALTY_L1L2)
