"""
Tests for AdaptiveMetropolisHastings.

These are split into three tiers:
  1. Unit tests on `_update_empir_cov` in isolation (no Pyro model needed).
  2. A formula test for the running acceptance-probability average.
  3. An integration test against a known Gaussian target, which exercises
     the full accept/reject loop and validates the sign convention on
     `target_ratio`.

Tier 1 and 2 construct a bare instance and hand-set only the attributes
`_update_empir_cov` / `sample` actually touch, so you don't need a real
Pyro model to run them. Adjust the import path below to match your module.
"""

import math
import torch
import pytest

from models.local_stats import AdaptiveMetropolisHastings  # <-- fix this import


def make_bare_kernel(dim, sd_scale=1.0, epsilon=1e-6, adapt_start=0):
    """Construct an instance without going through setup()/initialize_model."""
    kernel = AdaptiveMetropolisHastings.__new__(AdaptiveMetropolisHastings)
    kernel._dim = dim
    kernel.sd_scale = sd_scale
    kernel.epsilon = epsilon
    kernel.adapt_start = adapt_start
    kernel._t = 0
    kernel._accept_cnt = 0
    kernel._mean_accept_prob = 0.0
    kernel.c0 = torch.eye(dim, dtype=torch.float32)
    kernel.emp_cov = kernel.c0.clone()
    kernel._mean = torch.zeros(dim, dtype=torch.float32)
    kernel._scatter = torch.zeros(dim, dim, dtype=torch.float32)
    kernel._energy_last = 0.0
    return kernel


# ---------------------------------------------------------------------------
# Tier 1: _update_empir_cov correctness vs. direct batch computation
# ---------------------------------------------------------------------------

def batch_mean_cov(samples, sd_scale, epsilon, dim):
    """Directly compute what the running recursion is supposed to converge to."""
    X = torch.stack(samples)  # (t, dim)
    mean = X.mean(dim=0)
    centered = X - mean
    cov = sd_scale * (centered.t() @ centered) / X.shape[0]
    cov = cov + epsilon * torch.eye(dim, dtype=torch.float32)
    return mean, cov


def test_running_mean_matches_batch_mean():
    torch.manual_seed(0)
    dim = 3
    kernel = make_bare_kernel(dim)
    samples = [torch.randn(dim) for _ in range(20)]
    kernel._mean = samples[0].clone()

    for t, x in enumerate(samples[1:], start=2):
        kernel._t = t
        kernel._update_empir_cov(x)

    expected_mean = torch.stack(samples).mean(dim=0)
    assert torch.allclose(kernel._mean, expected_mean, atol=1e-5)


def test_running_cov_matches_batch_cov():
    """
    This is the test that catches a missing X_t @ X_t.T term: without it,
    the recursive covariance will NOT converge to the batch covariance
    of the same sample sequence.
    """
    torch.manual_seed(1)
    dim = 3
    sd_scale = 1.0
    epsilon = 1e-6
    kernel = make_bare_kernel(dim, sd_scale=sd_scale, epsilon=epsilon)

    samples = [torch.randn(dim) for _ in range(500)]  # long chain to wash out epsilon
    kernel._mean = samples[0].clone()
    kernel.emp_cov = torch.zeros(dim, dim, dtype=torch.float32)

    for t, x in enumerate(samples[1:], start=2):
        kernel._t = t
        kernel._update_empir_cov(x)

    _, expected_cov = batch_mean_cov(samples, sd_scale, epsilon, dim)
    assert torch.allclose(kernel.emp_cov, expected_cov, atol=1e-2), (
        f"recursive cov diverges from batch cov:\n{kernel.emp_cov}\nvs\n{expected_cov}"
    )


def test_emp_cov_is_symmetric_positive_definite():
    torch.manual_seed(2)
    dim = 4
    kernel = make_bare_kernel(dim)
    kernel._mean = torch.randn(dim)

    for t in range(2, 50):
        kernel._t = t
        kernel._update_empir_cov(torch.randn(dim))

    cov = kernel.emp_cov
    assert torch.allclose(cov, cov.t(), atol=1e-5), "covariance not symmetric"
    eigvals = torch.linalg.eigvalsh(cov)
    assert torch.all(eigvals > 0), f"covariance not PD, eigenvalues: {eigvals}"


# ---------------------------------------------------------------------------
# Tier 2: running acceptance-probability formula
# ---------------------------------------------------------------------------

def test_mean_accept_prob_matches_manual_average():
    torch.manual_seed(3)
    dim = 2
    kernel = make_bare_kernel(dim, adapt_start=1000)  # stay in c0 phase, keep it simple

    # monkeypatch a trivial potential_fn: standard normal, so acceptance is well-defined
    kernel.potential_fn = lambda p: 0.5 * (p["x"] ** 2).sum()
    kernel._site_names = ["x"]
    kernel._site_shapes = {"x": (dim,)}
    kernel._site_numels = {"x": dim}
    kernel._energy_last = kernel.potential_fn({"x": torch.zeros(dim)})

    params = {"x": torch.zeros(dim)}
    alphas = []
    n_steps = 25
    for _ in range(n_steps):
        params = kernel.sample(params)
        # after `sample`, kernel should have recorded the alpha used that step;
        # if it doesn't expose one, capture via return value of a small wrapper
        # instead. Adjust this line if `alpha` isn't stored on the instance.
        alphas.append(kernel._last_alpha)  # <-- add self._last_alpha = alpha in sample()

    expected = sum(alphas) / len(alphas)
    assert math.isclose(kernel._mean_accept_prob, expected, rel_tol=1e-4), (
        f"{kernel._mean_accept_prob} != {expected}"
    )


# ---------------------------------------------------------------------------
# Tier 3: integration test against a known Gaussian target
# ---------------------------------------------------------------------------

def test_recovers_known_gaussian_target():
    """
    Sets potential_fn to the negative log-density (up to a constant) of a
    known N(mu, Sigma) target, runs the full chain, and checks the empirical
    mean/covariance of accepted samples against the truth. This is the test
    that catches a flipped sign in `target_ratio`: with the wrong sign, the
    chain will systematically reject good moves and fail to concentrate near
    `mu`, or will accept everything indiscriminately (looks like an unbiased
    random walk instead of a valid MH chain).
    """
    torch.manual_seed(4)
    dim = 2
    true_mu = torch.tensor([2.0, -1.0])
    true_sigma = torch.tensor([[1.0, 0.3], [0.3, 0.5]])
    precision = torch.linalg.inv(true_sigma)

    def potential_fn(p):
        x = p["x"] - true_mu
        return 0.5 * (x @ precision @ x)

    kernel = make_bare_kernel(dim, sd_scale=2.4 ** 2 / dim, adapt_start=200)
    kernel.potential_fn = potential_fn
    kernel._site_names = ["x"]
    kernel._site_shapes = {"x": (dim,)}
    kernel._site_numels = {"x": dim}
    kernel._energy_last = potential_fn({"x": torch.zeros(dim)})
    kernel._mean = torch.zeros(dim)
    kernel.c0 = torch.eye(dim, dtype=torch.float32)

    params = {"x": torch.zeros(dim)}
    collected = []
    n_steps = 5000
    for _ in range(n_steps):
        params = kernel.sample(params)
        collected.append(params["x"].clone())

    burn = n_steps // 5
    samples = torch.stack(collected[burn:])
    est_mu = samples.mean(dim=0)
    est_sigma = torch.cov(samples.t())

    assert torch.allclose(est_mu, true_mu, atol=0.15), f"mean off: {est_mu} vs {true_mu}"
    assert torch.allclose(est_sigma, true_sigma, atol=0.2), f"cov off: {est_sigma} vs {true_sigma}"


    # test gibbs kernel:


from models.local_stats import GibbsKernel  
 
 
# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
 
def constant_lengthscale_fn(value=1.0):
    """Returns a lengthscale_fn that ignores X and returns a constant ell,
    shaped (N, 1) for 1D input or (N, d) for d-dim input depending on call site.
    """
    def fn(X):
        d = X.shape[1]
        return torch.full((X.shape[0], d), float(value), dtype=X.dtype)
    return fn
 
 
def rbf_reference(X, lengthscale, variance):
    """Ground truth ordinary RBF kernel (stationary lengthscale), used as a
    sanity check: when lengthscale_fn is constant, the Gibbs kernel should
    reduce exactly to an RBF kernel with that lengthscale."""
    diffs = X.unsqueeze(1) - X.unsqueeze(0)          # (N, N, d)
    sqdist = (diffs ** 2).sum(-1)                     # (N, N)
    return variance * torch.exp(-sqdist / (2 * lengthscale ** 2))
 
 
# ---------------------------------------------------------------------------
# 1. Variance must actually scale the kernel
# ---------------------------------------------------------------------------
 
class TestVarianceScaling:
    def test_diagonal_equals_variance(self):
        """K(x,x) should equal `variance` exactly, for any lengthscale fn.
        This is the test that catches the missing `K *= self.variance`.
        """
        variance = 1e-7
        X = torch.linspace(0, 1, 10).reshape(-1, 1)
        kernel = GibbsKernel(input_dim=1, lengthscale_fn=constant_lengthscale_fn(0.3),
                              variance=variance)
        K = kernel.forward(X)
        diag = torch.diag(K)
        assert torch.allclose(diag, torch.full_like(diag, variance), rtol=1e-5), (
            f"Expected diagonal == variance ({variance}), got {diag.max().item()}. "
            "This means `self.variance` is not being applied in forward()."
        )
 
    def test_diagonal_via_diag_flag_matches_full_matrix(self):
        """The diag=True fast path must agree with diag(full matrix)."""
        variance = 2.5
        X = torch.rand(15, 2)
        kernel = GibbsKernel(input_dim=2, lengthscale_fn=constant_lengthscale_fn(0.5),
                              variance=variance)
        K_full = kernel.forward(X)
        K_diag = kernel.forward(X, diag=True)
        assert torch.allclose(torch.diag(K_full), K_diag, rtol=1e-5)
 
    def test_variance_scales_linearly(self):
        """K should scale linearly in variance: K(2*var) == 2 * K(var)."""
        X = torch.rand(8, 1)
        lfn = constant_lengthscale_fn(0.4)
        k1 = GibbsKernel(input_dim=1, lengthscale_fn=lfn, variance=1.0)
        k2 = GibbsKernel(input_dim=1, lengthscale_fn=lfn, variance=2.0)
        K1 = k1.forward(X)
        K2 = k2.forward(X)
        assert torch.allclose(K2, 2 * K1, rtol=1e-5)
 
    def test_default_variance_is_one(self):
        """If variance=None, should default to 1.0 and diagonal should be 1."""
        X = torch.rand(6, 1)
        kernel = GibbsKernel(input_dim=1, lengthscale_fn=constant_lengthscale_fn(0.3))
        K = kernel.forward(X)
        assert torch.allclose(torch.diag(K), torch.ones(6), rtol=1e-5)
 
 
# ---------------------------------------------------------------------------
# 2. Reduces to ordinary RBF when lengthscale is constant
# ---------------------------------------------------------------------------
 
class TestReducesToRBF:
    def test_matches_rbf_1d(self):
        variance = 0.8
        lengthscale = 0.35
        X = torch.linspace(-2, 2, 12).reshape(-1, 1)
        kernel = GibbsKernel(input_dim=1, lengthscale_fn=constant_lengthscale_fn(lengthscale),
                              variance=variance)
        K = kernel.forward(X)
        K_ref = rbf_reference(X, lengthscale, variance)
        assert torch.allclose(K, K_ref, rtol=1e-4, atol=1e-6)
 
    def test_matches_rbf_2d(self):
        variance = 1.3
        lengthscale = 0.6
        X = torch.rand(10, 2)
        kernel = GibbsKernel(input_dim=2, lengthscale_fn=constant_lengthscale_fn(lengthscale),
                              variance=variance)
        K = kernel.forward(X)
        K_ref = rbf_reference(X, lengthscale, variance)
        assert torch.allclose(K, K_ref, rtol=1e-4, atol=1e-6)
 
 
# ---------------------------------------------------------------------------
# 3. Basic mathematical properties any valid kernel matrix must have
# ---------------------------------------------------------------------------
 
class TestKernelProperties:
    def test_symmetric(self):
        X = torch.rand(20, 2)
        kernel = GibbsKernel(input_dim=2, lengthscale_fn=constant_lengthscale_fn(0.4),
                              variance=0.5)
        K = kernel.forward(X)
        assert torch.allclose(K, K.T, atol=1e-8)
 
    def test_positive_semidefinite(self):
        X = torch.rand(25, 2)
        kernel = GibbsKernel(input_dim=2, lengthscale_fn=constant_lengthscale_fn(0.3),
                              variance=0.7)
        K = kernel.forward(X)
        eigvals = torch.linalg.eigvalsh(K)
        assert eigvals.min() > -1e-6, f"Smallest eigenvalue {eigvals.min()} < 0"
 
    def test_cholesky_succeeds_with_jitter(self):
        """Mirrors real usage: K + jitter should be Cholesky-decomposable,
        and the resulting L should reproduce K when L @ L.T is taken."""
        variance = 1e-7
        X = torch.rand(30, 2, dtype=torch.float64)
        kernel = GibbsKernel(input_dim=2, lengthscale_fn=constant_lengthscale_fn(0.3),
                              variance=variance)
        K = kernel.forward(X)
        jitter = 1e-6 * variance
        K_j = K + torch.eye(len(X), dtype=torch.float64) * jitter
        L = torch.linalg.cholesky(K_j).detach()
        assert torch.diag(L @ L.T).max() == pytest.approx(variance, rel=5e-2)
 
    def test_diagonal_is_constant_variance_regardless_of_lengthscale_variation(self):
        """K(x,x) must equal variance for every x, even when lengthscale
        varies across the input space (this is the defining property of the
        Gibbs kernel's normalization, and the thing most likely to silently
        break if someone 'simplifies' the prefactor term)."""
        variance = 3.0
        def varying_lengthscale_fn(X):
            # lengthscale grows with the first input dimension
            return (0.1 + X[:, 0:1].abs()).repeat(1, X.shape[1])
        X = torch.rand(20, 2)
        kernel = GibbsKernel(input_dim=2, lengthscale_fn=varying_lengthscale_fn,
                              variance=variance)
        K = kernel.forward(X)
        assert torch.allclose(torch.diag(K), torch.full((20,), variance), rtol=1e-4)
 
 
# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------
 
class TestInputValidation:
    def test_rejects_nonpositive_lengthscale_1d(self):
        bad_fn = lambda X: torch.zeros(X.shape[0], 1)
        kernel = GibbsKernel(input_dim=1, lengthscale_fn=bad_fn, variance=1.0)
        X = torch.rand(5, 1)
        with pytest.raises(ValueError, match="non-positive"):
            kernel._ell(X)
 
    def test_rejects_wrong_shape_from_lengthscale_fn_nd(self):
        wrong_shape_fn = lambda X: torch.ones(X.shape[0], X.shape[1] + 1)
        kernel = GibbsKernel(input_dim=2, lengthscale_fn=wrong_shape_fn, variance=1.0)
        X = torch.rand(5, 2)
        with pytest.raises(ValueError):
            kernel._ell_nd(X)
 
    def test_variance_must_be_positive(self):
        """PyroParam with constraints.positive should reject non-positive init."""
        with pytest.raises(ValueError, match="variance must be positive"):
            GibbsKernel(input_dim=1, lengthscale_fn=constant_lengthscale_fn(0.3),
                        variance=-1.0)
 
 
# ---------------------------------------------------------------------------
# 5. Cross-covariance (X != Z) sanity checks
# ---------------------------------------------------------------------------
 
class TestCrossCovariance:
    def test_shape_with_different_X_Z(self):
        kernel = GibbsKernel(input_dim=1, lengthscale_fn=constant_lengthscale_fn(0.3),
                              variance=1.0)
        X = torch.rand(7, 1)
        Z = torch.rand(4, 1)
        K = kernel.forward(X, Z)
        assert K.shape == (7, 4)
 
    def test_off_diagonal_bounded_by_variance(self):
        """|K(x, x')| <= variance for all x, x' (Cauchy-Schwarz for a valid kernel)."""
        variance = 2.0
        kernel = GibbsKernel(input_dim=1, lengthscale_fn=constant_lengthscale_fn(0.3),
                              variance=variance)
        X = torch.linspace(-3, 3, 20).reshape(-1, 1)
        K = kernel.forward(X)
        assert K.max() <= variance + 1e-6
        assert K.min() >= -variance - 1e-6
 
    def test_far_apart_points_have_near_zero_covariance(self):
        variance = 1.0
        kernel = GibbsKernel(input_dim=1, lengthscale_fn=constant_lengthscale_fn(0.1),
                              variance=variance)
        X = torch.tensor([[0.0], [100.0]])
        K = kernel.forward(X)
        assert K[0, 1].item() < 1e-6 * variance
 
 
if __name__ == "__main__":

    variance = torch.tensor(1e-3, dtype=torch.float64)
    kernel = GibbsKernel(input_dim=2, lengthscale_fn=constant_lengthscale_fn(0.3), variance=variance)
    print("param value:", kernel.variance.item(), kernel.variance.dtype)

    X = torch.rand(30, 2, dtype=torch.float64)
    K = kernel.forward(X)
    print("diag(K) max (no jitter):", torch.diag(K).max().item())

    jitter = 1e-6 * variance
    K_j = K + torch.eye(len(X), dtype=torch.float64) * jitter
    print("diag(K_j) max (with jitter):", torch.diag(K_j).max().item())

    L = torch.linalg.cholesky(K_j)
    print("diag(L @ L.T) max:", torch.diag(L @ L.T).max().item())


    # import sys
    # sys.exit(pytest.main([__file__, "-v"]))