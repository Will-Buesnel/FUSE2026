"""
I'm negl this is straight AI code.
"""

import torch
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from scipy.interpolate import UnivariateSpline
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel

import matplotlib.pyplot as plt # here for debugging matricies; ignore elsewise

class GibbsKernel(gp.kernels.Kernel):
    def __init__(self, input_dim, lengthscale_fn, variance=None, active_dims=None):
        """
        Args:
            input_dim (int): Dimension of input data (usually 1 for this variant).
            lengthscale_fn (callable): Maps X -> lengthscale tensor > 0, shape (N,) or (N,1).
            variance (Tensor): Scalar amplitude variance (sigma^2).
        """
        super().__init__(input_dim, active_dims)

        if variance is None:
            variance = torch.tensor(1.0)
        variance = torch.as_tensor(variance, dtype=torch.get_default_dtype())
        self.variance = pyro.nn.PyroParam(variance, dist.constraints.positive)

        self.lengthscale_fn = lengthscale_fn

    def _ell(self, X):
        ell = self.lengthscale_fn(X)
        ell = ell.reshape(X.size(0), -1)
        if ell.shape[1] != 1:
            raise ValueError(
                f"lengthscale_fn must return one lengthscale per input row; "
                f"got shape {tuple(ell.shape)} for input shape {tuple(X.shape)}"
            )
        if torch.any(ell <= 0):
            raise ValueError("lengthscale_fn produced non-positive lengthscale(s)")
        return ell

    def forward(self, X, Z=None, diag=False):
        X = self._slice_input(X)
        if Z is None:
            Z = X
        else:
            Z = self._slice_input(Z)

        ell_X = self._ell(X)                       # (N, 1)
        ell_Z = self._ell(Z)                        # (M, 1)

        ell2_sum = ell_X**2 + ell_Z.t()**2           # (N, M)
        ell_prod = 2 * (ell_X @ ell_Z.t())           # (N, M)
        prefactor = torch.sqrt(ell_prod / ell2_sum)

        X2 = X.pow(2).sum(dim=1, keepdim=True)
        Z2 = Z.pow(2).sum(dim=1, keepdim=True)
        dist2 = X2 - 2 * X @ Z.t() + Z2.t()
        dist2 = dist2.clamp(min=0.0)                 # guard against fp noise

        exponent = -dist2 / ell2_sum
        K = self.variance * prefactor * torch.exp(exponent)

        if diag:
            return K.diag()
        return K
    


import math
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model


class AdaptiveMetropolisHastings(MCMCKernel):
    r"""
    Gradient-free random-walk Metropolis kernel that adapts the FULL proposal
    covariance matrix from the empirical covariance of the chain history
    (Haario et al. 2001), rather than only a scalar step size.

    :param model: Python callable containing Pyro primitives.
    :param float init_step_size: initial isotropic step size, used before
        enough samples have accumulated to estimate covariance.
    :param float target_accept_prob: target acceptance rate for the initial
        scalar-adaptation phase.
    :param int adapt_start: number of iterations after which the empirical
        covariance starts being used for proposals (needs > dim samples).
    :param float epsilon: small ridge term added to the empirical covariance
        for numerical stability (avoids singular/near-singular covariance).
    :param float sd_scale: scaling factor on the empirical covariance;
        2.4^2 / dim is the theoretically optimal choice for Gaussian targets
        (Gelman, Roberts, Gilks 1996).
    """

    def __init__(
        self,
        model,
        init_step_size: float = 0.1,
        target_accept_prob: float = 0.234,
        adapt_start: int = 100,
        epsilon: float = 1e-6,
        sd_scale: float = None,
    ):
        self.model = model
        self.init_step_size = init_step_size
        self.target_accept_prob = target_accept_prob
        self.adapt_start = adapt_start # ideally is a value > dim^(3/2). theres nowhere to reference this, this is just intuition on my part that might be wrong.
        self.epsilon = epsilon
        self.sd_scale = sd_scale  # set to 2.4**2 / dim once dim is known, if None

        self._t = 0
        self._log_step_size = math.log(init_step_size)
        self._accept_cnt = 0
        self._mean_accept_prob = 0.0
        super().__init__()

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        (
            self._initial_params,
            self.potential_fn,
            self.transforms,
            self._prototype_trace,
        ) = initialize_model(
            self.model,
            model_args=args,
            model_kwargs=kwargs,
        )
        self._energy_last = self.potential_fn(self._initial_params)

        # flatten param dict -> single vector, keep shapes to unflatten later
        self._site_names = list(self._initial_params.keys())
        self._site_shapes = {k: v.shape for k, v in self._initial_params.items()}
        self._site_numels = {k: v.numel() for k, v in self._initial_params.items()}
        self._dim = sum(self._site_numels.values())

        self._site_slices = {}

        self._site_slices = {}

        # automatically create slices for each site in the flattened parameter vector.
        # this is due to the gaussian process values having strong correlation and therefore it is better setting-up to have them grouped together/
        # this means potentially at a later data I can have a random walk over the gaussian process values as a whole, rather than each value individually..
        idx = 0
        for name in self._site_names:
            n = self._site_numels[name]
            self._site_slices[name] = slice(idx, idx + n)
            idx += n



        if self.sd_scale is None:
            self.sd_scale = 2.4**2 / self._dim # 2.4 is taken from Gelman, wilks et al (even though we dont have access to this paper).

        # running mean/covariance (Welford-style update), in unconstrained flat space
        if self._t <= self._warmup_steps: # going to try only updating during warmup, since that is where the AM really improves over standard. 
            # otherwise, I would suppose it runs the risk of some type of 'forgetting' of the covariance structure.
            # update mean/covariance
            self._mean = self._flatten(self._initial_params).clone()
            self._cov = torch.eye(self._dim, dtype=self._mean.dtype) * (self.init_step_size**2)

    def _flatten(self, params):
        return torch.cat([params[k].reshape(-1) for k in self._site_names])

    def _unflatten(self, vec):
        out = {}
        idx = 0
        for k in self._site_names:
            n = self._site_numels[k]
            out[k] = vec[idx: idx + n].reshape(self._site_shapes[k])
            idx += n
        return out

    def sample(self, params):
        flat_params = self._flatten(params)

        if self._t in [self.adapt_start-395, self.adapt_start, self.adapt_start+1, self.adapt_start + 100, self.adapt_start + 1000]:
            print(
                self._t,
                "acc =", self._mean_accept_prob,
                "cov eig:",
                torch.linalg.eigvalsh(self._cov).min().item(),
                torch.linalg.eigvalsh(self._cov).median().item(),
                torch.linalg.eigvalsh(self._cov).max().item(),

            )
            print(
                "diag std:",
                torch.sqrt(torch.diag(self._cov)).min().item(),
                torch.sqrt(torch.diag(self._cov)).median().item(),
                torch.sqrt(torch.diag(self._cov)).max().item(),
            )

        if self._t < self.adapt_start:
            # isotropic phase, same scalar-adaptation as RandomWalkKernel
            step_size = math.exp(self._log_step_size)
            proposal = flat_params + step_size * torch.randn_like(flat_params)
        else:
            # full-covariance phase: propose from N(current, sd_scale * cov + eps*I)
            cov = self.sd_scale * self._cov + self.epsilon * torch.eye(self._dim, dtype=self._cov.dtype)
            try:
                L = torch.linalg.cholesky(cov)
            except torch._C._LinAlgError:
                print("Stopping: Covariance matrix is not positive definite. Cholesky decomposition failed.")
                eig_min = torch.linalg.eigvalsh(cov).min()
                print("Minimum eigenvalue:", eig_min.item())
                print("Maximum eigenvalue:", torch.linalg.eigvalsh(cov).max().item())                
                plt.imshow(cov.detach().numpy())
                plt.colorbar()
                plt.title("Covariance Matrix")
                plt.show()
                # if the minimum eigenvalue is negative, it indicates that the covariance matrix is not positive definite.
                # if it is really small and negative, it is likely due to numerical issues.
                # if it is large negative, something in the update rule is not working properly.
                # if it is massive and negative (i.e. np.inf), then the covariance matrix is likely diverging and the MCMC is not working properly..

                raise MCMCStop
            
                
            proposal = flat_params + L @ torch.randn(self._dim, dtype=flat_params.dtype)

        new_params = self._unflatten(proposal)
        energy_proposal = self.potential_fn(new_params)
        delta_energy = energy_proposal - self._energy_last
        accept_prob = (-delta_energy).exp().clamp(max=1.0).item()

        rand = pyro.sample("rand_t={}".format(self._t), dist.Uniform(0.0, 1.0))
        accepted = False
        if rand < accept_prob:
            accepted = True
            params = new_params
            self._energy_last = energy_proposal
            flat_params = proposal  # for the running-covariance update below

        # scalar step-size adaptation (only matters during isotropic phase)
        if self._t <= self._warmup_steps:
            adaptation_speed = max(0.001, 0.1 / math.sqrt(1 + self._t))
            self._log_step_size += adaptation_speed * (accept_prob - self.target_accept_prob)

        # online mean/covariance update (Welford), using post-step position
        self._t += 1
        n = self._t
        delta = flat_params - self._mean
        self._mean += delta / n
        delta2 = flat_params - self._mean
        self._cov += (torch.outer(delta, delta2) - self._cov) / n

        if self._t > self._warmup_steps:
            n_post = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n_post = self._t
        self._mean_accept_prob += (accept_prob - self._mean_accept_prob) / n_post

        return params.copy()

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def logging(self):
        return OrderedDict(
            [
                ("step size", "{:.2e}".format(math.exp(self._log_step_size))),
                ("acc. prob", "{:.3f}".format(self._mean_accept_prob)),
            ]
        )

    def diagnostics(self):
        return {
            "acceptance rate": self._accept_cnt / max(1, self._t - self._warmup_steps),
        }

class MCMCStop(Exception):
    pass



@staticmethod
def generate_length_scale_function(x: torch.Tensor, y: torch.Tensor, ell_min: float = 1e-3, ell_max: float = 1e3, roughness: float = 1e-2) -> callable:
    """
    Generates a lengthscale function based on input-output pairs.

    Parameters:
        x (torch.Tensor): Input tensor of shape (N, 1).
        y (torch.Tensor): Output tensor of shape (N, 1).
        ell_min (float): Minimum lengthscale value.
        ell_max (float): Maximum lengthscale value.
        roughness (float): Roughness parameter for the lengthscale function.

    Returns:
        callable: A function that takes an input tensor and returns the corresponding lengthscale tensor.
    """
    # Fit a simple linear model to the data

    ones = torch.ones_like(x)
    A = torch.cat([x.reshape(-1, 1), ones.reshape(-1, 1)], dim=1)  # Add a bias term
    solution, _, _, _, = torch.linalg.lstsq(y.reshape(-1, 1), A)
    slope, intercept = solution[0][0], solution[0][1]
    def lengthscale_fn(input_tensor: torch.Tensor) -> torch.Tensor:
        # Compute the lengthscale based on the linear model
        lengthscale = slope * input_tensor + intercept

        # Apply roughness and clamp to specified bounds
        lengthscale = torch.clamp(lengthscale, min=ell_min, max=ell_max)
        lengthscale = lengthscale + roughness * torch.randn_like(lengthscale)

        return lengthscale

    return lengthscale_fn


def create_posterior_distribution(kernel, X_train, y_train, X_star, noise_variance=1e-2):
    """
    Creates a posterior distribution for the Gaussian Process given training data.

    Parameters:
        kernel (GibbsKernel): The Gibbs kernel to use for the GP.
        X_train (torch.Tensor): Training input data of shape (N, D).
        y_train (torch.Tensor): Training output data of shape (N, 1).
        X_star (torch.Tensor): Test input data of shape (M, D).
        noise_variance (float): Variance of the observation noise.
    """
    K_xx = kernel(X_train, X_train)
    K_xs = kernel(X_train, X_star)
    K_ss = kernel(X_star, X_star)

    L = torch.linalg.solve(K_xx + noise_variance * torch.eye(len(X_train)), K_xs)

    posterior_cov = K_ss - K_xs.T @ L
    alpha = torch.linalg.solve(K_xx + noise_variance * torch.eye(len(X_train)), y_train)

    posterior_mean = K_xs.T @ alpha

    return posterior_mean, posterior_cov


def sample_from_posterior(posterior_mean, posterior_cov, num_samples=1):
    """
    Samples from the posterior distribution of the Gaussian Process.

    Parameters:
        posterior_mean (torch.Tensor): Mean of the posterior distribution of shape (M, 1).
        posterior_cov (torch.Tensor): Covariance matrix of the posterior distribution of shape (M, M).
        num_samples (int): Number of samples to draw from the posterior.

    Returns:
        torch.Tensor: Samples drawn from the posterior distribution of shape (num_samples, M).
    """
    mvn = dist.MultivariateNormal(posterior_mean.flatten(), covariance_matrix=posterior_cov)
    samples = mvn.sample((num_samples,))
    return samples



def generate_length_scale_function1(x: torch.Tensor, y: torch.Tensor, ell_min: float = 1e-3, ell_max: float = 1e3, roughness: float = 1e-2,spline_s: float = 0.0) -> callable:
    x_np = x.numpy().reshape(-1)
    y_np = y.numpy().reshape(-1)

    k = min(2, len(x) - 1)
    spline = UnivariateSpline(x_np, y_np, k=k, s=spline_s)
    dspline = spline.derivative()

    def lengthscale_fn(input_tensor: torch.Tensor) -> torch.Tensor:
        xt = input_tensor.reshape(1, -1)
        xt_clamp = torch.clamp(xt, min=x.min().item(), max=x.max().item())
        local_slope = dspline(xt_clamp)**2

        # high roughness => smaller lengthscale.
        lengthscale = ell_min + (ell_max - ell_min) / (1.0 + roughness * local_slope)

        return torch.tensor(lengthscale).reshape(-1, 1) # reshape to N,1

    return lengthscale_fn


def test_generate_length_scale_function():
    # Generate synthetic data
    x = torch.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * x + 5 + torch.randn_like(x) * 0.5  # Linear relationship with noise

    # Generate lengthscale function
    lengthscale_fn = generate_length_scale_function1(x, y)

    # Test the lengthscale function on new inputs
    test_inputs = torch.tensor([[0.0], [5.0], [10.0]])
    lengthscales = lengthscale_fn(test_inputs)

    print("Test Inputs:\n", test_inputs)
    print("Lengthscales:\n", lengthscales)


def test_generate_length_scale_function_with_kernel():
    # Generate synthetic data
    x = torch.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * x + 5 + torch.randn_like(x) * 0.5  # Linear relationship with noise

    # Generate lengthscale function
    lengthscale_fn = generate_length_scale_function1(x, y)
    print(lengthscale_fn(x))
    print(generate_length_scale_function(x, y)(x))
    # Create Gibbs kernel with the generated lengthscale function
    kernel = GibbsKernel(input_dim=1, lengthscale_fn=lengthscale_fn)

    # Test the kernel on new inputs
    kernel._ell(x)
    test_inputs = torch.tensor([[0.0], [5.0], [10.0]])
    #K = kernel.forward(test_inputs)

    print("Test Inputs:\n", test_inputs)
    print("Kernel Matrix:\n", K)


if __name__ == "__main__":

    #test_generate_length_scale_function()
    test_generate_length_scale_function_with_kernel()
        
