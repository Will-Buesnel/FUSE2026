"""
Initially this was written with the help of Claude while I got to grips with Pyro.

As I got further into the proejct I realised the implementations of these things were not correct in some cases, and unclear in almost all.

So I have rewritten parts where necesary. A 6 week project would not allow for a full rewrite;

A big help on this rewrite for the Adaptive Metropolis-Hastings was this medium article: https://medium.com/@soham.phanse/the-algorithms-that-unlock-bayesian-inference-part-2-adaptive-metropolis-hastings-9ef8322c0b8b
"""

from pathlib import Path

from functorch import dim
import torch
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from scipy.interpolate import UnivariateSpline
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.autoguide.initialization import init_to_median

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
        if variance <= 0:
            raise ValueError("variance must be positive")
        self.variance = pyro.nn.PyroParam(variance, dist.constraints.positive)

        self.lengthscale_fn = lengthscale_fn

    def _ell(self, X): # legacy 1d version of _ell_nd, returns (N, 1) where N is the number of input points.
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
    

    def _ell_nd(self, X):
        # n-dimensional version of _ell, returns (N, d) where d is the input dimension.
        ell = self.lengthscale_fn(X) # this should ideally return a tensor of shape (N, d) where d is the input dimension.
        if (ell.shape != (X.shape[0], self.input_dim)):
            raise ValueError(
                f"lengthscale_fn must return one lengthscale per input dimension; "
                f"got shape {tuple(ell.shape)} for input shape {tuple(X.shape)}"
            )
        ell = ell.reshape(X.size(0), -1)
        if torch.any(ell <= 0):
            raise ValueError("lengthscale_fn produced non-positive lengthscale(s)")
        return ell
    

    def forward(self, X, Z=None, diag=False):
        # X and Z are two sets of points we are computing covariances between. 
        # for our purposes we will only really use X. I have left the Z option as it is standard practice and might be used down the line.
        X = self._slice_input(X)
        if Z is None:
            Z = X
        else:
            Z = self._slice_input(Z)

        ell_X = self._ell_nd(X)                       # (N, d)
        ell_Z = self._ell_nd(Z)                        # (M, d)

        d = X.shape[1]  # input dimension
        K = torch.ones(X.shape[0], Z.shape[0], dtype=X.dtype, device=X.device)  # initialize K with ones

        for dim in range(d):
            ell_Xk = ell_X[:, dim:dim+1]  # (N, 1)
            ell_Zk = ell_Z[:, dim:dim+1]  # (M, 1)
            ell2_sum_k = ell_Xk**2 + ell_Zk.t()**2  # (N, M)
            ell_prod_k = 2 * (ell_Xk @ ell_Zk.t())
            prefactor_k = (ell_prod_k / ell2_sum_k).pow(1 / 2)  # (N, M)

            diff_k = X[:, dim:dim+1] - Z[:, dim:dim+1].t()  # (N, M) : difference between each pair of points in dimension k.
            dist2_k = diff_k**2  # (N, M)
            dist2_k = dist2_k.clamp(min=0.0)  # guard against fp noise


            K *= prefactor_k * torch.exp(-dist2_k / ( ell2_sum_k))  # again use the 1d equivalent.

        K *= self.variance  # scale by variance
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
        c0: torch.tensor | None = None, # it infers the dtype for any samples from this. Potentially this is not an optimal method, but it works.
    ):
        self.model = model
        self.init_step_size = init_step_size
        self.target_accept_prob = target_accept_prob
        self.adapt_start = adapt_start # t0 in the haario et al paper. This is the number of steps before we start adapting the covariance matrix.
        self.epsilon = epsilon
        self.sd_scale = sd_scale  # set to 2.4**2 / dim once dim is known, if None
        self.c0 = c0
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
            init_strategy=init_to_median(num_samples=15)
        )
        self._energy_last = self.potential_fn(self._initial_params)

        # flatten param dict -> single vector, keep shapes to unflatten later
        self._site_names = list(self._initial_params.keys())
        self._site_shapes = {k: v.shape for k, v in self._initial_params.items()}
        self._site_numels = {k: v.numel() for k, v in self._initial_params.items()}
        self._dim = sum(self._site_numels.values())
        print(f"AdaptiveMetropolisHastings: total dimension = {self._dim}, site names = {self._site_names}")
        if self.c0 is not None:
            self._cov = self.c0
            if self.c0.shape != (self._dim, self._dim):
                raise ValueError(f"c0 must be of shape ({self._dim}, {self._dim}), but got {self.c0.shape}")
        else:
            self.c0 = torch.eye(self._dim, dtype=torch.float32) * self.init_step_size**2  # initial covariance matrix, isotropic. default dtype is float32 for samples due to this.

        self.emp_cov = self.c0.clone()  # empirical covariance matrix, initialized to c0
        self._mean = self._flatten(self._initial_params).clone()  # running mean of the samples, initialized to initial params
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
            # technically speaking it is actually 2.38.

        self._scatter = torch.zeros(self._dim, self._dim, dtype=self.c0.dtype)

        
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

    def _update_empir_cov(self, proposal):

        print("Calculating empirical covariance matrix at step", self._t)
    
        X_t = proposal
        X_t_1bar = self._mean 
        t = self._t
        self._mean = self._mean + (X_t - self._mean) / t
        X_tbar = self._mean
        
        # going to use outer products to account for possible shape issues in proposal.    
        self.emp_cov = (t-1)/t * self.emp_cov + self.sd_scale/t * (t*torch.outer(X_t_1bar, X_t_1bar) - (t+1) * torch.outer(X_tbar, X_tbar)
        + torch.outer(X_t, X_t))
        + self.epsilon * torch.eye(self._dim, dtype=self.c0.dtype)  # add ridge term for numerical stability
        cond_number = torch.linalg.cond(self.emp_cov)
        if cond_number > 1e10:
            print(f"Warning: covariance matrix is ill-conditioned at step {self._t}. Condition number = {cond_number:.2e}")


    def _Welford_update_empir_cov(self, proposal):
        X_t = proposal
        t = self._t

        mean_prev = self._mean.clone()
        self._mean = self._mean + (X_t - self._mean) / t

        delta_prev = X_t - mean_prev      # X_t - mean_{t-1}
        delta_curr = X_t - self._mean     # X_t - mean_t

        # running scatter matrix update (Welford, multivariate). This is the runnning sum of variances..
        self._scatter = self._scatter + torch.outer(delta_prev, delta_curr)

        if t > 1:
            cov = self.sd_scale * self._scatter / (t - 1)
            cov = cov + self.epsilon * torch.eye(self._dim, dtype=self.c0.dtype)
            # force symmmetry to avoid numerical issues.
            self.emp_cov = 0.5 * (cov + cov.T)

        cond_number = torch.linalg.cond(self.emp_cov)
        if cond_number > 1e10:
            print(f"Warning: ill-conditioned at step {self._t}. cond = {cond_number:.2e}")
    

    def sample(self, params): 
        # turn params dtype to desired:
        params = {k: v.to(self.c0.dtype) for k, v in params.items()}
        flat_params = self._flatten(params)
        # debugging:

        # print(f"Step {self._t}: current params = {flat_params}")
        # print(f"Step {self._t}: current energy = {self._energy_last}")
        # print(f"Step {self._t}: current empirical covariance matrix = {self.emp_cov}")

        if self._t <= self.adapt_start:
            
            proposal = flat_params + torch.distributions.MultivariateNormal(loc=torch.zeros(self._dim, dtype=self.c0.dtype), covariance_matrix=self.c0).sample()
            use_cov = self.c0

        else:
            # use empirical covariance proposal
            try:
                torch.linalg.cholesky(self.emp_cov)
            except RuntimeError as e:
                # print the error message and the empirical covariance matrix for debugging
                print(f"Cholesky decomposition failed at step {self._t}. Empirical covariance matrix:\n{self.emp_cov}")
                # print the eigenvalues for debugging
                eigvals = torch.linalg.eigvalsh(self.emp_cov)
                print(f"Eigenvalues of empirical covariance matrix:\n{eigvals}")
                raise e
            proposal = flat_params + torch.distributions.MultivariateNormal(loc=torch.zeros(self._dim, dtype=self.c0.dtype), covariance_matrix=self.emp_cov).sample()
            use_cov = self.emp_cov

        
        proposal_params = self._unflatten(proposal)
        energy_proposal = self.potential_fn(proposal_params) # energy proposal approximates the target density fn given in the paper by pi.
        prev_energy = self._energy_last

        target_ratio = energy_proposal - prev_energy # I think you minus because these are log probs?
        # compute proposal density ratio. This is the ratio of the proposal density at the current point to the proposal density at the proposed point.
        # proposal_next_given_current = torch.distributions.MultivariateNormal(loc=flat_params, covariance_matrix=use_cov).log_prob(proposal)
        # proposal_current_given_next = torch.distributions.MultivariateNormal(loc=proposal, covariance_matrix=use_cov).log_prob(flat_params)
        # proposal_ratio = torch.exp(proposal_current_given_next - proposal_next_given_current)
        proposal_ratio = 1.0 # since the proposal is symmetric, the ratio is 1.0
        # compute acceptance probability alpha

        alpha = min(1.0, torch.exp(-target_ratio) * proposal_ratio)
        self._last_alpha = alpha


        # accept if alpha is greater than a uniformly distributed randomly sampled number
        accepted_state_flag = False
        params_before = flat_params.clone() # store the current params before we update them.

        if torch.rand(1).item() < alpha:
            self._accept_cnt += 1
            self._energy_last = energy_proposal
            
            flat_params = proposal
            params = proposal_params
            accepted_state_flag = True

        self._t += 1  # increment the step counter

        if self._t > 1:
                                self._Welford_update_empir_cov(proposal if accepted_state_flag else params_before) # update the empirical covariance matrix with the new sample.

         # update the mean acceptance probability
        self._mean_accept_prob = (self._mean_accept_prob * (self._t -1)+ alpha) / (self._t)

        # print the min median and max eigenvalues of the empirical covariance matrix for debugging
        if self._t % 50 == 0:
            eigvals = torch.linalg.eigvalsh(self.emp_cov)
            print(f"Step {self._t}: empirical covariance matrix eigenvalues: min={eigvals.min().item():.3e}, median={eigvals.median().item():.3e}, max={eigvals.max().item():.3e}")
            eigvals, eigvecs = torch.linalg.eigh(self.emp_cov)
            stiff_direction = eigvecs[:, 0]  # eigh returns ascending order, so index 0 = min eigenvalue
            print(f"Stiff direction: {stiff_direction}")
       
            
        # # debugging print statements
        # print(f"Step {self._t}: acceptance probability = {alpha:.4f}")
        # print(f"{proposal_ratio=}")
        # print(f"t={self._t}, target_ratio={target_ratio.item():.3f}, ||proposal-flat_params||={torch.norm(proposal-flat_params).item():.4f}")
        # tiny_proposal = flat_params + 1e-6 * torch.randn_like(flat_params)
        # print(self.potential_fn(self._unflatten(tiny_proposal)) - self._energy_last)

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

class InvalidProposal(Exception):
    pass

def lengthscale_func_2d(x, soc_transition=0.3, soc_steepness=30,
                         l_soc_low=0.02, l_soc_high=0.08,
                         l_temp=0.15, temp_min=5.0, temp_max=40.0, l_temp_min=0.05, l_temp_max=0.2):
    temp_raw = x[:, 0]
    soc = x[:, 1]

    # normalize temperature to [0, 1] so it's on a comparable scale to SOC
    temp = (temp_raw - temp_min) / (temp_max - temp_min)

    # smooth transition in SOC: steepness ~10-50 gives a soft knee,
    # not a step. sigmoid input is O(1) in soc-units, not O(1000).
    w = torch.sigmoid(soc_steepness * (soc - soc_transition))
    l_soc = l_soc_low + (l_soc_high - l_soc_low) * w

    return torch.stack([5 * torch.ones_like(temp), l_soc], dim=1)  # shape (N, 2). for now I will keep temp lengthscale constant.


def get_path_to_data_processed_dir() -> Path:
    """
    Get the path to the data/processed directory, which is assumed to be two levels up from this file.
    """
    current_file_path = Path(__file__).resolve()
    data_processed_dir = current_file_path.parents[2] / "data" / "processed"
    return data_processed_dir

def test_lengthscale_func_2d():
    # test actual parameters - read them in from data
    
    import pandas as pd
    import numpy as np

    param_df = pd.read_csv(get_path_to_data_processed_dir() / "MLP001_params.csv")
    temps, socs = param_df["Temperature_degC"].to_numpy(), param_df["SOC"].to_numpy()
    print("testing the lengthscale function:")
    lengthscales = lengthscale_func_2d(torch.tensor(np.column_stack([temps, socs]), dtype=torch.float32))
    l_temp, l_soc = lengthscales[:, 0], lengthscales[:, 1]
    print("l_soc min:", l_soc.min().item(), "l_soc max:", l_soc.max().item())
    print("l_temp min:", l_temp.min().item(), "l_temp max:", l_temp.max().item())

    print("testing the generation of K:")
    X = torch.tensor(np.column_stack([temps, socs]), dtype=torch.float32)
    kernel = GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d, variance=1.0) # with unit variance.
    K_star = kernel.forward(X)
    eigvals = torch.linalg.eigvalsh(K_star)
    print("Minimum eigenvalue for K:", eigvals.min().item())

    print("Maximum eigenvalue for K:", eigvals.max().item())


def visualise_lengthscale_func_2d():
    import matplotlib.pyplot as plt
    import numpy as np

    # create a grid of temperature and SOC values
    temp_vals = np.linspace(5, 40, 100)
    soc_vals = np.linspace(0.1, 1, 100)
    temp_grid, soc_grid = np.meshgrid(temp_vals, soc_vals)
    x_grid = np.column_stack([temp_grid.ravel(), soc_grid.ravel()])

    # compute lengthscales for the grid
    lengthscales = lengthscale_func_2d(torch.tensor(x_grid, dtype=torch.float32))
    l_temp, l_soc = lengthscales[:, 0].numpy(), lengthscales[:, 1].numpy()

    # reshape for plotting
    l_temp_grid = l_temp.reshape(temp_grid.shape)
    l_soc_grid = l_soc.reshape(soc_grid.shape)

    # plot lengthscale for temperature
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(temp_grid, soc_grid, l_temp_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Lengthscale (Temperature)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('SOC')
    plt.title('Lengthscale Function for Temperature')

    # plot lengthscale for SOC
    plt.subplot(1, 2, 2)
    plt.contourf(temp_grid, soc_grid, l_soc_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Lengthscale (SOC)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('SOC')
    plt.title('Lengthscale Function for SOC')

    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    visualise_lengthscale_func_2d() 
    
        
