"""
I'm negl this is straight AI code.
"""

import torch
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from scipy.interpolate import UnivariateSpline

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
        
