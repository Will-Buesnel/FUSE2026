import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

from models.coupled import CoupledModel, ThermalModel
from models.electrical import ElectricalModel
from models.parameters import get_all_parameter_interpolants, format_interpolants, get_parameter_function
from models.local_stats import GibbsKernel

class Simulator:
    """
    This will be what I will pass into the pyro inference engine. A wrapper of the coupled model, that allows me to easily propagate uncertainties through it.
    """

    def __init__(self, cycler_df, ocv_df, param_df, entropy_df, cell: Cell, gauss_interps=None, **kwargs):

        self.exp_df = cycler_df
        self.ocv_df = ocv_df
        self.param_df = param_df
        self.entropy_df = entropy_df
        self.cell = cell
        self.kwargs = kwargs

        self.elec_model = ElectricalModel()
        self.thermal_model = ThermalModel(c=cell.c, h=cell.h, T_inf_degC=cell.T_inf_degC)
        self.coupled_model = CoupledModel(self.elec_model, self.thermal_model) #
        self.thermal_model.entropy_coeff_func = cell.entropy_coeff_func
        self.elec_model.max_capacity_As = cell.capacity_Ah * 3600 # Ah to As

        self.set_current_func() # set the current function for the simulation.

        self.param_interpolants = get_all_parameter_interpolants(param_df, ocv_df) # set all parameter interpolants by default.
        # we will use gauss_interps to overwrite any of the default parameter interpolants with a stochastic version.

        if gauss_interps is not None:
            self.set_gauss_interps(gauss_interps)


    def set_gauss_interps(self, gauss_interps: list[tuple[str, dict]]):

        for name, hyperparams in gauss_interps:

            # create a new ParameterFunction with Gaussian noise for the specified parameter
            kernel = GibbsKernel(input_dim=2, lengthscale_fn=hyperparams['lengthscale_func'], variance=hyperparams['variance'])

            X = np.column_stack([self.param_df["Temperature_degC"].to_numpy(), self.param_df["SOC"].to_numpy()])
            cov = kernel.forward(torch.tensor(X,
                                               dtype=torch.float32)) + torch.eye(len(X)) * 1e-6  # add a small jitter for numerical stability

            L = torch.linalg.cholesky(cov)  # Cholesky decomposition of the covariance matrix. This is allowed to have negative & positive values
            eps_standardised = pyro.sample(f"eps_{name}_standardised", dist.Normal(torch.zeros(len(self.param_df)), torch.ones(len(self.param_df))).to_event(1))  # sample standard normal epsilons. Helps the randon walk not blow up when choosing next steps.
            eps_sample = L @ eps_standardised
            # debug: force eps_sample to be zero for now, to check that the model works without noise.
            # eps_sample = torch.zeros(len(self.param_df))  # debug: force eps_sample

            pyro.deterministic(f"eps_{name}_sample", eps_sample)  # record the sampled epsilons for debugging

            self.param_interpolants[name] = get_parameter_function(self.param_df, name, eps_sample.detach().numpy())  # detach to avoid backprop through the sampling process
                                                                                                                    # maybe it is better to take a copy for this?


            # debug; get the parameter function for temp = 25degC
            socs = np.linspace(0.1, 1, 100)
            r0_func = self.param_interpolants[name]
            r0_values = [r0_func(soc, 25) for soc in socs]
            pyro.deterministic(f"{name}_values_at_25degC", torch.tensor(r0_values, dtype=torch.float32))  # record the parameter values at 25degC for debugging


    def set_current_func(self):
        self.set_current_func = lambda t: np.interp(t,
                                                    self.exp_df["Elapsed Time[h]"].to_numpy() * 3600,
                                                    -self.exp_df["Current(A)"].to_numpy()) # flip sign of current to match equations given.
        

    def run_simulation(self, **kwargs) -> dict:
        
        # set the interpolants as funcs for the model
        for name, interpolant in format_interpolants(self.param_interpolants).items():
            setattr(self.elec_model, f"_{name}_interp", interpolant) 

        return self.coupled_model.simulate(
            t_max=self.exp_df["Elapsed Time[h]"].to_numpy()[-1] * 3600,  # convert to seconds.
            current_func = self.set_current_func,
            **kwargs
        )


class Cell:
    def __init__(self, name: str, capacity_Ah: float, c: float, h: float, c_p: float, rho: float, volume: float= 1, T_inf_degC: float = 25.0, entropy_coeff_func=lambda soc: 0):
        """
        Initialize a Cell instance.

        Parameters:
        name (str): The name of the cell.
        capacity_Ah (float): The capacity of the cell in ampere-hours.
        c (float): The heat capacity of the cell in J/K.
        h (float): The heat transfer coefficient of the cell in W/K.
        c_p (float): The specific heat capacity of the cell in J/kg/K.
        rho (float): The density of the cell in kg/m^3.
        volume (float): The volume of the cell in m^3.
        entropy_coeff_func (callable): A function that returns the entropic coefficient for a given SOC.
        """
        self.name = name
        self.capacity_Ah = capacity_Ah
        self.c = c
        self.h = h
        self.c_p = c_p
        self.rho = rho
        self.T_inf_degC = T_inf_degC # this isn't stricly a cell-related attribute, but its easy to have here and I view it as a property of the cell in the context of a given experiment.
        self.volume = volume
        self.entropy_coeff_func = entropy_coeff_func
