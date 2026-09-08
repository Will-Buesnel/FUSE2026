"""
Will Buesnel, Aug 26.
This module contains objects for running the forward model with greater encapsulation, and for running the Bayesian inference with pyro.
The Simulator class is a wrapper around the coupled model, which allows for easy propagation of uncertainties through the model.
The Cell class is a simple data class to hold the properties of a cell, and the BayesianModelParams class is a data class to hold the parameters for the Bayesian model.
"""

from pathlib import Path

import numpy as np
import torch
import pandas as pd
import pyro
import pyro.distributions as dist

from models.coupled import CoupledModel, ThermalModel
from models.electrical import ElectricalModel
from models.parameters import get_all_parameter_interpolants, format_interpolants, get_parameter_function
from models.local_stats import GibbsKernel

# for debugging:
from models.local_stats import InvalidProposal, lengthscale_func_2d
from utils import safe_cholesky
import matplotlib.pyplot as plt
class Simulator:
    """
    This will be what I will pass into the pyro inference engine. A wrapper of the coupled model, that allows me to easily propagate uncertainties through it.
    """

    def __init__(self, cycler_df, ocv_df, param_df, cell: Cell, gauss_interps=None, **kwargs):

        self.exp_df = cycler_df
        self.ocv_df = ocv_df
        self.param_df = param_df
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

    def generate_x_vals(self):
        deg_25_param_df = self.param_df[self.param_df["Temperature_degC"] == 25]  # filter the parameter dataframe for 25 degrees only
        return np.column_stack([deg_25_param_df["Temperature_degC"].to_numpy(), deg_25_param_df["SOC"].to_numpy()])

    def generate_general_x_vals(self):
        return np.column_stack([self.param_df["Temperature_degC"].to_numpy(), self.param_df["SOC"].to_numpy()])


    def set_gauss_interps(self, gauss_interps: list[tuple[str, dict]]):


        for name, hyperparams in gauss_interps:

            # create a new ParameterFunction with Gaussian noise for the specified parameter
            kernel = GibbsKernel(input_dim=2, lengthscale_fn=hyperparams['lengthscale_func'], variance=hyperparams['variance'])

            X = self.generate_x_vals()
            
            K = kernel.forward(torch.tensor(X, dtype=torch.float64))
            K_jittered = K + torch.eye(len(X), dtype=torch.float64) * 1e-6 * hyperparams['variance']   # make the jitter adapt to the variance of the model.
      
            try:
                L = safe_cholesky(K_jittered)  # Cholesky decomposition of the covariance matrix. This is allowed to have negative & positive values
            except RuntimeError as e:

                # diagnostics in case of error:
                asymmetry = torch.max(torch.abs(K_jittered - K_jittered.T))
                print("asymmetry:", asymmetry.item())
                eigvals = torch.linalg.eigvalsh(K_jittered)
                print("eigenvalues:", eigvals)
                print("minimum eigenvalue:", eigvals.min().item())
                print("variance hyperparam:", hyperparams['variance'])

                # produce a plot of the covariance via matrix.
                cov_np = K_jittered.detach().cpu().numpy()

                plt.imshow(cov_np, cmap="viridis")
                plt.colorbar(label="Covariance")
                plt.show()

                eig_K = torch.linalg.eigvalsh(K)
                eig_Kj = torch.linalg.eigvalsh(K_jittered)

                print("K min:", eig_K.min())
                print("K+jitter min:", eig_Kj.min())

                print("Eigenvalue shifts:")
                print(eig_Kj - eig_K)

                print(f"{hyperparams['variance']=}, {hyperparams['lengthscale_func']=}")

                raise InvalidProposal(f"Cholesky decomposition failed for parameter {name}. Check the covariance matrix and hyperparameters.") from e
            
            # going to try reducing the state space of the model by only sampling the epsilons for 25 degrees; the rest can go to zero. This is because the model is only trained at 25 degrees, so the other temperatures are not well constrained. This will reduce the state space and make the inference more stable.
            
            param_df_25_deg_indexes = self.param_df[self.param_df["Temperature_degC"] == 25].index.to_numpy()
            eps_standardised = torch.zeros(self.param_df.shape[0], dtype=L.dtype)  # initialise
            sampled_25_deg_values = pyro.sample(f"eps_{name}_standardised", dist.Normal(torch.zeros(len(param_df_25_deg_indexes), dtype=L.dtype), torch.ones(len(param_df_25_deg_indexes), dtype=L.dtype)).to_event(1))

            eps_standardised[param_df_25_deg_indexes] = sampled_25_deg_values
            print(f"eps_standardised for {name}: {eps_standardised.detach().numpy()}")
            eps_sample = L @ sampled_25_deg_values 

            eps_samples = torch.zeros(len(self.param_df), dtype=L.dtype)
            eps_samples[param_df_25_deg_indexes] = eps_sample  # only set the eps_samples for the 25 degree indexes; the rest are zero. This is because the model is only trained at 25 degrees, so the other temperatures are not well constrained. This will reduce the state space and make the inference more stable.
            
            pyro.deterministic(f"eps_{name}_sample", eps_sample)  # record the sampled epsilons for debugging
            #print(f"{eps_sample=}")
            self.param_interpolants[name] = get_parameter_function(self.param_df, name, eps_samples.detach().numpy())  # detach to avoid backprop through the sampling process
                                                                                                                    # maybe it is better to take a copy for this?


    def set_cell_capacity(self, capacity_Ah: float):
        self.cell.capacity_Ah = capacity_Ah
        self.elec_model.max_capacity_As = capacity_Ah * 3600  # Ah to As

    def set_current_func(self):
        self.current_time = self.exp_df["Elapsed Time[h]"].to_numpy() * 3600
        self.current = -self.exp_df["Current(A)"].to_numpy()

    def set_solver(self, solver: str, method: str = "Kvaerno3"):
        self.coupled_model.set_sol_function(solver, method=method)

    def current_func(self, t):
        return np.interp(t, self.current_time, self.current)

    def run_simulation(self, **kwargs) -> dict:

        # set the interpolants as funcs for the model
        for name, interpolant in format_interpolants(self.param_interpolants).items():
            setattr(self.elec_model, f"_{name}_interp", interpolant) 

        return self.coupled_model.simulate(y0=self.y0,
            t_max=self.exp_df["Elapsed Time[h]"].to_numpy()[-1] * 3600,  # convert to seconds.
            current_func = self.current_func,
            **kwargs
        )

    def set_bayesian_model_params(self, bayesian_model_params):
        print("Setting Bayesian model parameters...")
        self.bayesian_model_params = bayesian_model_params
        self.variational_param_dim = bayesian_model_params.get_total_dimensionality()

    def get_bayesian_model_params(self):
        return self.bayesian_model_params


class Cell:
    """
    A simple data class to hold the properties of a cell.
    """
    def __init__(self, name: str, capacity_Ah: float, c: float, h: float, c_p: float, rho: float, volume: float= 1, T_inf_degC: float = 25.0, entropy_coeff_func=lambda soc: 0):
        """
        Initialise a Cell instance.

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

    @staticmethod
    def get_standard_cell(entropyfunc):
        """
        Returns a standard mlpl001cell. can customise individual parameters if desired but this is a good starting point.
        """
        return Cell(
        name="MLP001",
        capacity_Ah=2.2,
        c = 42.9, # J K^-1
        h = 3.59, # J K^-1
        c_p = 887, # J kg^-1 K^-1
        rho = 2682, # kg m^-3,
        entropy_coeff_func = entropyfunc
    )


class EntropyCoeffFunc:
    def __init__(self, entropy_df):
        self.soc = entropy_df["SOC"].to_numpy()
        self.coeff = entropy_df["Entropic_Coefficient"].to_numpy()

    def __call__(self, soc):
        return np.interp(soc, self.soc, self.coeff)


class BayesianModelParams:
    """
    DataClass to hold the parameters for the Bayesian model. This is used to pass parameters to the model and guide functions.
    """
    def __init__(self, var_scaled_det: float = 1e-7, obs_scale_det: float = 1e-7):
        self.var_scaled_det = var_scaled_det
        self.obs_scale_det = obs_scale_det

    def set_gaussian_process_size(self, size: int = 25):
        self.gaussian_process_size = size

    def set_variational_params(self, variational_params_dict: dict = {"obs_scale": [1e-4, 1, True, 1e-2], "var_scaled": [1e-6, 1, True, 1e-6], "eps_R0 [Ohm]_standardised": [1, 23, True, 1e-2]}): # list of [scaling, size, is variational, c0scaling]
        self.variational_params = variational_params_dict
        
        # set dimensionality of the variational parameters
        self.set_total_dimensionality()


    def get_scaling(self, param_name: str) -> float:
        return self.variational_params.get(param_name)[0]  # return the first element of the list, which is the scaling factor for the parameter.

    def is_variational_param(self, param_name: str) -> bool:
        return self.variational_params.get(param_name)[2]  # return the third element of the list, which is a boolean indicating if the parameter is variational or not.

    def set_total_dimensionality(self):
        total_dim = 0
        for _, (_, size, is_variational, _) in self.variational_params.items():
            if is_variational:
                total_dim += size
        self.param_dim = total_dim

    def get_total_dimensionality(self) -> int:
        return self.param_dim

    def generate_c0(self):
        c0 = torch.zeros(self.param_dim, self.param_dim, dtype=torch.float64)
        running_idx = 0
        for param_name, (scaling, size, is_variational, c0_scaling) in self.variational_params.items():
            if is_variational:
                start_idx = running_idx
                end_idx = start_idx + size
                c0[start_idx:end_idx, start_idx:end_idx] = torch.eye(size, dtype=torch.float64) * c0_scaling # for now will assume the c0 scaling is always that parameters are independent.
                # this isn't necessarily true; but any additional bias would be better to be learned by the model rather than hardcoded in.
                running_idx = end_idx
        return c0

    def toDict(self):
        return {
            "var_scaled": self.var_scaled_det,
            "obs_scale": self.obs_scale_det,
            "variational_params": self.variational_params,
            "param_dim": self.param_dim
        }
    
def generate_standard_simulator(start_idx=0, stop_idx=70000, use_deq=True, stochastic=True, bayesian_model_params: BayesianModelParams = None):
    root = Path.cwd().resolve()
    processed_data_dir = root / "data" / "processed"
    param_df = pd.read_csv(processed_data_dir / "MLP001_params.csv")
    ocv_df = pd.read_csv(processed_data_dir / "MLP001_ocv.csv")
    entropy_df = pd.read_csv(processed_data_dir / "entropy_data_cell1.csv")

    if use_deq:
        wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_deq.csv")
        time_column = "deq_Elapsed Time[h]"
        print("Simulating with dequantised data, assuming uniform spacing between bins")
    else:
        wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_shortened.csv")
        time_column = "Elapsed Time[h]"
        print("Simulating with undequantised data, collapsed at the mean per unique time value.")

    # take only the first x records
    wltp_df = wltp_df.iloc[start_idx:stop_idx, :]


    entropyfunc = EntropyCoeffFunc(entropy_df)  # create an instance of the EntropyCoeffFunc class

    # Define cell properties
    lp_cell = Cell(
        name="MLP001",
        capacity_Ah= 2.132, # Ah. This is a reasonable estimate for this cell. This was found to reduce model error at steady states.
        c = 42.9, # J K^-1
        h = 3.59, # J K^-1
        c_p = 887, # J kg^-1 K^-1
        rho = 2682, # kg m^-3,
        entropy_coeff_func = entropyfunc
    )

    if stochastic:
        gauss_interps = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": 1e-7})]  # variational parameters for the Gaussian noise
    else:
        gauss_interps = None  # no Gaussian noise interpolants for the deterministic model

    sim = Simulator(wltp_df, ocv_df, param_df, lp_cell, gauss_interps=gauss_interps, t_eval = wltp_df[time_column].to_numpy() * 3600)
   
    # get initial soc via interpolation of the ocv_df
   
    initial_soc = 1.0
    initial_temp = 25.0  # initial temperature in degrees Celsius. Ideally I would use the one from the experimentdf, but I don't trust it.
    sim.y0 = [initial_soc, 0, 0, initial_temp]  

    if bayesian_model_params is not None:
        sim.set_bayesian_model_params(bayesian_model_params)  # set the Bayesian model parameters for the simulator
        print("Bayesian model parameters provided, running stochastic simulation with Bayesian inference.")
    else:
        print("No Bayesian model parameters provided, running deterministic simulation with default parameters.")
    return sim