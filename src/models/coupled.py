"Will Buesnel, Jul 26."
"""
Coupled model class, which will extend the electrical model class.

For now I will also have the thermal model module
If I add more models, I will put it in it's own module.
Thermal Parameters necessary (these are constant I believe):
Thermal mass c [J K^-1] * need to make sure to do degC->degK conversion
Heat transfer coefficient h [W K^-1]]
Far-field temperature T_inf [degC]
Temperature of the cell T [degC]

Electical parameters:
Current I [A]
Resistances R0, R1, R2 [Ohm]
SoC [fraction]
partial V_{oc}/ partial T: will need to use the approximation function for OCV, and somehow take the derivative w.r.t. T

State space is T(t). q(t) will also be calculated as it factors into T

Equations are taken from https://www.sciencedirect.com/science/article/pii/S2352152X25035698#sec2, eqs 5&4.
1 to 3 are already implemented in the electrical model.

Eq4 is dT/dt = (1/c) * (q(t) - h * (T(t) - T_inf))
Eq5 is q(t) = I(t)^2 * R0(T,Soc) + sum{over rc branches} {v^2_rc_i / R_i(T,SoC)}  - I(t)*T(t) * (partial V_{oc}/ partial T)

for Eq5, the first part we will call power_loss, the second part we will call power_rc, and the third part we will call power_ocv.
to get the partial derivative, could we again use interpolation?
I.e. compute full derivatve of v_oc wrt to time, 
for multiple different temperatures, then use the same interpolation method to get the partial derivative at a given temperature and SoC. 
for now, just ignore while we get everything else working, and then come back to this.

An improvement too think about is to keep the number of rc pairs abstract rather than hardcoded, to make swapping elec models easier
idk if it is necessary though.
"""

from typing import Callable

from models.electrical import ElectricalModel
from models.base import BaseModel
from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import diffrax as dx

class ThermalModel(BaseModel):

    state_names = ["T"]
    required_attrs = ["c", "h", "T_inf_degK"]  # thermal mass, heat transfer coefficient, far-field temperature, cell temperature

    def __init__(self, c: float, h: float, T_inf_degC: float, entropy_coeff_func: Callable | None = None):
        self.c = c
        self.h = h
        self.T_inf_degK = T_inf_degC + 273.15  # convert to Kelvin
        self.entropy_coeff_func = entropy_coeff_func

    def derivative(self, t, y, I, R0, R1, R2, v_rc1, v_rc2, partial_voc_partial_T, verbose: bool = False):
        """
        State: y = [T]. Returns dy/dt.
        to repeat: 
        Eq4 is dT/dt = (1/c) * (q(t) - h * (T(t) - T_inf))
        Eq5 is q(t) = I(t)^2 * R0(T,Soc) + sum{over rc branches} {v^2_rc_i / R_i(T,SoC)}  - I(t)*T(t) * (partial V_{oc}/ partial T)
        """
        
        T = y[0] + 273.15  # convert to Kelvin for calculations
        q = I**2 * R0 + (v_rc1**2 / R1) + (v_rc2**2 / R2) - I * T * partial_voc_partial_T
        dTdt = (1/self.c) * (q - self.h * (T - self.T_inf_degK))
        if verbose:
            print(f"At time {t:.2f}s: T={T:.2f}°K, dT/dt={dTdt:.4f}°K/s, q={q:.4f}W")
        return [dTdt]



class CoupledModel(BaseModel):

    def __init__(self, electrical_model: ElectricalModel, thermal_model: ThermalModel):
        super().__init__()
        self.electrical_model = electrical_model
        self.thermal_model = thermal_model
        self._get_sol = self._get_sol_solveivp  # default solver is solve_ivp, can be changed to diffrax if desired.
        self.solver_method = "BDF"  # default method for solve_ivp, can be changed to "RK45" or "RK23" if desired.

    def partial_voc_partial_T(self, soc, T, verbose=False):
        """
        Compute the partial derivative of V_oc with respect to T at a given soc and T.
        For now, just return 0. TODO: implement this later.
        """
        if verbose:
            print(f"Computing partial V_oc / partial T at soc={soc}, T={T}")
        return self.thermal_model.entropy_coeff_func(soc) if self.thermal_model.entropy_coeff_func is not None else 0.0


    def rhs(self, t, y, current_func: Callable|np.ndarray, verbose: bool = False, slow: bool = False):

        elec_state, thermal_state = self.unpack(y) 
        soc, v_rc1, v_rc2 = elec_state["soc"], elec_state["v_rc1"], elec_state["v_rc2"]
        T = thermal_state["T"] # error is that thermal state is a dict with nothing in it.
        if verbose:
            print("------------------------------")
            print(f"At time {t:.2f}s: soc={soc:.4f}, v_rc1={v_rc1:.4f}V, v_rc2={v_rc2:.4f}V, T={T:.2f}°C")
        # get electricl interpolated parameters:
        r0, r1, r2, c1, c2 = self.electrical_model.get_inter_params(soc, T, verbose)

        partial_voc_partial_T = self.partial_voc_partial_T(soc, T, verbose)

        # compute electrical derivatives:
        if callable(current_func):
            current = current_func(t)
        else:
            current = current_func[np.searchsorted(current_func[:, 0], t, side="right") - 1, 1]  # get the current at time t from the array

        dydt_elec = self.electrical_model.derivative(t, self.electrical_model.pack(soc=soc, v_rc1=v_rc1, v_rc2=v_rc2), current, T, (r0, r1, r2, c1, c2), verbose)
        if verbose:
            print(f"Electrical derivatives at t={t:.2f}s: {dydt_elec=}")

        dydt_thermal = self.thermal_model.derivative(t, self.thermal_model.pack(T=T), current, r0, r1, r2, v_rc1, v_rc2, partial_voc_partial_T, verbose)
        if slow:
            time.sleep(1)  # slow down the simulation for visualization purposes
        # repack states:
        return self.pack(elec_state=dydt_elec, thermal_state=dydt_thermal)
    
    
    def pack(self, **kwargs) -> list[float]:
        """Build a state vector from two packed state values, in the model's canonical order."""
        elec_state = kwargs["elec_state"]
        thermal_state = kwargs["thermal_state"]
        return elec_state + thermal_state

    def unpack(self, y) -> tuple[dict, dict]:
        """Inverse of pack — turn into two packed stated vectors"""
        elec_state_size = self.electrical_model.state_size
        elec_state = self.electrical_model.unpack(y[:elec_state_size])
        thermal_state = self.thermal_model.unpack(y[elec_state_size:])
        return elec_state, thermal_state

    def set_sol_function(self, solver: str, method: str = "BDF"):
        if solver == "solve_ivp":
            self._get_sol = self._get_sol_solveivp
        elif solver == "diffrax":
            self._get_sol = self._get_sol_diffrax
        else:
            raise ValueError(f"Unknown solver: {solver}. Must be one of 'solve_ivp' or 'diffrax'.")
        self.solver_method = method


    def _get_sol_solveivp(self,rhs_func: Callable,
                           y0, max_step, atol,
                            rtol, t_eval, args: tuple, t_span, **kwargs
                            ):
        
        return solve_ivp(  
                        fun=rhs_func,
                        y0=y0,
                        method=self.solver_method,
                        max_step=max_step,
                        atol=atol,
                        rtol=rtol,
                        t_eval=t_eval,
                        args=args,
                        t_span=t_span,
                        **kwargs
        )

    def _get_sol_diffrax(self, rhs_func: Callable,
                           y0, max_step=0.5, atol=1e-6,
                            rtol=1e-3, t_eval=None, args: tuple = (), t_span=(0, 1), **kwargs
                            ):

        current_func, verbose, slow = args
        def rhs(t, y, args):
            return rhs_func(t, y, current_func, verbose, slow)

        if self.solver_method != "Kvaerno3":
                raise ValueError(f"Unknown method: {self.solver_method}. Only 'Kvaerno3' is supported for this model. at the moment.")


        return dx.diffeqsolve(
            dx.ODETerm(rhs),
            dx.Kvaerno3(),
            t0=t_span[0],
            t1=t_span[1],
            dt0=max_step,
            y0=y0,
            args=(current_func, verbose, slow),
            saveat=dx.SaveAt(ts=t_eval),
            stepsize_controller=dx.PIDController(
                rtol=rtol,
                atol=atol,
            ),
        )
        

    def simulate(self,
                y0: list[float]= [1, 0, 0, 25],  # default initial conditions: soc=1.0, v_rc1=0, v_rc2=0, T=25C
                current_func: Callable | np.ndarray = lambda t: 0.0,
                t_max: float = 3600, # seconds
                max_step: float = 0.5, # seconds, change
                atol: float = 1e-6,
                rtol: float = 1e-3,
                t_eval: np.ndarray | None = None,
                pbar: bool = False,
                verbose: bool = False,
                slow: bool = False,
                **kwargs):
        """
        Simulate the coupled model over time.
        y0: initial state vector, should be a list of length equal to the sum of the state sizes of the electrical and thermal models.
        current_func: function of time that returns the current at that time.
        t_max: maximum time to simulate to.
        """
        # check that all required attributes are set for both models
        _check_ready(self)

        t_span = t_eval[0], t_eval[-1] if t_eval is not None else (0, 1)
        t_max = t_span[1]

        if verbose:
            print("Starting simulation with parameters:")
            print(f"  t_max: {t_max}")
            print(f"  max_step: {max_step}")
            print(f"  atol: {atol}")
            print(f"  rtol: {rtol}")
        
        if pbar is True:
            progbar = tqdm(total=t_max, unit="s", desc="Simulating")
            last_t = [0.0]  # mutable container so the closure can update it

            def sol_rhs(t, y, *args):
                progbar.update(t - last_t[0])
                last_t[0] = t
                return self.rhs(t, y, *args)
        else:
            sol_rhs = self.rhs

        try:
            sol = self._get_sol(  
                rhs_func=sol_rhs,
                y0=y0,
                max_step=max_step,
                atol=atol,
                rtol=rtol,
                t_eval=t_eval,
                args=(current_func, verbose, slow),
                t_span=t_span,
                **kwargs
            )

        finally:
            if pbar is True:
                progbar.close()

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        
        soc, v_rc1, v_rc2, T = sol.y
        r0, r1, r2, c1, c2 = self.electrical_model.get_inter_params(soc, T, verbose) 
        current = current_func(sol.t)
        # v_cell does not appear in the state vectors therefore it can be calculated in a vectorised way after the simulation
        v_cell = self.electrical_model._v_cell(soc, T, current, r0, v_rc1, v_rc2)

        return {
            "t [s]": sol.t,
            "soc": soc,
            "v_oc [V]": self.electrical_model._ocv_interp(soc, T),
            "v_rc1 [V]": v_rc1,
            "v_rc2 [V]": v_rc2,
            "v_cell [V]": v_cell,
            "T [°C]": T,
            "R0 [Ohm]": r0, "R1 [Ohm]": r1, "R2 [Ohm]": r2,
            "C1 [F]": c1, "C2 [F]": c2,
            "I [A]": current
        }
    

def _require_attrs(obj):
        missing = [a for a in obj.required_attrs if not hasattr(obj, a)]
        if missing:
            raise ValueError(f"{obj.__class__.__name__} must have {', '.join(missing)} set before simulation.")


def _check_ready(self):
    _require_attrs(
        self.electrical_model)
    _require_attrs(
        self.thermal_model,
    )


def test_simulation(y0: list[float], current_func: Callable | np.ndarray = lambda t: 0.0, t_max: float = 5.0, **kwargs):
    elec_model = ElectricalModel()
    # set interpolators for the electrical model
    from models.parameters import get_all_parameter_interpolants

    param_file_path = "data/processed/MLP001_params.csv"
    ocv_file_path = "data/processed/MLP001_ocv.csv"
    param_df, ocv_df = pd.read_csv(param_file_path), pd.read_csv(ocv_file_path)
    param_interpolants = get_all_parameter_interpolants(param_df, ocv_df)
    for name, interpolant in param_interpolants.items():
        name = name.split(" ")[0].split("[")[0].lower()  # take only the first part of the name, e.g. "R0 [Ohm]" -> "R0"
        setattr(elec_model, f"_{name}_interp", interpolant)

    
    elec_model.max_capacity_As = 2.2 * 3600  # 2.2 Ah in As

    thermal_model = ThermalModel(c=100, h=10, T_inf_degC=25)
    coupled_model = CoupledModel(elec_model, thermal_model)
    res = coupled_model.simulate(y0=y0, current_func=current_func, t_max=t_max, **kwargs)
    return res


def test_simulation_param_interp_scheme(y0: list[float], current_func: Callable, t_max: float = 5.0, param_interpolants=None, **kwargs):
    elec_model = ElectricalModel()

    # set interpolators for the electrical model
    for name, interpolant in param_interpolants.items():
        name = name.split(" ")[0].split("[")[0].lower()  # take only the first part of the name, e.g. "R0 [Ohm]" -> "R0"
        setattr(elec_model, f"_{name}_interp", interpolant)

    
    elec_model.max_capacity_As = 2.2 * 3600  # 2.2 Ah in As

    thermal_model = ThermalModel(c=100, h=10, T_inf_degC=25)
    coupled_model = CoupledModel(elec_model, thermal_model)
    res = coupled_model.simulate(y0=y0, current_func=current_func, t_max=t_max, **kwargs)
    return res


if __name__ == "__main__":
    # test the coupled model with a simple simulation
    res = test_simulation(y0=[1, 0, 0, 25], current_func=lambda t: -1.0, t_max=1000.0, verbose=False, pbar=True)
    # dummy plot of returned state variables over time:
    # plot v_cell, T, soc over time (on three separate axes)
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(res["t"], res["v_cell"])
    plt.ylabel("v_cell")
    plt.subplot(3, 1, 2)
    plt.plot(res["t"], res["T"])
    plt.ylabel("T")
    plt.subplot(3, 1, 3)
    plt.plot(res["t"], res["soc"])
    plt.ylabel("SoC")
    plt.xlabel("Time [s]")
    plt.show()
