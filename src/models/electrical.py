"Will Buesnel, Jul 26."

"""
Electrical model class, which will inherit from the base model class.

Parameters necessary:

    R_cell (float): The internal resistance of the cell in ohms. = R_{ohmic} + R_{ionic diffusion} +ß R_{charge transfer}
    SoC (float): The state of charge of the cell, expressed as a fraction between 0 and 1.
    Voc (float): The open-circuit voltage of the cell in volts. This is a function of SoC (and maybe temperature later on).
    ambient_temp (float): The ambient temperature in degrees Celsius.
    T: current temperature of the cell in degrees Celsius.
    current_func (callable): A function that returns the current at a given time. This can be a constant value or a time-varying function.
        (I is the current into or out of the cell and taken as positive during charge)
   Capacity (float): The nominal amount of charge within the cell in ampere-hours (Ah).
    

    Equations are taken from p.37 of Mark Blyth's slides: Thermal and electical models for batteries, Nov '24.

    You can add a 'slow' hyperparameter to better view how the model updates over time.
    I've also added a verbose option to print out variables at each timestep, and a progress bar option to show how far through the simulation we are.
"""

import numpy as np
from panel import state
from panel import state
from scipy.integrate import solve_ivp
from typing import Callable
from models.base import BaseModel
from tqdm import tqdm
import time

class ElectricalModel(BaseModel):
    state_names = ["soc", "v_rc1", "v_rc2"]
    required_attrs = [
            "max_capacity_As",
            "_ocv_interp",
            "_r0_interp",
            "_r1_interp",
            "_r2_interp",
            "_tau1_interp",
            "_tau2_interp",
            "_soc_ode",
            "_v_cell",
            "_ocv_interp",
        ]

    def __init__(self):
        pass


    def simulate(self, current_func: Callable[[float], float] = lambda t: 0.0,  # default to no current
                y0: list = [1, 0, 0],
                max_capacity_Ah: float = 2.2, # Ah
                ambient_temp: float = 25, # deg
                t_max: float = 3600, # seconds
                max_step: float = 1.0, # seconds, change
                atol: float = 1e-6,
                rtol: float = 1e-3,
                t_eval: np.ndarray | None = None,
                pbar: bool = False,
                verbose: bool = False,
                slow: bool = False):
        """
        Integrate the ECM state [SOC, v_rc1, v_rc2] from t=0 to t_max.
        Temperature is held at ambient_temp here as without thermal coupling it is essentially isothermal.
        """
        if verbose:
            print("Starting simulation with parameters:")
            print(f"  max_capacity: {max_capacity_Ah}")
            print(f"  ambient_temp: {ambient_temp}")
            print(f"  t_max: {t_max}")
            print(f"  max_step: {max_step}")
            print(f"  atol: {atol}")
            print(f"  rtol: {rtol}")

        self.max_capacity_As = max_capacity_Ah * 3600 # Ah 
        T = ambient_temp  # placeholder until coupled model drives this

        print(f"Running model with max_capacity={self.max_capacity_As} As, ambient_temp={T} degC, t_max={t_max} s, max_step={max_step} s, atol={atol}, rtol={rtol}")

        rhs = self.rhs
        if pbar:
            pbar = tqdm(total=t_max, unit="s", desc="Simulating")
            last_t = [0.0]  # mutable container so the closure can update it

            def rhs(t, y, *args):
                pbar.update(t - last_t[0])
                last_t[0] = t
                return self.rhs(t, y, *args)

        try:
            print("Starting simulation...")
            sol = solve_ivp(
                fun=rhs,
                t_span=(0.0, t_max),
                y0=y0,
                max_step=max_step,
                atol=atol,
                rtol=rtol,
                t_eval=t_eval,
                args=(current_func, T, verbose, slow),
                
            )

        finally:
            if pbar:
                pbar.close()

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        soc, v_rc1, v_rc2 = sol.y
        r0, r1, r2, c1, c2 = self.get_inter_params(soc, T, verbose)  # these are the final values at the end of the simulation, but we could also return them as arrays if needed.
        current = current_func(sol.t)
        # v_cell does not appear in the state vectors therefore it can be calculated in a vectorised way after the simulation
        v_cell = self._v_cell(soc, T, current, r0, v_rc1, v_rc2)

        return {
            "t": sol.t,
            "soc": soc,
            "v_oc": self._ocv_interp(soc, T),
            "v_rc1": v_rc1,
            "v_rc2": v_rc2,
            "v_cell": v_cell,
            "r0": r0, "r1": r1, "r2": r2,
            "c1": c1, "c2": c2,
            "current": current,
            "sol": sol,  # keep the dense_output object around for resampling if needed
        }
    

    def derivative(self, t, y, current: int, T: float, params: tuple, verbose: bool = False):
        state = self.unpack(y)

        soc, v_rc1, v_rc2 = state["soc"], state["v_rc1"], state["v_rc2"]
        r0, r1, r2, t1, t2 = params # unpack params.


        soc_dot = -self._soc_ode(current)
 
        v_rc1_dot = (current * r1 - v_rc1) / (t1)
        v_rc2_dot = (current * r2 - v_rc2) / (t2)

        if verbose:
            print(f"t={t:.2f}, soc={soc:.4f}, v_rc1={v_rc1:.4f}, v_rc2={v_rc2:.4f}, current={current:.4f}")
            print(f"r0={r0:.4f}, r1={r1:.4f}, r2={r2:.4f}, t1={t1:.4f}, t2={t2:.4f}")

            print(f"{soc_dot=:.4f}, {v_rc1_dot=:.4f}, {v_rc2_dot=:.4f}")

        return self.pack(soc=soc_dot, v_rc1=v_rc1_dot, v_rc2=v_rc2_dot)
    

    def rhs(self, t, y, current_func: Callable, T: float, verbose: bool = False, slow: bool = False):
        current = current_func(t)
        params = self.get_inter_params(self.unpack(y)["soc"], T, verbose)
        dydt = self.derivative(t, y, current, T, params, verbose)

        if slow:
            time.sleep(1)  # slow down the simulation for visualization purposes

        return dydt
    

    def get_inter_params(self, soc, T, verbose=False)-> tuple[float, float, float, float, float]:
        """Get the interpolated parameters r0, r1, r2, c1, c2 for the given soc and T."""
        # vectorised: soc, T can be scalars or arrays
        if verbose:
            print(f"Getting parameters for soc={soc}, T={T} deg C")
        r0 = self._r0_interp(soc, T)
        r1 = self._r1_interp(soc, T)
        r2 = self._r2_interp(soc, T)
        t1 = self._tau1_interp(soc, T)
        t2 = self._tau2_interp(soc, T)
        c1 = t1 / r1
        c2 = t2 / r2
        if verbose:
            print(f"r0={r0}, r1={r1}, r2={r2}, t1={t1}, t2={t2}")
        return r0, r1, r2, t1, t2

    def _soc_ode(self, current):
        capacity_As = self.max_capacity_As 
        return current / capacity_As

    def _v_cell(self, soc, T, current, r0, v_rc1, v_rc2):
        voc = self._ocv_interp(soc, T)
        return voc - current * r0 - v_rc1 - v_rc2
    

def _test_derivative_func():
    """Test the derivative function of the ElectricalModel class."""
    # Create a simple current function
    def current_func(t):
        return -1.0  # constant current of 1 A

    # Create an instance of the ElectricalModel with dummy parameters
    params = {
        "R_cell": 0.1,
        "SoC": 1.0,
        "Voc": 3.7,
        "ambient_temp": 25.0,
        "T": 25.0,
        "current_func": current_func,
        "max_capacity": 2.0,
    }
    model = ElectricalModel(params)

    # Set dummy parameter functions
    model.set_r0_interp(lambda T, soc: 0.1)
    model.set_r1_interp(lambda T, soc: 0.05)
    model.set_r2_interp(lambda T, soc: 0.02)
    model.set_c1_interp(lambda T, soc: 1000.0)
    model.set_c2_interp(lambda T, soc: 500.0)
    model.set_v_oc_interp(lambda soc, T: 3.7)
    

    # Test the _rhs function at t=0 with initial state
    t = 0.0
    y = [1.0, 0.0, 0.0]  # initial state: [soc, v_rc1, v_rc2]
    model.max_capacity_Ah = 2.0  # Set max_capacity for the model
    dy_dt = model.derivative(t, y, current_func, params["T"])
    # iterate it
    for i in range(10):
        dy_dt = model.derivative(t, y, current_func, params["T"])
        print(f"Iteration {i}: dy/dt at t={t}: {dy_dt}")
        print(f"Current state y: {y}")
        # Update the state y using Euler's method for demonstration purposes
        dt = 0.1  # time step
        y = [y[j] + dy_dt[j] * dt for j in range(len(y))]
        t += dt
    
    print("dy/dt at t=0:", dy_dt)

if __name__ == "__main__":
    pass 