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
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable
from base import BaseModel
from tqdm import tqdm
import time

class ElectricalModel(BaseModel):
    state_names = ["soc", "v_rc1", "v_rc2"]

    def __init__(self, params: dict):
        super().__init__(params)


    def simulate(self, current_func: Callable[[float], float],
                max_capacity: float,
                initial_soc: float,
                ambient_temp: float,
                t_max: float,
                max_step: float,
                atol: float,
                rtol: float,
                pbar: bool = False):
        """
        Integrate the ECM state [SOC, v_rc1, v_rc2] from t=0 to t_max.
        Temperature is held at ambient_temp here as without thermal coupling it is essentially isothermal.
        """
        self.max_capacity = max_capacity  # Ah
        T = ambient_temp  # placeholder until coupled model drives this

        y0 = [initial_soc, 0.0, 0.0]  # v_rc1, v_rc2 start at 0 as default values. If we have ICs I'd like to use them ideally.

        rhs = self._rhs
        if pbar:
            pbar = tqdm(total=t_max, unit="s", desc="Simulating")
            last_t = [0.0]  # mutable container so the closure can update it

            def rhs(t, y, *args):
                pbar.update(t - last_t[0])
                last_t[0] = t
                return self._rhs(t, y, *args)

        try:
            print("Starting simulation...")
            sol = solve_ivp(
                fun=rhs,
                t_span=(0.0, t_max),
                y0=y0,
                method="RK45",
                max_step=max_step,
                atol=atol,
                rtol=rtol,
                args=(current_func, T),
                dense_output=True,
            )

        finally:
            if pbar:
                pbar.close()

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        soc, v_rc1, v_rc2 = sol.y
        r0, r1, r2, c1, c2 = self._get_params(soc, T)  # these are the final values at the end of the simulation, but we could also return them as arrays if needed.
        current = current_func(sol.t)
        # v_cell does not appear in the state vectors therefore it can be calculated in a vectorised way after the simulation
        v_cell = self._v_cell(soc, T, current, r0, v_rc1, v_rc2)

        return {
            "t": sol.t,
            "soc": soc,
            "v_rc1": v_rc1,
            "v_rc2": v_rc2,
            "v_cell": v_cell,
            "r0": r0, "r1": r1, "r2": r2,
            "c1": c1, "c2": c2,
            "sol": sol,  # keep the dense_output object around for resampling if needed
        }

    def _rhs(self, t, y, current_func: Callable, T: float, verbose: bool = False):
        """State: y = [soc, v_rc1, v_rc2]. Returns dy/dt."""
        state = self.unpack(y)

        time.sleep(0.5)  # slow down the simulation for demonstration purposes
        soc, v_rc1, v_rc2 = state["soc"], state["v_rc1"], state["v_rc2"]
        current = current_func(t)

        r0, r1, r2, c1, c2 = self._get_params(soc, T) # non-vectorised operation.

        if verbose:
            print(f"t={t:.2f}, soc={soc:.4f}, v_rc1={v_rc1:.4f}, v_rc2={v_rc2:.4f}, current={current:.4f}")
            print(f"r0={r0:.4f}, r1={r1:.4f}, r2={r2:.4f}, c1={c1:.4f}, c2={c2:.4f}")

        soc_dot = self._soc_ode(current)
        v_rc1_dot = (current * r1 - v_rc1) / (r1 * c1)
        v_rc2_dot = (current * r2 - v_rc2) / (r2 * c2)


        return self.pack(soc=soc_dot, v_rc1=v_rc1_dot, v_rc2=v_rc2_dot)

    def _get_params(self, soc, T):
        r0 = self._r0_func(T, soc)
        r1 = self._r1_func(T, soc)
        r2 = self._r2_func(T, soc)
        c1 = self._c1_func(T, soc)
        c2 = self._c2_func(T, soc)
        return r0, r1, r2, c1, c2

    def _soc_ode(self, current):
        capacity_As = self.max_capacity * 3600.0  # Ah -> As. # TODO: is this correct?
        return current / capacity_As

    def _v_cell(self, soc, T, current, r0, v_rc1, v_rc2):
        voc = self._voc_func(soc, T)
        return voc - current * r0 - v_rc1 - v_rc2

    def set_r0_func(self, r0_func: Callable[[float, float], float]):
        self._r0_func = r0_func

    def set_r1_func(self, r1_func: Callable[[float, float], float]):
        self._r1_func = r1_func

    def set_r2_func(self, r2_func: Callable[[float, float], float]):
        self._r2_func = r2_func

    def set_c1_func(self, c1_func: Callable[[float, float], float]):
        self._c1_func = c1_func

    def set_c2_func(self, c2_func: Callable[[float, float], float]):
        self._c2_func = c2_func

    def set_v_oc_func(self, voc_func: Callable[[float], float]):
        self._voc_func = voc_func

def test_rhs_func():
    """Test the _rhs function of the ElectricalModel class."""
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
    model.set_r0_func(lambda T, soc: 0.1)
    model.set_r1_func(lambda T, soc: 0.05)
    model.set_r2_func(lambda T, soc: 0.02)
    model.set_c1_func(lambda T, soc: 1000.0)
    model.set_c2_func(lambda T, soc: 500.0)
    model.set_v_oc_func(lambda soc, T: 3.7)
    

    # Test the _rhs function at t=0 with initial state
    t = 0.0
    y = [1.0, 0.0, 0.0]  # initial state: [soc, v_rc1, v_rc2]
    model.max_capacity = 2.0  # Set max_capacity for the model
    dy_dt = model._rhs(t, y, current_func, params["T"])
    # iterate it
    for i in range(10):
        dy_dt = model._rhs(t, y, current_func, params["T"])
        print(f"Iteration {i}: dy/dt at t={t}: {dy_dt}")
        print(f"Current state y: {y}")
        # Update the state y using Euler's method for demonstration purposes
        dt = 0.1  # time step
        y = [y[j] + dy_dt[j] * dt for j in range(len(y))]
        t += dt
    
    print("dy/dt at t=0:", dy_dt)

if __name__ == "__main__":
    test_rhs_func()