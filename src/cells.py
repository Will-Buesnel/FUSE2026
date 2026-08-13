"""
Will Buesnel, Jul 26.
dataclass for having different batteries with different parameters.
"""

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

