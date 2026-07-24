"""
Will Buesnel, Jul 26.
dataclass for having different batteries with different parameters.
"""

class Cell:
    def __init__(self, name: str, capacity_Ah: float, c: float, h: float, c_p: float, rho: float, volume: float= 1):
        self.name = name
        self.capacity_Ah = capacity_Ah
        self.c = c
        self.h = h
        self.c_p = c_p
        self.rho = rho
        self.volume = volume

