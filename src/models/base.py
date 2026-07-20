"Will Buesnel, Jul 26."

"""
This will be a base model class which the electrical, thermal and (coupled) electro-thermal models will inherit from.
It is debatably overengineering for this project (at least initially), but it will make it easier to add the thermal and coupled models later on.
Especially if I want to try different thermal models to see what is most effective.
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, params: dict):
        self.params = params

    def update_params(self, new_params: dict):
        self.params.update(new_params)
    
    def get_params(self):
        return self.params

    def set_initial_conditions(self, initial_conditions: dict):
        self.initial_conditions = initial_conditions

    def get_initial_conditions(self):
        return getattr(self, 'initial_conditions', None)
    
    state_names: list[str] = []  # will override in subclasses, e.g. ["soc", "v_rc1", "v_rc2"] for elec & ["T"] for temp.

    @property # added the propety tag since you can never set the value of this; it is read-only.
    def state_size(self) -> int:
        return len(self.state_names)

    def pack(self, **kwargs) -> list[float]:
        """Build a state vector from named values, in the model's canonical order."""
        return [kwargs[name] for name in self.state_names]

    def unpack(self, y) -> dict:
        """Inverse of pack — turn a state vector (or array of them) into a named dict."""
        return dict(zip(self.state_names, y))
    
    @abstractmethod
    def simulate(self):
        pass



