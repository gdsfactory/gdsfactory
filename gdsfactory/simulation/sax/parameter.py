from typing import Optional

import numpy as np

from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack


class Parameter:
    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        nominal_value: Optional[float] = None,
        step: Optional[float] = None,
    ) -> None:
        """Generic parameter class for training Component models.

        Arguments:
            min_value: minimum value of the parameter. Default to layer thickness minus tolerance.
            max_value: maximum value of the parameter. Default to layer thickness plus tolerance.
            nominal_value: nominal value of the parameter. Default to layer thickness.
            step: size of the step going from min_value to max_value when generating data. Default to 3 steps between min and max.
        """

    def sample(self, rand_val: float = None):
        """Return a random value within a parameter allowable values.

        User can provide their own random number between 0 and 1 (mapping to a value between min and max), or default to uniform sampling.

        Arguments:
            rand_val: random value between 0 and 1, where 0 maps to min_value and 1 to max_value.

        """
        rand_val = rand_val or np.random.rand(1)[0]
        return self.min_value + (self.max_value - self.min_value) * rand_val

    def count(self):
        """Given min, max, and step, returns number of grid points."""
        return np.ceil(np.abs(self.max_value - self.min_value) / self.step)

    def arange(self):
        """Given min, max, and step, return array of values between min and max (inclusive)."""
        return np.arange(self.min_value, self.max_value + self.step / 2, self.step)


class LayerStackThickness(Parameter):
    def __init__(
        self,
        layerstack: Optional[LayerStack] = None,
        layername: Optional[str] = "core",
        **kwargs,
    ) -> None:
        """Layerstack thickness parameter.

        Arguments:
            layerstack: LayerStack
            layername: Name of the layer in the layerstack
        """
        super().__init__(**kwargs)
        self.layerstack = layerstack or get_layer_stack()
        self.layername = layername
        self.min_value = (
            kwargs["min_value"]
            or self.layerstack.layers[self.layername].thickness
            - self.layerstack.layers[self.layername].thickness_tolerance
        )
        self.max_value = (
            kwargs["max_value"]
            or self.layerstack.layers[self.layername].thickness
            + self.layerstack.layers[self.layername].thickness_tolerance
        )
        self.nominal_value = self.layerstack.layers[self.layername].thickness
        self.step = kwargs["step"] or np.abs(self.max_value - self.min_value) / 3
        self.current_value = None
        return None


class NamedParameter(Parameter):
    def __init__(self, **kwargs) -> None:
        """Parameter associated with the Component or simulation (e.g. wavelength)."""
        super().__init__(**kwargs)
        self.min_value = kwargs.get("min_value")
        self.max_value = kwargs.get("max_value")
        self.nominal_value = kwargs.get("nominal_value")
        self.step = kwargs.get("step")
        return None
