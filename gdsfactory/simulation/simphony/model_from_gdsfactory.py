from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from scipy.constants import speed_of_light
from simphony import Model
from simphony.pins import Pin, PinList
from simphony.tools import interpolate

import gdsfactory as gf
import gdsfactory.simulation.lumerical as sim
from gdsfactory.component import Component


class GDSFactorySimphonyWrapper(Model):
    """Take a GDSFactory component and convert it into a Simphony Model object."""

    def __init__(
        self,
        name: str = "",
        *,
        component: Component,
        dirpath=gf.PATH.sparameters,
        **kwargs,
    ) -> None:
        """Take a GDSFactory component and convert it into a Simphony Model object.

        Args:
            name: name of the model.
            component: component factory or instance.
            dirpath: sparameters directory.
            kwargs: settings.

        """
        pin_names, self.f, self.s = self._model_from_gdsfactory(
            component=component, dirpath=dirpath, **kwargs
        )

        pins = PinList(
            [Pin(component=self, name=pin_names[i]) for i in range(len(pin_names))]
        )

        freq_range = self.f[0], self.f[-1]

        self.wavelengths = speed_of_light / np.array(self.f)

        super().__init__(name, freq_range=freq_range, pins=pins)

    def s_parameters(self, freqs: np.array) -> np.ndarray:
        return interpolate(freqs, self.f, self.s)

    def _model_from_gdsfactory(
        self, component: Component, dirpath=gf.PATH.sparameters, **kwargs
    ) -> Tuple[List[str], Any, np.ndarray]:
        """Return simphony model from gdsfactory Component Sparameters.

        Args:
            component: component factory or instance.
            dirpath: sparameters directory.
            kwargs: settings.

        """
        kwargs.pop("function_name", "")
        kwargs.pop("module", "")
        component = gf.call_if_func(component, **kwargs)

        return sim.read_sparameters_lumerical(component=component, dirpath=dirpath)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = GDSFactorySimphonyWrapper(component=gf.c.mmi2x2())
    # c = GDSFactorySimphonyWrapper(component=gf.c.mmi2x2())
    # c = GDSFactorySimphonyWrapper(component=gf.c.bend_euler())
    # wav = np.linspace(1520, 1570, 1024) * 1e-9
    # f = speed_of_light / wav
    # s = c.s_parameters(freqs=f)

    wav = c.wavelengths
    s = c.s
    plt.plot(wav * 1e9, np.abs(s[:, 1] ** 2))
    plt.show()
