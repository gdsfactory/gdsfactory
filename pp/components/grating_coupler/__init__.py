from typing import Callable
import functools


def grating_coupler(component_factory: Callable) -> Callable:
    """ grating_coupler decorator """

    @functools.wraps(component_factory)
    def grating_coupler_component(*args, **kwargs):
        gc = component_factory(*args, **kwargs)
        assert hasattr(gc, "polarization") and gc.polarization in ["te", "tm"]
        assert hasattr(gc, "wavelength") and 500 < gc.wavelength < 2000
        return gc

    return grating_coupler_component
