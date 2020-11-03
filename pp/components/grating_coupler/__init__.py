from typing import Callable
import functools


def grating_coupler(component_factory: Callable) -> Callable:
    """ grating_coupler decorator """

    @functools.wraps(component_factory)
    def grating_coupler_component(*args, **kwargs):
        gc = component_factory(*args, **kwargs)
        assert hasattr(
            gc, "polarization"
        ), f"{gc.name} does not have polarization attribute"
        assert gc.polarization in [
            "te",
            "tm",
        ], f"{gc.name} polarization  should be 'te' or 'tm'"
        assert hasattr(
            gc, "wavelength"
        ), f"{gc.name} wavelength does not have wavelength attribute"
        assert (
            500 < gc.wavelength < 2000
        ), f"{gc.name} wavelength {gc.wavelength} should be in nm"
        if "W0" not in gc.ports:
            print(f"grating_coupler {gc.name} should have a W0 port. It has {gc.ports}")
        if "W0" in gc.ports and gc.ports["W0"].orientation != 180:
            print(
                f"grating_coupler {gc.name} W0 port should have orientation = 180 degrees. It has {gc.ports['W0'].orientation}"
            )
        return gc

    return grating_coupler_component
