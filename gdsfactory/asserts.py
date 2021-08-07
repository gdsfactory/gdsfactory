from gdsfactory.component import Component
from gdsfactory.config import __version__


def grating_coupler(gc: Component) -> None:
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


def version_equal_or_greater_than(version: str):
    """Raises error if version  is not high enough"""

    if __version__ < version:
        raise ValueError(
            f"gf.minimum required version = {version}\n"
            f"not compatible your current installed version {__version__}\n"
            "you can run:\n"
            "pip install gf.--upgrade\n"
            "to install a later version",
        )


if __name__ == "__main__":
    # equal_or_greater_than("3.0.0")
    # equal_or_greater_than("2.4.9")
    # equal_or_greater_than("2.4.8")
    version_equal_or_greater_than("2.5.0")
