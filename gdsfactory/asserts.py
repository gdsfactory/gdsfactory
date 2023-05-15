from __future__ import annotations

import warnings

from gdsfactory.component import Component
from gdsfactory.config import __version__


def grating_coupler(gc: Component) -> None:
    if not gc.info.get("polarization"):
        raise ValueError(f"{gc.name} does not have polarization")

    if gc.info.get("polarization") not in ["te", "tm", "dual"]:
        raise ValueError(f"{gc.name} polarization not 'te' or 'tm' or 'dual")

    if not gc.info.get("wavelength"):
        raise ValueError(f"{gc.name} wavelength does not have wavelength")
    if not (0.5 < gc.info["wavelength"] < 5.0):
        raise ValueError(f"{gc.name} wavelength {gc.wavelength} should be in um")

    if "o1" not in gc.ports:
        warnings.warn(
            f"grating_coupler {gc.name} should have a o1 port. It has {gc.ports}"
        )
    if "o1" in gc.ports and gc.ports["o1"].orientation != 180:
        warnings.warn(
            f"grating_coupler {gc.name} orientation = {gc.ports['o1'].orientation}"
            " should be 180 degrees"
        )


def version(
    requirement: str, current: str = __version__, package_name="gdsfactory"
) -> None:
    """Raises error if current version does not match requirement."""
    import semantic_version

    s = semantic_version.SimpleSpec(requirement)
    if not s.match(semantic_version.Version(current)):
        raise ValueError(
            f"{package_name} requirement {requirement}\n"
            f"not compatible your current installed version {current}\n"
            "you can run:\n"
            f"pip install {package_name} {requirement}\n"
        )


if __name__ == "__main__":
    # version(">=3.8.10")
    version("<=3.8.7")
