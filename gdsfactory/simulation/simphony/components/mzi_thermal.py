import numpy as np


def delta_temperature(wavelength, length, dn=1.87e-4):
    """Return the delta temperature for a pi phase shift on a MZI interferometer."""
    return wavelength / 2 / length / dn


def test_delta_temperature() -> None:
    dt = delta_temperature(1.55, 100)
    np.isclose(dt, 41.44385026737968)
    dt = delta_temperature(1.55, 1000)
    np.isclose(dt, 4.44385026737968)


if __name__ == "__main__":
    test_delta_temperature()
