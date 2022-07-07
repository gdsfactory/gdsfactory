"""Waiting for https://github.com/omry/omegaconf/issues/725
"""
import gdsfactory as gf
import numpy as np


@gf.cell
def straight(
    length: float = 10.0,
) -> gf.Component:
    c = gf.Component()
    c.info["width"] = np.float64(3.2)
    return c


if __name__ == "__main__":
    c = straight()
