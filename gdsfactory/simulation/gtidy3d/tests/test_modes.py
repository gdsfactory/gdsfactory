import numpy as np

from gdsfactory.simulation.gtidy3d.modes import Waveguide


def test_index():
    c = Waveguide(t_slab=0)
    c.compute_modes()
    n0 = abs(c.neffs[0])
    assert np.isclose(n0, 2.357, rtol=0.01), n0


if __name__ == "__main__":
    test_index()
