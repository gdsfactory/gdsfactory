import numpy as np
from gdsfactory.simulation.devsim import get_simulation_xsection
import gdsfactory as gf

PDK = gf.get_generic_pdk()
PDK.activate()


def test_pin_waveguide():
    """Test reverse bias waveguide."""
    nm = 1e-9
    c = get_simulation_xsection.PINWaveguide(
        wg_width=500 * nm,
        wg_thickness=220 * nm,
        slab_thickness=90 * nm,
    )
    c.ddsolver()
    n_dist = {}
    neffs = {}

    vmin = 0
    vmax = -5
    voltages = [vmin, vmax]
    Vstep = vmax

    for ind, voltage in enumerate(voltages):
        Vinit = 0 if ind == 0 else voltages[ind - 1]
        c.ramp_voltage(Vfinal=voltage, Vstep=Vstep, Vinit=Vinit)
        waveguide = c.make_waveguide(wavelength=1.55)
        waveguide.compute_modes(isolate=True)
        n_dist[voltage] = waveguide.nx
        neffs[voltage] = waveguide.neffs[0]

    dn = neffs[vmin] - neffs[vmax]
    assert np.isclose(dn.real, -0.00011342135795189279), dn.real


if __name__ == "__main__":
    nm = 1e-9
    c = get_simulation_xsection.PINWaveguide(
        wg_width=500 * nm,
        wg_thickness=220 * nm,
        slab_thickness=90 * nm,
    )
    c.ddsolver()
    n_dist = {}
    neffs = {}

    vmin = 0
    vmax = -5
    voltages = [vmin, vmax]
    Vstep = vmax

    for ind, voltage in enumerate(voltages):
        Vinit = 0 if ind == 0 else voltages[ind - 1]
        c.ramp_voltage(Vfinal=voltage, Vstep=Vstep, Vinit=Vinit)
        waveguide = c.make_waveguide(wavelength=1.55)
        waveguide.compute_modes(isolate=True)
        n_dist[voltage] = waveguide.nx
        neffs[voltage] = waveguide.neffs[0]

    dn = neffs[vmin] - neffs[vmax]
