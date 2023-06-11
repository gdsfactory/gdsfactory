import gdsfactory as gf
from gdsfactory.simulation.devsim import get_simulation_xsection

PDK = gf.get_generic_pdk()
PDK.activate()


def test_pin_waveguide() -> None:
    """Test reverse bias waveguide."""
    nm = 1e-9
    c = get_simulation_xsection.PINWaveguide(
        core_width=500 * nm,
        core_thickness=220 * nm,
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
        n_dist[voltage] = waveguide.index.values
        neffs[voltage] = waveguide.n_eff[0]

    # dn = neffs[vmin] - neffs[vmax]

    # TODO: Find a correct value to test devsim
    # assert np.isclose(dn.real, -0.00011342135795189279), dn.real


if __name__ == "__main__":
    nm = 1e-9
    c = get_simulation_xsection.PINWaveguide(
        core_width=500 * nm,
        core_thickness=220 * nm,
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
        n_dist[voltage] = waveguide.index.values
        neffs[voltage] = waveguide.n_eff[0]

    dn = neffs[vmin] - neffs[vmax]
