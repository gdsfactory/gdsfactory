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
    ramp_rate = -0.1

    for ind, voltage in enumerate(voltages):
        if ind == 0:
            Vinit = 0
        else:
            Vinit = voltages[ind - 1]

        c.ramp_voltage(Vfinal=voltage, Vstep=ramp_rate, Vinit=Vinit)
        waveguide = c.make_waveguide(wavelength=1.55)
        waveguide.compute_modes(isolate=True)
        n_dist[voltage] = waveguide.nx
        neffs[voltage] = waveguide.neffs[0]

    # dn = neffs[vmin] - neffs[vmax]
    assert neffs[vmin]


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
    ramp_rate = -0.1

    for ind, voltage in enumerate(voltages):
        if ind == 0:
            Vinit = 0
        else:
            Vinit = voltages[ind - 1]

        c.ramp_voltage(Vfinal=voltage, Vstep=ramp_rate, Vinit=Vinit)
        waveguide = c.make_waveguide(wavelength=1.55)
        waveguide.compute_modes(isolate=True)
        n_dist[voltage] = waveguide.nx
        neffs[voltage] = waveguide.neffs[0]

    dn = neffs[vmin] - neffs[vmax]
