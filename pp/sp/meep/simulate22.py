import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import pp
from pp.sp.meep.add_monitors import add_monitors


def simulate22(
    component,
    layer_core=1,
    SOURCE_LAYER=200,
    res=10,
    t_oxide=1.0,
    t_Si=0.22,
    t_air=0.78,
    dpml=1,
    si_zmin=0,
    clad_material=mp.Medium(epsilon=2.25),
    core_material=mp.Medium(epsilon=12),
    wavelength=1.55,
    dfcen=0.1,
    nf=30,
    ymargin=3,
    xmargin=0,
    three_d=False,
    run=False,
    run_until=600,
):
    """returns Sparameters for a 2 port device

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction

    Args:
        res: resolution (default 50 pixels/um)
        dpml: PML thickness (um)
        nf: number of frequencies
        df: 0.1*fcen # 10% bandwidth
    """

    if not isinstance(component, pp.Component):
        c = pp.load_component(component)
        component = c

    c = pp.extend_ports(component, length=2 * dpml)
    c.flatten()
    c.x = 0
    c.y = 0
    pp.show(c)

    gdspath = pp.write_gds(c)

    fcen = 1 / wavelength
    df = dfcen * fcen
    cell_thickness = dpml + t_oxide + t_Si + t_air + dpml

    cell_zmax = 0.5 * cell_thickness if three_d else 0
    cell_zmin = -0.5 * cell_thickness if three_d else 0

    si_zmax = 0.5 * t_Si if three_d else 10
    si_zmin = -0.5 * t_Si if three_d else -10

    geometry = mp.get_GDSII_prisms(core_material, gdspath, layer_core, si_zmin, si_zmax)
    cell = mp.GDSII_vol(gdspath, layer_core, cell_zmin, cell_zmax)
    cell.size = mp.Vector3(c.xsize - dpml + 2 * xmargin, cell.size[1], cell.size[2])
    cell.size += 2 * mp.Vector3(y=ymargin)
    cell_size = cell.size
    length = c.xsize - dpml + 2 * xmargin

    src_vol = mp.GDSII_vol(gdspath, SOURCE_LAYER, si_zmin, si_zmax)

    if three_d:
        oxide_center = mp.Vector3(z=-0.5 * t_oxide)
        oxide_size = mp.Vector3(cell.size.x, cell.size.y, t_oxide)
        oxide_layer = [
            mp.Block(material=clad_material, center=oxide_center, size=oxide_size)
        ]
        geometry = geometry + oxide_layer

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            size=src_vol.size,
            center=src_vol.center,
            eig_band=1,
            eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        resolution=res,
        cell_size=cell_size,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
    )

    m1 = sim.add_mode_monitor(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=[-length / 2, 0, 0], size=[0, cell_size.y, cell_size.z]),
    )
    m2 = sim.add_mode_monitor(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=[length / 2, 0, 0], size=[0, cell_size.y, cell_size.z]),
    )

    r = dict()
    if run:
        # sim.run(until_after_sources=until_after_sources)
        sim.run(until=run_until)

        # S parameters
        p1 = sim.get_eigenmode_coefficients(
            m1, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[0, 0, 0]
        p2 = sim.get_eigenmode_coefficients(
            m2, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[0, 0, 1]

        # transmittance
        t = abs(p2) ** 2 / abs(p1) ** 2

        # S parameters
        # Calculate the scattering params for each waveguide
        bands = [1, 2, 3]  # just look at first, second, and third, TE modes
        m1_results = sim.get_eigenmode_coefficients(
            m1, [1], eig_parity=(mp.ODD_Z + mp.EVEN_Y)
        ).alpha
        m2_results = sim.get_eigenmode_coefficients(
            m2, bands, eig_parity=(mp.ODD_Z + mp.EVEN_Y)
        ).alpha

        a1 = m1_results[:, :, 0]  # forward wave
        b1 = m1_results[:, :, 1]  # backward wave
        a2 = m2_results[:, :, 0]  # forward wave
        b2 = m2_results[:, :, 1]  # backward wave

        S12_mode1 = a2[0, :] / a1[0, :]
        S12_mode2 = a2[1, :] / a1[0, :]
        S12_mode3 = a2[2, :] / a1[0, :]

        S11 = b1 / a1
        S12 = b1 / a2
        S22 = b2 / a2
        S21 = b2 / a1

        freqs = np.array(mp.get_flux_freqs(m1))

        # visualize results
        plt.figure()
        plt.plot(
            1 / freqs,
            10 * np.log(np.abs(S12_mode1) ** 2),
            "-o",
            label="S12 Input Mode 1 Output Mode 1",
        )
        plt.plot(
            1 / freqs,
            10 * np.log(np.abs(S12_mode2) ** 2),
            "-o",
            label="S12 Input Mode 1 Output Mode 2",
        )
        plt.plot(
            1 / freqs,
            10 * np.log(np.abs(S12_mode3) ** 2),
            "-o",
            label="S12 Input Mode 1 Output Mode 3",
        )
        plt.ylabel("Power")
        plt.xlabel("Wavelength (microns)")
        plt.legend()
        plt.grid(True)
        plt.savefig("Results.png")
        plt.show()

        r = dict(S12_mode1, S12_mode2, S12_mode3, S11, S12, S21, S22, t=t)

    if not three_d:
        sim.plot2D(
            output_plane=mp.Volume(center=mp.Vector3(), size=cell.size),
            fields=mp.Ez,
            field_parameters={"interpolation": "spline36", "cmap": "RdBu"},
        )
    r["sim"] = sim
    return r


if __name__ == "__main__":
    c = pp.c.waveguide(length=2)
    cm = add_monitors(c)
    pp.show(cm)

    r = simulate22(cm, run=True)
    print(r)
