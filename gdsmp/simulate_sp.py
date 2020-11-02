import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import pp
from gdsmp.add_monitors import add_monitors


def simulate22(
    component,
    layer_core=1,
    SOURCE_LAYER=200,
    res=20,
    t_oxide=1.0,
    t_Si=0.22,
    t_air=0.78,
    dpml=1,
    si_zmin=0,
    clad_material=mp.Medium(epsilon=2.25),
    core_material=mp.Medium(epsilon=12),
    ymargin=3,
    xmargin=0,
    three_d=False,
    run=True,
    wavelengths=np.linspace(1.5, 1.6, 10),
    monitor_point=(0, 0, 0),
    dfcen=0.2,
):
    """returns Sparameters for a 2 port device

    https://support.lumerical.com/hc/en-us/articles/360042095873-Metamaterial-S-parameter-extraction

    Args:
        component: gdsfactory Component or GDSpath
        layer_core: GDS layer for the Component material
        SOURCE_LAYER: for the monitor
        res: resolution (pixels/um) For example: (10: 100nm step size)
        dpml: PML thickness (um)
        run: if True runs simulation
        wavelengths: iterable of wavelengths to simulate
    """

    if not isinstance(component, pp.Component):
        component = pp.import_gds(component)

    c = pp.extend_ports(component, length=2 * dpml)
    c.flatten()
    c.x = 0
    c.y = 0
    pp.show(c)

    gdspath = pp.write_gds(c)

    freqs = 1 / wavelengths
    fcen = np.mean(freqs)
    frequency_width = dfcen * fcen
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
            src=mp.GaussianSource(fcen, fwidth=frequency_width),
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
        freqs,
        mp.FluxRegion(center=[-length / 2, 0, 0], size=[0, cell_size.y, cell_size.z]),
    )
    m2 = sim.add_mode_monitor(
        freqs,
        mp.FluxRegion(center=[length / 2, 0, 0], size=[0, cell_size.y, cell_size.z]),
    )

    r = dict(sim=sim)
    if run:
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                dt=50, c=mp.Ez, pt=monitor_point, decay_by=1e-9
            )
        )

        # look at simulation and measure component that we want to measure (Ez component)
        # when it decays below a certain
        # monitor_point is a monitor point
        # 1e-9 field threshold
        # call this function every 50 time spes

        # S parameters
        p1 = sim.get_eigenmode_coefficients(
            m1, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[
            0, 0, 0
        ]  # 0 means forward, 1: backward
        p2 = sim.get_eigenmode_coefficients(
            m2, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[0, 0, 0]

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

        # visualize results
        plt.figure()
        plt.plot(
            wavelengths,
            10 * np.log(np.abs(S12_mode1) ** 2),
            "-o",
            label="S12 Input Mode 1 Output Mode 1",
        )
        plt.plot(
            wavelengths,
            10 * np.log(np.abs(S12_mode2) ** 2),
            "-o",
            label="S12 Input Mode 1 Output Mode 2",
        )
        plt.plot(
            wavelengths,
            10 * np.log(np.abs(S12_mode3) ** 2),
            "-o",
            label="S12 Input Mode 1 Output Mode 3",
        )
        plt.ylabel("Power")
        plt.xlabel("Wavelength (microns)")
        plt.legend()
        plt.grid(True)

        r.update(
            dict(
                S12_mode1=S12_mode1,
                S12_mode2=S12_mode2,
                S12_mode3=S12_mode3,
                S11=S11,
                S12=S12,
                S21=S21,
                S22=S22,
                t=t,
                cell_size=cell_size,
            )
        )

    return r


def plot_fields(sim, cell_size):
    sim.plot2D(
        output_plane=mp.Volume(center=mp.Vector3(), size=cell_size),
        fields=mp.Ez,
        field_parameters={"interpolation": "spline36", "cmap": "RdBu"},
    )


if __name__ == "__main__":
    c = pp.c.waveguide(length=2)
    cm = add_monitors(c)
    pp.show(cm)

    r = simulate22(cm, run=True)
    print(r)

    sim = r["sim"]
    cell_size = r["cell_size"]
    plt.show()
