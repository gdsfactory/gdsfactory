import meep as mp
import pp

from gdsmp.add_monitors import add_monitors


def simulate4(
    component,
    layer_core=1,
    SOURCE_LAYER=200,
    PORT1_LAYER=201,
    PORT2_LAYER=202,
    PORT3_LAYER=203,
    PORT4_LAYER=204,
    res=10,
    t_oxide=1.0,
    t_Si=0.22,
    t_air=0.78,
    dpml=1,
    si_zmin=0,
    clad_material=mp.Medium(epsilon=2.25),
    core_material=mp.Medium(epsilon=12),
    wavelength=1.55,
    dfcen=0.2,
    ymargin=3,
    xmargin=0,
    three_d=False,
    run=False,
):
    """Simulate a 4 port device

    Args:
        res: resolution (default 50 pixels/um)

    """
    if not isinstance(component, pp.Component):
        component = pp.load_component(component)

    c = pp.extend_ports(component, length=2 * dpml)
    c.x = 0
    c.y = 0
    c.flatten()
    pp.show(c)

    gdspath = pp.write_gds(c)

    fcen = 1 / wavelength
    df = dfcen * fcen
    cell_thickness = dpml + t_oxide + t_Si + t_air + dpml

    cell_zmax = 0.5 * cell_thickness if three_d else 0
    cell_zmin = -0.5 * cell_thickness if three_d else 0
    si_zmax = 0.5 * t_Si if three_d else 10
    si_zmin = -0.5 * t_Si if three_d else -10

    cell = mp.GDSII_vol(gdspath, layer_core, cell_zmin, cell_zmax)
    cell.size = mp.Vector3(c.xsize - dpml + 2 * xmargin, cell.size[1], cell.size[2])
    cell.size += 2 * mp.Vector3(y=ymargin)

    geometry = mp.get_GDSII_prisms(core_material, gdspath, layer_core, si_zmin, si_zmax)

    src_vol = mp.GDSII_vol(gdspath, SOURCE_LAYER, si_zmin, si_zmax)

    p1 = mp.GDSII_vol(gdspath, PORT1_LAYER, si_zmin, si_zmax)
    p2 = mp.GDSII_vol(gdspath, PORT2_LAYER, si_zmin, si_zmax)
    p3 = mp.GDSII_vol(gdspath, PORT3_LAYER, si_zmin, si_zmax)
    p4 = mp.GDSII_vol(gdspath, PORT4_LAYER, si_zmin, si_zmax)

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
        cell_size=cell.size,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
    )

    m1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p1))
    m2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p2))
    m3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p3))
    m4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p4))

    r = dict()
    if run:
        sim.run(until_after_sources=100)

        # S parameters
        p1 = sim.get_eigenmode_coefficients(
            m1, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[0, 0, 0]
        p2 = sim.get_eigenmode_coefficients(
            m2, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[0, 0, 1]
        p3 = sim.get_eigenmode_coefficients(
            m3, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[0, 0, 0]
        p4 = sim.get_eigenmode_coefficients(
            m4, [1], eig_parity=mp.NO_PARITY if three_d else mp.EVEN_Y + mp.ODD_Z
        ).alpha[0, 0, 0]

        # transmittance
        t2 = abs(p2) ** 2 / abs(p1) ** 2
        t3 = abs(p3) ** 2 / abs(p1) ** 2
        t4 = abs(p4) ** 2 / abs(p1) ** 2

        r = dict(p1=p1, p2=p2, p3=p3, p4=p4, t2=t2, t3=t3, t4=t4)

    if not three_d:
        sim.plot2D(
            output_plane=mp.Volume(center=mp.Vector3(), size=cell.size),
            fields=mp.Ez,
            field_parameters={"interpolation": "spline36", "cmap": "RdBu"},
        )
    r["sim"] = sim
    return r


if __name__ == "__main__":
    c = pp.c.coupler(length=0.1)
    cm = add_monitors(c)
    pp.show(cm)
    r = simulate4(cm)
