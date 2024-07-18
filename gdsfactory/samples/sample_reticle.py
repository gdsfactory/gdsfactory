from __future__ import annotations

import gdsfactory as gf


@gf.cell(check_ports=False)
def spiral_gc(**kwargs):
    c = gf.c.spiral(**kwargs)
    c = gf.routing.add_fiber_array(c)
    c.info["doe"] = "spirals_sc"
    c.info["measurement"] = "optical_loopback4"
    c.info["analysis"] = "optical_loopback4_spirals"
    return c


@gf.cell(check_ports=False)
def mzi_gc(length_x=10, **kwargs):
    c = gf.components.mzi2x2_2x2_phase_shifter(
        length_x=length_x, auto_rename_ports=False, **kwargs
    )
    c = gf.routing.add_pads_top(c, port_names=["top_l_e2", "top_r_e2"])
    c = gf.routing.add_fiber_array(c)
    c.info["doe"] = "mzi"
    c.info["measurement"] = "optical_loopback4"
    c.info["analysis"] = "optical_loopback4_mzi"
    return c


@gf.cell(check_ports=False)
def sample_reticle(grid: bool = False) -> gf.Component:
    """Returns MZI with TE grating couplers."""
    from gdsfactory.generic_tech.cells import (
        add_fiber_array_optical_south_electrical_north,
    )

    mzis = [mzi_gc(length_x=lengths) for lengths in [100, 200, 300]]

    spirals = [spiral_gc(length=length) for length in [0, 100, 200]]

    rings = []
    for length_x in [10, 20, 30]:
        ring = gf.components.ring_single_heater(length_x=length_x)
        ring_te = add_fiber_array_optical_south_electrical_north(
            component=ring,
            electrical_port_names=["l_e2", "r_e2"],
        )
        ring_te.name = f"ring_{length_x}"
        rings.append(ring_te)

    copies = 3
    components = mzis * copies + rings * copies + spirals * copies

    if grid:
        return gf.grid(components)
    c = gf.pack(components)
    if len(c) > 1:
        c = gf.pack(c)[0]
    return c[0]


if __name__ == "__main__":
    c = sample_reticle()
    # gdspath = c.write_gds("mask.gds")
    # csvpath = write_labels(gdspath, prefixes=[""], layer_label="TEXT")
    # df = pd.read_csv(csvpath)
    # print(df)
    c.show()
