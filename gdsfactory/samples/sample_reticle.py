from __future__ import annotations

from typing import Any

import gdsfactory as gf


@gf.cell
def spiral_gc(**kwargs: Any) -> gf.Component:
    """Returns a spiral double with Grating Couplers.

    Args:
        kwargs: additional settings.

    Keyword Args:
        length: length of the spiral straight section.
        bend: bend component.
        straight: straight component.
        cross_section: cross_section component.
        spacing: spacing between the spiral loops.
        n_loops: number of loops.
    """
    c = gf.c.spiral(**kwargs)
    c = gf.routing.add_fiber_array(c)
    c.info["doe"] = "spirals_sc"
    c.info["measurement"] = "optical_spectrum"
    c.info["measurement_parameters"] = (
        "{'wl_start': 1.5, 'wl_stop': 1.6, 'wl_step': 0.001}"
    )
    c.info["analysis"] = "[power_envelope]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_optical"] = 4
    c.info["ports_electrical"] = 0
    c.info.update(kwargs)
    return c


@gf.cell
def mzi_gc(length_x: float = 10, **kwargs: Any) -> gf.Component:
    """Returns a MZI with Grating Couplers.

    Args:
        length_x: length of the MZI.
        kwargs: additional settings.
    """
    c = gf.components.mzi2x2_2x2_phase_shifter(
        length_x=length_x, auto_rename_ports=False, **kwargs
    )
    c = gf.routing.add_pads_top(c, port_names=["top_l_e1", "top_r_e3"])
    c = gf.routing.add_fiber_array(c)
    c.info["doe"] = "mzi"
    c.info["measurement"] = "optical_spectrum"
    c.info["measurement_parameters"] = (
        "{'wl_start': 1.5, 'wl_stop': 1.6, 'wl_step': 0.001}"
    )
    c.info["analysis"] = "[fsr]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_electrical"] = 2
    c.info["ports_optical"] = 6
    c.info["length_x"] = length_x
    c.info.update(kwargs)
    return c


@gf.cell
def sample_reticle(grid: bool = False) -> gf.Component:
    """Returns MZI with TE grating couplers."""
    from gdsfactory.generic_tech.cells import (
        add_fiber_array_optical_south_electrical_north,
    )

    mzis = [mzi_gc(length_x=lengths) for lengths in [100, 200, 300]]
    spirals = [spiral_gc(length=length) for length in [0, 100, 200]]
    rings: list[gf.Component] = []
    for length_x in [10, 20, 30]:
        ring = gf.components.ring_single_heater(length_x=length_x)
        c = add_fiber_array_optical_south_electrical_north(
            component=ring,
            electrical_port_names=["l_e2", "r_e2"],
        )
        c.name = f"ring_{length_x}"
        c.info["doe"] = "ring_length_x"
        c.info["measurement"] = "optical_spectrum"
        c.info["measurement_parameters"] = (
            "{'wl_start': 1.5, 'wl_stop': 1.6, 'wl_step': 0.001}"
        )
        c.info["ports_electrical"] = 2
        c.info["ports_optical"] = 4
        c.info["analysis"] = "[fsr]"
        c.info["analysis_parameters"] = "[]"
        rings.append(c)

    copies = 3  # number of copies of each component
    components = mzis * copies + rings * copies + spirals * copies
    if grid:
        return gf.grid(components)
    components = gf.pack(components)
    if len(components) > 1:
        components = gf.pack(components)
    return components[0]


if __name__ == "__main__":
    c = sample_reticle()
    # gdspath = c.write_gds("mask.gds")
    # csvpath = write_labels(gdspath, prefixes=[""], layer_label="TEXT")
    # df = pd.read_csv(csvpath)
    # print(df)
    c.show()
