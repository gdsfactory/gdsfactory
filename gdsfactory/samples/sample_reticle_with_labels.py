from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.typings import LayerSpec, Ports

layer_label = "TEXT"


def label_farthest_right_port(
    component: gf.Component, ports: Ports, layer: LayerSpec, text: str
) -> gf.Component:
    """Adds a label to the right of the farthest right port in a given component.

    Args:
        component: The component to which the label is added.
        ports: A list of ports to evaluate for positioning the label.
        layer: The layer on which the label will be added.
        text: The text to display in the label.
    """
    rightmost_port = max(ports, key=lambda port: port.dx)

    component.add_label(
        text=text,
        position=rightmost_port.center,
        layer=layer,
    )
    return component


def spiral_gc(length: float = 0, **kwargs: Any) -> gf.Component:
    """Returns a spiral double with Grating Couplers.

    Args:
        length: length of the spiral straight section.
        kwargs: additional settings.

    Keyword Args:
        bend: bend component.
        straight: straight component.
        cross_section: cross_section component.
        spacing: spacing between the spiral loops.
        n_loops: number of loops.
    """
    c0 = gf.c.spiral(length=length, **kwargs)
    c = gf.routing.add_fiber_array(c0)
    c.info["doe"] = "spirals_sc"
    c.info["measurement"] = "optical_spectrum"
    c.info["analysis"] = "[power_envelope]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_optical"] = 4
    c.info["ports_electrical"] = 0
    c.info.update(kwargs)

    c.name = f"spiral_gc_{length}"
    label_farthest_right_port(c, c.ports, layer=layer_label, text=f"opt-4-{c.name}")
    return c


def mzi_gc(length_x: float = 10, **kwargs: Any) -> gf.Component:
    """Returns a MZI with Grating Couplers.

    Args:
        length_x: length of the MZI.
        kwargs: additional settings.
    """
    c = gf.components.mzi2x2_2x2_phase_shifter(
        length_x=length_x, auto_rename_ports=False, **kwargs
    )
    c = gf.routing.add_pads_top(c, port_names=("top_l_e1", "top_r_e3"))
    c.name = f"mzi_{length_x}"
    c = gf.routing.add_fiber_array(c)

    c.info["doe"] = "mzi"
    c.info["measurement"] = "optical_spectrum"
    c.info["analysis"] = "[fsr]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_electrical"] = 2
    c.info["ports_optical"] = 6
    c.info["length_x"] = length_x
    c.info.update(kwargs)

    c.name = f"mzi_gc_{length_x}"
    label_farthest_right_port(
        c,
        c.ports.filter(port_type="vertical_te"),
        layer=layer_label,
        text=f"opt-{c.info['ports_optical']}-{c.name}",
    )
    label_farthest_right_port(
        c,
        c.ports.filter(port_type="electrical"),
        layer=layer_label,
        text=f"elec-{c.info['ports_electrical']}-{c.name}",
    )
    return c


def sample_reticle_with_labels(grid: bool = False) -> gf.Component:
    """Returns MZI with TE grating couplers."""
    mzis = [mzi_gc(length_x=lengths) for lengths in [100, 200, 300]]
    spirals = [spiral_gc(length=length) for length in [0, 100, 200]]
    rings: list[gf.Component] = []
    for length_x in [10, 20, 30]:
        ring = gf.components.ring_single_heater(length_x=length_x)
        c = gf.c.add_fiber_array_optical_south_electrical_north(
            component=ring,
            electrical_port_names=["l_e2", "r_e2"],
            pad=gf.c.pad,
            grating_coupler=gf.c.grating_coupler_te,
            cross_section_metal="metal3",
        ).copy()
        c.name = f"ring_{length_x}"
        c.info["doe"] = "ring_length_x"
        c.info["measurement"] = "optical_spectrum"
        c.info["ports_electrical"] = 2
        c.info["ports_optical"] = 4
        c.info["analysis"] = "[fsr]"
        c.info["analysis_parameters"] = "[]"
        label_farthest_right_port(
            c,
            c.ports.filter(port_type="vertical_te"),
            layer=layer_label,
            text=f"opt-{c.info['ports_optical']}-{c.name}",
        )
        label_farthest_right_port(
            c,
            c.ports.filter(port_type="electrical"),
            layer=layer_label,
            text=f"elec-{c.info['ports_electrical']}-{c.name}",
        )
        rings.append(c)

    copies = 3  # number of copies of each component
    components = mzis * copies + rings * copies + spirals * copies
    if grid:
        return gf.grid(components)
    components_packed = gf.pack(components)
    if len(components_packed) > 1:
        components_packed = gf.pack(components_packed)
    return components_packed[0]


if __name__ == "__main__":
    import pandas as pd

    c = sample_reticle_with_labels()
    # c.name = "sample_reticle_with_labels"
    # c = spiral_gc()
    # c = mzi_gc()
    gdspath = c.write_gds()
    csvpath = gf.labels.write_labels(gdspath, layer_label=layer_label)
    df = pd.read_csv(csvpath)
    df = df.sort_values(by=["text"])
    print(df)
    c.show()
