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
        position=rightmost_port.dcenter,
        layer=layer,
    )
    return component


def resistance_sheet(width: float = 5, **kwargs: Any) -> gf.Component:
    """Returns a resistance sheet.

    Args:
        width: width of the resistance sheet.
        kwargs: additional settings.
    """
    c = gf.components.resistance_sheet(width=width, **kwargs)
    c.info["doe"] = "resistance_sheet"
    c.info["measurement"] = "iv"
    c.info["measurement_parameters"] = (
        "{'i_start': 0, 'i_stop': 0.005, 'i_step': 0.001}"
    )
    c.info["analysis"] = "[iv]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_optical"] = 0
    c.info["ports_electrical"] = 2
    c.info.update(kwargs)
    label_farthest_right_port(c, c.ports, layer=layer_label, text=f"elec-4-{c.name}")
    return c  # type: ignore[no-any-return]


def via_chain(
    num_vias: int = 100, component_name: str = "via_chain", **kwargs: Any
) -> gf.Component:
    """Returns a chain of vias.

    Args:
        num_vias: number of vias in the chain.
        component_name: name of the component.
        kwargs: additional settings.
    """
    c = gf.Component()
    component_name = f"{component_name}_{num_vias}"
    c0 = gf.components.via_chain(num_vias=num_vias, **kwargs)
    r = c << c0
    r.rotate(-90)  # type: ignore
    c.add_ports(r.ports)
    c.name = f"{c0.name}r90"

    c = gf.routing.add_electrical_pads_top(c, spacing=(0, 20))
    c.info["doe"] = "via_chain"

    c.info["measurement"] = "iv"
    c.info["measurement_parameters"] = (
        "{'i_start': 0, 'i_stop': 0.005, 'i_step': 0.001}"
    )
    c.info["analysis"] = "[iv]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_optical"] = 0
    c.info["ports_electrical"] = 2

    c.name = component_name
    label_farthest_right_port(
        c, c.ports, layer=layer_label, text=f"elec-4-{component_name}"
    )
    return c


def sample_reticle_with_labels(grid: bool = False) -> gf.Component:
    """Returns electrical test structures."""
    res = [resistance_sheet(width=width) for width in [5, 10, 15]]
    via_chains_ = [via_chain(num_vias=num_vias) for num_vias in [100, 200, 500]]

    copies = 3  # number of copies of each component
    components = res * copies + via_chains_ * copies
    return gf.grid(components) if grid else gf.pack(components)[0]


if __name__ == "__main__":
    c = sample_reticle_with_labels(grid=False)
    gdspath = c.write_gds()
    csvpath = gf.labels.write_labels(gdspath, layer_label=layer_label)

    import pandas as pd  # type: ignore

    df = pd.read_csv(csvpath)
    df = df.sort_values(by=["text"])
    print(df)
    c.show()
