from __future__ import annotations

__all__ = ["alignment_mark_cross"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags={"type": "pcms"})
def alignment_mark_cross(
    arm_width: float = 2.0,
    arm_length: float = 20.0,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Custom alignment cross mark for lithography alignment.

    Creates a cross shape centered at the origin composed of two
    overlapping rectangles (horizontal and vertical arms).

    Args:
        arm_width: Width of each arm in um.
        arm_length: Length of each arm in um (full extent from tip to tip
            is 2 * arm_length).
        layer: Layer specification for the cross geometry.
        port_type: Optional port type. If provided, ports are added at the
            four arm tips.
    """
    c = Component()

    hw = arm_width / 2

    # Horizontal arm
    c.add_polygon(
        [(-arm_length, -hw), (arm_length, -hw), (arm_length, hw), (-arm_length, hw)],
        layer=layer,
    )

    # Vertical arm
    c.add_polygon(
        [(-hw, -arm_length), (hw, -arm_length), (hw, arm_length), (-hw, arm_length)],
        layer=layer,
    )

    if port_type is not None:
        c.add_port(
            name="o1",
            center=(-arm_length, 0),
            width=arm_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            name="o2",
            center=(arm_length, 0),
            width=arm_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            name="o3",
            center=(0, -arm_length),
            width=arm_width,
            orientation=270,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            name="o4",
            center=(0, arm_length),
            width=arm_width,
            orientation=90,
            layer=layer,
            port_type=port_type,
        )

    return c


if __name__ == "__main__":
    c = alignment_mark_cross()
    c.show()
