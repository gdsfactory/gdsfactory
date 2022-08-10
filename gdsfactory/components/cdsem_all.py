"""CD SEM structures."""
from functools import partial
from typing import Optional, Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.cdsem_bend180 import cdsem_bend180
from gdsfactory.components.cdsem_straight import cdsem_straight
from gdsfactory.components.cdsem_straight_density import cdsem_straight_density
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentSpec, CrossSectionSpec

text_rectangular_mini = partial(text_rectangular, size=1)


@cell
def cdsem_all(
    widths: Tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    dense_lines_width: Optional[float] = 0.3,
    dense_lines_width_difference: float = 20e-3,
    dense_lines_gap: float = 0.3,
    dense_lines_labels: Tuple[str, ...] = ("DL", "DM", "DH"),
    straight: ComponentSpec = straight_function,
    bend90: Optional[ComponentSpec] = bend_circular,
    cross_section: CrossSectionSpec = strip,
    text: ComponentSpec = text_rectangular_mini,
) -> Component:
    """Column with all optical PCMs.

    Args:
        widths: for straight lines.
        dense_lines_width: in um.
        dense_lines_width_difference: in um.
        dense_lines_gap: in um.
        dense_lines_labels: strings.
        straight: spec.
        bend90: spec.
        cross_section: spec.
        text: sepc.
    """
    c = Component()
    _c1 = cdsem_straight(
        widths=widths,
        cross_section=cross_section,
    )

    all_devices = [_c1]

    if bend90:
        all_devices += [
            cdsem_bend180(
                width=width,
                straight=straight,
                bend90=bend90,
                cross_section=cross_section,
                text=text,
            )
            for width in widths
        ]

    if dense_lines_width:
        density_params = [
            (
                dense_lines_width - dense_lines_width_difference,
                dense_lines_gap - dense_lines_width_difference,
                dense_lines_labels[0],
            ),
            (dense_lines_width, dense_lines_gap, dense_lines_labels[1]),
            (
                dense_lines_width + dense_lines_width_difference,
                dense_lines_gap + dense_lines_width_difference,
                dense_lines_labels[2],
            ),
        ]

        all_devices += [
            cdsem_straight_density(
                widths=[w] * 10,
                gaps=[g] * 10,
                label=lbl,
                cross_section=cross_section,
                text=text,
            )
            for w, g, lbl in density_params
        ]

    [c.add_ref(d) for d in all_devices]
    c.align(elements="all", alignment="xmin")
    c.distribute(elements="all", direction="y", spacing=5, separation=True)
    return c


if __name__ == "__main__":
    c = cdsem_all()
    c.show(show_ports=True)
