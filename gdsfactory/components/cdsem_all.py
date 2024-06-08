"""CD SEM structures."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.cdsem_bend180 import cdsem_bend180
from gdsfactory.components.cdsem_straight import cdsem_straight
from gdsfactory.components.cdsem_straight_density import cdsem_straight_density
from gdsfactory.typings import ComponentFactory, ComponentSpec, CrossSectionSpec


@gf.cell
def cdsem_all(
    widths: tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    dense_lines_width: float | None = 0.3,
    dense_lines_width_difference: float = 20e-3,
    dense_lines_gap: float = 0.3,
    dense_lines_labels: tuple[str, ...] = ("DL", "DM", "DH"),
    straight: ComponentSpec = "straight",
    bend90: ComponentSpec | None = "bend_circular",
    cross_section: CrossSectionSpec = "strip",
    text: ComponentFactory = "text_rectangular_mini",
    spacing: float = 5,
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
        text: spec.
        spacing: from group to group.
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
                widths=(w,) * 10,
                gaps=(g,) * 10,
                label=lbl,
                cross_section=cross_section,
                text=text,
            )
            for w, g, lbl in density_params
        ]

    ymin = 0
    for d in all_devices:
        ref = c.add_ref(d)
        ref.dxmin = 0
        ref.dymin = ymin
        ymin += ref.dysize + spacing

    return c


if __name__ == "__main__":
    c = cdsem_all()
    c.show()
