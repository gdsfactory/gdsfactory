from __future__ import annotations

import gdsfactory as gf
from gdsfactory.cross_section import cross_section

LENGTH = 0.5
CELL_NAME = "straight_L500n"
CUSTOM_LABEL = "straight_cband"


def test_label_fiber_array_custom(
    length: float = LENGTH, cell_name: str = CELL_NAME
) -> None:
    c = gf.components.straight(length=length, cross_section=cross_section)

    assert len(c.labels) == 0, len(c.labels)

    cte = gf.routing.add_fiber_array(
        component=c,
        with_loopback=False,
        component_name=CUSTOM_LABEL,
        cross_section=cross_section,
        layer_label=None,
        decorator=gf.add_labels.add_labels_to_ports,
    )
    assert len(cte.labels) == 2, len(cte.labels)
    l0 = cte.labels[0].text
    l1 = cte.labels[1].text
    assert l0 == "o1", l0
    assert l1 == "o2", l1
