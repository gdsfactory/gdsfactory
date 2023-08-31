from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.cross_section import cross_section

LENGTH = 0.5
CELL_NAME = "straight_L500n"
CUSTOM_LABEL = "straight_cband"


def test_label_fiber_array_custom(
    length: float = LENGTH, cell_name: str = CELL_NAME
) -> None:
    c = gf.components.straight(length=LENGTH, cross_section=cross_section)

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

    gc_name = "grating_coupler_elliptical_trenches_taper_angle35"
    assert l0 == f"opt-{gc_name}-{CUSTOM_LABEL}-o1", l0
    assert l1 == f"opt-{gc_name}-{CUSTOM_LABEL}-o2", l1


def test_label_fiber_single_custom(num_regression, check: bool = True) -> None:
    c = gf.components.straight(length=3, cross_section=cross_section)
    assert len(c.labels) == 0, len(c.labels)

    cte = gf.routing.add_fiber_single(
        component=c,
        with_loopback=True,
        component_name=CUSTOM_LABEL,
        cross_section=cross_section,
        layer_label=None,
        decorator=gf.add_labels.add_labels_to_ports,
    )
    assert len(cte.labels) == 4, len(cte.labels)
    labels = {
        label.text: np.array(
            [
                label.origin[0],
                label.origin[1],
                label.layer,
            ]
        )
        for label in cte.labels
    }
    if check and labels:
        num_regression.check(labels)
    else:
        for key in labels:
            print(key)


if __name__ == "__main__":
    test_label_fiber_array_custom()

    # c = gf.components.straight(length=LENGTH, cross_section=cross_section)
    # c = gf.routing.add_fiber_array(
    #     component=c,
    #     with_loopback=False,
    #     component_name=CUSTOM_LABEL,
    #     cross_section=cross_section,
    #     decorator=gf.add_labels.add_labels_to_ports
    # )

    # c = gf.routing.add_fiber_single(
    #     component=c,
    #     with_loopback=True,
    #     component_name=CUSTOM_LABEL,
    #     cross_section=cross_section,
    #     decorator=gf.add_labels.add_labels_to_ports_opt
    # )

    # test_label_fiber_array_custom()
    test_label_fiber_single_custom(None, check=False)

    # c = gf.components.straight()
    # assert len(c.labels) == 0

    # c = gf.routing.add_fiber_array(component=c, with_loopback=True)
    # print(len(c.labels))
    # c.show(show_ports=True)
