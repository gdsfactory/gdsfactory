import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component

LENGTH = 0.5
CELL_NAME = "straight_L500n"
CUSTOM_LABEL = "straight_cband"


def test_label_fiber_array_custom(length=LENGTH, cell_name=CELL_NAME) -> Component:
    c = gf.components.straight(length=LENGTH)

    assert len(c.labels) == 0, len(c.labels)

    cte = gf.routing.add_fiber_array(
        component=c, with_loopback=False, component_name=CUSTOM_LABEL
    )
    assert len(cte.labels) == 2, len(cte.labels)
    l0 = cte.labels[0].text
    l1 = cte.labels[1].text

    assert l0 == f"opt_te_1530_({CUSTOM_LABEL})_0_o1", l0
    assert l1 == f"opt_te_1530_({CUSTOM_LABEL})_1_o2", l1
    return cte


def test_label_fiber_single_custom(num_regression, check=True):
    c = gf.components.straight(length=3)
    assert len(c.labels) == 0

    cte = gf.routing.add_fiber_single(
        component=c, with_loopback=True, component_name=CUSTOM_LABEL
    )
    assert len(cte.labels) == 4
    labels = {
        label.text: np.array(
            [
                label.position[0],
                label.position[1],
                label.layer,
            ]
        )
        for label in cte.labels
    }
    if check and labels:
        num_regression.check(labels)
    else:
        for key in labels.keys():
            print(key)
    return cte


if __name__ == "__main__":
    c = test_label_fiber_array_custom()
    # c = test_label_fiber_single_custom(None, check=False)
    c.show()

    # c = gf.components.straight()
    # assert len(c.labels) == 0

    # c = gf.routing.add_fiber_array(component=c, with_loopback=True)
    # print(len(c.labels))
    # c.show()
