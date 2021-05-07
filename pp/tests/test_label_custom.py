import numpy as np

import pp
from pp.component import Component

LENGTH = 0.5
CELL_NAME = "straight_L500n"
CUSTOM_LABEL = "straight_cband"


def test_label_fiber_array_custom(length=LENGTH, cell_name=CELL_NAME) -> Component:
    c = pp.components.straight(length=LENGTH)

    assert len(c.labels) == 0
    # nlabels = len(c.labels)

    cte = pp.routing.add_fiber_array(
        component=c, with_align_ports=False, component_name=CUSTOM_LABEL
    )
    assert len(cte.labels) == 2
    l0 = cte.labels[0].text
    l1 = cte.labels[1].text

    print(l0)
    print(l1)
    assert l0 == f"opt_te_1530_({CUSTOM_LABEL})_0_W0"
    assert l1 == f"opt_te_1530_({CUSTOM_LABEL})_1_E0"

    return c


def test_label_fiber_single_custom(num_regression, check=True):
    c = pp.components.straight(length=3)
    assert len(c.labels) == 0

    cte = pp.routing.add_fiber_single(
        component=c, with_align_ports=True, component_name=CUSTOM_LABEL
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
    # c = test_label_fiber_array_custom()
    c = test_label_fiber_single_custom(None, check=False)
    c.show()

    # c = pp.components.straight()
    # assert len(c.labels) == 0

    # c = pp.routing.add_fiber_array(component=c, with_align_ports=True)
    # print(len(c.labels))
    # c.show()
