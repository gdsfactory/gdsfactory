import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component

LENGTH = 0.5
CELL_NAME = "straight_L500n"


def test_label_fiber_array(length=LENGTH, cell_name=CELL_NAME) -> Component:
    c = gf.components.straight(length=LENGTH)

    assert len(c.labels) == 0
    # nlabels = len(c.labels)

    cte = gf.routing.add_fiber_array(component=c, with_loopback=False)
    assert len(cte.labels) == 2
    l0 = cte.labels[0].text
    l1 = cte.labels[1].text

    assert l0 == f"opt_te_1530_({cell_name})_0_W0"
    assert l1 == f"opt_te_1530_({cell_name})_1_E0"

    cte1 = gf.routing.add_fiber_array(
        component=c, with_loopback=True, nlabels_loopback=1
    )
    assert len(cte1.labels) == 3

    cte2 = gf.routing.add_fiber_array(
        component=c, with_loopback=True, nlabels_loopback=2
    )
    assert len(cte2.labels) == 4

    return c


def test_label_fiber_single_loopback(length=LENGTH, cell_name=CELL_NAME) -> Component:
    """Test that adds the correct label for measurements."""
    c = gf.components.straight(length=length)
    assert len(c.labels) == 0
    # nlabels = len(c.labels)

    c = gf.routing.add_fiber_single(component=c, with_loopback=True)
    print(len(c.labels))
    assert len(c.labels) == 4

    l0 = c.labels[0].text
    l1 = c.labels[1].text
    l2 = c.labels[2].text
    l3 = c.labels[3].text

    print(l0)
    print(l1)
    print(l2)
    print(l3)

    assert l0 == f"opt_te_1530_({cell_name})_0_W0"
    assert l1 == f"opt_te_1530_({cell_name})_0_E0"
    assert l2 == f"opt_te_1530_(loopback_{cell_name})_0_E0"
    assert l3 == f"opt_te_1530_(loopback_{cell_name})_1_W0"

    return c


def test_labels_fiber_array(num_regression):
    c = gf.components.straight(length=3)
    assert len(c.labels) == 0

    cte = gf.routing.add_fiber_array(component=c, with_loopback=True)
    assert len(cte.labels) == 4  # Loopback does not have labels
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
    if labels:
        num_regression.check(labels)


def test_labels_fiber_single(num_regression):
    c = gf.components.straight(length=3)
    assert len(c.labels) == 0

    cte = gf.routing.add_fiber_single(component=c, with_loopback=True)
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
    if labels:
        num_regression.check(labels)


if __name__ == "__main__":
    # c = test_label_fiber_array()
    c = test_label_fiber_array()
    # c = test_label_fiber_single_loopback()
    c.show()

    # c = gf.components.straight()
    # assert len(c.labels) == 0

    # c = gf.routing.add_fiber_array(component=c, with_loopback=True)
    # print(len(c.labels))
    # c.show()
