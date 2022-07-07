import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component

LENGTH = 0.5

straight = gf.partial(
    gf.components.straight,
    with_bbox=True,
    cladding_layers=None,
    add_pins=None,
    add_bbox=None,
)


def test_label_fiber_array(length=LENGTH) -> Component:
    c = straight(length=LENGTH)
    cell_name = c.name

    assert len(c.labels) == 0, len(c.labels)

    cte = gf.routing.add_fiber_array(component=c, with_loopback=False)
    assert len(cte.labels) == 2, len(cte.labels)
    l0 = cte.labels[0].text
    l1 = cte.labels[1].text

    assert l0 == f"opt_te_1530_({cell_name})_0_o1", l0
    assert l1 == f"opt_te_1530_({cell_name})_1_o2", l1

    cte1 = gf.routing.add_fiber_array(
        component=c, with_loopback=True, nlabels_loopback=1
    )
    assert len(cte1.labels) == 3, len(cte1.labels)

    cte2 = gf.routing.add_fiber_array(
        component=c, with_loopback=True, nlabels_loopback=2
    )
    assert len(cte2.labels) == 4, len(cte2.labels)
    return c


def test_label_fiber_single_loopback(length=LENGTH) -> Component:
    """Test that adds the correct label for measurements."""
    c = straight(length=length)
    cell_name = c.name

    assert len(c.labels) == 0, len(c.labels)
    # nlabels = len(c.labels)

    c = gf.routing.add_fiber_single(component=c, with_loopback=True)
    assert len(c.labels) == 4, c.labels

    l0 = c.labels[0].text
    l1 = c.labels[1].text
    l2 = c.labels[2].text
    l3 = c.labels[3].text

    assert l0 == f"opt_te_1530_({cell_name})_0_o1", l0
    assert l1 == f"opt_te_1530_({cell_name})_0_o2", l1
    assert l2 == f"opt_te_1530_(loopback_{cell_name})_0_o2", l2
    assert l3 == f"opt_te_1530_(loopback_{cell_name})_1_o1", l3

    return c


def test_labels_fiber_array(num_regression) -> None:
    c = straight(length=3)
    assert len(c.labels) == 0, len(c.labels)

    # Loopback does not have labels
    cte = gf.routing.add_fiber_array(component=c, with_loopback=True)
    assert len(cte.labels) == 4, len(c.labels)
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


def test_labels_fiber_single(num_regression) -> None:
    c = straight(length=3)
    assert len(c.labels) == 0, len(c.labels)

    cte = gf.routing.add_fiber_single(component=c, with_loopback=True)
    assert len(cte.labels) == 4, len(cte.labels)
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
    # c = test_label_fiber_array()
    c = test_label_fiber_single_loopback()
    c.show(show_ports=True)

    # c = gf.components.straight()
    # assert len(c.labels) == 0

    # c = gf.routing.add_fiber_array(component=c, with_loopback=True)
    # print(len(c.labels))
    # c.show(show_ports=True)
