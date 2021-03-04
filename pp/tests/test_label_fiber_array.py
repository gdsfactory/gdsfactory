import numpy as np

import pp
from pp.component import Component


def test_label_fiber_array() -> Component:
    c = pp.c.waveguide()
    assert len(c.labels) == 0

    c = pp.routing.add_fiber_array(c, with_align_ports=False)
    assert len(c.labels) == 2
    l0 = c.labels[0].text
    l1 = c.labels[1].text

    print(l0)
    print(l1)
    assert l0 == "opt_te_1530_(waveguide)_0_W0"
    assert l1 == "opt_te_1530_(waveguide)_1_E0"

    return c


def test_label_fiber_array_align_ports() -> Component:
    """Test that adds the correct label for measurements."""
    c = pp.c.waveguide()
    assert len(c.labels) == 0

    c = pp.routing.add_fiber_single(c, with_align_ports=True)
    c.show()
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

    assert l0 == "opt_te_1530_(waveguide)_0_W0"
    assert l1 == "opt_te_1530_(waveguide)_0_E0"
    assert l2 == "opt_te_1530_(loopback_waveguide)_0_E0"
    assert l3 == "opt_te_1530_(loopback_waveguide)_1_W0"

    return c


def test_labels_fiber_array(num_regression):
    c = pp.c.waveguide()
    assert len(c.labels) == 0

    c = pp.routing.add_fiber_array(c, with_align_ports=True)
    assert len(c.labels) == 2  # Loopback does not have labels
    labels = {
        label.text: np.array(
            [
                label.position[0],
                label.position[1],
                label.layer,
            ]
        )
        for label in c.labels
    }
    if labels:
        num_regression.check(labels)


def test_labels_fiber_single(num_regression):
    c = pp.c.waveguide()
    assert len(c.labels) == 0

    c = pp.routing.add_fiber_single(c, with_align_ports=True)
    assert len(c.labels) == 4
    labels = {
        label.text: np.array(
            [
                label.position[0],
                label.position[1],
                label.layer,
            ]
        )
        for label in c.labels
    }
    if labels:
        num_regression.check(labels)


if __name__ == "__main__":
    # c = test_label_fiber_array()
    # c = test_label_fiber_array_align_ports()
    # c.show()

    c = pp.c.waveguide()
    assert len(c.labels) == 0

    c = pp.routing.add_fiber_array(c, with_align_ports=True)
    print(len(c.labels))
    c.show()
