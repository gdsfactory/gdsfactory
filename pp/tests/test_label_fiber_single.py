import pp
from pp.component import Component

LENGTH = 0.5
CELL_NAME = "straight_L500n"


def test_label_fiber_single(length=LENGTH, cell_name=CELL_NAME) -> Component:
    """Test that add_fiber single adds the correct label for measurements."""
    c = pp.components.straight(length=length)

    assert len(c.labels) == 0

    cte = pp.routing.add_fiber_single(component=c, with_align_ports=False)
    assert len(cte.labels) == 2
    l0 = cte.labels[0].text
    l1 = cte.labels[1].text

    print(l0)
    print(l1)
    assert l0 == f"opt_te_1530_({cell_name})_0_W0"
    assert l1 == f"opt_te_1530_({cell_name})_0_E0"

    return c


def test_label_fiber_single_align_ports(
    length=LENGTH, cell_name=CELL_NAME
) -> Component:
    """Test that add_fiber single adds the correct label for measurements."""
    c = pp.components.straight(length=length)

    assert len(c.labels) == 0

    cte = pp.routing.add_fiber_single(component=c, with_align_ports=True)
    print(len(cte.labels))
    assert len(cte.labels) == 4

    l0 = cte.labels[0].text
    l1 = cte.labels[1].text
    l2 = cte.labels[2].text
    l3 = cte.labels[3].text

    print(l0)
    print(l1)
    print(l2)
    print(l3)

    assert l0 == f"opt_te_1530_({cell_name})_0_W0"
    assert l1 == f"opt_te_1530_({cell_name})_0_E0"
    assert l2 == f"opt_te_1530_(loopback_{cell_name})_0_E0"
    assert l3 == f"opt_te_1530_(loopback_{cell_name})_1_W0"

    return c


if __name__ == "__main__":
    # c = test_label_fiber_single()
    c = test_label_fiber_single_align_ports()
    c.show()
