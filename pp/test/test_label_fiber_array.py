import pp


def test_label_fiber_array():
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


def test_label_fiber_array_align_ports():
    """Test that adds the correct label for measurements."""
    c = pp.c.waveguide()
    assert len(c.labels) == 0

    c = pp.routing.add_fiber_single(c, with_align_ports=True)
    pp.show(c)
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


if __name__ == "__main__":
    # c = test_label_fiber_array()
    c = test_label_fiber_array_align_ports()
    pp.show(c)
