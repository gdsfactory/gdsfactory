import gdsfactory as gf


def test_two_copies_in_one():
    c = gf.Component()
    c1 = gf.components.straight()
    c2 = c1.copy()
    c3 = c1.copy()
    c3.add_label("I'm different")

    c << c1
    r2 = c << c2
    r3 = c << c3
    r2.movey(-100)
    r3.movey(-200)
    c.write_gds("two_copies_in_one.gds")
    assert c2.name != c3.name


def test_copied_cell_keeps_info():
    c1 = gf.components.straight()
    c2 = c1.copy()
    assert (
        len(c1.info) > 0
    ), "This test doesn't make any sense unless there is some info to copy"
    assert c1.info == c2.info


if __name__ == "__main__":
    c = gf.Component()
    c1 = gf.components.straight()
    c2 = c1.copy()

    c << c1
    c << c2
    c.show(show_ports=True)
