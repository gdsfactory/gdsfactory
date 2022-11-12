import gdsfactory as gf


def test_get_ports_sort_clockwise() -> None:
    """.. code::

        3   4
        |___|_
    2 -|      |- 5
       |      |
    1 -|______|- 6
        |   |
        8   7

    """
    c = gf.Component()
    nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    ref = c << nxn
    p = ref.get_ports_list(clockwise=True)
    p1 = p[0]
    p8 = p[-1]

    assert p1.name == "o1", p1.name
    assert p1.orientation == 180, p1.orientation
    assert p8.name == "o8", p8.name
    assert p8.orientation == 270, p8.orientation


def test_get_ports_sort_counter_clockwise() -> None:
    """.. code::

        4   3
        |___|_
    5 -|      |- 2
       |      |
    6 -|______|- 1
        |   |
        7   8

    """
    c = gf.Component()
    nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    ref = c << nxn
    p = ref.get_ports_list(clockwise=False)
    p1 = p[0]
    p8 = p[-1]
    assert p1.name == "o6", p1.name
    assert p1.orientation == 0, p1.orientation
    assert p8.name == "o7", p8.name
    assert p8.orientation == 270, p8.orientation


if __name__ == "__main__":
    test_get_ports_sort_counter_clockwise()
    test_get_ports_sort_clockwise()

    # c = gf.Component()
    # nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    # ref = c << nxn
    # p = ref.get_ports_list(clockwise=False)
    # p1 = p[0]
    # p8 = p[-1]

    # assert p1.name == "o6", p1.name
    # assert p1.orientation == 0, p1.orientation
    # assert p8.name == "o7", p8.name
    # assert p8.orientation == 270, p8.orientation
