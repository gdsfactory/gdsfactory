import gdsfactory as gf


@gf.cell
def straight_with_padding(default: float = 3.0) -> gf.Component:
    c = gf.c.straight()
    c = c.add_padding(default=default)
    return c


if __name__ == "__main__":
    c1 = straight_with_padding(default=1)
    c2 = straight_with_padding(default=3)
    assert c1.name != c2.name
