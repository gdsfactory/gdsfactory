import gdsfactory as gf


@gf.cell
def straight_with_padding(default: float = 3.0) -> gf.Component:
    c = gf.c.straight()
    c.add_padding(default=default)
    return c


@gf.cell
def straight_with_padding_solution(default: float = 3.0) -> gf.Component:
    c = gf.Component()
    component = gf.c.straight()
    c << component
    c.add_padding(default=default)
    c.copy_child_info(component)
    c.add_ports(component.ports)
    return c


if __name__ == "__main__":
    c1 = straight_with_padding(default=1)
    c2 = straight_with_padding(default=3)
    # c1 = straight_with_padding_solution(default=1)
    # c2 = straight_with_padding_solution(default=3)

    print(c1.name)
    print(c2.name)
    assert c1.name != c2.name
