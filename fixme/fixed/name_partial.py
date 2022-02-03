import gdsfactory as gf


@gf.cell
def straight_with_padding(padding: float = 3.0) -> gf.Component:
    """
    Adding padding to a cached component should raise MutabilityError

    Args:
        default: default padding on all sides
    """
    c = gf.components.straight()
    c.add_padding(default=padding)  # add padding to original cell
    return c


@gf.cell
def straight_with_padding_container(padding: float = 3.0) -> gf.Component:
    """Solution1: create new component (container)"""
    c = gf.Component()
    component = gf.components.straight()
    c << component
    c.add_padding(default=padding)
    c.copy_child_info(component)
    c.add_ports(component.ports)
    return c


@gf.cell
def straight_with_padding_copy(padding: float = 3.0) -> gf.Component:
    """Solution2: create component copy
    Args:
        default: default padding on all sides
    """
    c = gf.components.straight()
    c = c.copy()
    c.add_padding(default=padding)  # add padding to original cell
    return c


if __name__ == "__main__":
    # c1 = straight_with_padding(padding=1)
    # c2 = straight_with_padding(padding=3)

    # c1 = straight_with_padding_container(default=1)
    # c2 = straight_with_padding_container(default=3)

    c1 = straight_with_padding_copy(padding=1)
    c2 = straight_with_padding_copy(padding=3)

    print(c1.name)
    print(c2.name)
    assert c1.name != c2.name, f"{c1.name} and {c2.name} must be different"
