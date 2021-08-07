import gdsfactory as gf


def test_name_from_args() -> None:
    name = "my_name"
    c = gf.Component(name)
    print(c)
    assert c.name == name


def test_name_in_kwargs() -> None:
    name = "my_name"
    c = gf.Component(name=name)
    print(c)
    assert c.name == name


def test_cell() -> None:
    c = gf.components.straight(length=11)
    print(c)
    assert c.name == "straight_L11"


if __name__ == "__main__":
    test_cell()
    test_name_from_args()
    test_name_in_kwargs()
