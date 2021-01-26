import pp


def test_name_from_args() -> None:
    name = "my_name"
    c = pp.Component(name)
    print(c)
    assert c.name == name


def test_name_in_kwargs() -> None:
    name = "my_name"
    c = pp.Component(name=name)
    print(c)
    assert c.name == name


def test_cell() -> None:
    c = pp.c.waveguide(length=11)
    print(c)
    assert c.name == "waveguide_L11"


if __name__ == "__main__":
    test_cell()
    test_name_from_args()
    test_name_in_kwargs()
