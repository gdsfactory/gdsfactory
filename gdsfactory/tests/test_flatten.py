import gdsfactory as gf


def test_flatten() -> None:
    c1 = gf.components.mzi()
    c2 = c1.flatten()

    assert len(c1.references) > 0, f"{len(c1.references)}"
    assert len(c2.references) == 0, f"{len(c2.references)}"
    assert c1.name != c2.name, f"{c1.name} must be different from {c2.name}"


def test_two_flats_in_one():
    c = gf.Component()
    c1 = gf.components.straight()
    c2 = c1.flatten()
    c3 = c1.flatten()
    c3.add_label("I'm different")

    c << c1
    r2 = c << c2
    r3 = c << c3
    r2.movey(-100)
    r3.movey(-200)
    assert c2.name != c3.name


def test_flattened_cell_keeps_info():
    c1 = gf.components.straight()
    c2 = c1.flatten()
    assert (
        len(c1.info) > 0
    ), "This test doesn't make any sense unless there is some info to copy"
    assert c1.info == c2.info


if __name__ == "__main__":
    c1 = gf.components.mzi()
    c2 = c1.flatten(single_layer=(2, 0))

    assert len(c1.references) > 0, f"{len(c1.references)}"
    assert len(c2.references) == 0, f"{len(c2.references)}"
    assert c1.name != c2.name, f"{c1.name} must be different from {c2.name}"

    c2.show()
