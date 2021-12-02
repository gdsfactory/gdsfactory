import gdsfactory as gf


def test_flatten():
    c1 = gf.c.mzi()
    c2 = c1.flatten()

    assert len(c1.references) > 0, f"{len(c1.references)}"
    assert len(c2.references) == 0, f"{len(c2.references)}"
    assert c1.name != c2.name, f"{c1.name} must be different from {c2.name}"


if __name__ == "__main__":
    c1 = gf.c.mzi()
    c2 = c1.flatten(single_layer=(2, 0))

    assert len(c1.references) > 0, f"{len(c1.references)}"
    assert len(c2.references) == 0, f"{len(c2.references)}"
    assert c1.name != c2.name, f"{c1.name} must be different from {c2.name}"

    c2.show()
