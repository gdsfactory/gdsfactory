import gdsfactory as gf


@gf.cell
def inner(a=1):
    c = gf.Component()
    c.add_ref(gf.components.rectangle(size=(a, a)))
    return c


@gf.cell
def outer(b=1):
    return inner(b)


def test_double_decorated_cell():
    c = outer(b=10)
    d = c.settings.model_dump()
    assert d == dict(b=10), d


if __name__ == "__main__":
    test_double_decorated_cell()
    # c = outer(b=10)
    # print(c.settings)
    # c.show()
