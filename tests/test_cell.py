import gdsfactory as gf
from gdsfactory import partial


@gf.cell
def inner(a=1) -> gf.Component:
    c = gf.Component()
    c.add_ref(gf.components.rectangle(size=(a, a)))
    return c


@gf.cell
def outer(b=1) -> gf.Component:
    return inner(b)


def test_double_decorated_cell() -> None:
    c = outer(b=10)
    d = c.settings.model_dump()
    assert d == dict(b=10), d


def test_partial() -> None:
    x1 = partial(gf.cross_section.cross_section, layer=(2, 0), width=0.6)
    x2 = partial(gf.cross_section.cross_section, layer=(2, 0), width=0.6)
    b1 = gf.components.bend_euler(cross_section=x1)
    b2 = gf.components.bend_euler(cross_section=x2)
    assert id(b1) == id(b2)


if __name__ == "__main__":
    test_partial()
