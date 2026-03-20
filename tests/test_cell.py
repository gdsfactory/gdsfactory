import gdsfactory as gf
from gdsfactory import partial


@gf.cell
def inner(a: int = 1) -> gf.Component:
    c = gf.Component()
    c.add_ref(gf.components.rectangle(size=(a, a)))
    return c


@gf.cell
def outer(b: int = 1) -> gf.Component:
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
    assert b1.base is b2.base


def test_schematic_cell() -> None:
    def my_straight_schematic(width: float = 1.1, length: float = 3.5) -> gf.Schematic:
        s = gf.Schematic()
        s.info["c"] = 5
        s.create_port("o1", "strip", x=0, y=0, orientation=180)
        s.create_port("o2", "strip", x=length, y=0, orientation=0)
        return s

    @gf.cell(schematic_function=my_straight_schematic)
    def my_straight(width: float, length: float) -> gf.Component:
        return gf.Path(path=[(0, 0), (length, 0)]).extrude(
            cross_section="strip", width=1.1
        )

    assert gf.kcl.layout.cell("my_straight_W1p1_L5") is None
    factory = gf.kcl.factories["my_straight"]
    schematic = factory.get_schematic(width=1.1, length=5)
    assert schematic == my_straight_schematic(width=1.1, length=5)
    assert gf.kcl.layout.cell("my_straight_W1p1_L5") is None
    c = my_straight(1.1, 5)
    # bug in kfactory
    # assert schematic == c.schematic
    assert c.cell_index() == gf.kcl.layout.cell("my_straight_W1p1_L5").cell_index()


if __name__ == "__main__":
    test_partial()
