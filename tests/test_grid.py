import gdsfactory as gf


@gf.cell
def swatch(index: int) -> gf.Component:
    return gf.components.rectangle(size=(1, 1), layer=(index + 1, 0))


def test_grid_with_None_ports(rows: int = 3, columns: int = 4) -> None:
    swatches = [swatch(index) for index in range(11)]
    c = gf.grid(
        components=swatches,
        spacing=(1, 1),
        shape=(rows, columns),
        align_x="x",
        align_y="y",
    )
    assert c


def test_grid_with_ports(rows: int = 3, columns: int = 4) -> None:
    n = 3
    components = [gf.c.rectangle(size=(1, 1)) for _ in range(3)]
    c = gf.grid(components=components)
    assert len(c.ports) == n * 4, len(c.ports)


if __name__ == "__main__":
    # test_grid_with_None_ports()
    test_grid_with_ports()
