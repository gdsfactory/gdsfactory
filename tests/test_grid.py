import pytest

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
        align_x="xmin",
        align_y="ymin",
    )
    assert c


def test_grid_with_ports(rows: int = 3, columns: int = 4) -> None:
    n = 3
    components = [gf.c.rectangle(size=(1, 1)) for _ in range(3)]
    c = gf.grid(components=components)
    assert len(c.ports) == n * 4, len(c.ports)


def _label_and_anchor_positions(
    c: gf.Component,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    labels = [
        (i.dtrans.disp.x, i.dtrans.disp.y)
        for i in c.insts
        if i.cell.name.startswith("text_")
    ]
    anchors = [
        i.dsize_info.center for i in c.insts if i.cell.name.startswith("rectangle")
    ]
    return labels, anchors


@pytest.mark.parametrize("text_rotation", [0, 90, 180, 270])
def test_grid_with_text_rotation_keeps_labels_on_anchors(text_rotation: int) -> None:
    components = [gf.c.rectangle(size=(1, 1)) for _ in range(3)]
    c = gf.grid_with_text(
        components, shape=(1, 3), spacing=(5, 5), text_rotation=text_rotation
    )
    labels, anchors = _label_and_anchor_positions(c)
    assert len(labels) == 3
    assert labels == anchors


def test_grid_with_text_mirror_keeps_labels_on_anchors() -> None:
    components = [gf.c.rectangle(size=(1, 1)) for _ in range(3)]
    c = gf.grid_with_text(components, shape=(1, 3), spacing=(5, 5), text_mirror=True)
    labels, anchors = _label_and_anchor_positions(c)
    assert len(labels) == 3
    assert labels == anchors
