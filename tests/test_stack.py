import pytest

import gdsfactory as gf
from gdsfactory.stack import stack_add


def _rect(width: float = 10.0, height: float = 5.0) -> gf.Component:
    return gf.components.rectangle(size=(width, height))


# ── empty history ────────────────────────────────────────────────────────────


def test_empty_refs_appends_and_returns_ref() -> None:
    c = gf.Component()
    refs: list = []
    ref = stack_add(c, refs, _rect())
    assert ref is not None
    assert len(refs) == 1
    assert refs[0] is ref


# ── vertical stacking ────────────────────────────────────────────────────────


def test_vertical_flush() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect())
    r2 = stack_add(c, refs, _rect())
    assert r2.ymax == pytest.approx(r1.ymin, abs=0.001)


def test_vertical_with_spacing() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect())
    r2 = stack_add(c, refs, _rect(), spacing=2.0)
    assert r2.ymax == pytest.approx(r1.ymin - 2.0, abs=0.001)


def test_vertical_three_refs_chain() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect())
    r2 = stack_add(c, refs, _rect(), spacing=1.0)
    r3 = stack_add(c, refs, _rect(), spacing=1.0)
    assert r2.ymax == pytest.approx(r1.ymin - 1.0, abs=0.001)
    assert r3.ymax == pytest.approx(r2.ymin - 1.0, abs=0.001)


# ── horizontal stacking ──────────────────────────────────────────────────────


def test_horizontal_flush() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(), direction="horizontal")
    r2 = stack_add(c, refs, _rect(), direction="horizontal")
    assert r2.xmin == pytest.approx(r1.xmax, abs=0.001)


def test_horizontal_with_spacing() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(), direction="horizontal")
    r2 = stack_add(c, refs, _rect(), direction="horizontal", spacing=3.0)
    assert r2.xmin == pytest.approx(r1.xmax + 3.0, abs=0.001)


# ── alignment ────────────────────────────────────────────────────────────────


def test_align_center_vertical() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(width=10.0))
    r2 = stack_add(c, refs, _rect(width=6.0), align="center")
    assert r2.x == pytest.approx(r1.x, abs=0.001)


def test_align_min_vertical() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(width=10.0))
    r2 = stack_add(c, refs, _rect(width=6.0), align="min")
    assert r2.xmin == pytest.approx(r1.xmin, abs=0.001)


def test_align_max_vertical() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(width=10.0))
    r2 = stack_add(c, refs, _rect(width=6.0), align="max")
    assert r2.xmax == pytest.approx(r1.xmax, abs=0.001)


def test_align_center_horizontal() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(height=10.0), direction="horizontal")
    r2 = stack_add(c, refs, _rect(height=6.0), direction="horizontal", align="center")
    assert r2.y == pytest.approx(r1.y, abs=0.001)


def test_align_min_horizontal() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(height=10.0), direction="horizontal")
    r2 = stack_add(c, refs, _rect(height=6.0), direction="horizontal", align="min")
    assert r2.ymin == pytest.approx(r1.ymin, abs=0.001)


def test_align_max_horizontal() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(height=10.0), direction="horizontal")
    r2 = stack_add(c, refs, _rect(height=6.0), direction="horizontal", align="max")
    assert r2.ymax == pytest.approx(r1.ymax, abs=0.001)


# ── rotation ─────────────────────────────────────────────────────────────────


def test_rotate_then_stack_vertical() -> None:
    """Rotation happens before positioning — bbox of rotated ref is used."""
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(width=10.0, height=5.0))
    r2 = stack_add(c, refs, _rect(width=10.0, height=5.0), rotate=90)
    assert r2.ymax == pytest.approx(r1.ymin, abs=0.001)


# ── rows / columns array refs ────────────────────────────────────────────────


def test_rows_array_bounding_box_used_for_stacking() -> None:
    """The array's full bounding box drives the next ref's position."""
    c = gf.Component()
    refs: list = []
    # 3 rows of 5x5 rects with 7um pitch -> total height = 5 + 2*(7) = 19um? No:
    # row_pitch is origin-to-origin, so ymin of array = 0, ymax = 5 + 2*7 = 19
    r1 = stack_add(c, refs, _rect(width=5.0, height=5.0), rows=3, row_pitch=7.0)
    r2 = stack_add(c, refs, _rect(width=5.0, height=5.0))
    assert r2.ymax == pytest.approx(r1.ymin, abs=0.001)


def test_columns_array_bounding_box_used_for_stacking() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(
        c,
        refs,
        _rect(width=5.0, height=5.0),
        direction="horizontal",
        columns=3,
        column_pitch=7.0,
    )
    r2 = stack_add(c, refs, _rect(width=5.0, height=5.0), direction="horizontal")
    assert r2.xmin == pytest.approx(r1.xmax, abs=0.001)


# ── edge cases ───────────────────────────────────────────────────────────────


def test_negative_spacing_overlaps() -> None:
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect())
    r2 = stack_add(c, refs, _rect(), spacing=-2.0)
    # ymax = r1.ymin - (-2) = r1.ymin + 2  →  overlap of 2um
    assert r2.ymax == pytest.approx(r1.ymin + 2.0, abs=0.001)


def test_refs_list_grows_correctly() -> None:
    c = gf.Component()
    refs: list = []
    for _ in range(5):
        stack_add(c, refs, _rect())
    assert len(refs) == 5


def test_no_align_does_not_move_x() -> None:
    """Without align, orthogonal position is whatever add_ref gives (origin)."""
    c = gf.Component()
    refs: list = []
    r1 = stack_add(c, refs, _rect(width=10.0))
    r2 = stack_add(c, refs, _rect(width=6.0))  # no align
    # r2 was placed at origin; stacking only moves y — x stays at origin xmin=0
    assert r2.xmin == pytest.approx(0.0, abs=0.001)
    assert r2.ymax == pytest.approx(r1.ymin, abs=0.001)
