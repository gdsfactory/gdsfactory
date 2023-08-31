# from pprint import pprint

from __future__ import annotations

import jsondiff

import gdsfactory as gf


def test_read_gds_hash() -> None:
    gdspath = gf.PATH.gdsdir / "straight.gds"
    c = gf.import_gds(gdspath, unique_names=False)
    h = "c956390621a5322a185cd20b0072a778fc613195"
    assert c.hash_geometry() == h, f"h = {c.hash_geometry()!r}"


# def test_read_gds_with_settings(data_regression: DataRegressionFixture) -> None:
#     gdspath = gf.PATH.gdsdir / "straight.gds"
#     c = gf.import_gds(gdspath)
#     data_regression.check(c.to_dict())


def test_read_gds_equivalent() -> None:
    """Ensures Component from GDS + YAML loads same component settings."""
    c1 = gf.components.straight(length=1.234)
    gdspath = gf.PATH.gdsdir / "straight.gds"

    c2 = gf.import_gds(gdspath, read_metadata=True, unique_names=False)
    d1 = c1.to_dict()
    d2 = c2.to_dict()
    d1.pop("name")
    d2.pop("name")
    d = jsondiff.diff(d1, d2)

    # pprint(d1)
    # pprint(d2)
    # pprint(d)
    assert len(d) == 0, f"{c1.name} != {c2.name}"


def test_build_and_import() -> None:
    """Create a cell and then import the same cell from GDS."""
    gdspath = gf.PATH.gdsdir / "straight.gds"
    c = gf.Component("build_and_import")
    c << gf.components.straight(length=1.234)
    c << gf.import_gds(gdspath, unique_names=False)
    c.write_gds()


def test_import_and_build() -> None:
    """Import a same cell from GDS and then create the same cell."""
    gdspath = gf.PATH.gdsdir / "straight.gds"
    c = gf.Component("build_and_import")
    c << gf.import_gds(gdspath, unique_names=False)
    c << gf.components.straight(length=1.234)
    c.write_gds()


def _write() -> None:
    c1 = gf.components.straight(length=1.234)
    gdspath = gf.PATH.gdsdir / "straight.gds"
    c1.write_gds(gdspath=gdspath, with_metadata=True)
    c1.show()
    c1.pprint()


if __name__ == "__main__":
    # c1 = gf.components.straight(length=1.234)
    # gdspath = gf.PATH.gdsdir / "straight.gds"

    # c2 = gf.import_gds(gdspath, read_metadata=True, unique_names=False)
    # d1 = c1.to_dict()
    # d2 = c2.to_dict()
    # d = jsondiff.diff(d1, d2)
    # test_build_and_import()
    test_read_gds_equivalent()
