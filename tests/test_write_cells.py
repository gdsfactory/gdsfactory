from gdsfactory.config import PATH
from gdsfactory.write_cells import write_cells, write_cells_recursively


def test_write_cells_recursively() -> None:
    gdspath = PATH.gdsdir / "mzi2x2.gds"
    gdspaths = write_cells_recursively(gdspath=gdspath, dirpath="extra/gds")
    assert len(gdspaths) == 13, len(gdspaths)


def test_write_cells() -> None:
    gdspath = PATH.gdsdir / "alphabet_3top_cells.gds"
    gdspaths = write_cells(gdspath=gdspath, dirpath="extra/gds")
    assert len(gdspaths) == 3, len(gdspaths)
