import pytest

from gdsfactory.component import GDSDIR_TEMP
from gdsfactory.config import PATH
from gdsfactory.write_cells import write_cells, write_cells_recursively


@pytest.mark.skip("TODO")
def test_write_cells_recursively() -> None:
    gdspath = PATH.gdsdir / "mzi2x2.gds"
    gdspaths = write_cells_recursively(gdspath=gdspath, dirpath=GDSDIR_TEMP)
    assert len(gdspaths) == 10, len(gdspaths)


@pytest.mark.skip("TODO")
def test_write_cells() -> None:
    gdspath = PATH.gdsdir / "alphabet_3top_cells.gds"
    gdspaths = write_cells(gdspath=gdspath, dirpath=GDSDIR_TEMP)
    assert len(gdspaths) == 4, len(gdspaths)
