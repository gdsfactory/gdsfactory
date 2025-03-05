import pytest

import gdsfactory as gf
from gdsfactory.config import GDSDIR_TEMP, PATH
from gdsfactory.write_cells import (
    get_import_gds_script,
    get_script,
    write_cells,
    write_cells_recursively,
)


def test_write_cells_recursively() -> None:
    gdspath = PATH.gdsdir / "mzi2x2.gds"
    gf.clear_cache()
    gdspaths = write_cells_recursively(gdspath=gdspath, dirpath=GDSDIR_TEMP)
    assert len(gdspaths) == 10, len(gdspaths)


def test_write_cells() -> None:
    gdspath = PATH.gdsdir / "alphabet_3top_cells.gds"
    gf.clear_cache()
    gdspaths = write_cells(gdspath=gdspath, dirpath=GDSDIR_TEMP)
    assert len(gdspaths) == 3, len(gdspaths)


def test_get_import_gds_script() -> None:
    path = GDSDIR_TEMP / "test.gds"
    gf.components.mzi().write_gds(path)
    script = get_import_gds_script(GDSDIR_TEMP)
    assert script, "Script should not be empty"

    script = get_script(path, module="test_module")
    assert "test_module.test()" in script


def test_get_import_gds_script_no_dir() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        get_import_gds_script("nonexistent_dir")


def test_get_import_gds_script_empty_dir() -> None:
    dirpath = GDSDIR_TEMP / "empty_dir"
    dirpath.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match=r"No GDS files found at .*"):
        get_import_gds_script(dirpath)
