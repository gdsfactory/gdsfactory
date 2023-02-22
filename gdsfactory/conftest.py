from __future__ import annotations

import pytest
from _pytest.fixtures import SubRequest

import gdsfactory as gf
from gdsfactory.config import diff_path

PDK = gf.get_generic_pdk()
PDK.activate()


@pytest.fixture
def datadir(original_datadir, tmpdir):
    return original_datadir


@pytest.fixture(scope="session")
def show_diffs(request: SubRequest) -> None:
    c = gf.read.from_gdspaths(diff_path.glob("*.gds"))
    c.show(show_ports=True)


collect_ignore = ["difftest.py"]
