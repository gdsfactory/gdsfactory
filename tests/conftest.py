import pathlib

import pytest

from gdsfactory.config import PATH


@pytest.fixture(scope="session", autouse=True)
def activate_generic_pdk() -> None:
    """Activate the generic PDK for all tests."""
    from gdsfactory.gpdk import PDK

    PDK.activate()
    return
    # No teardown needed - PDK state is global


@pytest.fixture(scope="session")
def datadir() -> pathlib.Path:
    return PATH.repo / "tests/test-data-regression"


@pytest.fixture(scope="session")
def original_datadir() -> pathlib.Path:
    return PATH.repo / "tests/test-data-regression"
