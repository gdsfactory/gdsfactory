import pathlib

import pytest

from gdsfactory.config import PATH


@pytest.fixture(scope="session")
def datadir() -> pathlib.Path:
    return PATH.repo / "test-data-regression"


@pytest.fixture(scope="session")
def original_datadir() -> pathlib.Path:
    return PATH.repo / "test-data-regression"
