import pathlib

import pytest

from gdsfactory.config import PATH, logger


@pytest.fixture(scope="session")
def datadir() -> pathlib.Path:
    return PATH.repo / "test-data-regression"


@pytest.fixture(scope="session")
def original_datadir() -> pathlib.Path:
    return PATH.repo / "test-data-regression"


@pytest.fixture
def caplog(caplog):
    """Support `loguru` logger in pytest caplog fixture. See https://github.com/Delgan/loguru/issues/59."""
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)
