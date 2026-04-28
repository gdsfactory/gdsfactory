import pathlib
import platform
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import kfactory as kf
import pytest
from hypothesis import settings
from pytest_regressions.file_regression import FileRegressionFixture

from gdsfactory.config import CONF, PATH

settings.register_profile("default", max_examples=200)
settings.load_profile("default")


@pytest.fixture(scope="session", autouse=True)
def activate_generic_pdk() -> None:
    """Activate the generic PDK for all tests."""
    from gdsfactory.gpdk import PDK

    PDK.activate()

    # Don't embed klayout version/context info in GDS files,
    # so that binary comparisons are stable across environments.
    CONF.write_kfactory_settings = False

    CONF.on_placer_error = "warning"
    CONF.on_collision = "warning"

    return


@pytest.fixture
def kcl(request: pytest.FixtureRequest) -> kf.KCLayout:
    """Create a per-test KCLayout namespace for parallel test isolation."""
    from kfactory.kcell import clean_name

    return kf.KCLayout(name=clean_name(request.node.name))


def _layout_xor(
    obtained_filename: pathlib.Path,
    expected_filename: pathlib.Path,
    tolerance: int = 0,
    raises: Literal["error", "warning"] = "error",
) -> None:
    """Compare two GDS/OAS files using KLayout's LayoutDiff.

    Compares the entire GDS (not just shapes), including metadata.
    """
    diff = kf.kdb.LayoutDiff()
    ly_a = kf.kdb.Layout()
    ly_a.read(str(obtained_filename))
    ly_b = kf.kdb.Layout()
    ly_b.read(str(expected_filename))

    flags = (
        kf.kdb.LayoutDiff.Verbose
        | kf.kdb.LayoutDiff.WithMetaInfo
        | kf.kdb.LayoutDiff.NoLayerNames
    )

    if not diff.compare(ly_a, ly_b, flags=flags, tolerance=tolerance):
        match raises:
            case "error":
                raise AssertionError(
                    f"Layouts {str(obtained_filename)!r} and {str(expected_filename)!r} differ!"
                )
            case "warning":
                warnings.warn(
                    f"Layouts {str(obtained_filename)!r} and {str(expected_filename)!r} differ!",
                    stacklevel=3,
                )


@pytest.fixture
def gds_regression(
    file_regression: FileRegressionFixture,
) -> Callable[[kf.ProtoTKCell[Any]], None]:
    """Regression test fixture that compares the entire GDS file, not just shapes.

    Uses file_regression to store/compare raw GDS bytes, with XOR-based
    comparison on failure to detect geometry differences.
    """
    saveopts = kf.save_layout_options()

    # Stricter on Linux (CI), warn on other platforms
    raises: Literal["error", "warning"] = (
        "error" if platform.system() == "Linux" else "warning"
    )

    def _check(
        c: kf.ProtoTKCell[Any],
        tolerance: int = 0,
    ) -> None:
        c.kcl.layout.clear_meta_info()

        file_regression.check(
            c.write_bytes(saveopts, convert_external_cells=True),
            binary=True,
            extension=".gds",
            check_fn=partial(_layout_xor, tolerance=tolerance, raises=raises),
        )

    return _check


@pytest.fixture(scope="session")
def datadir() -> pathlib.Path:
    return PATH.repo / "tests/test-data-regression"


@pytest.fixture(scope="session")
def original_datadir() -> pathlib.Path:
    return PATH.repo / "tests/test-data-regression"
