import shutil

import pytest
from _pytest.fixtures import SubRequest

import gdsfactory as gf
from gdsfactory.config import CONFIG, diff_path

# from gdsfactory import clear_cache


@pytest.fixture(scope="session", autouse=False)
def cleandir(request: SubRequest) -> None:
    # clear_cache()
    build_folder = CONFIG["build_directory"]
    module_path = CONFIG["module_path"]

    if diff_path.exists():
        shutil.rmtree(diff_path)

    if build_folder.exists() and "noautofixt" not in request.keywords:
        shutil.rmtree(build_folder)

    for build_folder in module_path.glob("**/build"):
        shutil.rmtree(build_folder)


@pytest.fixture(scope="session")
def show_diffs(request: SubRequest) -> None:
    c = gf.component_from.gdspaths(diff_path.glob("*.gds"))
    c.show()


collect_ignore = ["difftest.py"]
