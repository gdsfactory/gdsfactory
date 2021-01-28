import shutil

import pytest
from _pytest.fixtures import SubRequest

from pp.config import CONFIG, diff_path
from pp.merge_cells import merge_cells

# from pp import clear_cache


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
    c = merge_cells(diff_path.glob("*.gds"))
    c.show()
