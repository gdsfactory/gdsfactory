import shutil

import pytest
from _pytest.fixtures import SubRequest

from pp import CONFIG

# from pp import clear_cache


@pytest.fixture(autouse=True)
def cleandir(request: SubRequest) -> None:
    # clear_cache()
    build_folder = CONFIG["build_directory"]
    module_path = CONFIG["module_path"]

    if build_folder.exists() and "noautofixt" not in request.keywords:
        shutil.rmtree(build_folder)

    for build_folder in module_path.glob("**/build"):
        shutil.rmtree(build_folder)
