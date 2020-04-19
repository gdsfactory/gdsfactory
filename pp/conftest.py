import shutil
import pytest

from pp import CONFIG


@pytest.fixture(autouse=True)
def cleandir(request):
    build_folder = CONFIG["build_directory"]
    module_path = CONFIG["module_path"]

    if build_folder.exists() and "noautofixt" not in request.keywords:
        shutil.rmtree(build_folder)

    for build_folder in module_path.glob("**/build"):
        shutil.rmtree(build_folder)
