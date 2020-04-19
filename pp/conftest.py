import shutil
import pytest

from pp import CONFIG


@pytest.fixture(autouse=True)
def cleandir(request):
    build_folder = CONFIG["build_directory"]
    if build_folder.exists() and "noautofixt" in request.keywords:
        shutil.rmtree(build_folder)
