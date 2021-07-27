"""Package configuration.

FIXME: tidy3d does not have __version__
# import tidy3d as td
# logger.info(f"tidy3d {td.__version__}")
"""


import pathlib
from loguru import logger
import pp


__version__ = "0.0.1"
home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
sparameters = repo_path / "sparameters"
sparameters.mkdir(exist_ok=True, parents=True)

# home_path = pathlib.Path.home() / ".gdsfactory"
# home_path.mkdir(exist_ok=True, parents=True)
# logpath = home_path / "gtidy3d.log"
logpath = cwd / "gtidy3d.log"

logger.info(f"gdsfactory {pp.__version__}")
logger.info(f"gtidy3d {__version__}")
logger.add(sink=logpath)


class Path:
    module = module_path
    repo = repo_path
    sparameters = repo_path / "sparameters"
    results = home / ".tidy3d"


PATH = Path()
PATH.results.mkdir(exist_ok=True)

__all__ = ["PATH"]
if __name__ == "__main__":
    print(PATH.repo)
