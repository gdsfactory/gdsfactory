"""Store path."""

__all__ = ["PATH"]

import pathlib

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
module = pathlib.Path(__file__).parent.absolute()
repo = module.parent


class Path:
    module = module
    repo = repo
    gds = module / "gds"
    klayout = module / "klayout"


PATH = Path()
