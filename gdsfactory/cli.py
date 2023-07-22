"""Command line interface. Type `gf` into a terminal."""

from __future__ import annotations

import pathlib
from typing import Optional

from click.core import Context, Option

import gdsfactory
from gdsfactory.config import cwd, print_config
from gdsfactory.config import print_version as _print_version
from gdsfactory.config import print_version_pdks, print_version_raw
from gdsfactory.generic_tech import LAYER
from gdsfactory.install import install_gdsdiff, install_klayout_package
from gdsfactory.technology import lyp_to_dataclass
from gdsfactory.typings import PathType
from gdsfactory.write_cells import write_cells as write_cells_to_separate_gds

try:
    import rich_click as click
except ImportError:
    import click

VERSION = "6.114.1"
LAYER_LABEL = LAYER.LABEL


def print_version(ctx: Context, param: Option, value: bool) -> None:
    """Prints the version."""
    if not value or ctx.resilient_parsing:
        return
    _print_version()
    ctx.exit()


@click.group()
def version() -> None:
    """Commands for printing gdsfactory extension versions."""
    pass


# install
@click.group()
def install() -> None:
    """Commands install."""
    pass


@click.command(name="config")
@click.argument("key", required=False, default=None)
def config_get(key: str) -> None:
    """Shows key values from CONFIG."""
    print_config(key)


# GDS
@click.group()
def gds() -> None:
    """Commands for dealing with GDS files."""
    pass


@click.command(name="layermap_to_dataclass")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--force", "-f", default=False, help="Force deletion", is_flag=True)
def layermap_to_dataclass(filepath, force: bool) -> None:
    """Converts KLayout LYP to a dataclass."""
    filepath_lyp = pathlib.Path(filepath)
    filepath_py = filepath_lyp.with_suffix(".py")
    if not filepath_lyp.exists():
        raise FileNotFoundError(f"{filepath_lyp} not found")
    if not force and filepath_py.exists():
        raise FileExistsError(f"found {filepath_py}")
    lyp_to_dataclass(lyp_filepath=filepath_lyp)


@click.command(name="write_cells")
@click.argument("gdspath")
@click.argument("dirpath", required=False, default=None)
def write_cells(gdspath, dirpath) -> None:
    """Write each all level cells into separate GDS files."""
    write_cells_to_separate_gds(gdspath=gdspath, dirpath=dirpath)


@click.command(name="merge_gds")
@click.argument("dirpath", required=False, default=None)
@click.argument("gdspath", required=False, default=None)
def merge_gds(
    dirpath: Optional[PathType] = None, gdspath: Optional[PathType] = None
) -> None:
    """Merges GDS cells from a directory into a single GDS."""
    dirpath = dirpath or pathlib.Path.cwd()
    gdspath = gdspath or pathlib.Path.cwd() / "merged.gds"

    dirpath = pathlib.Path(dirpath)

    c = gdsfactory.read.from_gdsdir(dirpath=dirpath)
    c.write_gds(gdspath=gdspath)
    c.show(show_ports=True)


DEFAULT_PORT = 8765
DEFAULT_HOST = "localhost"


@click.option(
    "--pdk",
    type=click.STRING,
    default="generic",
    help="Process Design Kit (PDK) to activate",
    show_default=True,
)
@click.option(
    "--host",
    "-h",
    type=click.STRING,
    default=DEFAULT_HOST,
    help="Host to run server on",
    show_default=True,
)
@click.option(
    "--port",
    "-p",
    type=click.INT,
    help=f"Port to run server on - defaults to {DEFAULT_PORT}",
    default=DEFAULT_PORT,
    show_default=True,
)
@click.command()
def web(
    pdk: str,
    host: str,
    port: int,
) -> None:
    """Opens web viewer."""
    import os

    import uvicorn

    from gdsfactory.plugins.web.main import app

    os.environ["PDK"] = pdk

    uvicorn.run(app, host=host, port=port)


@click.argument("path", type=click.Path(exists=True), required=False, default=cwd)
@click.command()
def watch(path=cwd) -> None:
    """Filewatch a folder for changes in python or pic.yaml files."""
    from gdsfactory.watch import watch

    path = pathlib.Path(path)
    path = path.parent if path.is_dir() else path
    watch(str(path))


# INIT
@click.group()
def init() -> None:
    """Commands for initializing projects."""
    pass


@click.command()
def notebooks() -> None:
    """Convert notebooks in py to ipynb."""
    from gdsfactory.install import convert_py_to_ipynb

    convert_py_to_ipynb()


# EXTRA
@click.command()
@click.argument("filename")
def show(filename: str) -> None:
    """Show a GDS file using klive."""
    gdsfactory.show(filename)


@click.command()
@click.argument("gdspath1")
@click.argument("gdspath2")
@click.option("--xor", "-x", default=False, help="include xor", is_flag=True)
def diff(gdspath1: str, gdspath2: str, xor: bool = False) -> None:
    """Show boolean difference between two GDS files."""
    from gdsfactory.difftest import diff

    diff(gdspath1, gdspath2, xor=xor)


@click.command()
def klayout_genericpdk() -> None:
    """Install Klayout generic PDK."""
    install_klayout_package()


@click.command()
def git_diff() -> None:
    """Install git diff."""
    install_gdsdiff()


@click.command()
def raw() -> None:
    """Show installed plugin versions."""
    print_version_raw()


@click.command()
def pdks() -> None:
    """Show installed PDK versions."""
    print_version_pdks()


@click.group()
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show the version number.",
)
def cli() -> None:
    """`gf` is the gdsfactory command line tool."""


gds.add_command(layermap_to_dataclass)
gds.add_command(write_cells)
gds.add_command(merge_gds)
gds.add_command(show)
gds.add_command(diff)

install.add_command(klayout_genericpdk)
install.add_command(git_diff)

version.add_command(raw)
version.add_command(pdks)

init.add_command(notebooks)

cli.add_command(web)
# watch.add_command(watch_yaml)

cli.add_command(gds)
cli.add_command(install)
cli.add_command(watch)
cli.add_command(version)
cli.add_command(init)


if __name__ == "__main__":
    # cli()
    print_version()
