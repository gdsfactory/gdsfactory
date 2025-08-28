from __future__ import annotations

import pathlib
from enum import Enum

import typer
from kfactory.cli.build import build

from gdsfactory import show as _show
from gdsfactory.config import print_version_plugins
from gdsfactory.difftest import diff
from gdsfactory.install import install_gdsdiff, install_klayout_package
from gdsfactory.read.from_updk import from_updk
from gdsfactory.watch import watch as _watch

app = typer.Typer()
app.command()(build)


class Migration(str, Enum):
    """Available Migrations."""

    upgrade7to8unsafe = "7to8-unsafe"
    upgrade7to8 = "7to8"


@app.command()
def layermap_to_dataclass(
    filepath: str,
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion"),
) -> None:
    """Converts KLayout LYP to a dataclass."""
    from gdsfactory.technology import lyp_to_dataclass

    filepath_lyp = pathlib.Path(filepath)
    filepath_py = filepath_lyp.with_suffix(".py")
    if not filepath_lyp.exists():
        raise FileNotFoundError(f"{filepath_lyp} not found")
    if not force and filepath_py.exists():
        raise FileExistsError(f"found {filepath_py}")
    lyp_to_dataclass(lyp_filepath=filepath_lyp)


@app.command()
def write_cells(
    gdspath: str,
    dirpath: str = typer.Option(
        "", "--dirpath", help="Directory path to write GDS files to"
    ),
    recursively: bool = typer.Option(True, help="Write cells recursively"),
) -> None:
    """Write each all level cells into separate GDS files."""
    from gdsfactory.write_cells import write_cells as write_cells_top_cells
    from gdsfactory.write_cells import write_cells_recursively

    if recursively:
        write_cells_recursively(gdspath=gdspath, dirpath=dirpath)
    else:
        write_cells_top_cells(gdspath=gdspath, dirpath=dirpath)


@app.command()
def merge_gds(
    dirpath: str = typer.Option(
        "", "--dirpath", help="Directory containing GDS files to merge"
    ),
    gdspath: str = typer.Option("", "--gdspath", help="Output GDS file path"),
) -> None:
    """Merges GDS cells from a directory into a single GDS."""
    from gdsfactory.read.from_gdspaths import from_gdsdir

    dirpath_path = pathlib.Path(dirpath) if dirpath else pathlib.Path.cwd()
    gdspath_path = (
        pathlib.Path(gdspath) if gdspath else pathlib.Path.cwd() / "merged.gds"
    )
    c = from_gdsdir(dirpath=dirpath_path)
    c.write_gds(gdspath=gdspath_path)
    c.show()


@app.command()
def watch(
    path: str = typer.Argument(str(pathlib.Path.cwd()), help="Folder to watch"),
    pdk: str = typer.Option(None, "--pdk", "-pdk", help="PDK name"),
    run_main: bool = typer.Option(False, "--run-main", "-rm", help="Run main"),
    run_cells: bool = typer.Option(False, "--run-cells", "-rc", help="Run cells"),
    pre_run: bool = typer.Option(
        False, "--pre-run", "-p", help="Build all cells on startup"
    ),
    overwrite: bool = typer.Option(True, help="Overwrite existing cells"),
) -> None:
    """Filewatch a folder for changes in *.py or *.pic.yml files.

    If a file changes, it will run the main function and show the cells.

    Args:
        path: folder to watch.
        pdk: process design kit.
        run_main: run the main function.
        run_cells: run the cells.
        pre_run: build all cells on startup.
        overwrite: overwrite existing cells.
    """
    path_path = pathlib.Path(path)
    path_path = path_path if path_path.is_dir() else path_path.parent
    path = str(path_path.absolute())
    if overwrite:
        from gdsfactory import CONF

        CONF.cell_overwrite_existing = True
    _watch(path, pdk=pdk, run_main=run_main, run_cells=run_cells, pre_run=pre_run)


@app.command()
def show(filename: str) -> None:
    """Show a GDS file using klive."""
    _show(filename)


@app.command()
def gds_diff(
    gdspath1: str, gdspath2: str, xor: bool = False, show: bool = False
) -> None:
    """Show boolean difference between two GDS files."""
    diff(gdspath1, gdspath2, xor=xor, show=show)


@app.command()
def install_klayout_genericpdk() -> None:
    """Install Klayout generic PDK."""
    install_klayout_package()


@app.command()
def install_git_diff() -> None:
    """Install git diff."""
    install_gdsdiff()


@app.command()
def version() -> None:
    """Show installed plugin versions."""
    print_version_plugins()


@app.command(name="from-updk")
def from_updk_command(
    filepath: str,
    filepath_out: str = typer.Option(
        "", "--output", "-o", help="Output Python file path"
    ),
) -> None:
    """Writes a PDK in python from uPDK YAML spec."""
    filepath_path = pathlib.Path(filepath)
    filepath_out_path = (
        pathlib.Path(filepath_out) if filepath_out else filepath_path.with_suffix(".py")
    )
    from_updk(filepath, filepath_out=filepath_out_path)


if __name__ == "__main__":
    app()
