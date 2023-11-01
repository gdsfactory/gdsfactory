from __future__ import annotations

import pathlib

import typer

from gdsfactory.config import print_version_pdks, print_version_plugins
from gdsfactory.difftest import diff
from gdsfactory.install import install_gdsdiff, install_klayout_package
from gdsfactory.technology import lyp_to_dataclass

app = typer.Typer()


@app.command()
def layermap_to_dataclass(
    filepath: str,
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion"),
) -> None:
    """Converts KLayout LYP to a dataclass."""

    filepath_lyp = pathlib.Path(filepath)
    filepath_py = filepath_lyp.with_suffix(".py")
    if not filepath_lyp.exists():
        raise FileNotFoundError(f"{filepath_lyp} not found")
    if not force and filepath_py.exists():
        raise FileExistsError(f"found {filepath_py}")
    lyp_to_dataclass(lyp_filepath=filepath_lyp)


@app.command()
def write_cells(gdspath: list[str], dirpath: str = None) -> None:
    """Write each all level cells into separate GDS files."""
    from gdsfactory.write_cells import write_cells as write_cells_to_separate_gds

    for path in gdspath:
        write_cells_to_separate_gds(gdspath=path, dirpath=dirpath)


@app.command()
def merge_gds(dirpath: str = None, gdspath: str = None) -> None:
    """Merges GDS cells from a directory into a single GDS."""
    from gdsfactory.read.from_gdspaths import from_gdsdir

    dirpath = dirpath or pathlib.Path.cwd()
    gdspath = gdspath or pathlib.Path.cwd() / "merged.gds"

    dirpath = pathlib.Path(dirpath)

    c = from_gdsdir(dirpath=dirpath)
    c.write_gds(gdspath=gdspath)
    c.show(show_ports=True)


@app.command()
def watch(
    path: str = str(pathlib.Path.cwd()),
    pdk: str = typer.Option(None, "--pdk", "-pdk", help="PDK name"),
) -> None:
    """Filewatch a folder for changes in *.py or *.pic.yml files."""
    from gdsfactory.watch import watch

    p = pathlib.Path(path)
    if not p.exists():
        raise ValueError(f"Invalid path passed to watch command: {p}")
    p = p.parent if p.is_file() else p
    watch(str(p), pdk=pdk)


@app.command()
def show(filename: str) -> None:
    """Show a GDS file using klive."""
    from gdsfactory.show import show

    show(filename)


@app.command()
def gds_diff(gdspath1: str, gdspath2: str, xor: bool = False) -> None:
    """Show boolean difference between two GDS files."""

    diff(gdspath1, gdspath2, xor=xor)


@app.command()
def install_klayout_genericpdk() -> None:
    """Install Klayout generic PDK."""

    install_klayout_package()


@app.command()
def install_git_diff() -> None:
    """Install git diff."""

    install_gdsdiff()


@app.command()
def print_plugins() -> None:
    """Show installed plugin versions."""

    print_version_plugins()


@app.command()
def print_pdks() -> None:
    """Show installed PDK versions."""

    print_version_pdks()


@app.command(name="from_updk")
def from_updk_command(filepath: str, filepath_out: str = None) -> None:
    """Writes a PDK in python from uPDK YAML spec."""
    from gdsfactory.read.from_updk import from_updk

    filepath = pathlib.Path(filepath)
    filepath_out = filepath_out or filepath.with_suffix(".py")
    from_updk(filepath, filepath_out=filepath_out)


@app.command(name="text_from_pdf")
def text_from_pdf_command(filepath: str) -> None:
    """Converts a PDF to text."""
    import pdftotext

    with open(filepath, "rb") as f:
        pdf = pdftotext.PDF(f)

    # Read all the text into one string
    text = "\n".join(pdf)
    filepath = pathlib.Path(filepath)
    f = filepath.with_suffix(".md")
    f.write_text(text)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:  # No arguments provided
        sys.argv.append("--help")
    app()
