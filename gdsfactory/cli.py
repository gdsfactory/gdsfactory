from __future__ import annotations

import os
import pathlib

import typer

app = typer.Typer()

VERSION = "7.7.0"


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
def write_cells(gdspath: str, dirpath: str = None) -> None:
    """Write each all level cells into separate GDS files."""
    from gdsfactory.write_cells import write_cells as write_cells_to_separate_gds

    write_cells_to_separate_gds(gdspath=gdspath, dirpath=dirpath)


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
def web(pdk: str = "generic", host: str = "localhost", port: int = 8765) -> None:
    """Opens web viewer."""
    import uvicorn

    os.environ["PDK"] = pdk
    uvicorn.run("gplugins.web.main:app", host=host, port=port, reload=True)


@app.command()
def watch(
    path: str = str(pathlib.Path.cwd()),
    pdk: str = typer.Option(None, "--pdk", "-pdk", help="PDK name"),
) -> None:
    """Filewatch a folder for changes in *.py or *.pic.yml files."""
    from gdsfactory.watch import watch

    path = pathlib.Path(path)
    path = path.parent if path.is_dir() else path
    watch(str(path), pdk=pdk)


@app.command()
def show(filename: str) -> None:
    """Show a GDS file using klive."""
    from gdsfactory.show import show

    show(filename)


@app.command()
def gds_diff(gdspath1: str, gdspath2: str, xor: bool = False) -> None:
    """Show boolean difference between two GDS files."""
    from gdsfactory.difftest import diff

    diff(gdspath1, gdspath2, xor=xor)


@app.command()
def install_klayout_genericpdk() -> None:
    """Install Klayout generic PDK."""
    from gdsfactory.install import install_klayout_package

    install_klayout_package()


@app.command()
def install_git_diff() -> None:
    """Install git diff."""
    from gdsfactory.install import install_gdsdiff

    install_gdsdiff()


@app.command()
def print_plugins() -> None:
    """Show installed plugin versions."""
    from gdsfactory.config import print_version_plugins

    print_version_plugins()


@app.command()
def print_pdks() -> None:
    """Show installed PDK versions."""
    from gdsfactory.config import print_version_pdks

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
