from __future__ import annotations

import pathlib
import re
from difflib import unified_diff
from enum import Enum
from typing import Annotated, Optional

import typer
from rich import print as pprint

from gdsfactory import show as _show
from gdsfactory.config import print_version_plugins
from gdsfactory.difftest import diff
from gdsfactory.install import install_gdsdiff, install_klayout_package
from gdsfactory.read.from_updk import from_updk
from gdsfactory.watch import watch as _watch

app = typer.Typer()


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
def write_cells(gdspath: str, dirpath: str = "", recursively: bool = True) -> None:
    """Write each all level cells into separate GDS files."""
    from gdsfactory.write_cells import write_cells as write_cells_top_cells
    from gdsfactory.write_cells import write_cells_recursively

    if recursively:
        write_cells_recursively(gdspath=gdspath, dirpath=dirpath)
    else:
        write_cells_top_cells(gdspath=gdspath, dirpath=dirpath)


@app.command()
def merge_gds(dirpath: str = "", gdspath: str = "") -> None:
    """Merges GDS cells from a directory into a single GDS."""
    from gdsfactory.read.from_gdspaths import from_gdsdir

    dirpath_path = pathlib.Path(dirpath) or pathlib.Path.cwd()
    gdspath_path = pathlib.Path(gdspath) or pathlib.Path.cwd() / "merged.gds"
    c = from_gdsdir(dirpath=dirpath_path)
    c.write_gds(gdspath=gdspath_path)
    c.show()


@app.command()
def watch(
    path: str = str(pathlib.Path.cwd()),
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
def version() -> None:
    """Show installed plugin versions."""
    print_version_plugins()


@app.command(name="from-updk")
def from_updk_command(filepath: str, filepath_out: str = "") -> None:
    """Writes a PDK in python from uPDK YAML spec."""
    filepath_path = pathlib.Path(filepath)
    filepath_out_path = filepath_out or filepath_path.with_suffix(".py")
    from_updk(filepath, filepath_out=filepath_out_path)


@app.command()
def text_from_pdf(filepath: str) -> None:
    """Converts a PDF to text."""
    import pdftotext

    with open(filepath, "rb") as f:
        pdf = pdftotext.PDF(f)

    # Read all the text into one string
    text = "\n".join(pdf)
    filepath_path = pathlib.Path(filepath)
    filepath_path_with_suffix = filepath_path.with_suffix(".md")
    filepath_path_with_suffix.write_text(text)


@app.command()
def migrate(
    migration: Annotated[
        Migration,
        typer.Option(
            case_sensitive=False,
            help="Choices of migrations. See the migration guide for more explanation "
            "https://gdsfactory.github.io/gdsfactory/migration.html",
        ),
    ],
    input: Annotated[pathlib.Path, typer.Argument(help="Input folder or file.")],
    output: Annotated[
        Optional[pathlib.Path],  # noqa: UP007
        typer.Argument(
            help="Output folder or file. If inplace is set, this argument will be ignored"
        ),
    ] = None,
    inplace: Annotated[
        bool,
        typer.Option(
            "--inplace",
            "-i",
            help="If set, the migration will overwrite the input folder"
            " or file and ignore any given output path.",
        ),
    ] = False,
) -> None:
    """Migrates python scripts to new syntax.

    It will only update `.py` files unless input is an exact file and not a directory.
    """
    if migration in [Migration.upgrade7to8, Migration.upgrade7to8unsafe]:
        to_be_replaced = {
            "center",
            "dmirror",
            "dmove",
            "dmovex",
            "dmovey",
            "drotate",
            "dsize_info",
            "dx",
            "dxmin",
            "dxmax",
            "dxsize",
            "dy",
            "dymin",
            "dymax",
            "dysize",
        }
        input = input.resolve()
        if output is None:
            if not inplace:
                raise ValueError(
                    "If inplace is not set, an output directory must be set."
                )
            output = input
        output.resolve()
        if migration == Migration.upgrade7to8unsafe:
            pattern1 = re.compile(
                r"\b(" + "|".join(r"d\." + _r for _r in to_be_replaced) + r")\b"
            )
            pattern2 = re.compile(r"\b(" + "|".join(to_be_replaced) + r")\b")
            replacement = r"d\1"
        else:
            pattern1 = re.compile(
                r"(?<=\.)(" + "|".join(r"d\." + _r for _r in to_be_replaced) + r")\b"
            )
            pattern2 = re.compile(r"(?<=\.)(" + "|".join(to_be_replaced) + r")\b")
            replacement = r"d\1"

        if not input.is_dir():
            if output.is_dir():
                output = output / input.name
            elif output.suffix == ".py":
                output.parent.mkdir(parents=True, exist_ok=True)
            else:
                output = output / input.name
                output.parent.mkdir(parents=True, exist_ok=True)

            with open(input, encoding="utf-8") as file:
                content = file.read()
            new_content = pattern2.sub(replacement, pattern1.sub(replacement, content))
            if output == input:
                if content != new_content:
                    with open(output, "w", encoding="utf-8") as file:
                        file.write(new_content)
                    pprint(f"Updated [bold violet]{output}[/]")
                    pprint(
                        "\n".join(
                            unified_diff(
                                a=content.splitlines(),
                                b=new_content.splitlines(),
                                fromfile=str(input.resolve()),
                                tofile=str(output.resolve()),
                            )
                        )
                    )
            else:
                with open(output, "w", encoding="utf-8") as file:
                    file.write(new_content)
                if content != new_content:
                    pprint(f"Updated [bold violet]{output}[/]")
                    pprint(
                        "\n".join(
                            unified_diff(
                                a=content,
                                b=new_content,
                                fromfile=str(input),
                                tofile=str(output),
                            )
                        )
                    )
        elif output == input:
            for inp in input.rglob("*.py"):
                with open(inp, encoding="utf-8") as file:
                    content = file.read()
                new_content = pattern2.sub(
                    replacement, pattern1.sub(replacement, content)
                )
                if content != new_content:
                    out = output / inp.relative_to(input)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with open(out, "w", encoding="utf-8") as file:
                        file.write(new_content)
                    pprint(f"Updated [bold violet]{out}[/]")
                    pprint(
                        "\n".join(
                            unified_diff(
                                a=content.splitlines(),
                                b=new_content.splitlines(),
                                fromfile=str(inp.resolve()),
                                tofile=str(out.resolve()),
                            )
                        )
                    )

        else:
            for inp in input.rglob("*.py"):
                with open(inp, encoding="utf-8") as file:
                    content = file.read()
                new_content = pattern2.sub(
                    replacement, pattern1.sub(replacement, content)
                )
                out = output / inp.relative_to(input)
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", encoding="utf-8") as file:
                    file.write(new_content)
                if content != new_content:
                    pprint(f"Updated [bold violet]{out}[/]")
                    pprint(
                        "\n".join(
                            unified_diff(
                                a=content.splitlines(),
                                b=new_content.splitlines(),
                                fromfile=str(inp.resolve()),
                                tofile=str(out.resolve()),
                            )
                        )
                    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:  # No arguments provided
        sys.argv.append("--help")
    app()
