"""Symlink tech to klayout."""

import os
import pathlib
import shutil
import sys

import typer

from mypdk import __version__

app = typer.Typer()


def remove_path_or_dir(dest: pathlib.Path) -> None:
    """Remove path or directory."""
    if dest.is_dir():
        if sys.platform == "win32":
            shutil.rmtree(dest)
        else:
            dest.unlink()
    else:
        dest.unlink()


def make_link(src: str, dest: str, overwrite: bool = True) -> None:
    """Make a link from src to dest."""
    dest_path = pathlib.Path(dest)
    src_path = pathlib.Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} does not exist")
    if dest_path.exists() and (not overwrite):
        print(f"{src_path} already exists")
        return
    if dest_path.exists() or dest_path.is_symlink():
        print(f"removing {src_path} already installed")
        remove_path_or_dir(dest_path)
    try:
        os.symlink(src_path, dest_path, target_is_directory=True)
    except OSError:
        shutil.copytree(src_path, dest_path)
    print("link made:")
    print(f"From: {src_path}")
    print(f"To:   {dest_path}")


@app.command()
def install() -> None:
    """Install Klayout layermap."""
    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    cwd = pathlib.Path(__file__).resolve().parent
    home = pathlib.Path.home()
    dest_folder = home / klayout_folder / "tech"
    dest_folder.mkdir(exist_ok=True, parents=True)
    src = cwd / "klayout"
    dest = dest_folder / "mypdk"
    make_link(src=str(src), dest=str(dest))


@app.command()
def version() -> None:
    """Print version."""
    print(__version__)
