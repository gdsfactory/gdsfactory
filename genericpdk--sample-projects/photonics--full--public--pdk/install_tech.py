"""Symlink tech to klayout."""

import os
import pathlib
import shutil


def remove_path_or_dir(dest: pathlib.Path):
    """Remove a path or directory."""
    if dest.is_dir():
        os.unlink(dest)
    else:
        os.remove(dest)


def make_link(src, dest, overwrite: bool = True) -> None:
    """Make a symbolic link from src to dest."""
    dest = pathlib.Path(dest)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")

    if dest.exists() and not overwrite:
        print(f"{dest} already exists")
        return
    if dest.exists() or dest.is_symlink():
        print(f"removing {dest} already installed")
        remove_path_or_dir(dest)
    try:
        os.symlink(src, dest, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dest)
    print("link made:")
    print(f"From: {src}")
    print(f"To:   {dest}")
