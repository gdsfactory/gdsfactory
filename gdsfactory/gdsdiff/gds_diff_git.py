"""A GDS diff tool which can be called by git when doing.

`git diff ...`

The function needs to take the arguments as described in
https://git-scm.com/docs/git/2.18.0#git-codeGITEXTERNALDIFFcode

"""
from __future__ import annotations

import subprocess
import sys

from gdsfactory.gdsdiff.gdsdiff import gdsdiff
from gdsfactory.show import show


def gds_diff_git(
    path, curr_file, old_file, old_hex, old_mode, new_file, new_hex, new_mode
) -> None:
    """Show diffs for two files when running git diff.

    Args:
        path: script to run path.
        curr_file: current GDS file.
        old_file: old GDS.
        old_hex: ignore.
        old_mode: ignore.
        new_file: new GDS file.
        new_hex: ignore.
        new_mode: ignore.
    """
    diff = gdsdiff(curr_file, new_file, xor=False)
    show(diff)
    p = subprocess.Popen(["strmcmp", "-u", curr_file, new_file], stdin=subprocess.PIPE)
    if p.stdout:
        print(p.stdout.read())
    if p.stderr:
        print(p.stderr.read())


if __name__ == "__main__":
    gds_diff_git(*sys.argv)
