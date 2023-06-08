import sys

from gdsfactory.difftest import diff


def gdsdiff_git(
    path: str = "",
    curr_file: str = "",
    old_file: str = "",
    old_hex: str = "",
    old_mode: str = "",
    new_file: str = "",
    new_hex: str = "",
    new_mode: str = "",
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
    diff(old_file, curr_file, xor=True)


if __name__ == "__main__":
    gdsdiff_git(*sys.argv)
