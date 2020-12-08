"""
A GDS diff tool which can be called by git when doing

`git diff ...`

The function needs to take the arguments as described in
https://git-scm.com/docs/git/2.18.0#git-codeGITEXTERNALDIFFcode

"""
import sys

from pp.gdsdiff.gdsdiff import gdsdiff
from pp.write_component import show


def gds_diff_git(
    path, curr_file, old_file, old_hex, old_mode, new_file, new_hex, new_mode
):
    """
    We do not use most of the arguments
    """
    print(old_hex, "->", new_hex)
    diff = gdsdiff(old_file, new_file)
    show(diff)


if __name__ == "__main__":
    # for f in sys.argv:
    # print (f)
    gds_diff_git(*sys.argv)
