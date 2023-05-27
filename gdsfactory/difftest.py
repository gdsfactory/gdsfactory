"""GDS regression test. Adapted from lytest.

TODO: adapt it into pytest_regressions
from __future__ import annotations
from pytest_regressions.file_regression import FileRegressionFixture
class GdsRegressionFixture(FileRegressionFixture):
    def check(self,
        contents,
        extension=".gds",
        basename=None,
        fullpath=None,
        binary=False,
        obtained_filename=None,
        check_fn=None,
            ):
        try:
            difftest(c)
"""
import pathlib
import shutil
from typing import Optional

import gdsfactory as gf
from gdsfactory.config import PATH, logger, GDSDIR_TEMP

from kfactory import KCell, KCLayout, kdb


class GeometryDifference(Exception):
    pass


def difftest(
    component: gf.Component,
    test_name: Optional[gf.Component] = None,
    dirpath: Optional[pathlib.Path] = None,
    xor: bool = True,
) -> None:
    """Avoids GDS regressions tests on the GeometryDifference.

    If files are the same it returns None. If files are different runs XOR
    between new component and the GDS reference stored in dirpath and
    raises GeometryDifference if there are differences and show differences in KLayout.

    If it runs for the fist time it just stores the GDS reference.

    Args:
        component: to test if it has changed.
        test_name: used to store the GDS file.
        dirpath: default directory for storing reference files.
    """
    # containers function_name is different from component.name
    # we store the container with a different name from original component
    test_name = test_name or (
        f"{component.function_name}_{component.name}"
        if hasattr(component, "function_name")
        and component.name != component.function_name
        else f"{component.name}"
    )
    filename = f"{test_name}.gds"
    dirpath = dirpath or PATH.cwd
    dirpath_ref = dirpath / "gds_ref"
    dirpath_run = GDSDIR_TEMP 

    ref_file = dirpath_ref / f"{test_name}.gds"
    run_file = dirpath_run / filename

    run = gf.get_component(component)
    run_file = run.write_gds(gdspath=run_file)

    if not ref_file.exists():
        component.write_gds(gdspath=ref_file)
        raise AssertionError(
            f"Reference GDS file for {test_name!r} not found. Writing to {ref_file!r}"
        )

    ref = read_top_cell(ref_file)
    run = read_top_cell(run_file)
    ld = kdb.LayoutDiff()

    if not ld.compare(ref._kdb_cell, run._kdb_cell, kdb.LayoutDiff.Verbose):
        c = KCell(f"{test_name}_diffs")
        refdiff = KCell(f"{test_name}_ref")
        rundiff = KCell(f"{test_name}_run")

        refdiff.copy_tree(ref._kdb_cell)
        rundiff.copy_tree(run._kdb_cell)
        c << refdiff
        c << rundiff

        if xor:
            diff = KCell(f"{test_name}_diff")

            for layer in c.kcl.layer_infos():
                layer = ref.layer(layer)
                region_run = kdb.Region(run.begin_shapes_rec(layer))
                region_ref = kdb.Region(ref.begin_shapes_rec(layer))

                region_diff = region_run - region_ref

                if not region_diff.is_empty():
                    layer_tuple = c.kcl.layer_infos()[layer]
                    region_xor = region_ref ^ region_run
                    diff.shapes(layer).insert(region_xor)
            c << diff

        c.show()

        print(
            f"\ngds_run {filename!r} changed from gds_ref {str(ref_file)!r}\n"
            "You can check the differences in Klayout GUI or run XOR with\n"
            f"gf gds diff --xor {ref_file} {run_file}\n"
        )

        try:
            overwrite(ref_file, run_file)
        except OSError as exc:
            raise GeometryDifference(
                "\n"
                f"{filename!r} changed from reference {str(ref_file)!r}\n"
                "To step over each error you can run `pytest -s`\n"
                "So you can check the differences in Klayout GUI\n"
            ) from exc


def overwrite(ref_file, run_file):
    val = input("Save current GDS as the new reference (Y)? [Y/n]")
    if val.upper().startswith("N"):
        raise GeometryDifference

    logger.info(f"deleting file {str(ref_file)!r}")
    ref_file.unlink()
    shutil.copy(run_file, ref_file)
    raise GeometryDifference


def read_top_cell(arg0):
    kcl = KCLayout()
    kcl.read(arg0)
    return kcl[0]


if __name__ == "__main__":
    difftest(gf.components.mzi, "mzi")
