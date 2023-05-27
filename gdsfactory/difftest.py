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
from gdsfactory.config import PATH, logger

from kfactory import KCell, KCLayout, kdb


class GeometryDifference(Exception):
    pass


def difftest(
    component: gf.Component,
    test_name: Optional[gf.Component] = None,
    dirpath: Optional[pathlib.Path] = PATH.gdslib,
    dirpath_ref: Optional[pathlib.Path] = PATH.gds_ref,
    dirpath_run: Optional[pathlib.Path] = PATH.gds_run,
    dirpath_diff: Optional[pathlib.Path] = PATH.gds_diff,
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
        dirpath: default directory for storing reference/run/diff files.
        dirpath_ref: optional directory for storing reference files.
        dirpath_run: optional directory for storing run files.
        dirpath_diff: optional directory for storing diff files.
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
    dirpath_ref = dirpath_ref or dirpath / "gds_ref"
    dirpath_run = dirpath_run or dirpath / "gds_run"
    dirpath_diff = dirpath_diff or dirpath / "gds_diff"

    ref_file = dirpath_ref / f"{component.name}.gds"
    run_file = dirpath_run / filename

    dirpath_diff / filename

    ref = gf.get_component(component)
    run = gf.get_component(test_name)

    ref_file = ref.write_gds()
    run_file = run.write_gds()

    ref = KCLayout()
    ref.read(ref_file)
    ref = ref[0]

    run = KCLayout()
    run.read(run_file)
    run = run[0]

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
            val = input("Save current GDS as the new reference (Y)? [Y/n]")
            if val.upper().startswith("N"):
                raise

            logger.info(f"deleting file {str(ref_file)!r}")
            ref_file.unlink()
            shutil.copy(run_file, ref_file)
            raise
        except OSError as exc:
            raise GeometryDifference(
                "\n"
                f"{filename!r} changed from reference {str(ref_file)!r}\n"
                "To step over each error you can run `pytest -s`\n"
                "So you can check the differences in Klayout GUI\n"
            ) from exc


if __name__ == "__main__":
    difftest(gf.components.mzi(delta_length=100), "mzi")
