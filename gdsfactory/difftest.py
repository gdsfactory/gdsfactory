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
import filecmp
import os
import pathlib
import shutil
from typing import Optional

from gdsfactory.component import Component
from gdsfactory.config import PATH, logger
from gdsfactory.gdsdiff.gdsdiff import gdsdiff


class GeometryDifference(Exception):
    pass


def run_xor(file1, file2, tolerance: int = 1, verbose: bool = False) -> None:
    """Returns nothing.

    Raises a GeometryDifference if there are differences detected.

    Args:
        file1: ref gdspath.
        file2: run gdspath.
        tolerance: in nm.
        verbose: prints output.
    """
    import klayout.db as kdb

    l1 = kdb.Layout()
    l1.read(file1)

    l2 = kdb.Layout()
    l2.read(file2)

    # Check that same set of layers are present
    layer_pairs = []
    for ll1 in l1.layer_indices():
        li1 = l1.get_info(ll1)
        ll2 = l2.find_layer(l1.get_info(ll1))
        if ll2 is None:
            raise GeometryDifference(
                f"Layer {li1} of layout {file1!r} not present in layout {file2!r}."
            )

        layer_pairs.append((ll1, ll2))

    for ll2 in l2.layer_indices():
        li2 = l2.get_info(ll2)
        ll1 = l1.find_layer(l2.get_info(ll2))
        if ll1 is None:
            raise GeometryDifference(
                f"Layer {li2} of layout {file2!r} not present in layout {file1!r}."
            )

    # Check that topcells are the same
    tc1_names = [tc.name for tc in l1.top_cells()]
    tc2_names = [tc.name for tc in l2.top_cells()]
    tc1_names.sort()
    tc2_names.sort()
    if tc1_names != tc2_names:
        raise GeometryDifference(
            f"Missing topcell on one of the layouts, or name differs:\n{tc1_names!r}\n{tc2_names!r}"
        )
    topcell_pairs = [(l1.cell(tc1_n), l2.cell(tc1_n)) for tc1_n in tc1_names]
    # Check that dbu are the same
    if (l1.dbu - l2.dbu) > 1e-6:
        raise GeometryDifference(
            f"Database unit of layout {file1!r} ({l1.dbu}) differs from that of layout {file2!r} ({l2.dbu})."
        )

    # Run the difftool
    diff = False
    for tc1, tc2 in topcell_pairs:
        for ll1, ll2 in layer_pairs:
            r1 = kdb.Region(tc1.begin_shapes_rec(ll1))
            r2 = kdb.Region(tc2.begin_shapes_rec(ll2))

            rxor = r1 ^ r2

            if tolerance > 0:
                rxor.size(-tolerance)

            if not rxor.is_empty():
                diff = True
                if verbose:
                    print(
                        f"{rxor.size()} differences found in {tc1.name!r} on layer {l1.get_info(ll1)}."
                    )

            elif verbose:
                print(
                    f"No differences found in {tc1.name!r} on layer {l1.get_info(ll1)}."
                )

    if diff:
        fn_abgd = []
        for fn in [file1, file2]:
            head, tail = os.path.split(fn)
            abgd = os.path.join(os.path.basename(head), tail)
            fn_abgd.append(abgd)
        raise GeometryDifference(
            "Differences found between layouts {} and {}".format(*fn_abgd)
        )


def difftest(
    component: Component,
    test_name: Optional[str] = None,
    dirpath: pathlib.Path = PATH.gdsdiff,
) -> None:
    """Avoids GDS regressions tests on the GeometryDifference.

    If files are the same it returns None. If files are different runs XOR
    between new component and the GDS reference stored in dirpath and
    raises GeometryDifference if there are differences and show differences in KLayout.

    If it runs for the fist time it just stores the GDS reference.

    Args:
        component: to test if it has changed.
        test_name: used to store the GDS file.
        dirpath: defaults to cwd refers to where the test is being invoked.
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
    ref_file = dirpath / "gds_ref" / filename
    run_file = dirpath / "gds_run" / filename
    diff_file = dirpath / "gds_diff" / filename

    component.write_gds(gdspath=run_file)

    if not ref_file.exists():
        component.write_gds(gdspath=ref_file)
        raise AssertionError(
            f"Reference GDS file for {test_name!r} not found. Writing to {ref_file!r}"
        )

    if filecmp.cmp(ref_file, run_file, shallow=False):
        return

    try:
        run_xor(str(ref_file), str(run_file), tolerance=1, verbose=False)
    except GeometryDifference as error:
        logger.error(error)
        diff = gdsdiff(ref_file, run_file, name=test_name, xor=False)
        diff.write_gds(diff_file)
        diff.show(show_ports=False)
        print(
            f"\ngds_run {filename!r} changed from gds_ref {str(ref_file)!r}\n"
            "You can check the differences in Klayout GUI or run XOR with\n"
            f"gf gds diff --xor {ref_file} {run_file}\n"
        )

        try:
            val = input(
                "Save current GDS as new reference (Y) or show differences (d)? [Y/n/d]"
            )
            if val.upper().startswith("N"):
                raise
            xor = val.upper().startswith("D")
            if xor:
                diff = gdsdiff(ref_file, run_file, name=test_name, xor=xor)
                diff.write_gds(diff_file)
                diff.show(show_ports=False)

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
    import gdsfactory as gf

    c = gf.components.straight()
    difftest(c)
    # test_component(c, None, None)
