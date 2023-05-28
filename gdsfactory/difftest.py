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
import os
import pathlib
import shutil
from typing import Optional

import gdsfactory as gf
from gdsfactory.config import PATH, logger

from kfactory import KCell, KCLayout, kdb


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
    component: gf.Component,
    test_name: Optional[gf.Component] = None,
    dirpath: Optional[pathlib.Path] = PATH.gdslib,
    dirpath_ref: Optional[pathlib.Path] = PATH.gds_ref,
    dirpath_run: Optional[pathlib.Path] = PATH.gds_run,
    dirpath_diff: Optional[pathlib.Path] = PATH.gds_diff,
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

    ref_file = dirpath_ref / filename
    run_file = dirpath_run / filename

    diff_file = dirpath_diff / filename

    ref = gf.get_component(component)
    if isinstance(test_name, gf.Component):
        comp = gf.get_component(test_name)
        run_file = comp.write_gds()

    run_file = ref.write_gds()

    ref = KCLayout()
    ref.read(str(run_file))
    ref = ref[0]

    comp = KCLayout()
    comp.read(str(ref_file))
    comp = comp[0]

    ld = kdb.LayoutDiff()

    a_regions: dict[int, kdb.Region] = {}
    a_texts: dict[int, kdb.Texts] = {}
    b_regions: dict[int, kdb.Region] = {}
    b_texts: dict[int, kdb.Texts] = {}

    def get_region(key, regions: dict[int, kdb.Region]) -> kdb.Region:
        if key not in regions:
            reg = kdb.Region()
            regions[key] = reg
            return reg
        else:
            return regions[key]

    def get_texts(key, texts_dict: dict[int, kdb.Texts]) -> kdb.Texts:
        if key not in texts_dict:
            texts = kdb.Texts()
            texts_dict[key] = texts
            return texts
        else:
            return texts_dict[key]

    def polygon_diff_a(anotb: kdb.Polygon, prop_id: int):
        get_region(ld.layer_index_a(), a_regions).insert(anotb)

    def polygon_diff_b(bnota: kdb.Polygon, prop_id: int):
        get_region(ld.layer_index_b(), b_regions).insert(bnota)

    def cell_diff_a(anotb: kdb.Cell):
        get_region(ld.layer_index_a(), a_regions).insert(
            anotb.begin_shapes_rec(ld.layer_index_a())
        )

    def cell_diff_b(anotb: kdb.Cell):
        get_region(ld.layer_index_b(), b_regions).insert(
            anotb.begin_shapes_rec(ld.layer_index_b())
        )

    def text_diff_a(anotb: kdb.Text, prop_id: int):
        get_texts(ld.layer_index_a(), a_texts).insert(anotb)

    def text_diff_b(bnota: kdb.Text, prop_id: int):
        get_texts(ld.layer_index_b(), b_texts).insert(bnota)

    ld.on_cell_in_a_only = lambda anotb: cell_diff_a(anotb)
    ld.on_cell_in_b_only = lambda anotb: cell_diff_b(anotb)
    ld.on_polygon_in_a_only = lambda anotb, prop_id: polygon_diff_a(anotb, prop_id)
    ld.on_polygon_in_b_only = lambda anotb, prop_id: polygon_diff_b(anotb, prop_id)
    ld.on_text_in_a_only = lambda anotb, prop_id: text_diff_a(anotb, prop_id)
    ld.on_text_in_b_only = lambda anotb, prop_id: text_diff_b(anotb, prop_id)

    if not ld.compare(ref._kdb_cell, comp._kdb_cell, kdb.LayoutDiff.Verbose):
        ref_2 = KCell(ref.name)
        comp_2 = KCell("reference")
        for layer, region in a_regions.items():
            idx = ref_2.kcl.layer(ref.kcl.layer_infos()[layer])
            ref_2.shapes(idx).insert(region)

        for layer, region in b_regions.items():
            idx = comp_2.kcl.layer(comp.kcl.layer_infos()[layer])
            comp_2.shapes(idx).insert(region)

        for layer, region in a_texts.items():
            idx = ref_2.kcl.layer(ref.kcl.layer_infos()[layer])
            ref_2.shapes(idx).insert(region)

        for layer, region in b_texts.items():
            idx = comp_2.kcl.layer(comp.kcl.layer_infos()[layer])
            comp_2.shapes(idx).insert(region)

        c = KCell("Polygon diffs")
        c << ref_2
        c << comp_2

        print(
            f"\ngds_run {filename!r} changed from gds_ref {str(ref_file)!r}\n"
            "You can check the differences in Klayout GUI or run XOR with\n"
            f"gf gds diff --xor {ref_file} {run_file}\n"
        )

        val = input(
            "Save current GDS as new reference (Y) or show differences (d)? [Y/n/d]"
        )
        if val.upper().startswith("N"):
            raise GeometryDifference(
                "\n"
                f"{filename!r} changed from reference {str(ref_file)!r}\n"
                "To step over each error you can run `pytest -s`\n"
                "So you can check the differences in Klayout GUI\n"
            )
        xor = val.upper().startswith("D")
        if xor:
            c.write(diff_file)
            c.show()

            val = input("Save current GDS as the new reference (Y)? [Y/n]")
            if val.upper().startswith("N"):
                raise GeometryDifference(
                    "\n"
                    f"{filename!r} changed from reference {str(ref_file)!r}\n"
                    "To step over each error you can run `pytest -s`\n"
                    "So you can check the differences in Klayout GUI\n"
                )

        logger.info(f"deleting file {str(ref_file)!r}")
        ref_file.unlink()
        shutil.copy(run_file, ref_file)


if __name__ == "__main__":
    difftest(gf.components.mmi1x2())
