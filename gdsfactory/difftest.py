"""Layout Diff utils."""

import pathlib
from typing import Optional

from kfactory import KLib, kdb

import gdsfactory as gf
from gdsfactory.config import PATH


def difftest(component: gf.Component, 
    comp: gf.Component,
    dirpath: pathlib.Path = PATH.gdslib,
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
    ref = gf.get_component(ref)
    comp = gf.get_component(comp)

    ref.write_gds("ref.gds")
    comp.write_gds("comp.gds")

    ref = KLib()
    ref.read("ref.gds")
    ref = ref[0]

    comp = KLib()
    comp.read("comp.gds")
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
        get_region(ld.layer_index_a, a_regions).insert(anotb)

    def polygon_diff_b(bnota: kdb.Polygon, prop_id: int):
        get_region(ld.layer_index_b, b_regions).insert(bnota)

    def text_diff_a(anotb: kdb.Text, prop_id: int):
        get_texts(ld.layer_index_a(), a_texts)

    def text_diff_b(bnota: kdb.Text, prop_id: int):
        get_texts(ld.layer_index_b(), b_texts)

    ld.on_polygon_in_a_only(
        lambda poly_anotb, propid: polygon_diff_a(poly_anotb, propid)
    )

    ld.on_polygon_in_b_only(
        lambda poly_anotb, propid: polygon_diff_b(poly_anotb, propid)
    )

    ld.on_text_in_a_only(lambda anotb, propid: text_diff_a(anotb, propid))

    ld.on_text_in_b_only(lambda anotb, propid: text_diff_b(anotb, propid))

    if not ld.compare(ref._kdb_cell, comp._kdb_cell, kdb.LayoutDiff.Verbose):
        ref = KCell()
        comp = KCell()
        for layer, region in a_regions:
            ref.shapes(layer).insert(region)

        for layer, region in b_regions:
            comp.shapes(layer).inser(region)

        for layer, region in a_texts:
            ref.shapes(layer).insert(region)

        for layer, region in b_texts:
            comp.shapes(layer).insert(region)

        c = KCell()
        c << ref
        c << comp

        


if __name__ == "__main__":
    difftest(gf.components.mzi(), gf.components.mzi())
