"""GDS regression test. Inspired by lytest."""
import filecmp
import pathlib
import shutil

import gdsfactory as gf
from gdsfactory.config import CONF, PATH, logger
from gdsfactory.name import clean_name


class GeometryDifference(Exception):
    pass


PathType = pathlib.Path | str


def diff(
    ref_file: PathType, run_file: PathType, xor: bool = True, test_name: str = ""
) -> bool:
    """Returns True if files are different, prints differences and shows them in klayout.

    Args:
        ref_file: reference (old) file.
        run_file: run (new) file.
        xor: runs xor on every layer between ref and run files.
        test_name: prefix for the new cell.
    """
    try:
        from kfactory import KCell, kdb
    except ImportError as e:
        print(
            "You can install `pip install gdsfactory[kfactory]` for using maskprep. "
            "And make sure you use python >= 3.10"
        )
        raise e
    ref = read_top_cell(ref_file)
    run = read_top_cell(run_file)
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

    def cell_diff_a(cell: kdb.Cell):
        print(f"{cell.name} only in old")

    def cell_diff_b(cell: kdb.Cell):
        print(f"{cell.name} only in new")

    def text_diff_a(anotb: kdb.Text, prop_id: int):
        get_texts(ld.layer_index_a(), a_texts).insert(anotb)

    def text_diff_b(bnota: kdb.Text, prop_id: int):
        get_texts(ld.layer_index_b(), b_texts).insert(bnota)

    ld.on_cell_in_a_only = lambda anotb: cell_diff_a(anotb)
    ld.on_cell_in_b_only = lambda anotb: cell_diff_b(anotb)
    ld.on_text_in_a_only = lambda anotb, prop_id: text_diff_a(anotb, prop_id)
    ld.on_text_in_b_only = lambda anotb, prop_id: text_diff_b(anotb, prop_id)

    ld.on_polygon_in_a_only = lambda anotb, prop_id: polygon_diff_a(anotb, prop_id)
    ld.on_polygon_in_b_only = lambda anotb, prop_id: polygon_diff_b(anotb, prop_id)

    if CONF.difftest_ignore_cell_name_differences:
        ld.on_cell_name_differs = lambda anotb: print(f"cell name differs {anotb.name}")
        equal = ld.compare(
            ref._kdb_cell, run._kdb_cell, kdb.LayoutDiff.SmartCellMapping, 1
        )
    else:
        equal = ld.compare(ref._kdb_cell, run._kdb_cell, kdb.LayoutDiff.Verbose, 1)

    if not equal:
        c = KCell(f"{test_name}_difftest")
        refdiff = KCell(f"{test_name}_old")
        rundiff = KCell(f"{test_name}_new")

        refdiff.copy_tree(ref._kdb_cell)
        rundiff.copy_tree(run._kdb_cell)
        _ = c << refdiff
        _ = c << rundiff

        if xor:
            diff = KCell(f"{test_name}_xor")

            for layer in c.kcl.layer_infos():
                if layer in run.kcl.layer_infos() and layer in ref.kcl.layer_infos():
                    layer_ref = ref.layer(layer)
                    layer_run = run.layer(layer)

                    region_run = kdb.Region(run.begin_shapes_rec(layer_run))
                    region_ref = kdb.Region(ref.begin_shapes_rec(layer_ref))
                    region_diff = region_run - region_ref

                    if not region_diff.is_empty():
                        layer_id = c.layer(layer)
                        region_xor = region_ref ^ region_run
                        diff.shapes(layer_id).insert(region_xor)
                        is_sliver = region_xor.sized(-1).is_empty()
                        message = f"{test_name}: XOR difference on layer {layer}"
                        if is_sliver:
                            message += " (sliver or label)"
                        print(message)
                elif layer in run.kcl.layer_infos():
                    layer_id = run.layer(layer)
                    region = kdb.Region(run.begin_shapes_rec(layer_id))
                    diff.shapes(layer_id).insert(region)
                    print(f"{test_name}: layer {layer} only exists in updated cell")
                elif layer in ref.kcl.layer_infos():
                    layer_id = ref.layer(layer)
                    region = kdb.Region(ref.begin_shapes_rec(layer_id))
                    diff.shapes(layer_id).insert(region)
                    print(f"{test_name}: layer {layer} missing from updated cell")

            _ = c << diff

        c.show()
        return True
    return False


def difftest(
    component: gf.Component,
    test_name: gf.Component | None = None,
    dirpath: pathlib.Path = PATH.gds_ref,
    xor: bool = True,
    dirpath_run: pathlib.Path = PATH.gds_run,
) -> None:
    """Avoids GDS regressions tests on the GeometryDifference.

    If files are the same it returns None. If files are different runs XOR
    between new component and the GDS reference stored in dirpath and
    raises GeometryDifference if there are differences and show differences in KLayout.

    If it runs for the fist time it just stores the GDS reference.

    Args:
        component: to test if it has changed.
        test_name: used to store the GDS file.
        dirpath: directory where reference files are stored.
        xor: runs XOR.
        dirpath_run: directory to store gds file generated by the test.
    """
    test_name = test_name or (
        f"{component.function_name}_{component.name}"
        if hasattr(component, "function_name")
        and component.name != component.function_name
        else f"{component.name}"
    )
    filename = f"{test_name}.gds"
    dirpath_ref = dirpath
    dirpath_ref.mkdir(exist_ok=True, parents=True)
    dirpath_run.mkdir(exist_ok=True, parents=True)

    ref_file = dirpath_ref / f"{clean_name(test_name)}.gds"
    run_file = dirpath_run / filename

    component = gf.get_component(component)
    run_file = component.write_gds(gdspath=run_file)

    if not ref_file.exists():
        shutil.copy(run_file, ref_file)
        raise AssertionError(
            f"Reference GDS file for {test_name!r} not found. Writing to {ref_file!r}"
        )

    if filecmp.cmp(ref_file, run_file, shallow=False):
        return

    if diff(ref_file=ref_file, run_file=run_file, xor=xor, test_name=test_name):
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
                f"{filename!r} changed from reference {str(ref_file)!r}. "
                "Run `pytest -s` to step and check differences in klayout GUI."
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
    from kfactory import KCLayout

    kcl = KCLayout()
    kcl.read(arg0)
    return kcl[kcl.top_cell().name]


if __name__ == "__main__":
    # print([i.name for i in c.get_dependencies()])
    # c.show()
    # c.name = "mzi"
    c = gf.components.straight(length=20, layer=(1, 0))
    c.name = "a:demo"
    c.show()
    difftest(c, "straight", dirpath=PATH.cwd)

    # component = gf.components.mzi()
    # test_name = "mzi"
    # filename = f"{test_name}.gds"
    # dirpath = PATH.cwd
    # dirpath_ref = dirpath / "gds_ref"
    # dirpath_run = GDSDIR_TEMP

    # ref_file = dirpath_ref / f"{test_name}.gds"
    # run_file = dirpath_run / filename
    # run = gf.get_component(component)
    # run_file = run.write_gds(gdspath=run_file)

    # if not ref_file.exists():
    #     component.write_gds(gdspath=ref_file)
    #     raise AssertionError(
    #         f"Reference GDS file for {test_name!r} not found. Writing to {ref_file!r}"
    #     )

    # ref = read_top_cell(ref_file)
    # run = read_top_cell(run_file)
    # ld = kdb.LayoutDiff()

    # print(ld.compare(ref._kdb_cell, run._kdb_cell))
