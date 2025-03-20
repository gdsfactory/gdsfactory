"""GDS regression test. Inspired by lytest."""

import filecmp
import pathlib
import shutil

import kfactory as kf
from kfactory import DKCell, KCLayout, kdb, logger

import gdsfactory as gf
from gdsfactory.config import CONF, PATH
from gdsfactory.name import clean_name, get_name_short


class GeometryDifference(Exception):
    pass


PathType = pathlib.Path | str


def xor(
    old: KCLayout,
    new: KCLayout,
    test_name: str = "",
    ignore_sliver_differences: bool | None = None,
    ignore_cell_name_differences: bool | None = None,
    ignore_label_differences: bool | None = None,
    stagger: bool = True,
) -> DKCell:
    """Returns XOR of two layouts.

    Args:
        old: reference layout.
        new: run layout.
        test_name: prefix for the new cell.
        ignore_sliver_differences: if True, ignores any sliver differences in the XOR result. If None (default), defers to the value set in CONF.difftest_ignore_sliver_differences
        ignore_cell_name_differences: if True, ignores any cell name differences. If None (default), defers to the value set in CONF.difftest_ignore_cell_name_differences
        ignore_label_differences: if True, ignores any label differences when run in XOR mode. If None (default) defers to the value set in CONF.difftest_ignore_label_differences
        stagger: if True, staggers the old/new/xor views. If False, all three are overlaid.
    """
    if ignore_sliver_differences is None:
        ignore_sliver_differences = CONF.difftest_ignore_sliver_differences

    if ignore_cell_name_differences is None:
        ignore_cell_name_differences = CONF.difftest_ignore_cell_name_differences

    if ignore_label_differences is None:
        ignore_label_differences = CONF.difftest_ignore_label_differences

    if old.kcl.dbu != new.kcl.dbu:
        raise ValueError(
            f"dbu is different in old {old.kcl.dbu} and new {new.kcl.dbu} cells"
        )

    equivalent = True
    ld = kdb.LayoutDiff()

    a_regions: dict[int, kdb.Region] = {}
    a_texts: dict[int, kdb.Texts] = {}
    b_regions: dict[int, kdb.Region] = {}
    b_texts: dict[int, kdb.Texts] = {}

    def get_region(key: int, regions: dict[int, kdb.Region]) -> kdb.Region:
        if key not in regions:
            reg = kdb.Region()
            regions[key] = reg
            return reg
        else:
            return regions[key]

    def get_texts(key: int, texts_dict: dict[int, kdb.Texts]) -> kdb.Texts:
        if key not in texts_dict:
            texts = kdb.Texts()
            texts_dict[key] = texts
            return texts
        else:
            return texts_dict[key]

    def polygon_diff_a(anotb: kdb.Polygon, prop_id: int) -> None:
        get_region(ld.layer_index_a(), a_regions).insert(anotb)

    def polygon_diff_b(bnota: kdb.Polygon, prop_id: int) -> None:
        get_region(ld.layer_index_b(), b_regions).insert(bnota)

    def cell_diff_a(cell: kdb.Cell) -> None:
        nonlocal equivalent
        print(f"{cell.name} only in old")
        if not ignore_cell_name_differences:
            equivalent = False

    def cell_diff_b(cell: kdb.Cell) -> None:
        nonlocal equivalent
        print(f"{cell.name} only in new")
        if not ignore_cell_name_differences:
            equivalent = False

    def text_diff_a(anotb: kdb.Text, prop_id: int) -> None:
        print("Text only in old")
        get_texts(ld.layer_index_a(), a_texts).insert(anotb)

    def text_diff_b(bnota: kdb.Text, prop_id: int) -> None:
        print("Text only in new")
        get_texts(ld.layer_index_b(), b_texts).insert(bnota)

    ld.on_cell_in_a_only = lambda anotb: cell_diff_a(anotb)  # type: ignore[assignment]
    ld.on_cell_in_b_only = lambda anotb: cell_diff_b(anotb)  # type: ignore[assignment]
    ld.on_text_in_a_only = lambda anotb, prop_id: text_diff_a(anotb, prop_id)  # type: ignore[assignment]
    ld.on_text_in_b_only = lambda anotb, prop_id: text_diff_b(anotb, prop_id)  # type: ignore[assignment]

    ld.on_polygon_in_a_only = lambda anotb, prop_id: polygon_diff_a(anotb, prop_id)  # type: ignore[assignment]
    ld.on_polygon_in_b_only = lambda anotb, prop_id: polygon_diff_b(anotb, prop_id)  # type: ignore[assignment]

    if ignore_cell_name_differences:
        ld.on_cell_name_differs = lambda anotb: print(f"cell name differs {anotb.name}")  # type: ignore[assignment]
        equal = ld.compare(
            old.kdb_cell,
            new.kdb_cell,
            kdb.LayoutDiff.SmartCellMapping | kdb.LayoutDiff.Verbose,
            1,
        )
    else:
        equal = ld.compare(old.kdb_cell, new.kdb_cell, kdb.LayoutDiff.Verbose, 1)

    if not ignore_label_differences and (a_texts or b_texts):
        equivalent = False
    if equal:
        return gf.Component(name="xor_empty")
    c = DKCell(name=f"{test_name}_difftest")
    ref = old
    run = new

    old_kcell = DKCell(name=f"{test_name}_old")
    new_kcell = DKCell(name=f"{test_name}_new")

    old_kcell.copy_tree(ref.kdb_cell)
    new_kcell.copy_tree(run.kdb_cell)

    old_kcell.name = f"{test_name}_old"
    new_kcell.name = f"{test_name}_new"

    old_ref = c << old_kcell
    new_ref = c << new_kcell

    dy = 10
    if stagger:
        old_ref.movey(+old_kcell.ysize + dy)
        new_ref.movey(-old_kcell.ysize - dy)

    layer_label = kf.kcl.layout.layer(1, 0)
    c.shapes(layer_label).insert(kf.kdb.DText("old", old_ref.dtrans))
    c.shapes(layer_label).insert(kf.kdb.DText("new", new_ref.dtrans))
    c.shapes(layer_label).insert(
        kf.kdb.DText(
            "xor", kf.kdb.DTrans(new_ref.xmin, old_ref.ymax - old_ref.ysize - dy)
        )
    )

    print("Running XOR on differences...")
    # assume equivalence until we find XOR differences, determined significant by the settings
    diff = DKCell(name=f"{test_name}_xor")

    for layer in c.kcl.layer_infos():
        # exists in both
        if (
            new_kcell.kcl.layout.find_layer(layer) is not None
            and old_kcell.kcl.layout.find_layer(layer) is not None
        ):
            layer_ref = old_kcell.layer(layer)
            layer_run = new_kcell.layer(layer)

            region_run = kdb.Region(new_kcell.begin_shapes_rec(layer_run))
            region_ref = kdb.Region(old_kcell.begin_shapes_rec(layer_ref))
            region_diff = region_run ^ region_ref

            if not region_diff.is_empty():
                layer_id = c.layer(layer)
                region_xor = region_ref ^ region_run
                diff.shapes(layer_id).insert(region_xor)
                xor_w_tolerance = region_xor.sized(-1)
                is_sliver = xor_w_tolerance.is_empty()
                message = f"{test_name}: XOR difference on layer {layer}"
                if is_sliver:
                    message += " (sliver)"
                    if not ignore_sliver_differences:
                        equivalent = False
                else:
                    equivalent = False
                print(message)
        # only in new
        elif new_kcell.kcl.layout.find_layer(layer) is not None:
            layer_id = new_kcell.layer(layer)
            region = kdb.Region(new_kcell.begin_shapes_rec(layer_id))
            diff.shapes(c.kcl.layer(layer)).insert(region)
            print(f"{test_name}: layer {layer} only exists in updated cell")
            equivalent = False

        # only in old
        elif old_kcell.kcl.layout.find_layer(layer) is not None:
            layer_id = old_kcell.layer(layer)
            region = kdb.Region(old_kcell.begin_shapes_rec(layer_id))
            diff.shapes(c.kcl.layer(layer)).insert(region)
            print(f"{test_name}: layer {layer} missing from updated cell")
            equivalent = False

        _ = c << diff
    return c


def diff(
    ref_file: PathType,
    run_file: PathType,
    xor: bool = True,
    test_name: str = "",
    ignore_sliver_differences: bool | None = None,
    ignore_cell_name_differences: bool | None = None,
    ignore_label_differences: bool | None = None,
    show: bool = False,
    stagger: bool = True,
) -> bool:
    """Returns True if files are different, prints differences and shows them in klayout.

    Args:
        ref_file: reference (old) file.
        run_file: run (new) file.
        xor: runs xor on every layer between old and run files.
        test_name: prefix for the new cell.
        ignore_sliver_differences: if True, ignores any sliver differences in the XOR result. If None (default), defers to the value set in CONF.difftest_ignore_sliver_differences
        ignore_cell_name_differences: if True, ignores any cell name differences. If None (default), defers to the value set in CONF.difftest_ignore_cell_name_differences
        ignore_label_differences: if True, ignores any label differences when run in XOR mode. If None (default) defers to the value set in CONF.difftest_ignore_label_differences
        show: shows diff in klayout.
        stagger: if True, staggers the old/new/xor views. If False, all three are overlaid.
    """
    old = read_top_cell(pathlib.Path(ref_file))
    new = read_top_cell(pathlib.Path(run_file))

    if ignore_sliver_differences is None:
        ignore_sliver_differences = CONF.difftest_ignore_sliver_differences

    if ignore_cell_name_differences is None:
        ignore_cell_name_differences = CONF.difftest_ignore_cell_name_differences

    if ignore_label_differences is None:
        ignore_label_differences = CONF.difftest_ignore_label_differences

    if old.kcl.dbu != new.kcl.dbu:
        raise ValueError(
            f"dbu is different in old {old.kcl.dbu} {ref_file!r} and new {new.kcl.dbu} {run_file!r} files"
        )

    equivalent = True
    ld = kdb.LayoutDiff()

    a_regions: dict[int, kdb.Region] = {}
    a_texts: dict[int, kdb.Texts] = {}
    b_regions: dict[int, kdb.Region] = {}
    b_texts: dict[int, kdb.Texts] = {}

    def get_region(key: int, regions: dict[int, kdb.Region]) -> kdb.Region:
        if key not in regions:
            reg = kdb.Region()
            regions[key] = reg
            return reg
        else:
            return regions[key]

    def get_texts(key: int, texts_dict: dict[int, kdb.Texts]) -> kdb.Texts:
        if key not in texts_dict:
            texts = kdb.Texts()
            texts_dict[key] = texts
            return texts
        else:
            return texts_dict[key]

    def polygon_diff_a(anotb: kdb.Polygon, prop_id: int) -> None:
        get_region(ld.layer_index_a(), a_regions).insert(anotb)

    def polygon_diff_b(bnota: kdb.Polygon, prop_id: int) -> None:
        get_region(ld.layer_index_b(), b_regions).insert(bnota)

    def cell_diff_a(cell: kdb.Cell) -> None:
        nonlocal equivalent
        print(f"{cell.name} only in old")
        if not ignore_cell_name_differences:
            equivalent = False

    def cell_diff_b(cell: kdb.Cell) -> None:
        nonlocal equivalent
        print(f"{cell.name} only in new")
        if not ignore_cell_name_differences:
            equivalent = False

    def text_diff_a(anotb: kdb.Text, prop_id: int) -> None:
        print("Text only in old")
        get_texts(ld.layer_index_a(), a_texts).insert(anotb)

    def text_diff_b(bnota: kdb.Text, prop_id: int) -> None:
        print("Text only in new")
        get_texts(ld.layer_index_b(), b_texts).insert(bnota)

    ld.on_cell_in_a_only = cell_diff_a  # type: ignore[assignment]
    ld.on_cell_in_b_only = cell_diff_b  # type: ignore[assignment]
    ld.on_text_in_a_only = text_diff_a  # type: ignore[assignment]
    ld.on_text_in_b_only = text_diff_b  # type: ignore[assignment]

    ld.on_polygon_in_a_only = polygon_diff_a  # type: ignore[assignment]
    ld.on_polygon_in_b_only = polygon_diff_b  # type: ignore[assignment]

    if ignore_cell_name_differences:
        ld.on_cell_name_differs = lambda anotb: print(f"cell name differs {anotb.name}")  # type: ignore[assignment]
        equal = ld.compare(
            old.kdb_cell,
            new.kdb_cell,
            kdb.LayoutDiff.SmartCellMapping | kdb.LayoutDiff.Verbose,
            1,
        )
    else:
        equal = ld.compare(old.kdb_cell, new.kdb_cell, kdb.LayoutDiff.Verbose, 1)

    if not ignore_label_differences and (a_texts or b_texts):
        equivalent = False

    if not equal:
        c = DKCell(name=f"{test_name}_difftest")
        ref = old
        run = new

        old = DKCell(name=f"{test_name}_old")
        new = DKCell(name=f"{test_name}_new")

        old.copy_tree(ref.kdb_cell)
        new.copy_tree(run.kdb_cell)

        old.name = f"{test_name}_old"
        new.name = f"{test_name}_new"

        old_ref = c << old
        new_ref = c << new

        dy = 10
        if stagger:
            old_ref.movey(+old.ysize + dy)
            new_ref.movey(-old.ysize - dy)

        layer_label = kf.kcl.layout.layer(1, 0)
        c.shapes(layer_label).insert(kf.kdb.DText("old", old_ref.dtrans))
        c.shapes(layer_label).insert(kf.kdb.DText("new", new_ref.dtrans))
        c.shapes(layer_label).insert(
            kf.kdb.DText(
                "xor", kf.kdb.DTrans(new_ref.xmin, old_ref.ymax - old_ref.ysize - dy)
            )
        )

        if xor:
            print("Running XOR on differences...")
            # assume equivalence until we find XOR differences, determined significant by the settings
            diff = DKCell(name=f"{test_name}_xor")

            for layer in c.kcl.layer_infos():
                # exists in both
                if (
                    new.kcl.layout.find_layer(layer) is not None
                    and old.kcl.layout.find_layer(layer) is not None
                ):
                    layer_ref = old.layer(layer)
                    layer_run = new.layer(layer)

                    region_run = kdb.Region(new.begin_shapes_rec(layer_run))
                    region_ref = kdb.Region(old.begin_shapes_rec(layer_ref))
                    region_diff = region_run ^ region_ref

                    if not region_diff.is_empty():
                        layer_id = c.layer(layer)
                        region_xor = region_ref ^ region_run
                        diff.shapes(layer_id).insert(region_xor)
                        xor_w_tolerance = region_xor.sized(-1)
                        is_sliver = xor_w_tolerance.is_empty()
                        message = f"{test_name}: XOR difference on layer {layer}"
                        if is_sliver:
                            message += " (sliver)"
                            if not ignore_sliver_differences:
                                equivalent = False
                        else:
                            equivalent = False
                        print(message)
                # only in new
                elif new.kcl.layout.find_layer(layer) is not None:
                    layer_id = new.layer(layer)
                    region = kdb.Region(new.begin_shapes_rec(layer_id))
                    diff.shapes(c.kcl.layer(layer)).insert(region)
                    print(f"{test_name}: layer {layer} only exists in updated cell")
                    equivalent = False

                # only in old
                elif old.kcl.layout.find_layer(layer) is not None:
                    layer_id = old.layer(layer)
                    region = kdb.Region(old.begin_shapes_rec(layer_id))
                    diff.shapes(c.kcl.layer(layer)).insert(region)
                    print(f"{test_name}: layer {layer} missing from updated cell")
                    equivalent = False

            _ = c << diff
            if equivalent:
                print("No significant XOR differences between layouts!")
        else:
            # if no additional xor verification, the two files are not equivalent
            equivalent = False

        if show and not equivalent:
            c.show()
        return not equivalent
    return False


def difftest(
    component: gf.Component,
    test_name: str | None = None,
    dirpath: pathlib.Path = PATH.gds_ref,
    xor: bool = True,
    dirpath_run: pathlib.Path = PATH.gds_run,
    ignore_sliver_differences: bool | None = None,
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
        ignore_sliver_differences: if True, ignores any sliver differences in the XOR result.
            If None (default), defers to the value set in CONF.difftest_ignore_sliver_differences
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

    filename = get_name_short(clean_name(test_name), max_cellname_length=32)

    ref_file = dirpath_ref / f"{filename}.gds"
    run_file = dirpath_run / f"{filename}.gds"

    component = gf.get_component(component)
    run_file = component.write_gds(gdspath=run_file)

    if not ref_file.exists():
        shutil.copy(run_file, ref_file)
        raise AssertionError(
            f"Reference GDS file for {test_name!r} not found. Writing to {ref_file!r}"
        )

    if filecmp.cmp(ref_file, run_file, shallow=False):
        return

    if diff(
        ref_file=ref_file,
        run_file=run_file,
        xor=xor,
        test_name=test_name,
        ignore_sliver_differences=ignore_sliver_differences,
        show=True,
    ):
        print(
            f"\ngds_run {filename!r} changed from gds_ref {str(ref_file)!r}\n"
            "You can check the differences in Klayout GUI or run XOR with\n"
            f"gf gds-diff --xor {ref_file} {run_file}\n"
        )
        try:
            overwrite(ref_file, run_file)
        except OSError as exc:
            raise GeometryDifference(
                "\n"
                f"{filename!r} changed from reference {str(ref_file)!r}. "
                "Run `pytest -s` to step and check differences in klayout GUI."
            ) from exc


def overwrite(ref_file: pathlib.Path, run_file: pathlib.Path) -> None:
    val = input("Save current GDS as the new reference (Y)? [Y/n]")
    if val.upper().startswith("N"):
        raise GeometryDifference

    logger.info(f"deleting file {str(ref_file)!r}")
    ref_file.unlink()
    shutil.copy(run_file, ref_file)
    raise GeometryDifference


def read_top_cell(arg0: pathlib.Path) -> kf.DKCell:
    kcl = KCLayout(name=str(arg0))
    kcl.read(arg0)
    kcell = kcl.dkcells[kcl.top_cell().name]

    if hasattr(kcl, "cross_sections"):
        for cross_section in kcl.cross_sections.cross_sections.values():
            kf.kcl.get_symmetrical_cross_section(cross_section)
    return kcell


if __name__ == "__main__":
    from gdsfactory.components.waveguides.straight import straight

    # print([i.name for i in c.get_dependencies()])
    # c.show()
    # c.name = "mzi"
    c = straight(length=10)
    difftest(c, test_name="straight", dirpath=PATH.cwd)
    c.show()
    # c.write_gds()

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
    # print(ld.compare(ref.kdb_cell, run.kdb_cell))
