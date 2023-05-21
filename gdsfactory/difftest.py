"""Layout Diff utils."""

from kfactory import KLib, kdb

import gdsfactory as gf


def difftest(ref: gf.Component, comp: gf.Component):
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

    assert ld.compare(ref._kdb_cell, comp._kdb_cell, kdb.LayoutDiff.Verbose)


if __name__ == "__main__":
    difftest(gf.components.mzi(), gf.components.mzi())
