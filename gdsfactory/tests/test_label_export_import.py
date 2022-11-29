import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.difftest import difftest
from gdsfactory.read.labels import add_port_markers


def test_label_export_import() -> None:
    def mzi_te(**kwargs):
        gc = gf.c.grating_coupler_elliptical_tm()
        c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
        c = gf.routing.add_fiber_single(
            c, get_input_label_text_function=None, grating_coupler=gc
        )
        gf.labels.add_label_yaml(c)
        c = c.rotate(-90)
        return c

    c = gf.grid(
        [
            mzi_te(),
            mzi_te(),
            gf.functions.rotate(mzi_te),
            gf.labels.add_label_yaml(
                gf.functions.mirror(gf.routing.add_fiber_single(gf.components.mmi1x2))
            ),
        ]
    )
    gdspath = c.write_gds("mask.gds")
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefix="component_name"
    )

    component = add_port_markers(gdspath=gdspath, csvpath=csvpath, marker_size=40)
    difftest(component, test_name="label_export_import", dirpath=PATH.gdsdiff)


if __name__ == "__main__":
    test_label_export_import()
