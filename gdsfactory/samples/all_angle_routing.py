from __future__ import annotations

import pathlib

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.read import cell_from_yaml_template, from_yaml
from gdsfactory.routing.factories import routing_strategy

SAMPLE_DIR = pathlib.Path(__file__).parent / "all_angle_routing"


@cell
def demo_all_angle_routing() -> Component:
    """Demonstrate all-angle routing."""
    file = SAMPLE_DIR / "aar_basic_01.pic.yml"
    return from_yaml(file)


def get_yaml_pics():
    files = SAMPLE_DIR.glob("*.pic.yml")
    pics = {}
    for file in files:
        name = file.name[: -len(".pic.yml")]
        pic = cell_from_yaml_template(file, name, routing_strategy=routing_strategy)
        pics[name] = pic
    return pics


if __name__ == "__main__":
    from gdsfactory import grid
    from gdsfactory.pdk import get_active_pdk

    # IMPORTANT: always use this gds write flag when using non-manhattan features
    get_active_pdk().gds_write_settings.flatten_invalid_refs = True

    c = demo_all_angle_routing()
    pics = get_yaml_pics()
    # get all the render-able pics: those with "error" in the name intentionally demonstrate errors
    good_pics = [pic for pic_name, pic in pics.items() if "error" not in pic_name]
    c = grid(good_pics)

    c.show(show_ports=True)
