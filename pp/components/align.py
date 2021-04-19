from typing import Optional, Tuple

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.grating_coupler.grating_coupler_tree import grating_coupler_tree
from pp.components.rectangle import rectangle


@cell
def align_wafer(
    width: float = 10.0,
    spacing: float = 10.0,
    cross_length: float = 80.0,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layer_cladding: Optional[Tuple[int, int]] = None,
    square_corner: str = "bottom_left",
) -> Component:
    """Returns cross inside a frame to align wafer."""
    c = pp.Component()
    cross = pp.components.cross(length=cross_length, width=width, layer=layer)
    c.add_ref(cross)

    b = cross_length / 2 + spacing + width / 2
    w = width

    rh = rectangle(size=(2 * b + w, w), layer=layer, centered=True)
    rtop = c.add_ref(rh)
    rbot = c.add_ref(rh)
    rtop.movey(+b)
    rbot.movey(-b)

    rv = rectangle(size=(w, 2 * b), layer=layer, centered=True)
    rl = c.add_ref(rv)
    rr = c.add_ref(rv)
    rl.movex(-b)
    rr.movex(+b)

    wsq = (cross_length + 2 * spacing) / 4
    square_mark = c << rectangle(size=(wsq, wsq), layer=layer, centered=True)
    a = width / 2 + wsq / 2 + spacing

    corner_to_position = {
        "bottom_left": (-a, -a),
        "bottom_right": (a, -a),
        "top_right": (a, a),
        "top_left": (-a, a),
    }

    square_mark.move(corner_to_position[square_corner])

    if layer_cladding:
        rc_tile_excl = rectangle(
            size=(2 * (b + spacing), 2 * (b + spacing)),
            layer=layer_cladding,
            centered=True,
        )
        c.add_ref(rc_tile_excl)

    return c


@cell
def add_frame(
    component: Component, width: float = 10.0, spacing: float = 10.0, layer=pp.LAYER.WG
) -> Component:
    """Returns component with a frame around it.

    Args:
        component: Component to frame
        width: of the frame
        spacing: of component to frame

    """
    c = pp.Component(f"{component.name}_f")
    cref = c.add_ref(component)
    cref.move(-c.size_info.center)
    y = (
        max([component.size_info.height, component.size_info.width]) / 2
        + spacing
        + width / 2
    )
    x = y
    w = width

    rh = rectangle(size=(2 * y + w, w), layer=layer, centered=True)
    rtop = c.add_ref(rh)
    rbot = c.add_ref(rh)
    rtop.movey(+y)
    rbot.movey(-y)

    rv = rectangle(size=(w, 2 * y), layer=layer, centered=True)
    rl = c.add_ref(rv)
    rr = c.add_ref(rv)
    rl.movex(-x)
    rr.movex(+x)
    c.absorb(cref)
    return c


@cell
def triangle(x, y, layer=1):
    c = pp.Component()
    points = [[x, 0], [0, 0], [0, y]]
    c.add_polygon(points, layer=layer)
    return c


@cell
def align_cryo_bottom_right(x=60, y=60, layer=1):
    c = align_cryo_top_left()
    cr = c.ref(rotation=180)
    cc = pp.Component()
    cc.add(cr)
    return cc


@cell
def align_cryo_top_right(x=60, y=60, layer=1):
    c = align_cryo_top_left()
    cr = c.ref(rotation=270)
    cc = pp.Component()
    cc.add(cr)
    return cc


@cell
def align_cryo_bottom_left(x=60, y=60, layer=1):
    c = align_cryo_top_left()
    cr = c.ref(rotation=90)
    cc = pp.Component()
    cc.add(cr)
    return cc


@cell
def align_cryo_top_left(x=60, y=60, s=0.2, layer=1):
    c = pp.Component()
    points = [[0, 0], [s, 0], [x - s, y - s], [x - s, y], [0, y]]
    c.add_polygon(points, layer=layer)
    cc = add_frame(component=c)
    return cc


@cell
def align_tree_top_left(**kwargs):
    c = pp.Component()
    c.name = "grating_coupler_tree_tl"
    gc = grating_coupler_tree(**kwargs)
    gc_ref = c.add_ref(gc)
    gc_ref.move(-gc.size_info.center)
    align = align_cryo_top_left()
    c.add_ref(align)
    return c


@cell
def align_tree_top_left_with_cross(**kwargs):
    c = pp.Component()
    c.name = "grating_coupler_tree_tl_x"
    gc = grating_coupler_tree(component_name=c.name, **kwargs)
    gc_ref = c.add_ref(gc)
    gc_ref.move(-gc.size_info.center)
    align = align_cryo_top_left()
    c.add_ref(align)
    align2 = align_wafer()
    align2_ref = c.add_ref(align2)
    align2_ref.movex(gc_ref.xmin - align2.size_info.width / 2)
    return c


@cell
def align_tree_top_right(**kwargs):
    c = pp.Component()
    c.name = "grating_coupler_tree_tr"
    gc = grating_coupler_tree(component_name=c.name, **kwargs)
    gc_ref = c.add_ref(gc)
    gc_ref.move(-gc.size_info.center)
    align = align_cryo_top_right()
    c.add_ref(align)
    return c


@cell
def align_tree_bottom_left(**kwargs):
    c = pp.Component()
    c.name = "grating_coupler_tree_bl"
    gc = grating_coupler_tree(component_name=c.name, **kwargs)
    gc_ref = c.add_ref(gc)
    gc_ref.move(-gc.size_info.center)
    align = align_cryo_bottom_left()
    c.add_ref(align)
    return c


@cell
def align_tree_bottom_right(**kwargs):
    c = pp.Component()
    c.name = "grating_coupler_tree_br"
    gc = grating_coupler_tree(component_name=c.name, **kwargs)
    gc_ref = c.add_ref(gc)
    gc_ref.move(-gc.size_info.center)
    align = align_cryo_bottom_right()
    c.add_ref(align)
    return c


if __name__ == "__main__":
    # c = pp.components.straight()
    # c = add_frame(component=c)
    # c = align_wafer()

    # c = align_tree_top_left_with_cross()
    c = align_tree_top_left()
    # c = triangle(x=60, y=60)
    # c = align_wafer()
    # c = pp.components.cross(length=80, width=10)
    # c = add_frame(component=c)
    # c = align_cryo_top_right()
    # c = align_cryo_top_left()
    # c = align_cryo_bottom_right()
    c.show()
