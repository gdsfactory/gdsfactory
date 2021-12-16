"""pack a list of components into as few components as possible.
adapted from phidl.geometry.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rectpack
from pydantic import validate_arguments

from gdsfactory.component import Component
from gdsfactory.name import get_name_short
from gdsfactory.types import (
    Anchor,
    ComponentFactory,
    ComponentOrFactory,
    Float2,
    Number,
)


def _pack_single_bin(
    rect_dict: Dict[int, Tuple[Number, Number]],
    aspect_ratio: Tuple[Number, Number],
    max_size: Tuple[float, float],
    sort_by_area: bool,
    density: float,
) -> Tuple[Dict[int, Tuple[Number, Number, Number, Number]], Dict[Any, Any]]:
    """Packs a dict of rectangles {id:(w,h)} and tries to
    pack it into a bin as small as possible with aspect ratio `aspect_ratio`
    Will iteratively grow the bin size until everything fits or the bin size
    reaches `max_size`.

    Args:
        rect_dict: dict of rectangles {id: (w, h)} to pack
        aspect_ratio:
        max_size: tuple of max X, Y size
        sort_by_area: sorts components by area
        density: of packing, closer to 1 packs tighter (more compute heavy)

    Returns:
        packed rectangles dict {id:(x,y,w,h)}
        dict of remaining unpacked rectangles
    """

    # Compute total area and use it for an initial estimate of the bin size
    total_area = 0
    for r in rect_dict.values():
        total_area += r[0] * r[1]
    aspect_ratio = np.asarray(aspect_ratio) / np.linalg.norm(aspect_ratio)  # Normalize

    # Setup variables
    box_size = np.asarray(aspect_ratio * np.sqrt(total_area), dtype=np.float64)
    box_size = np.clip(box_size, None, max_size)
    if sort_by_area:
        rp_sort = rectpack.SORT_AREA
    else:
        rp_sort = rectpack.SORT_NONE

    # Repeatedly run the rectangle-packing algorithm with increasingly larger
    # areas until everything fits or we've reached the maximum size
    while True:
        # Create the pack object
        rect_packer = rectpack.newPacker(
            mode=rectpack.PackingMode.Offline,
            pack_algo=rectpack.MaxRectsBlsf,
            sort_algo=rp_sort,
            bin_algo=rectpack.PackingBin.BBF,
            rotation=False,
        )

        # Add each rectangle to the pack, create a single bin, and pack
        for rid, r in rect_dict.items():
            rect_packer.add_rect(width=r[0], height=r[1], rid=rid)
        rect_packer.add_bin(width=box_size[0], height=box_size[1])
        rect_packer.pack()

        # Adjust the box size for next time
        box_size *= density  # Increase area to try to fit
        box_size = np.clip(box_size, None, max_size)

        # Quit the loop if we've packed all the rectangles or reached the max size
        if len(rect_packer.rect_list()) == len(rect_dict):
            break
        if all(box_size >= max_size):
            break

    # Separate packed from unpacked rectangles, make dicts of form {id:(x,y,w,h)}
    packed_rect_dict = {r[-1]: r[:-1] for r in rect_packer[0].rect_list()}
    unpacked_rect_dict = {}
    for k, v in rect_dict.items():
        if k not in packed_rect_dict:
            unpacked_rect_dict[k] = v

    return packed_rect_dict, unpacked_rect_dict


@validate_arguments
def pack(
    component_list: List[ComponentOrFactory],
    spacing: float = 10.0,
    aspect_ratio: Float2 = (1.0, 1.0),
    max_size: Tuple[Optional[float], Optional[float]] = (None, None),
    sort_by_area: bool = True,
    density: float = 1.1,
    precision: float = 1e-2,
    text: Optional[ComponentFactory] = None,
    text_prefix: str = "",
    text_offsets: Tuple[Float2, ...] = ((0, 0),),
    text_anchors: Tuple[Anchor, ...] = ("cc",),
    name_prefix: Optional[str] = None,
    rotation: int = 0,
    h_mirror: bool = False,
    v_mirror: bool = False,
) -> List[Component]:
    """Pack a list of components into as few Components as possible.

    Adapted from phidl.geometry

    Args:
        component_list: list or tuple
        spacing: Minimum distance between adjacent shapes
        aspect_ratio: (width, height) ratio of the rectangular bin
        max_size: Limits the size into which the shapes will be packed
        sort_by_area: Pre-sorts the shapes by area
        density: Values closer to 1 pack tighter but require more computation
        precision: Desired precision for rounding vertex coordinates.
        text: Optional function to add text labels.
        text_prefix: for labels. For example. 'A' will produce 'A1', 'A2', ...
        text_offsets: relative to component size info anchor. Defaults to center.
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        name_prefix: for each packed component (avoids the Unnamed cells warning).
            Note that the suffix contains a uuid so the name will not be deterministic
        rotation: for each component in degrees
        h_mirror: horizontal mirror in y axis (x, 1) (1, 0). This is the most common.
        v_mirror: vertical mirror using x axis (1, y) (0, y)
    """
    if density < 1.01:
        raise ValueError(
            "pack() `density` argument is too small. "
            "The density argument must be >= 1.01"
        )

    # Santize max_size variable
    max_size = [np.inf if v is None else v for v in max_size]
    max_size = np.asarray(max_size, dtype=np.float64)  # In case it's integers
    max_size = max_size / precision

    component_list = [
        component() if callable(component) else component
        for component in component_list
    ]

    # Convert Components to rectangles
    rect_dict = {}
    for n, D in enumerate(component_list):
        w, h = (D.size + spacing) / precision
        w, h = int(w), int(h)
        if (w > max_size[0]) or (h > max_size[1]):
            raise ValueError(
                f"pack() failed because Component {D.name!r} has x or y "
                "dimension larger than `max_size` and cannot be packed.\n"
                f"size = {w*precision, h*precision}, max_size = {max_size*precision}"
            )
        rect_dict[n] = (w, h)

    packed_list = []
    while len(rect_dict) > 0:
        (packed_rect_dict, rect_dict) = _pack_single_bin(
            rect_dict,
            aspect_ratio=aspect_ratio,
            max_size=max_size,
            sort_by_area=sort_by_area,
            density=density,
        )
        packed_list.append(packed_rect_dict)

    components_packed_list = []
    index = 0
    for i, rect_dict in enumerate(packed_list):
        name = get_name_short(f"{name_prefix or 'pack'}_{i}")
        packed = Component(name, with_uuid=True)
        packed.info["components"] = {}
        for n, rect in rect_dict.items():
            x, y, w, h = rect
            xcenter = x + w / 2 + spacing / 2
            ycenter = y + h / 2 + spacing / 2
            component = component_list[n]
            d = component.ref(rotation=rotation, h_mirror=h_mirror, v_mirror=v_mirror)
            packed.add(d)
            if hasattr(component, "settings"):
                packed.info["components"][component.name] = component.settings
            d.center = (xcenter * precision, ycenter * precision)

            if text:
                for text_offset, text_anchor in zip(text_offsets, text_anchors):
                    label = packed << text(f"{text_prefix}{index}")
                    label.move(
                        (np.array(text_offset) + getattr(d.size_info, text_anchor))
                    )
                index += 1

        components_packed_list.append(packed)

    if len(components_packed_list) > 1:
        groups = len(components_packed_list)
        warnings.warn(f"unable to pack in one component, creating {groups} components")

    return components_packed_list


def test_pack() -> Component:
    import gdsfactory as gf

    component_list = [
        gf.components.ellipse(radii=tuple(np.random.rand(2) * n + 2)) for n in range(2)
    ]
    component_list += [
        gf.components.rectangle(size=tuple(np.random.rand(2) * n + 2)) for n in range(2)
    ]

    components_packed_list = pack(
        component_list,  # Must be a list or tuple of Components
        spacing=1.25,  # Minimum distance between adjacent shapes
        aspect_ratio=(2, 1),  # (width, height) ratio of the rectangular bin
        max_size=(None, None),  # Limits the size into which the shapes will be packed
        density=1.05,  # Values closer to 1 pack tighter but require more computation
        sort_by_area=True,  # Pre-sorts the shapes by area
    )
    c = components_packed_list[0]  # Only one bin was created, so we plot that
    assert len(c.get_dependencies()) == 4
    return c


def test_pack_with_settings() -> Component:
    import gdsfactory as gf

    component_list = [
        gf.components.rectangle(size=(i, i), port_type=None) for i in range(1, 10)
    ]
    component_list += [
        gf.components.rectangle(size=(i, i), port_type=None) for i in range(1, 10)
    ]

    components_packed_list = pack(
        component_list,  # Must be a list or tuple of Components
        spacing=1.25,  # Minimum distance between adjacent shapes
        aspect_ratio=(2, 1),  # (width, height) ratio of the rectangular bin
        # max_size=(None, None),  # Limits the size into which the shapes will be packed
        max_size=(20, 20),  # Limits the size into which the shapes will be packed
        density=1.05,  # Values closer to 1 pack tighter but require more computation
        sort_by_area=True,  # Pre-sorts the shapes by area
        precision=1e-3,
    )
    c = components_packed_list[0]
    # print(len(c.get_dependencies()))
    return c


if __name__ == "__main__":
    # test_pack()
    import gdsfactory as gf

    # c = test_pack_with_settings()
    # c = test_pack()
    # c.show()
    # c.pprint()
    # c.write_gds_with_metadata("mask.gds")

    p = pack(
        [gf.components.triangle(x=i) for i in range(1, 10)],
        spacing=20.0,
        max_size=(100, 100),
        text=gf.partial(gf.c.text, justify="center"),
        text_prefix="R",
        name_prefix="demo",
        text_anchor="nc",
        text_offset=(-10, 0),
        v_mirror=True,
    )
    c = p[0]
    print(c.name)
    c.show()
