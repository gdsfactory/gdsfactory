"""pack a list of components into as few components as possible.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import Anchor, ComponentSpec, Float2, Number, Size


def _pack_single_bin(
    rect_dict: dict[int, tuple[Number, Number]],
    aspect_ratio: tuple[Number, Number],
    max_size: Size,
    sort_by_area: bool,
    density: float,
) -> tuple[dict[int, tuple[Number, Number, Number, Number]], dict[Any, Any]]:
    """Packs a dict of rectangles {id:(w,h)} and tries to.

    Pack it into a bin as small as possible with aspect ratio `aspect_ratio`
    Will iteratively grow the bin size until everything fits or the bin size
    reaches `max_size`.

    Args:
        rect_dict: dict of rectangles {id: (w, h)} to pack.
        aspect_ratio: x, y.
        max_size: tuple of max X, Y size.
        sort_by_area: sorts components by area.
        density: of packing, closer to 1 packs tighter (more compute heavy).

    Returns:
        packed rectangles dict {id:(x,y,w,h)}. dict of remaining unpacked rectangles.

    """
    import rectpack

    # Compute total area and use it for an initial estimate of the bin size
    total_area = sum(r[0] * r[1] for r in rect_dict.values())
    aspect_ratio = np.asarray(aspect_ratio) / np.linalg.norm(aspect_ratio)  # Normalize

    # Setup variables
    box_size = np.asarray(aspect_ratio * np.sqrt(total_area), dtype=np.float64)
    box_size = np.clip(box_size, None, max_size)
    rp_sort = rectpack.SORT_AREA if sort_by_area else rectpack.SORT_NONE
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
    unpacked_rect_dict = {
        k: v for k, v in rect_dict.items() if k not in packed_rect_dict
    }

    return packed_rect_dict, unpacked_rect_dict


class TextFunction(Protocol):
    def __call__(self, text: str) -> Component: ...


def pack(
    component_list: Sequence[ComponentSpec],
    spacing: float = 10.0,
    aspect_ratio: Float2 = (1.0, 1.0),
    max_size: tuple[float | None, float | None] = (None, None),
    sort_by_area: bool = True,
    density: float = 1.1,
    precision: float = 1e-2,
    text: TextFunction | None = None,
    text_prefix: str = "",
    text_mirror: bool = False,
    text_rotation: int = 0,
    text_offsets: tuple[Float2, ...] = ((0, 0),),
    text_anchors: tuple[Anchor, ...] = ("cc",),
    name_prefix: str | None = None,
    rotation: int = 0,
    h_mirror: bool = False,
    v_mirror: bool = False,
    add_ports_prefix: bool = True,
    add_ports_suffix: bool = False,
) -> list[Component]:
    """Pack a list of components into as few Components as possible.

    Args:
        component_list: list or tuple.
        spacing: Minimum distance between adjacent shapes.
        aspect_ratio: (width, height) ratio of the rectangular bin.
        max_size: Limits the size into which the shapes will be packed.
        sort_by_area: Pre-sorts the shapes by area.
        density: Values closer to 1 pack tighter but require more computation.
        precision: Desired precision for rounding vertex coordinates.
        text: Optional function to add text labels.
        text_prefix: for labels. For example. 'A' will produce 'A1', 'A2', ...
        text_mirror: if True mirrors text.
        text_rotation: Optional text rotation.
        text_offsets: relative to component size info anchor. Defaults to center.
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        name_prefix: for each packed component (avoids the Unnamed cells warning). \
                Note that the suffix contains a uuid so the name will not be deterministic.
        rotation: optional component rotation in degrees.
        h_mirror: horizontal mirror in y axis (x, 1) (1, 0). This is the most common.
        v_mirror: vertical mirror using x axis (1, y) (0, y).
        add_ports_prefix: adds port names with prefix.
        add_ports_suffix: adds port names with suffix.

    .. plot::
        :include-source:

        import gdsfactory as gf
        from functools import partial

        components = [gf.components.triangle(x=i) for i in range(1, 10)]
        c = gf.pack(
            components,
            spacing=20.0,
            max_size=(100, 100),
            text=partial(gf.components.text, justify="center"),
            text_prefix="R",
            name_prefix="demo",
            text_anchors=["nc"],
            text_offsets=[(-10, 0)],
            v_mirror=True,
        )
        c[0].plot()

    """
    if density < 1.01:
        raise ValueError(
            "pack() `density` argument is too small. "
            "The density argument must be >= 1.01"
        )

    # Sanitize max_size variable
    max_size_filtered = tuple(np.inf if v is None else v for v in max_size)
    max_size_array: npt.NDArray[np.floating[Any]] = np.asarray(
        max_size_filtered, dtype=np.float64
    )  # In case it's integers
    max_size_array = max_size_array / precision
    max_size_tuple: tuple[float, float] = tuple(max_size_array)  # type: ignore[assignment]

    components = [gf.get_component(component) for component in component_list]

    # Convert Components to rectangles
    rect_dict: dict[int, tuple[Number, Number]] = {}
    for n, D in enumerate(components):
        size = np.array([D.dxsize, D.dysize])
        w, h = (size + spacing) / precision
        w, h = int(w), int(h)
        if (w > max_size_tuple[0]) or (h > max_size_tuple[1]):
            raise ValueError(
                f"pack() failed because Component {D.name!r} has x or y "
                "dimension larger than `max_size` and cannot be packed.\n"
                f"size = {(w * precision, h * precision)}, max_size = {max_size_array}"
            )
        rect_dict[n] = (w, h)

    packed_list: list[dict[int, tuple[Number, Number, Number, Number]]] = []
    while rect_dict:
        (packed_rect_dict, rect_dict) = _pack_single_bin(
            rect_dict,
            aspect_ratio=aspect_ratio,
            max_size=max_size_tuple,
            sort_by_area=sort_by_area,
            density=density,
        )
        packed_list.append(packed_rect_dict)

    components_packed_list: list[Component] = []
    index = 0
    for rect_dict_ in packed_list:
        packed = Component()
        for n, rect in rect_dict_.items():
            x, y, w, h = rect
            xcenter = x + w / 2 + spacing / 2
            ycenter = y + h / 2 + spacing / 2
            component = components[n]
            d = packed << component
            if rotation:
                d.drotate(rotation)
            if h_mirror:
                d.mirror_x()
            if v_mirror:
                d.mirror_y()
            d.dcenter = tuple(snap_to_grid((xcenter * precision, ycenter * precision)))  # type: ignore[assignment]
            if add_ports_prefix:
                packed.add_ports(d.ports, prefix=f"{index}_")
            elif add_ports_suffix:
                packed.add_ports(d.ports, suffix=f"_{index}")
            else:
                try:
                    packed.add_ports(d.ports)
                except ValueError:
                    packed.add_ports(d.ports, suffix=f"_{index}")

            index += 1

            if text:
                for text_offset, text_anchor in zip(text_offsets, text_anchors):
                    label = packed << text(f"{text_prefix}{index}")
                    if text_mirror:
                        label.dmirror()
                    if text_rotation:
                        label.drotate(text_rotation)
                    label.dmove(
                        np.array(text_offset) + getattr(d.dsize_info, text_anchor)
                    )

        components_packed_list.append(packed)

    if len(components_packed_list) > 1:
        groups = len(components_packed_list)
        warnings.warn(f"unable to pack in one component, creating {groups} components")

    return components_packed_list


if __name__ == "__main__":
    # test_pack()
    component_list = [
        gf.components.ellipse(
            radii=(np.random.rand() * n + 2, np.random.rand() * n + 2)
        )
        for n in range(10)
    ]
    component_list += [
        gf.components.rectangle(
            size=(np.random.rand() * n + 2, np.random.rand() * n + 2)
        )
        for n in range(10)
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

    # p = pack(
    #     [gf.components.straight(length=i) for i in [1, 1]],
    #     spacing=20.0,
    #     max_size=(100, 100),
    #     text=partial(gf.components.text, justify="center"),
    #     text_prefix="R",
    #     name_prefix="demo",
    #     text_anchors=["nc"],
    #     text_offsets=[(-10, 0)],
    #     text_mirror=True,
    #     v_mirror=True,
    # )
    # c = p[0]
    c.show()
