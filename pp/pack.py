""" adapted from phidl.Geometry
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import rectpack
from numpy import ndarray

from pp.component import Component


def _pack_single_bin(
    rect_dict: Dict[int, Tuple[int, int]],
    aspect_ratio: Tuple[int, int],
    max_size: ndarray,
    sort_by_area: bool,
    density: float,
    precision: float,
) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[Any, Any]]:
    """Packs a dict of rectangles {id:(w,h)} and tries to
    pack it into a bin as small as possible with aspect ratio `aspect_ratio`
    Will iteratively grow the bin size until everything fits or the bin size
    reaches `max_size`.

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
        elif all(box_size >= max_size):
            break

    # Separate packed from unpacked rectangles, make dicts of form {id:(x,y,w,h)}
    packed_rect_dict = {r[-1]: r[:-1] for r in rect_packer[0].rect_list()}
    unpacked_rect_dict = {}
    for k, v in rect_dict.items():
        if k not in packed_rect_dict:
            unpacked_rect_dict[k] = v

    return (packed_rect_dict, unpacked_rect_dict)


def pack(
    D_list: List[Component],
    spacing: int = 10,
    aspect_ratio: Tuple[int, int] = (1, 1),
    max_size: Tuple[None, None] = (None, None),
    sort_by_area: bool = True,
    density: float = 1.1,
    precision: float = 1e-2,
) -> List[Component]:
    """Pack a list of components into as few Components as possible.

    Args:
        D_list: Must be a list or tuple of Components
        spacing: Minimum distance between adjacent shapes
        aspect_ratio: (width, height) ratio of the rectangular bin
        max_size: Limits the size into which the shapes will be packed
        density:  Values closer to 1 pack tighter but require more computation
        sort_by_area (Boolean): Pre-sorts the shapes by area
    """

    if density < 1.01:
        raise ValueError(
            "pack() was given a `density` argument that is"
            + " too small.  The density argument must be >= 1.01"
        )

    # Santize max_size variable
    max_size = [np.inf if v is None else v for v in max_size]
    max_size = np.asarray(max_size, dtype=np.float64)  # In case it's integers
    max_size = max_size / precision

    # Convert Components to rectangles
    rect_dict = {}
    for n, D in enumerate(D_list):
        w, h = (D.size + spacing) / precision
        w, h = int(w), int(h)
        if (w > max_size[0]) or (h > max_size[1]):
            raise ValueError(
                "pack() failed because one of the objects (D)"
                + "in `D_list` is has an x or y dimension larger than `max_size` and "
                + "so cannot be packed"
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
            precision=precision,
        )
        packed_list.append(packed_rect_dict)

    D_packed_list = []
    for rect_dict in packed_list:
        D_packed = Component()
        for n, rect in rect_dict.items():
            x, y, w, h = rect
            xcenter = x + w / 2 + spacing / 2
            ycenter = y + h / 2 + spacing / 2
            d = D_packed.add_ref(D_list[n])
            d.center = (xcenter * precision, ycenter * precision)
        D_packed_list.append(D_packed)

    return D_packed_list


def _demo():
    import phidl.geometry as pg

    import pp

    D_list = [pg.ellipse(radii=np.random.rand(2) * n + 2) for n in range(50)]
    D_list += [pg.rectangle(size=np.random.rand(2) * n + 2) for n in range(50)]

    D_packed_list = pack(
        D_list,  # Must be a list or tuple of Components
        spacing=1.25,  # Minimum distance between adjacent shapes
        aspect_ratio=(2, 1),  # (width, height) ratio of the rectangular bin
        max_size=(None, None),  # Limits the size into which the shapes will be packed
        density=1.05,  # Values closer to 1 pack tighter but require more computation
        sort_by_area=True,  # Pre-sorts the shapes by area
    )
    D = D_packed_list[0]  # Only one bin was created, so we plot that
    pp.show(D)  # show it in klayout


def test_pack() -> None:
    import phidl.geometry as pg

    D_list = [pg.ellipse(radii=np.random.rand(2) * n + 2) for n in range(2)]
    D_list += [pg.rectangle(size=np.random.rand(2) * n + 2) for n in range(2)]

    D_packed_list = pack(
        D_list,  # Must be a list or tuple of Components
        spacing=1.25,  # Minimum distance between adjacent shapes
        aspect_ratio=(2, 1),  # (width, height) ratio of the rectangular bin
        max_size=(None, None),  # Limits the size into which the shapes will be packed
        density=1.05,  # Values closer to 1 pack tighter but require more computation
        sort_by_area=True,  # Pre-sorts the shapes by area
    )
    c = D_packed_list[0]  # Only one bin was created, so we plot that
    # print(len(c.get_dependencies()))
    assert len(c.get_dependencies()) == 4


if __name__ == "__main__":
    test_pack()

    # import phidl.geometry as pg
    # spacing = 1
    # ellipses = pack(
    #     [pg.ellipse(radii=np.random.rand(2) * n + 2) for n in range(50)],
    #     spacing=spacing,
    # )[0]
    # ellipses.name = "ellipses"
    # rectangles = pack(
    #     [pg.rectangle(size=np.random.rand(2) * n + 2) for n in range(50)],
    #     spacing=spacing,
    # )[0]
    # rectangles.name = "rectangles"
    # p = pack([ellipses, rectangles])
    # pp.show(p[0])
